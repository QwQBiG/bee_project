"""使用现有 YOLO 权重生成统一格式的预标注，供人工校正。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, VideoAnnotation


CLASS_ALIASES = {
    "bee": "bee",
    "workerbee": "worker_bee",
    "worker": "worker_bee",
    "dronebee": "drone_bee",
    "drone": "drone_bee",
    "queenbee": "queen_bee",
    "queen": "queen_bee",
    "pollenbee": "pollen_bee",
    "pollenloadedbee": "pollen_bee",
    "varroamite": "varroa_mite",
    "varroa": "varroa_mite",
}


def canonical_class(name: str) -> str:
    key = "".join(character for character in name.lower() if character.isalnum())
    return CLASS_ALIASES.get(key, "bee")


def frame_indices(record: Dict, frames_dir: Path) -> Iterable[tuple[int, Path]]:
    video_dir = frames_dir / record["video_id"]
    for frame_index in record.get("planned_frames", []):
        stem = f"frame_{int(frame_index):08d}"
        image = next((video_dir / f"{stem}{suffix}" for suffix in (".jpg", ".jpeg", ".png")
                      if (video_dir / f"{stem}{suffix}").exists()), None)
        if image is None:
            raise FileNotFoundError(f"找不到预标注抽帧: {video_dir / stem}")
        yield int(frame_index), image


def result_instances(result, frame_index: int, names: Dict[int, str]) -> List[BeeInstance]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    keypoints = getattr(result, "keypoints", None)
    keypoint_xy = keypoints.xy.detach().cpu().tolist() if keypoints is not None else []
    confidence_tensor = getattr(keypoints, "conf", None)
    keypoint_conf = confidence_tensor.detach().cpu().tolist() if confidence_tensor is not None else []
    instances = []
    for index, (coordinates, class_id, confidence) in enumerate(zip(xyxy, classes, confidences)):
        x1, y1, x2, y2 = coordinates
        points = []
        if index < len(keypoint_xy):
            labels = ("head", "thorax", "abdomen_tip")
            for point_index, (x, y) in enumerate(keypoint_xy[index][:3]):
                score = keypoint_conf[index][point_index] if index < len(keypoint_conf) else 1.0
                visibility = 2 if score >= 0.5 else 1 if score > 0 else 0
                points.append(Keypoint(labels[point_index], float(x), float(y), visibility))
        instances.append(BeeInstance(
            instance_id=f"{frame_index}-{index}",
            bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            category=canonical_class(names.get(int(class_id), "bee")),
            keypoints=points, source="prediction", confidence=float(confidence),
        ))
    return instances


def prelabel(manifest_path: Path, frames_dir: Path, model_path: Path,
             output_dir: Path, device: str | None = None, confidence: float = 0.15,
             scene: str | None = None, imgsz: int | None = None, preprocess: str = "raw") -> Dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("预标注需要安装 ultralytics") from error
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = YOLO(str(model_path))
    name_items = model.names.items() if hasattr(model.names, "items") else enumerate(model.names)
    names = {int(key): str(value) for key, value in name_items}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"videos": 0, "frames": 0, "instances": 0}
    if preprocess not in {"raw", "inside_ir_demo"}:
        raise ValueError(f"不支持的预处理模式: {preprocess}")
    enhancer = None
    cv2 = None
    if preprocess == "inside_ir_demo":
        import cv2 as cv2_module
        from tracking.inside_tracker import InfraredImageEnhancer
        cv2 = cv2_module
        enhancer = InfraredImageEnhancer()
    for record in manifest.get("videos", []):
        if scene is not None and record.get("scene") != scene:
            continue
        frames = []
        for frame_index, image in frame_indices(record, frames_dir):
            source = str(image)
            if enhancer is not None:
                frame = cv2.imread(str(image))
                if frame is None:
                    raise RuntimeError(f"无法读取预标注图片: {image}")
                source = enhancer.denoise(
                    enhancer.enhance(frame))
            predict_args = {"source": source, "conf": confidence,
                            "device": device, "verbose": False}
            if imgsz is not None:
                predict_args["imgsz"] = imgsz
            results = model.predict(**predict_args)
            instances = result_instances(results[0], frame_index, names)
            timestamp = frame_index / float(record["fps"]) * 1000.0
            frames.append(FrameAnnotation(frame_index, timestamp, instances))
            summary["frames"] += 1
            summary["instances"] += len(instances)
        annotation = VideoAnnotation(
            video_id=record["video_id"], source_path=record["source_path"],
            scene=record.get("scene", "unknown"), width=int(record["width"]),
            height=int(record["height"]), fps=float(record["fps"]),
            frame_count=int(record["frame_count"]), frames=frames,
            metadata={"split": record.get("split", "unassigned"),
                      "sha256": record.get("sha256"), "prelabel_model": str(model_path),
                      "prelabel_imgsz": imgsz,
                      "prelabel_preprocess": preprocess},
        )
        annotation.save(output_dir / f"{record['video_id']}.json")
        summary["videos"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 YOLO 生成待人工校正的蜜蜂预标注")
    parser.add_argument("manifest")
    parser.add_argument("frames")
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--device")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, help="YOLO 推理尺寸；小目标场景可对照 640 与 1280")
    parser.add_argument("--scene",
                        choices=["inside_ir", "inside_visible", "outside_entrance"],
                        help="只处理指定场景，便于巢内外分别使用对应权重")
    parser.add_argument("--preprocess", choices=["raw", "inside_ir_demo"], default="raw",
                        help="巢内任务使用 inside_ir_demo 以对齐 Demo 的增强与降噪流程")
    args = parser.parse_args()
    try:
        summary = prelabel(Path(args.manifest), Path(args.frames), Path(args.model),
                           Path(args.output), args.device, args.confidence,
                           args.scene, args.imgsz, args.preprocess)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"预标注完成：{summary['frames']} 帧、{summary['instances']} 个候选实例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
