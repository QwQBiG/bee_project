"""制作低工作量的单蜂姿态种子标注任务。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DATA_YAML = """path: ./
train: train.txt
kpt_shape: [3, 3]
flip_idx: [0, 1, 2]
names:
  0: bee
"""


def map_apic_keypoints(raw: dict[str, Any]) -> list[tuple[float, float]]:
    """将 APIC 的 32 点映射为 head、thorax、abdomen_tip。"""
    try:
        head_parts = [raw[str(index)] for index in range(4)]
        thorax = raw["4"]
        abdomen_tip = raw["5"]
    except (KeyError, TypeError) as error:
        raise ValueError("APIC 标注缺少头、胸或腹部末端关键点") from error
    head = (
        sum(float(point[0]) for point in head_parts) / len(head_parts),
        sum(float(point[1]) for point in head_parts) / len(head_parts),
    )
    return [head, (float(thorax[0]), float(thorax[1])),
            (float(abdomen_tip[0]), float(abdomen_tip[1]))]


def _point_extent(raw: dict[str, Any], width: int, height: int,
                  padding: float) -> tuple[float, float, float, float]:
    points = [(float(value[0]), float(value[1])) for value in raw.values()]
    x1 = max(0.0, min(point[0] for point in points) - padding)
    y1 = max(0.0, min(point[1] for point in points) - padding)
    x2 = min(float(width), max(point[0] for point in points) + padding)
    y2 = min(float(height), max(point[1] for point in points) + padding)
    return x1, y1, x2, y2


def make_pose_row(raw: dict[str, Any], width: int, height: int) -> str:
    points = map_apic_keypoints(raw)
    body_length = math.dist(points[0], points[2])
    x1, y1, x2, y2 = _point_extent(
        raw, width, height, max(6.0, body_length * 0.08))
    values: list[float | int] = [
        0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
        (x2 - x1) / width, (y2 - y1) / height,
    ]
    for x, y in points:
        values.extend((max(0.0, min(x / width, 1.0)),
                       max(0.0, min(y / height, 1.0)), 2))
    return " ".join(
        str(value) if isinstance(value, int) else f"{value:.6f}"
        for value in values)


def select_apic_records(dataset_root: Path, detector_path: Path, count: int,
                        device: str | None = None) -> list[dict[str, Any]]:
    import cv2
    from ultralytics import YOLO

    records = json.loads(
        (dataset_root / "data" / "pose_dataset.json").read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for record in records:
        image_path = dataset_root / "data" / record["path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        try:
            points = map_apic_keypoints(record["keypoints"])
        except ValueError:
            continue
        body_length = math.dist(points[0], points[2])
        if not 45 <= body_length <= min(width, height) * 0.82:
            continue
        if any(not (5 <= x < width - 5 and 5 <= y < height - 5)
               for x, y in points):
            continue
        angle = math.atan2(points[0][1] - points[2][1],
                           points[0][0] - points[2][0])
        angle_bin = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
        sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
        if sharpness < 8:
            continue
        contrast = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std())
        candidate = dict(record)
        candidate.update({"image_path": image_path, "width": width,
                          "height": height, "body_length": body_length,
                          "sharpness": sharpness, "contrast": contrast,
                          "angle_bin": angle_bin})
        candidates.append(candidate)

    model = YOLO(str(detector_path))
    predict_args: dict[str, Any] = {
        "source": [str(item["image_path"]) for item in candidates],
        "conf": 0.10, "imgsz": 640, "agnostic_nms": True, "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device
    results = model.predict(**predict_args)
    single_bee: list[dict[str, Any]] = []
    for candidate, result in zip(candidates, results):
        boxes = getattr(result, "boxes", None)
        confidences = (boxes.conf.detach().cpu().tolist()
                       if boxes is not None and boxes.conf is not None else [])
        detected_bees = sum(confidence >= 0.20 for confidence in confidences)
        if detected_bees != 1:
            continue
        candidate["detected_bees"] = detected_bees
        candidate["quality"] = (
            math.log1p(candidate["sharpness"])
            + candidate["body_length"] / 180
            + candidate["contrast"] / 100)
        single_bee.append(candidate)
    single_bee.sort(key=lambda item: item["quality"], reverse=True)

    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    angle_counts: dict[int, int] = defaultdict(int)
    for source_limit, angle_limit in ((1, 4), (2, 6), (3, count)):
        for candidate in single_bee:
            if candidate in selected:
                continue
            source_group = Path(candidate["path"]).stem.split("_bee_id")[0]
            angle_bin = candidate["angle_bin"]
            if (source_counts[source_group] >= source_limit
                    or angle_counts[angle_bin] >= angle_limit):
                continue
            source_counts[source_group] += 1
            angle_counts[angle_bin] += 1
            selected.append(candidate)
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"检测器确认的清晰单蜂 APIC 样本只有 {len(selected)} 张")
    return selected


def write_cvat_archive(output: Path,
                       items: Iterable[tuple[str, bytes, str]]) -> dict[str, int]:
    materialized = list(items)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", DATA_YAML.encode("utf-8"))
        archive.writestr(
            "train.txt", "".join(
                f"./images/train/{name}\n" for name, _, _ in materialized
            ).encode("utf-8"))
        for name, image_bytes, label in materialized:
            archive.writestr(f"images/train/{name}", image_bytes)
            archive.writestr(
                f"labels/train/{Path(name).stem}.txt",
                (label.rstrip() + "\n" if label.strip() else "").encode("utf-8"))
    return {"images": len(materialized),
            "annotation_rows": sum(bool(label.strip()) for _, _, label in materialized)}


def extract_ir_single_bees(video_path: Path, detector_path: Path, count: int,
                            sample_count: int, device: str | None = None
                            ) -> tuple[list[tuple[str, bytes, str]], list[dict[str, Any]]]:
    """用检测框挑选居中且邻居尽量少的红外单蜂裁剪图。"""
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from tracking.inside_tracker import InfraredImageEnhancer

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"无法读取红外视频: {video_path}")
    indices = np.linspace(0, frame_count - 1,
                          min(sample_count, frame_count), dtype=int).tolist()
    enhancer = InfraredImageEnhancer()
    frames: list[Any] = []
    valid_indices: list[int] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        frames.append(enhancer.denoise(enhancer.enhance(frame)))
        valid_indices.append(frame_index)
    capture.release()
    if not frames:
        raise RuntimeError("红外视频没有可读取帧")

    model = YOLO(str(detector_path))
    predict_args: dict[str, Any] = {
        "source": frames, "conf": 0.18, "imgsz": 960, "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device
    results = model.predict(**predict_args)
    candidates: list[dict[str, Any]] = []
    for frame_index, frame, result in zip(valid_indices, frames, results):
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                   for box in coordinates]
        height, width = frame.shape[:2]
        frame_candidates: list[dict[str, Any]] = []
        for box_index, (box, confidence) in enumerate(zip(coordinates, confidences)):
            if confidence < 0.35:
                continue
            x1, y1, x2, y2 = box
            box_width, box_height = x2 - x1, y2 - y1
            longest = max(box_width, box_height)
            if longest < 24 or min(box_width, box_height) < 12:
                continue
            side = int(max(96, min(224, math.ceil(longest * 1.55))))
            center_x, center_y = centers[box_index]
            crop_x = int(round(center_x - side / 2))
            crop_y = int(round(center_y - side / 2))
            if crop_x < 0 or crop_y < 0 or crop_x + side > width or crop_y + side > height:
                continue
            neighbors = sum(
                index != box_index
                and abs(other_x - center_x) < side * 0.43
                and abs(other_y - center_y) < side * 0.43
                for index, (other_x, other_y) in enumerate(centers))
            crop = frame[crop_y:crop_y + side, crop_x:crop_x + side]
            crop = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_CUBIC)
            sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            score = confidence * 3 + min(sharpness / 1800, 2) - neighbors * 0.8
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            frame_candidates.append({
                "frame_index": frame_index, "crop": [crop_x, crop_y, side, side],
                "center": [center_x, center_y], "confidence": float(confidence),
                "neighbors": int(neighbors), "sharpness": sharpness,
                "score": score, "bytes": encoded.tobytes(),
            })
        if frame_candidates:
            candidates.append(max(frame_candidates, key=lambda item: item["score"]))

    candidates.sort(key=lambda item: (item["neighbors"], -item["score"]))
    selected: list[dict[str, Any]] = []
    minimum_gap = max(2, frame_count // max(count * 5, 1))
    for candidate in candidates:
        if any(abs(candidate["frame_index"] - item["frame_index"]) < minimum_gap
               for item in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"只筛到 {len(selected)} 张合格红外单蜂图，目标为 {count} 张")
    selected.sort(key=lambda item: item["frame_index"])
    items, records = [], []
    for index, candidate in enumerate(selected, 1):
        name = f"02_ir_oist_{index:03d}.jpg"
        items.append((name, candidate.pop("bytes"), ""))
        candidate.pop("center", None)
        candidate.pop("score", None)
        records.append({"task_file": name, **candidate,
                        "annotation_policy": "仅标中央完整蜜蜂；边缘残蜂忽略"})
    return items, records


def build_task(apic_root: Path, ir_video: Path, detector: Path,
               visible_detector: Path,
               output_root: Path, visible_count: int = 20, ir_count: int = 12,
               ir_samples: int = 64, device: str | None = None,
               overwrite: bool = False) -> dict[str, Any]:
    archive_path = output_root / (
        f"cvat_easy_pose_seed_{visible_count + ir_count}_single_task.zip")
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在: {archive_path}")
    if not (apic_root / "data" / "pose_dataset.json").is_file():
        raise FileNotFoundError(f"找不到 APIC 数据集: {apic_root}")
    if not ir_video.is_file():
        raise FileNotFoundError(ir_video)
    if not detector.is_file():
        raise FileNotFoundError(detector)
    if not visible_detector.is_file():
        raise FileNotFoundError(visible_detector)

    preview_visible = output_root / "preview" / "visible"
    preview_ir = output_root / "preview" / "ir"
    preview_visible.mkdir(parents=True, exist_ok=True)
    preview_ir.mkdir(parents=True, exist_ok=True)
    selected_visible = select_apic_records(
        apic_root, visible_detector, visible_count, device)
    visible_items: list[tuple[str, bytes, str]] = []
    visible_records: list[dict[str, Any]] = []
    for index, record in enumerate(selected_visible, 1):
        suffix = Path(record["path"]).suffix.lower()
        name = f"01_visible_apic_{index:03d}{suffix}"
        image_bytes = Path(record["image_path"]).read_bytes()
        label = make_pose_row(record["keypoints"], record["width"], record["height"])
        visible_items.append((name, image_bytes, label))
        (preview_visible / name).write_bytes(image_bytes)
        visible_records.append({
            "task_file": name, "source_path": record["path"],
            "source_split": record.get("set"),
            "body_length_px": round(record["body_length"], 3),
            "sharpness": round(record["sharpness"], 3),
            "mapping": {"head": "mean(0..3)", "thorax": "4",
                        "abdomen_tip": "5"},
            "annotation_policy": "检查并纠正已有三点；不要无故重画",
        })

    ir_items, ir_records = extract_ir_single_bees(
        ir_video, detector, ir_count, ir_samples, device)
    for name, image_bytes, _ in ir_items:
        (preview_ir / name).write_bytes(image_bytes)
    package_report = write_cvat_archive(
        archive_path, [*visible_items, *ir_items])
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "version": "1.0", "purpose": "easy_pose_seed_review",
        "archive": archive_path.name, "sha256": digest,
        "keypoints": ["head", "thorax", "abdomen_tip"],
        "counts": {"total_images": visible_count + ir_count,
                   "visible_with_human_source_prelabels": visible_count,
                   "ir_manual_central_bee": ir_count,
                   **package_report},
        "sources": {
            "visible": {
                "name": "apic.ai bee pose dataset", "records": visible_records,
                "selection_detector": str(visible_detector),
                "url": "https://github.com/apic-ai/apic-bee-pose-dataset",
                "license": "CC BY-NC-SA 4.0; academic/non-commercial use only",
                "attribution": "apic.ai bee pose dataset by apic.ai GmbH",
            },
            "ir": {
                "name": "OIST M13 waggle-dance infrared supplemental video",
                "source_path": str(ir_video), "detector": str(detector),
                "records": ir_records,
                "url": "https://www.oist.jp/research/research-units/bptu/honeybee_tracking_datasets",
                "note": "仅限本地研究与比赛验证；不要把第三方原视频或裁剪图提交到公开仓库",
            },
        },
    }
    manifest_path = output_root / "task_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    guide_path = output_root / "使用说明.txt"
    guide_path.write_text(
        f"只上传这一个文件：{archive_path.name}\n\n"
        f"它只含 train 一个分组，因此会生成 1 个 CVAT Task，不会再出现 train/val 两份。\n"
        f"第 1–{visible_count} 张：已有真实来源三点预标注，只检查头、胸、腹末端并纠错。\n"
        f"第 {visible_count + 1}–{visible_count + ir_count} 张：每张只标中央完整蜜蜂，"
        "边缘截断的蜜蜂忽略。\n"
        "先完成这一小批，不需要现在硬做 60 或 100 张。导出格式选择 "
        "Ultralytics YOLO Pose 1.0。\n",
        encoding="utf-8")
    return {"archive": str(archive_path), "manifest": str(manifest_path),
            "guide": str(guide_path), "sha256": digest, **package_report}


def main() -> int:
    parser = argparse.ArgumentParser(description="制作清晰、低密度的姿态种子标注任务")
    parser.add_argument("--apic-root", type=Path, required=True)
    parser.add_argument("--ir-video", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--visible-detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-count", type=int, default=20)
    parser.add_argument("--ir-count", type=int, default=12)
    parser.add_argument("--ir-samples", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = build_task(
            args.apic_root, args.ir_video, args.detector, args.visible_detector,
            args.output,
            args.visible_count, args.ir_count, args.ir_samples,
            args.device, args.force)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError,
            OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
