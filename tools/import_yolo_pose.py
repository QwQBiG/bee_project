"""将 CVAT 导出的 Ultralytics YOLO Pose 标注转回统一 JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import BeeInstance, FrameAnnotation, Keypoint, KEYPOINT_NAMES, VideoAnnotation


def parse_pose_row(line: str, width: int, height: int, classes: List[str],
                   frame_index: int, row_index: int,
                   source: str = "interpolated") -> BeeInstance:
    fields = line.split()
    expected = 5 + len(KEYPOINT_NAMES) * 3
    if len(fields) not in {expected, expected + 1}:
        raise ValueError(f"帧 {frame_index} 第 {row_index + 1} 行字段数应为 {expected} 或 {expected + 1}")
    try:
        values = [float(value) for value in fields]
    except ValueError as error:
        raise ValueError(f"帧 {frame_index} 第 {row_index + 1} 行包含非数字") from error
    class_id = int(values[0])
    if values[0] != class_id or not 0 <= class_id < len(classes):
        raise ValueError(f"帧 {frame_index} 第 {row_index + 1} 行类别无效")
    cx, cy, box_width, box_height = values[1:5]
    x = (cx - box_width / 2) * width
    y = (cy - box_height / 2) * height
    bbox = [x, y, box_width * width, box_height * height]
    keypoints = []
    offset = 5
    for point_index, name in enumerate(KEYPOINT_NAMES):
        px, py, raw_visibility = values[offset + point_index * 3:offset + point_index * 3 + 3]
        visibility = int(raw_visibility)
        if raw_visibility != visibility or visibility not in {0, 1, 2}:
            raise ValueError(f"帧 {frame_index} 第 {row_index + 1} 行关键点可见性无效")
        if visibility:
            keypoints.append(Keypoint(name, px * width, py * height, visibility))
    track_id = int(values[expected]) if len(values) == expected + 1 else None
    return BeeInstance(
        instance_id=f"{frame_index}-{row_index}",
        bbox=bbox,
        category=classes[class_id],
        track_id=track_id,
        keypoints=keypoints,
        source=source,
    )


def import_dataset(dataset_root: Path, mapping_path: Path, output_root: Path,
                   reviewed: bool = False) -> Dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    classes = [str(item) for item in mapping["classes"]]
    videos = mapping.get("videos", {})
    images = mapping.get("images", {})
    if not videos or not images:
        raise ValueError("annotation_map.json 缺少 videos 或 images")
    label_index: Dict[str, Path] = {}
    for label_path in dataset_root.rglob("*.txt"):
        if label_path.name in {"train.txt", "val.txt", "test.txt"}:
            continue
        if label_path.stem in label_index:
            raise ValueError(f"重复标注文件: {label_path.stem}")
        label_index[label_path.stem] = label_path
    grouped: Dict[str, List[FrameAnnotation]] = {video_id: [] for video_id in videos}
    total_instances = 0
    source = "manual" if reviewed else "interpolated"
    for stem, image_info in images.items():
        if stem not in label_index:
            raise FileNotFoundError(f"找不到标注文件: {stem}.txt")
        video_id = str(image_info["video_id"])
        frame_index = int(image_info["frame_index"])
        width, height = int(image_info["width"]), int(image_info["height"])
        lines = [line.strip() for line in label_index[stem].read_text(
            encoding="utf-8").splitlines() if line.strip()]
        instances = [parse_pose_row(line, width, height, classes, frame_index, index, source)
                     for index, line in enumerate(lines)]
        fps = float(videos[video_id]["fps"])
        grouped[video_id].append(FrameAnnotation(
            frame_index, frame_index / fps * 1000.0, instances))
        total_instances += len(instances)
    output_root.mkdir(parents=True, exist_ok=True)
    for video_id, frames in grouped.items():
        video = videos[video_id]
        metadata = dict(video.get("metadata", {}))
        metadata["annotation_import"] = "cvat_yolo_pose"
        metadata["review_status"] = "reviewed" if reviewed else "unreviewed"
        annotation = VideoAnnotation(
            video_id=video_id,
            source_path=str(video["source_path"]),
            scene=str(video["scene"]),
            width=int(video["width"]),
            height=int(video["height"]),
            fps=float(video["fps"]),
            frame_count=int(video["frame_count"]),
            frames=sorted(frames, key=lambda item: item.frame_index),
            metadata=metadata,
            schema_version=str(video.get("schema_version", "1.0")),
        )
        errors = annotation.validate(require_manual=reviewed)
        if errors:
            raise ValueError(f"{video_id}: {'; '.join(errors)}")
        annotation.save(output_root / f"{video_id}.json")
    return {"videos": len(grouped), "frames": len(images), "instances": total_instances}


def main() -> int:
    parser = argparse.ArgumentParser(description="导入 CVAT YOLO Pose 标注")
    parser.add_argument("dataset", help="CVAT 导出的目录或 ZIP")
    parser.add_argument("mapping", help="任务包中的 annotation_map.json")
    parser.add_argument("output", help="统一 JSON 输出目录")
    parser.add_argument("--reviewed", action="store_true",
                        help="仅在任务已完成人工标注和复核后使用；否则保持非金标准来源")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    try:
        if dataset.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as directory:
                shutil.unpack_archive(str(dataset), directory)
                summary = import_dataset(
                    Path(directory), Path(args.mapping), Path(args.output), args.reviewed)
        else:
            summary = import_dataset(dataset, Path(args.mapping), Path(args.output), args.reviewed)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"已导入 {summary['videos']} 个视频、{summary['frames']} 帧、{summary['instances']} 个实例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
