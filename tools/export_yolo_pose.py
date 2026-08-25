"""将统一标注导出为 YOLO Pose 数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import BeeInstance, KEYPOINT_NAMES, VideoAnnotation


DEFAULT_CLASSES = ["bee", "worker_bee", "drone_bee", "queen_bee", "pollen_bee", "varroa_mite"]


def normalize(value: float, maximum: int) -> float:
    return max(0.0, min(float(value) / maximum, 1.0))


def pose_row(instance: BeeInstance, width: int, height: int,
             class_to_id: Dict[str, int], category_override: str | None = None,
             include_track_id: bool = False) -> str:
    category = category_override or instance.category
    if category not in class_to_id:
        raise ValueError(f"未知类别: {category}")
    x, y, box_width, box_height = instance.bbox
    values: List[float | int] = [
        class_to_id[category],
        normalize(x + box_width / 2, width), normalize(y + box_height / 2, height),
        normalize(box_width, width), normalize(box_height, height),
    ]
    keypoints = {item.name: item for item in instance.keypoints}
    for name in KEYPOINT_NAMES:
        keypoint = keypoints.get(name)
        if keypoint is None or keypoint.visibility == 0:
            values.extend((0.0, 0.0, 0))
        else:
            values.extend((normalize(keypoint.x, width), normalize(keypoint.y, height),
                           keypoint.visibility))
    if include_track_id and instance.track_id is not None:
        values.append(instance.track_id)
    return " ".join(str(value) if isinstance(value, int) else f"{value:.6f}" for value in values)


def find_frame(frames_root: Path, video_id: str, frame_index: int) -> Path:
    stem = f"frame_{frame_index:08d}"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = frames_root / video_id / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到抽帧图片: {frames_root / video_id / stem}")


def export_annotations(annotation_files: Iterable[Path], frames_root: Path,
                       output_root: Path, include_non_manual: bool = False,
                       split_override: str | None = None,
                       scene_filter: str | None = None,
                       collapse_to_bee: bool = False,
                       include_track_ids: bool = False) -> Dict:
    export_classes = ["bee"] if collapse_to_bee else DEFAULT_CLASSES
    class_to_id = {name: index for index, name in enumerate(export_classes)}
    summary = {"videos": 0, "frames": 0, "instances": 0, "skipped_instances": 0}
    source_counts = {"manual": 0, "prediction": 0, "interpolated": 0}
    pose_incomplete = 0
    subset_images = {"train": [], "val": [], "test": []}
    image_map = {}
    video_map = {}
    for annotation_file in annotation_files:
        annotation = VideoAnnotation.load(annotation_file)
        if scene_filter is not None and annotation.scene != scene_filter:
            continue
        video_map[annotation.video_id] = {
            "source_path": annotation.source_path,
            "scene": annotation.scene,
            "width": annotation.width,
            "height": annotation.height,
            "fps": annotation.fps,
            "frame_count": annotation.frame_count,
            "metadata": annotation.metadata,
            "schema_version": annotation.schema_version,
        }
        errors = annotation.validate(require_manual=False)
        if errors:
            raise ValueError(f"{annotation_file}: {'; '.join(errors)}")
        split = split_override or str(annotation.metadata.get("split", "train"))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"{annotation_file}: split 无效: {split}")
        summary["videos"] += 1
        for frame in annotation.frames:
            image_source = find_frame(frames_root, annotation.video_id, frame.frame_index)
            output_name = f"{annotation.video_id}_{frame.frame_index:08d}{image_source.suffix.lower()}"
            image_output = output_root / "images" / split / output_name
            label_output = output_root / "labels" / split / f"{Path(output_name).stem}.txt"
            subset_images[split].append(image_output.relative_to(output_root).as_posix())
            image_map[Path(output_name).stem] = {
                "video_id": annotation.video_id,
                "frame_index": frame.frame_index,
                "width": annotation.width,
                "height": annotation.height,
            }
            rows = []
            for instance in frame.instances:
                if instance.source != "manual" and not include_non_manual:
                    summary["skipped_instances"] += 1
                    continue
                source_counts[instance.source] += 1
                points = {point.name: point for point in instance.keypoints}
                if (set(points) != set(KEYPOINT_NAMES)
                        or points.get("head") is None or points["head"].visibility == 0
                        or points.get("abdomen_tip") is None
                        or points["abdomen_tip"].visibility == 0):
                    pose_incomplete += 1
                category_override = "bee" if collapse_to_bee else None
                rows.append(pose_row(instance, annotation.width, annotation.height,
                                     class_to_id, category_override, include_track_ids))
            image_output.parent.mkdir(parents=True, exist_ok=True)
            label_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_source, image_output)
            label_output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            summary["frames"] += 1
            summary["instances"] += len(rows)
    for subset, images in subset_images.items():
        (output_root / f"{subset}.txt").write_text(
            "\n".join(images) + ("\n" if images else ""), encoding="utf-8")
    mapping = {"schema_version": "1.0", "keypoints": list(KEYPOINT_NAMES),
               "classes": export_classes, "videos": video_map, "images": image_map}
    (output_root / "annotation_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(export_classes))
    yaml_text = ("path: ./\n"
                 "train: train.txt\nval: val.txt\ntest: test.txt\n"
                 "kpt_shape: [3, 3]\nflip_idx: [0, 1, 2]\nnames:\n" + names + "\n")
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    pose_labels_ready = (
        summary["instances"] > 0
        and source_counts["prediction"] == 0
        and source_counts["interpolated"] == 0
        and pose_incomplete == 0)
    split_counts = {name: len(images) for name, images in subset_images.items()}
    dataset_meta = {
        "task": "pose",
        "track_ids_exported": include_track_ids,
        "pose_labels_ready": pose_labels_ready,
        "training_ready": pose_labels_ready
        and split_counts["train"] > 0 and split_counts["val"] > 0,
        "source_counts": source_counts,
        "pose_incomplete_instances": pose_incomplete,
        "split_counts": split_counts,
    }
    (output_root / "dataset_meta.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 YOLO Pose 头胸腹关键点数据集")
    parser.add_argument("annotations", help="统一 JSON 标注目录")
    parser.add_argument("frames", help="prepare_unlabeled_dataset.py 生成的 frames 目录")
    parser.add_argument("output", help="YOLO Pose 数据集输出目录")
    parser.add_argument("--include-non-manual", action="store_true",
                        help="包含 prediction/interpolated；正式金标准集不建议使用")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="覆盖原标注 split；制作待标注任务时可使用 train")
    parser.add_argument("--archive",
                        help="可选 ZIP 输出路径，供 CVAT 导入")
    parser.add_argument("--scene",
                        choices=["inside_ir", "inside_visible", "outside_entrance"],
                        help="只导出指定场景")
    parser.add_argument("--collapse-to-bee", action="store_true",
                        help="将模型细分类别统一为 bee，避免把未经确认的蜂种写入姿态任务")
    parser.add_argument("--include-track-ids", action="store_true",
                        help="仅供 CVAT 轨迹交换；正式 Ultralytics 训练标签不得使用")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_source = Path(args.annotations)
    files = ([annotation_source] if annotation_source.is_file()
             else sorted(annotation_source.rglob("*.json")))
    if not files:
        print("ERROR: 没有找到统一标注 JSON", file=sys.stderr)
        return 2
    try:
        summary = export_annotations(files, Path(args.frames), Path(args.output),
                                     args.include_non_manual, args.split, args.scene,
                                     args.collapse_to_bee, args.include_track_ids)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"已导出 {summary['frames']} 帧、{summary['instances']} 个实例")
    if args.archive:
        archive_path = Path(args.archive)
        archive_base = archive_path.with_suffix("") if archive_path.suffix.lower() == ".zip" else archive_path
        result = shutil.make_archive(str(archive_base), "zip", root_dir=Path(args.output))
        print(f"CVAT 任务包: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
