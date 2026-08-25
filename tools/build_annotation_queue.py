"""为 YOLO Pose 任务生成可复现的人工复核优先队列。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple


@dataclass
class FrameCandidate:
    image_id: str
    video_id: str
    frame_index: int
    image_path: str
    label_path: str
    detection_count: int
    small_object_fraction: float
    edge_object_fraction: float
    overlapping_object_fraction: float
    mean_box_area: float
    priority_score: float = 0.0
    review_reasons: Tuple[str, ...] = ()


def _box_iou(first: Tuple[float, float, float, float],
             second: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    right, bottom = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def parse_yolo_boxes(label_path: Path) -> List[Tuple[float, float, float, float]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"{label_path}:{line_number} 字段不足")
        try:
            x, y, width, height = map(float, fields[1:5])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} 坐标不是数字") from exc
        if width <= 0 or height <= 0 or not all(0 <= value <= 1 for value in (x, y, width, height)):
            raise ValueError(f"{label_path}:{line_number} 归一化框无效")
        boxes.append((x, y, width, height))
    return boxes


def frame_features(boxes: List[Tuple[float, float, float, float]],
                   small_area: float = 0.001,
                   edge_margin: float = 0.05,
                   overlap_iou: float = 0.10) -> Dict[str, float]:
    count = len(boxes)
    if count == 0:
        return {"detection_count": 0, "small_object_fraction": 0.0,
                "edge_object_fraction": 0.0, "overlapping_object_fraction": 0.0,
                "mean_box_area": 0.0}
    small = sum(width * height <= small_area for _, _, width, height in boxes)
    edge = sum(x - width / 2 <= edge_margin or y - height / 2 <= edge_margin or
               x + width / 2 >= 1 - edge_margin or y + height / 2 >= 1 - edge_margin
               for x, y, width, height in boxes)
    overlapping = set()
    for first in range(count):
        for second in range(first + 1, count):
            if _box_iou(boxes[first], boxes[second]) >= overlap_iou:
                overlapping.update((first, second))
    return {
        "detection_count": count,
        "small_object_fraction": small / count,
        "edge_object_fraction": edge / count,
        "overlapping_object_fraction": len(overlapping) / count,
        "mean_box_area": sum(width * height for _, _, width, height in boxes) / count,
    }


def load_task_candidates(task_dir: Path) -> List[FrameCandidate]:
    map_path = task_dir / "annotation_map.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"缺少 {map_path}")
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    images = mapping.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("annotation_map.json 不包含 images")
    image_paths = {item.stem: item for item in task_dir.glob("images/**/*") if item.is_file()}
    label_paths = {item.stem: item for item in task_dir.glob("labels/**/*.txt")}
    candidates = []
    for image_id, metadata in images.items():
        image_path = image_paths.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"找不到图像: {image_id}")
        label_path = label_paths.get(image_id, task_dir / "labels" / f"{image_id}.txt")
        features = frame_features(parse_yolo_boxes(label_path))
        candidates.append(FrameCandidate(
            image_id=image_id,
            video_id=str(metadata["video_id"]),
            frame_index=int(metadata["frame_index"]),
            image_path=str(image_path.relative_to(task_dir)).replace("\\", "/"),
            label_path=(str(label_path.relative_to(task_dir)).replace("\\", "/")
                        if label_path.is_file() else ""),
            **features,
        ))
    return sorted(candidates, key=lambda item: (item.video_id, item.frame_index))


def _normalize(values: Iterable[float]) -> List[float]:
    values = list(values)
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def score_candidates(candidates: List[FrameCandidate]) -> List[FrameCandidate]:
    if not candidates:
        return []
    count_scores = _normalize(item.detection_count for item in candidates)
    area_difficulty = _normalize(-item.mean_box_area for item in candidates)
    count_median = median(item.detection_count for item in candidates)
    for item, count_score, area_score in zip(candidates, count_scores, area_difficulty):
        possible_miss = 1.0 if item.detection_count == 0 else max(
            0.0, (count_median - item.detection_count) / max(count_median, 1))
        item.priority_score = round(
            0.25 * count_score +
            0.25 * item.overlapping_object_fraction +
            0.20 * item.small_object_fraction +
            0.10 * item.edge_object_fraction +
            0.10 * area_score +
            0.10 * possible_miss,
            6,
        )
        reasons = []
        if item.overlapping_object_fraction >= 0.2:
            reasons.append("目标重叠或遮挡")
        if item.small_object_fraction >= 0.4:
            reasons.append("小目标较多")
        if item.edge_object_fraction >= 0.2:
            reasons.append("边缘目标需查漏")
        if item.detection_count >= max(count_median, 1):
            reasons.append("高密度画面")
        if possible_miss >= 0.5:
            reasons.append("低检出画面需检查漏标")
        item.review_reasons = tuple(reasons or ("时间覆盖样本",))
    return candidates


def select_diverse_queue(candidates: List[FrameCandidate], count: int,
                         min_frame_gap: int) -> List[FrameCandidate]:
    if count < 1 or min_frame_gap < 0:
        raise ValueError("count 必须为正数且 min_frame_gap 不能为负数")
    ranked = sorted(candidates, key=lambda item: (-item.priority_score, item.frame_index))
    selected = []
    # 先按时间轴分层，避免队列全部集中在同一小段或同类画面。
    videos: Dict[str, List[FrameCandidate]] = {}
    for item in candidates:
        videos.setdefault(item.video_id, []).append(item)
    if len(videos) > 1:
        video_items = sorted(videos.items())
        base_quota, remainder = divmod(count, len(video_items))
        balanced = []
        for index, (_, items) in enumerate(video_items):
            quota = min(base_quota + int(index < remainder), len(items))
            if quota:
                balanced.extend(select_diverse_queue(items, quota, min_frame_gap))
        if len(balanced) < min(count, len(candidates)):
            chosen = {item.image_id for item in balanced}
            for item in ranked:
                if item.image_id not in chosen:
                    balanced.append(item)
                    chosen.add(item.image_id)
                    if len(balanced) == min(count, len(candidates)):
                        break
        return sorted(balanced[:count],
                      key=lambda value: (-value.priority_score, value.video_id, value.frame_index))
    if len(videos) == 1 and count <= len(candidates):
        timeline = next(iter(videos.values()))
        first_frame = min(item.frame_index for item in timeline)
        last_frame = max(item.frame_index for item in timeline)
        span = max(last_frame - first_frame + 1, 1)
        bins: List[List[FrameCandidate]] = [[] for _ in range(count)]
        for item in timeline:
            bucket = min(int((item.frame_index - first_frame) / span * count), count - 1)
            bins[bucket].append(item)
        for bucket in bins:
            for item in sorted(bucket, key=lambda value: (-value.priority_score, value.frame_index)):
                if all(abs(item.frame_index - previous.frame_index) >= min_frame_gap
                       for previous in selected):
                    selected.append(item)
                    break
    if len(selected) >= count:
        return sorted(selected[:count], key=lambda value: (-value.priority_score, value.frame_index))
    for item in ranked:
        if item in selected:
            continue
        if all(item.video_id != previous.video_id or
               abs(item.frame_index - previous.frame_index) >= min_frame_gap
               for previous in selected):
            selected.append(item)
            if len(selected) == count:
                return sorted(selected, key=lambda value: (-value.priority_score, value.frame_index))
    for item in ranked:
        if item not in selected:
            selected.append(item)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda value: (-value.priority_score, value.frame_index))


def write_queue(task_dir: Path, selected: List[FrameCandidate], output_dir: Path,
                total_candidates: int, min_frame_gap: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_images = output_dir / "selected_images"
    selected_labels = output_dir / "selected_labels"
    selected_images.mkdir(exist_ok=True)
    selected_labels.mkdir(exist_ok=True)
    for index, item in enumerate(selected, 1):
        image_source = task_dir / item.image_path
        image_target = selected_images / f"{index:02d}{image_source.suffix.lower()}"
        shutil.copy2(image_source, image_target)
        if item.label_path:
            label_source = task_dir / item.label_path
            label_target = selected_labels / f"{index:02d}.txt"
            shutil.copy2(label_source, label_target)

    payload = {
        "schema_version": "1.0",
        "method": "heuristic_review_priority",
        "is_model_uncertainty": False,
        "task_dir": str(task_dir.resolve()),
        "total_candidates": total_candidates,
        "selected_count": len(selected),
        "min_frame_gap": min_frame_gap,
        "limitations": "YOLO Pose 文本不含预测置信度；本队列依据框密度、重叠、小目标、边缘目标和时间分散度排序。",
        "queue": [{"rank": index, **asdict(item)} for index, item in enumerate(selected, 1)],
    }
    (output_dir / "review_queue.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["rank", "image_id", "video_id", "frame_index", "priority_score",
              "detection_count", "small_object_fraction", "edge_object_fraction",
              "overlapping_object_fraction", "review_reasons", "image_path", "label_path"]
    with (output_dir / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(selected, 1):
            row = asdict(item)
            row["rank"] = index
            row["review_reasons"] = "；".join(item.review_reasons)
            writer.writerow({key: row[key] for key in fields})
    lines = ["# 人工复核优先队列", "",
             "> 本清单是启发式复核顺序，不是模型不确定性或准确率结果。", "",
             "| 顺序 | 帧号 | 候选框 | 优先分 | 复核原因 |", "|---:|---:|---:|---:|---|"]
    for index, item in enumerate(selected, 1):
        lines.append(f"| {index} | {item.frame_index} | {item.detection_count} | "
                     f"{item.priority_score:.3f} | {'；'.join(item.review_reasons)} |")
    lines.extend(["", "复核时需要删除误检框、补充漏检框，并为每个有效实例标注 head、thorax、abdomen_tip。", ""])
    (output_dir / "review_queue.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 YOLO Pose 人工复核优先队列")
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--min-frame-gap", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = score_candidates(load_task_candidates(args.task_dir))
    selected = select_diverse_queue(candidates, args.count, args.min_frame_gap)
    write_queue(args.task_dir, selected, args.output, len(candidates), args.min_frame_gap)
    print(json.dumps({"selected": len(selected), "total": len(candidates),
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
