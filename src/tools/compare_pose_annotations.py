"""比较两份人工姿态标注并生成裁决清单。"""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from annotation.schema import BeeInstance, VideoAnnotation


def bbox_iou(first: List[float], second: List[float]) -> float:
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(first: List[BeeInstance], second: List[BeeInstance],
                 min_iou: float) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    candidates = sorted(
        ((bbox_iou(left.bbox, right.bbox), left_index, right_index)
         for left_index, left in enumerate(first)
         for right_index, right in enumerate(second)),
        reverse=True,
    )
    used_first, used_second, matches = set(), set(), []
    for overlap, left_index, right_index in candidates:
        if overlap < min_iou:
            break
        if left_index in used_first or right_index in used_second:
            continue
        used_first.add(left_index)
        used_second.add(right_index)
        matches.append((left_index, right_index, overlap))
    return (matches,
            [index for index in range(len(first)) if index not in used_first],
            [index for index in range(len(second)) if index not in used_second])


def _point_map(instance: BeeInstance) -> Dict[str, object]:
    return {point.name: point for point in instance.keypoints}


def compare_pair(first: BeeInstance, second: BeeInstance, overlap: float,
                 max_keypoint_error: float) -> Dict:
    first_points, second_points = _point_map(first), _point_map(second)
    names = ("head", "thorax", "abdomen_tip")
    diagonal = max(hypot((first.bbox[2] + second.bbox[2]) / 2,
                         (first.bbox[3] + second.bbox[3]) / 2), 1.0)
    point_errors, visibility_matches, missing = {}, [], []
    for name in names:
        if name not in first_points or name not in second_points:
            missing.append(name)
            continue
        left, right = first_points[name], second_points[name]
        point_errors[name] = hypot(left.x - right.x, left.y - right.y) / diagonal
        visibility_matches.append(left.visibility == right.visibility)
    reversal = False
    if all(name in first_points and name in second_points for name in ("head", "abdomen_tip")):
        direct = (hypot(first_points["head"].x - second_points["head"].x,
                        first_points["head"].y - second_points["head"].y) +
                  hypot(first_points["abdomen_tip"].x - second_points["abdomen_tip"].x,
                        first_points["abdomen_tip"].y - second_points["abdomen_tip"].y))
        swapped = (hypot(first_points["head"].x - second_points["abdomen_tip"].x,
                         first_points["head"].y - second_points["abdomen_tip"].y) +
                   hypot(first_points["abdomen_tip"].x - second_points["head"].x,
                         first_points["abdomen_tip"].y - second_points["head"].y))
        reversal = swapped + 0.05 * diagonal < direct
    issues = []
    if overlap < 0.7:
        issues.append("bbox_disagreement")
    if missing:
        issues.append("missing_keypoints")
    if any(error > max_keypoint_error for error in point_errors.values()):
        issues.append("keypoint_disagreement")
    if visibility_matches and not all(visibility_matches):
        issues.append("visibility_disagreement")
    if reversal:
        issues.append("head_tail_reversal")
    return {
        "first_instance_id": first.instance_id,
        "second_instance_id": second.instance_id,
        "bbox_iou": round(overlap, 6),
        "keypoint_normalized_errors": {key: round(value, 6)
                                       for key, value in point_errors.items()},
        "missing_keypoints": missing,
        "visibility_agreement": (sum(visibility_matches) / len(visibility_matches)
                                 if visibility_matches else None),
        "head_tail_reversal": reversal,
        "issues": issues,
    }


def compare_annotations(first: VideoAnnotation, second: VideoAnnotation,
                        min_match_iou: float = 0.3,
                        max_keypoint_error: float = 0.15) -> Dict:
    if not 0 <= min_match_iou <= 1 or max_keypoint_error < 0:
        raise ValueError("比较阈值无效")
    for field in ("video_id", "width", "height", "frame_count"):
        if getattr(first, field) != getattr(second, field):
            raise ValueError(f"两份标注的 {field} 不一致")
    errors = first.validate(require_manual=True) + second.validate(require_manual=True)
    if errors:
        raise ValueError("输入不是有效人工标注：" + "；".join(errors[:10]))
    first_frames = {frame.frame_index: frame for frame in first.frames}
    second_frames = {frame.frame_index: frame for frame in second.frames}
    frame_ids = sorted(set(first_frames) | set(second_frames))
    frame_reports, all_ious, all_point_errors, visibility, reversals = [], [], [], [], 0
    unmatched_first_total = unmatched_second_total = 0
    for frame_id in frame_ids:
        left = first_frames.get(frame_id)
        right = second_frames.get(frame_id)
        left_instances = left.instances if left else []
        right_instances = right.instances if right else []
        matches, unmatched_left, unmatched_right = greedy_match(
            left_instances, right_instances, min_match_iou)
        pairs = []
        for left_index, right_index, overlap in matches:
            pair = compare_pair(left_instances[left_index], right_instances[right_index],
                                overlap, max_keypoint_error)
            pairs.append(pair)
            all_ious.append(overlap)
            all_point_errors.extend(pair["keypoint_normalized_errors"].values())
            if pair["visibility_agreement"] is not None:
                visibility.append(pair["visibility_agreement"])
            reversals += int(pair["head_tail_reversal"])
        unmatched_first_total += len(unmatched_left)
        unmatched_second_total += len(unmatched_right)
        needs_adjudication = bool(unmatched_left or unmatched_right or
                                  any(pair["issues"] for pair in pairs))
        frame_reports.append({
            "frame_index": frame_id,
            "first_instance_count": len(left_instances),
            "second_instance_count": len(right_instances),
            "matched_pairs": pairs,
            "unmatched_first": [left_instances[index].instance_id for index in unmatched_left],
            "unmatched_second": [right_instances[index].instance_id for index in unmatched_right],
            "needs_adjudication": needs_adjudication,
        })
    adjudication = [item["frame_index"] for item in frame_reports if item["needs_adjudication"]]
    return {
        "report_type": "manual_pose_annotation_agreement_screening",
        "is_formal_model_metric": False,
        "video_id": first.video_id,
        "thresholds": {"min_match_iou": min_match_iou,
                       "max_keypoint_normalized_error": max_keypoint_error},
        "summary": {
            "frames_compared": len(frame_ids),
            "matched_instances": len(all_ious),
            "unmatched_first": unmatched_first_total,
            "unmatched_second": unmatched_second_total,
            "mean_bbox_iou": round(mean(all_ious), 6) if all_ious else None,
            "mean_keypoint_normalized_error": (round(mean(all_point_errors), 6)
                                                if all_point_errors else None),
            "mean_visibility_agreement": round(mean(visibility), 6) if visibility else None,
            "head_tail_reversal_count": reversals,
            "adjudication_frame_count": len(adjudication),
            "adjudication_frames": adjudication,
        },
        "frames": frame_reports,
        "limitations": "使用 IoU 贪心匹配筛查双标差异；高密度歧义帧必须由第三人裁决，不能自动合并。",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两份人工整蜂框与三关键点标注")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-match-iou", type=float, default=0.3)
    parser.add_argument("--max-keypoint-error", type=float, default=0.15)
    parser.add_argument("--strict", action="store_true", help="存在待裁决帧时返回退出码 1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = compare_annotations(
            VideoAnnotation.load(args.first), VideoAnnotation.load(args.second),
            args.min_match_iou, args.max_keypoint_error)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 1 if args.strict and report["summary"]["adjudication_frame_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
