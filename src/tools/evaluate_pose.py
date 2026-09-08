"""评测整蜂框、头胸腹关键点和身体朝向。"""

from __future__ import annotations

import argparse
import json
from math import atan2, degrees, hypot
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import BeeInstance, VideoAnnotation


def bbox_iou(first: List[float], second: List[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_instances(truth: List[BeeInstance], predictions: List[BeeInstance],
                    iou_threshold: float) -> List[Tuple[int, int, float]]:
    candidates = []
    for truth_index, gt_instance in enumerate(truth):
        for prediction_index, predicted in enumerate(predictions):
            overlap = bbox_iou(gt_instance.bbox, predicted.bbox)
            if overlap >= iou_threshold:
                candidates.append((overlap, truth_index, prediction_index))
    used_truth, used_predictions, matches = set(), set(), []
    for overlap, truth_index, prediction_index in sorted(candidates, reverse=True):
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, overlap))
    return matches


def orientation(instance: BeeInstance) -> float | None:
    points = {item.name: item for item in instance.keypoints if item.visibility > 0}
    if "head" not in points or "abdomen_tip" not in points:
        return None
    head, abdomen = points["head"], points["abdomen_tip"]
    if head.x == abdomen.x and head.y == abdomen.y:
        return None
    return degrees(atan2(head.y - abdomen.y, head.x - abdomen.x)) % 360.0


def angle_error(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def evaluate_pose(ground_truth: VideoAnnotation, predictions: VideoAnnotation,
                  iou_threshold: float = 0.5, pck_threshold: float = 0.1) -> Dict:
    if ground_truth.video_id != predictions.video_id:
        raise ValueError("video_id 不一致")
    if ground_truth.width != predictions.width or ground_truth.height != predictions.height:
        raise ValueError("视频尺寸不一致")
    pose_errors = ground_truth.validate_pose_gold()
    if pose_errors:
        raise ValueError("ground_truth 不是有效的姿态金标准标注: " + "; ".join(pose_errors))
    truth_frames = {item.frame_index: item for item in ground_truth.frames}
    prediction_frames = {item.frame_index: item for item in predictions.frames}
    truth_count = prediction_count = matched_count = 0
    normalized_errors, angular_errors, overlaps = [], [], []
    correct_keypoints = total_keypoints = head_tail_reversals = 0
    for frame_index in sorted(set(truth_frames) | set(prediction_frames)):
        truth = truth_frames.get(frame_index)
        predicted = prediction_frames.get(frame_index)
        gt_instances = truth.instances if truth else []
        pred_instances = predicted.instances if predicted else []
        truth_count += len(gt_instances)
        prediction_count += len(pred_instances)
        matches = match_instances(gt_instances, pred_instances, iou_threshold)
        matched_count += len(matches)
        for truth_index, prediction_index, overlap in matches:
            gt_instance, pred_instance = gt_instances[truth_index], pred_instances[prediction_index]
            overlaps.append(overlap)
            gt_points = {item.name: item for item in gt_instance.keypoints if item.visibility > 0}
            pred_points = {item.name: item for item in pred_instance.keypoints if item.visibility > 0}
            diagonal = hypot(gt_instance.bbox[2], gt_instance.bbox[3])
            for name, gt_point in gt_points.items():
                if name not in pred_points or diagonal <= 0:
                    total_keypoints += 1
                    continue
                pred_point = pred_points[name]
                error = hypot(gt_point.x - pred_point.x, gt_point.y - pred_point.y) / diagonal
                normalized_errors.append(error)
                total_keypoints += 1
                correct_keypoints += error <= pck_threshold
            gt_angle, pred_angle = orientation(gt_instance), orientation(pred_instance)
            if gt_angle is not None and pred_angle is not None:
                error = angle_error(gt_angle, pred_angle)
                angular_errors.append(error)
                head_tail_reversals += error > 90.0
    precision = matched_count / max(prediction_count, 1)
    recall = matched_count / max(truth_count, 1)
    return {
        "ground_truth_instances": truth_count,
        "predicted_instances": prediction_count,
        "matched_instances": matched_count,
        "detection_precision_at_iou": round(precision, 6),
        "detection_recall_at_iou": round(recall, 6),
        "mean_matched_iou": round(statistics.mean(overlaps), 6) if overlaps else None,
        "pck_threshold": pck_threshold,
        "pck": round(correct_keypoints / max(total_keypoints, 1), 6),
        "keypoints_evaluated": total_keypoints,
        "mean_normalized_keypoint_error": round(statistics.mean(normalized_errors), 6)
        if normalized_errors else None,
        "orientation_samples": len(angular_errors),
        "mean_orientation_error_degrees": round(statistics.mean(angular_errors), 6)
        if angular_errors else None,
        "head_tail_reversal_rate": round(head_tail_reversals / max(len(angular_errors), 1), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评测头胸腹关键点与朝向")
    parser.add_argument("ground_truth")
    parser.add_argument("predictions")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--pck", type=float, default=0.1)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        report = evaluate_pose(VideoAnnotation.load(args.ground_truth),
                               VideoAnnotation.load(args.predictions), args.iou, args.pck)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
