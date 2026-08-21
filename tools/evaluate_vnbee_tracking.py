"""使用 VnBeeTracking GroundTruth 评估巢外检测与多目标跟踪。

标注格式为 MOT 风格：frame,id,x,y,w,h,conf,class,visibility。
脚本只依赖项目现有的 OpenCV、NumPy、SciPy 和 Ultralytics。
"""

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.outside_tracker import OutsideHiveTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt", required=True, help="VnBeeTracking GroundTruth .txt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=0, help="0 表示使用视频全部帧")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    return parser.parse_args()


def load_ground_truth(path: str) -> Dict[int, List[dict]]:
    """读取 frame,id,x,y,w,h,... 格式的 GroundTruth。"""
    frames: Dict[int, List[dict]] = defaultdict(list)
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [item.strip() for item in line.replace(" ", ",").split(",") if item.strip()]
        if len(fields) < 6:
            raise ValueError(f"GT 第 {line_number} 行字段不足：{raw_line!r}")
        try:
            frame_id, track_id = int(float(fields[0])), int(float(fields[1]))
            bbox = [float(value) for value in fields[2:6]]
        except ValueError as exc:
            raise ValueError(f"GT 第 {line_number} 行无法解析：{raw_line!r}") from exc
        frames[frame_id].append({"track_id": track_id, "bbox": bbox})
    if not frames:
        raise ValueError(f"GT 文件没有有效标注：{path}")
    return dict(frames)


def iou_xywh(first: List[float], second: List[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + max(aw, 0.0), ay1 + max(ah, 0.0)
    bx2, by2 = bx1 + max(bw, 0.0), by1 + max(bh, 0.0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - intersection
    return intersection / union if union > 0 else 0.0


def match_frame(gt_rows: List[dict], predictions: List[dict], threshold: float):
    if not gt_rows or not predictions:
        return [], list(range(len(gt_rows))), list(range(len(predictions)))
    matrix = np.array(
        [[iou_xywh(gt["bbox"], pred["bbox"]) for pred in predictions] for gt in gt_rows],
        dtype=np.float32,
    )
    gt_indices, pred_indices = linear_sum_assignment(1.0 - matrix)
    matches = [
        (int(gt_index), int(pred_index), float(matrix[gt_index, pred_index]))
        for gt_index, pred_index in zip(gt_indices, pred_indices)
        if matrix[gt_index, pred_index] >= threshold
    ]
    matched_gt = {item[0] for item in matches}
    matched_pred = {item[1] for item in matches}
    return (
        matches,
        [index for index in range(len(gt_rows)) if index not in matched_gt],
        [index for index in range(len(predictions)) if index not in matched_pred],
    )


def build_tracker_config(config_path: str, tracker_type: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    section = copy.deepcopy(config.get("outside_tracker") or config.get("tracker") or {})
    section["tracker_type"] = tracker_type
    section["tracker_config"] = f"{tracker_type}.yaml"
    return section


def evaluate(args: argparse.Namespace) -> dict:
    ground_truth = load_ground_truth(args.gt)
    config = build_tracker_config(args.config, args.tracker)
    tracker = OutsideHiveTracker(config)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_limit = args.frames if args.frames > 0 else None
    frame_number = 0
    gt_total = 0
    detector_total = detector_tp = detector_fp = detector_fn = 0
    detector_iou_sum = 0.0
    track_total = track_tp = track_fp = track_fn = 0
    track_iou_sum = 0.0
    id_switches = 0
    previous_assignment: Dict[int, int] = {}
    association_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    started = time.perf_counter()

    while frame_limit is None or frame_number < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        tracks, detections = tracker.process_frame(frame)
        detector_predictions = [
            {"track_id": -1, "bbox": list(detection["bbox"])}
            for detection in detections
        ]
        predictions = [
            {"track_id": int(track.track_id), "bbox": list(track.bbox)}
            for track in tracks
        ]
        gt_rows = ground_truth.get(frame_number, [])
        detector_matches, detector_unmatched_gt, detector_unmatched_pred = match_frame(
            gt_rows, detector_predictions, args.iou
        )
        matches, unmatched_gt, unmatched_pred = match_frame(gt_rows, predictions, args.iou)
        gt_total += len(gt_rows)
        detector_total += len(detector_predictions)
        detector_tp += len(detector_matches)
        detector_fn += len(detector_unmatched_gt)
        detector_fp += len(detector_unmatched_pred)
        detector_iou_sum += sum(item[2] for item in detector_matches)
        track_total += len(predictions)
        track_tp += len(matches)
        track_fn += len(unmatched_gt)
        track_fp += len(unmatched_pred)
        track_iou_sum += sum(item[2] for item in matches)

        current_assignment = {}
        for gt_index, pred_index, _ in matches:
            gt_id = int(gt_rows[gt_index]["track_id"])
            pred_id = int(predictions[pred_index]["track_id"])
            if gt_id in previous_assignment and previous_assignment[gt_id] != pred_id:
                id_switches += 1
            current_assignment[gt_id] = pred_id
            association_counts[(gt_id, pred_id)] += 1
        previous_assignment = current_assignment

    capture.release()
    elapsed = time.perf_counter() - started

    gt_ids = sorted({gt_id for gt_id, _ in association_counts})
    pred_ids = sorted({pred_id for _, pred_id in association_counts})
    association_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    gt_index = {track_id: index for index, track_id in enumerate(gt_ids)}
    pred_index = {track_id: index for index, track_id in enumerate(pred_ids)}
    for (gt_id, pred_id), count in association_counts.items():
        association_matrix[gt_index[gt_id], pred_index[pred_id]] = count
    if association_matrix.size:
        idtp = int(association_matrix[linear_sum_assignment(-association_matrix)].sum())
    else:
        idtp = 0
    idfn = gt_total - idtp
    idfp = track_total - idtp

    detector_precision = detector_tp / detector_total if detector_total else 0.0
    detector_recall = detector_tp / gt_total if gt_total else 0.0
    detector_f1 = (
        2 * detector_precision * detector_recall / (detector_precision + detector_recall)
        if detector_precision + detector_recall else 0.0
    )
    track_precision = track_tp / track_total if track_total else 0.0
    track_recall = track_tp / gt_total if gt_total else 0.0
    track_f1 = (
        2 * track_precision * track_recall / (track_precision + track_recall)
        if track_precision + track_recall else 0.0
    )
    mota = 1.0 - (track_fn + track_fp + id_switches) / gt_total if gt_total else 0.0
    idf1 = 2 * idtp / (2 * idtp + idfn + idfp) if (2 * idtp + idfn + idfp) else 0.0
    return {
        "tracker": args.tracker,
        "video": args.video,
        "ground_truth": args.gt,
        "frames_evaluated": frame_number,
        "source_fps": source_fps,
        "processing_seconds": elapsed,
        "throughput_fps": frame_number / elapsed if elapsed else None,
        "iou_threshold": args.iou,
        "ground_truth_objects": gt_total,
        "predicted_detections": detector_total,
        "detector_precision": detector_precision,
        "detector_recall": detector_recall,
        "detector_f1": detector_f1,
        "detector_mean_matched_iou": detector_iou_sum / detector_tp if detector_tp else 0.0,
        "predicted_tracks": track_total,
        "tracking_precision": track_precision,
        "tracking_recall": track_recall,
        "tracking_f1": track_f1,
        "tracking_mean_matched_iou": track_iou_sum / track_tp if track_tp else 0.0,
        "mota": mota,
        "idf1": idf1,
        "id_switches": id_switches,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "metric_note": "MOTA/IDF1 are computed on IoU-matched detections; HOTA is not included in this dependency-free baseline.",
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
