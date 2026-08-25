"""比较人工进出事件与算法预测，输出事件级 Precision、Recall 和 F1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import TemporalEvent, VideoAnnotation


def crossing_frame(event: TemporalEvent) -> int:
    value = event.attributes.get("crossing_frame")
    return int(value) if value is not None else (event.start_frame + event.end_frame) // 2


def _match_type(ground_truth: List[TemporalEvent], predictions: List[TemporalEvent],
                tolerance_frames: int, require_track_id: bool) -> Dict:
    candidates: List[Tuple[int, int, int]] = []
    for gt_index, truth in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            difference = abs(crossing_frame(truth) - crossing_frame(prediction))
            same_track = bool(set(truth.track_ids) & set(prediction.track_ids))
            if difference <= tolerance_frames and (same_track or not require_track_id):
                candidates.append((difference, gt_index, pred_index))
    matched_truth, matched_prediction, matches = set(), set(), []
    for difference, gt_index, pred_index in sorted(candidates):
        if gt_index in matched_truth or pred_index in matched_prediction:
            continue
        matched_truth.add(gt_index)
        matched_prediction.add(pred_index)
        matches.append({"ground_truth_index": gt_index, "prediction_index": pred_index,
                        "frame_error": difference})
    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "count_absolute_error": abs(len(predictions) - len(ground_truth)),
        "matches": matches,
    }


def evaluate_events(ground_truth: Iterable[TemporalEvent], predictions: Iterable[TemporalEvent],
                    tolerance_frames: int = 5, require_track_id: bool = False) -> Dict:
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames 不能为负数")
    truth = [item for item in ground_truth if item.event_type in {"entering", "leaving"}]
    predicted = [item for item in predictions if item.event_type in {"entering", "leaving"}]
    per_type = {}
    for event_type in ("entering", "leaving"):
        per_type[event_type] = _match_type(
            [item for item in truth if item.event_type == event_type],
            [item for item in predicted if item.event_type == event_type],
            tolerance_frames, require_track_id)
    totals = {name: sum(item[name] for item in per_type.values())
              for name in ("true_positive", "false_positive", "false_negative")}
    precision = totals["true_positive"] / max(totals["true_positive"] + totals["false_positive"], 1)
    recall = totals["true_positive"] / max(totals["true_positive"] + totals["false_negative"], 1)
    totals.update({
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 6),
        "count_absolute_error": abs(len(predicted) - len(truth)),
    })
    return {"tolerance_frames": tolerance_frames, "require_track_id": require_track_id,
            "overall": totals, "per_type": per_type}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测进出巢事件预测")
    parser.add_argument("ground_truth")
    parser.add_argument("predictions")
    parser.add_argument("--tolerance-frames", type=int, default=5)
    parser.add_argument("--require-track-id", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    truth = VideoAnnotation.load(args.ground_truth)
    predictions = VideoAnnotation.load(args.predictions)
    if truth.video_id != predictions.video_id:
        print("ERROR: 两个标注文件的 video_id 不一致", file=sys.stderr)
        return 2
    report = evaluate_events(truth.events, predictions.events,
                             args.tolerance_frames, args.require_track_id)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
