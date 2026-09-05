"""Measure the actual ONNX/IoU batch baseline on a paired VnBee clip.

This diagnostic uses the existing local video and matching GT. It neither
trains nor tunes models. The short clip is not the official hidden test set.
"""
import argparse
import json
import tempfile
from pathlib import Path

import cv2

from inference.algorithm_cli import load_runtime_config, run_detection, match_ious
from tools.evaluate_vnbee_tracking import load_ground_truth, compute_map50_95, match_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_runtime_config(str(root / "configs/algorithm_config.json"))
    config["_scene"] = "outside"
    gt = load_ground_truth(str(root / "tmp/2022-04-08-12-30.txt"))
    capture = cv2.VideoCapture(str(root / "data/external/vnbee/2022-04-08-12-30.mp4"))
    output = root / "output/batch_baseline_20260905"
    output.mkdir(parents=True, exist_ok=True)
    predictions, ground_truth = {}, {}
    previous_boxes, previous_ids, assignment = [], [], {}
    next_id, fp, fn, switches, total_gt = 1, 0, 0, 0, 0
    try:
        with tempfile.TemporaryDirectory(dir=output) as temp:
            for frame_id in range(1, args.frames + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                path = Path(temp) / "im_0001.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError("image write failed")
                rows, _ = run_detection(path, config, conf_override=0, topk=200)
                predictions[frame_id], ground_truth[frame_id] = rows, gt.get(frame_id, [])
                decided = [row for row in rows if row["confidence"] >= config["detector"]["outside"]["conf"]]
                boxes = [row["bbox"] for row in decided]
                links, ids = match_ious(boxes, previous_boxes), []
                for index in range(len(boxes)):
                    if index in links:
                        ids.append(previous_ids[links[index]])
                    else:
                        ids.append(next_id)
                        next_id += 1
                matches, missed, extra = match_frame(ground_truth[frame_id], decided, .5)
                fp, fn = fp + len(extra), fn + len(missed)
                total_gt += len(ground_truth[frame_id])
                for gi, pi, _ in matches:
                    gid = ground_truth[frame_id][gi]["track_id"]
                    switches += int(gid in assignment and assignment[gid] != ids[pi])
                    assignment[gid] = ids[pi]
                previous_boxes, previous_ids = boxes, ids
    finally:
        capture.release()
    result = {"frames": len(predictions), "gt_objects": total_gt,
              "detection": compute_map50_95(ground_truth, predictions),
              "tracking_iou_baseline": {"fp": fp, "fn": fn, "id_switches": switches,
                  "mota_diagnostic": 1 - (fp + fn + switches) / total_gt if total_gt else None},
              "note": "Short public clip diagnostic; not official TrackEval or competition score."}
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
