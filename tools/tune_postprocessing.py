"""Small development-only search. Previously viewed holdout is regression only."""
import hashlib
import json
import math
from pathlib import Path

import cv2
from inference.box_calibration import calibrate_detections

from tools.compare_onnx_candidates import detection_metrics, mot_metrics, tracking_predictions
from tools.evaluate_vnbee_tracking import load_ground_truth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/postprocessing_experiment"
BASE = dict(backend="bytetrack", track_high_thresh=.25, track_low_thresh=.05,
            new_track_thresh=.25, track_buffer=10, match_thresh=.8, fuse_score=True)


def resize_predictions(predictions, factor, width, height):
    if not math.isfinite(factor) or factor <= 0 or min(width, height) <= 0:
        raise ValueError("positive finite scale and image dimensions required")
    return {fid: calibrate_detections(rows, factor, width, height)
            for fid, rows in predictions.items()}


def load_predictions(split):
    path = ROOT / f"output/onnx_candidates/original_0_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "artifacts/models/hive_entrance_bee_yolov8n.onnx").read_bytes()).hexdigest()
    if payload["sha256"] != digest:
        raise ValueError("prediction cache weight mismatch")
    return {int(k): v for k, v in payload["predictions"].items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(ROOT / "data/external/vnbee/2022-04-08-12-30.mp4"))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if min(width, height) <= 0:
        raise ValueError("video dimensions unavailable")
    gt = load_ground_truth(str(ROOT / "tmp/2022-04-08-12-30.txt"))
    dev = load_predictions("development")
    dev_gt = {fid: gt.get(fid, []) for fid in dev}
    # Exploratory post-NMS scaling; no confidence deletion or extra candidates.
    scales = []
    for factor in (1., 1.05, 1.1, 1.15):
        metrics = detection_metrics(dev_gt, resize_predictions(dev, factor, width, height))
        scales.append({"scale": factor, "metrics": metrics})
        print(f"scale {factor}: {metrics['map_50_95']:.6f}", flush=True)
    chosen = max(scales, key=lambda x: x["metrics"]["map_50_95"])
    # A small development gain alone is insufficient to change geometry.
    factor = chosen["scale"] if chosen["metrics"]["map_50_95"] >= scales[0]["metrics"]["map_50_95"]+.005 else 1.
    options = [BASE]
    options += [{**BASE, "track_buffer": b} for b in (5, 20)]
    options += [{**BASE, "match_thresh": t} for t in (.7, .9)]
    options += [{**BASE, "fuse_score": False},
                {**BASE, "track_low_thresh": .025},
                {**BASE, "track_high_thresh": .2, "new_track_thresh": .2}]
    ranked = []
    # Keep tracking boxes unchanged to isolate association from geometry.
    for option in options:
        metrics = mot_metrics(dev_gt, tracking_predictions(dev, option))
        ranked.append({"options": option, "metrics": metrics})
        print(f"tracker {option}: {metrics}", flush=True)
    baseline = ranked[0]["metrics"]
    eligible = [r for r in ranked if r["metrics"]["mota"] >= baseline["mota"]
                and r["metrics"]["idf1"] >= baseline["idf1"]]
    best = max(eligible, key=lambda r: r["metrics"]["mota"]+r["metrics"]["idf1"])
    report = {"scales": scales, "tracking": ranked,
              "locked": {"detection_scale": factor, "tracking": best["options"]},
              "note": "Same-video previously viewed regression, not independent validation."}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    held = load_predictions("holdout")
    held_gt = {fid: gt.get(fid, []) for fid in held}
    report["regression"] = {
        "baseline_detection": detection_metrics(held_gt, held),
        "candidate_detection": detection_metrics(held_gt, resize_predictions(held, factor, width, height)),
        "baseline_tracking": mot_metrics(held_gt, tracking_predictions(held, BASE)),
        "candidate_tracking": mot_metrics(held_gt, tracking_predictions(held, best["options"]))}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["regression"], indent=2), flush=True)


if __name__ == "__main__":
    main()
