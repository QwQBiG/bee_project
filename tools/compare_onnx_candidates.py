"""Development-only selection followed by one temporal holdout comparison.

Run with the experiment dependencies on PYTHONPATH. Public GT and predictions
stay under ignored output/. No official competition images are exported.
"""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from inference.algorithm_cli import load_runtime_config, match_ious
from inference.tiled_detection import detect_array
from tools.evaluate_vnbee_tracking import load_ground_truth, compute_map50_95, iou_xywh
from tracking.onnx_bytetrack import OnnxByteTracker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/onnx_candidates"
MODELS = {
    "current": "artifacts/models/bee_outside_ft_v1.onnx",
    "original": "artifacts/models/hive_entrance_bee_yolov8n.onnx",
    "heavy": "runs/outside_ft/n_e30_heavy/weights/best.onnx",
}
WINDOWS = {"development": (1, 120), "holdout": (481, 600)}


def predict(model, tiled, split):
    cfg = load_runtime_config(str(ROOT / "configs/algorithm_config.json"))
    cfg["_scene"], cfg["_device"] = "outside", "cpu"
    cfg["detector"]["outside"]["model"] = str(ROOT / MODELS[model])
    cfg["detector"]["outside"]["class_ids"] = [0, 1, 2, 3] if model == "original" else [0]
    cfg["tiling"] = {"enabled": tiled, "overlap": .2, "include_full": True}
    begin, end = WINDOWS[split]
    digest = hashlib.sha256((ROOT / MODELS[model]).read_bytes()).hexdigest()
    cache = OUT / f"{model}_{int(tiled)}_{split}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data["sha256"] == digest:
            return {int(k): v for k, v in data["predictions"].items()}, data["seconds"]
    capture = cv2.VideoCapture(str(ROOT / "data/external/vnbee/2022-04-08-12-30.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, begin - 1)
    predictions = {}
    started = time.perf_counter()
    try:
        for frame_id in range(begin, end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"missing video frame {frame_id}")
            predictions[frame_id], _ = detect_array(frame, cfg, conf_override=0, topk=600)
    finally:
        capture.release()
    seconds = time.perf_counter() - started
    cache.write_text(json.dumps({"sha256": digest, "seconds": seconds,
                                "predictions": predictions}), encoding="utf-8")
    print(f"inference {model} tiled={tiled} {split}: {seconds:.1f}s", flush=True)
    return predictions, seconds


def tracking_predictions(predictions, options):
    previous_boxes, previous_ids, next_id = [], [], 1
    tracker = OnnxByteTracker(options) if options["backend"] == "bytetrack" else None
    result = {}
    for fid, detections in sorted(predictions.items()):
        if tracker:
            rows = tracker.update(detections)
        else:
            detections = [d for d in detections[:300] if d["confidence"] >= options["threshold"]]
            boxes = [d["bbox"] for d in detections]
            matches, ids = match_ious(boxes, previous_boxes), []
            for index in range(len(boxes)):
                if index in matches:
                    ids.append(previous_ids[matches[index]])
                else:
                    ids.append(next_id)
                    next_id += 1
            rows = [{"track_id": tid, "bbox": box} for tid, box in zip(ids, boxes)]
            previous_boxes, previous_ids = boxes, ids
        result[fid] = rows
    return result


def mot_metrics(gt, predictions):
    import motmetrics as mm
    accumulator = mm.MOTAccumulator(auto_id=True)
    for fid in sorted(gt):
        targets, rows = gt[fid], predictions.get(fid, [])
        distances = np.full((len(targets), len(rows)), np.nan)
        for i, target in enumerate(targets):
            for j, row in enumerate(rows):
                overlap = iou_xywh(target["bbox"], row["bbox"])
                if overlap >= .5:
                    distances[i, j] = 1 - overlap
        accumulator.update([g["track_id"] for g in targets],
                           [p["track_id"] for p in rows], distances)
    metrics = ["mota", "idf1", "num_switches", "precision", "recall", "mostly_tracked", "mostly_lost"]
    frame = mm.metrics.create().compute(accumulator, metrics=metrics, name="result")
    return {name: float(frame.loc["result", name]) for name in metrics}


def detection_metrics(gt, predictions):
    return compute_map50_95(gt, {fid: rows[:200] for fid, rows in predictions.items()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_gt = load_ground_truth(str(ROOT / "tmp/2022-04-08-12-30.txt"))
    ground_truth = {split: {fid: all_gt.get(fid, []) for fid in range(begin, end+1)}
                    for split, (begin, end) in WINDOWS.items()}
    dev = ground_truth["development"]
    report = {"windows": WINDOWS, "device": "cpu", "development": {}}
    candidates = {}
    for model in MODELS:
        rows, seconds = predict(model, False, "development")
        candidates[(model, False)] = rows
        report["development"][model] = {
            "detection": detection_metrics(dev, rows), "seconds": seconds}
    best_model = max(MODELS, key=lambda m: report["development"][m]["detection"]["map_50_95"])
    tiled_rows, seconds = predict(best_model, True, "development")
    candidates[(best_model, True)] = tiled_rows
    tiled_metrics = detection_metrics(dev, tiled_rows)
    report["development"][best_model+"_tiled"] = {"detection": tiled_metrics, "seconds": seconds}
    # Require an absolute 0.01 AP gain before paying roughly 5x inference cost.
    use_tiles = tiled_metrics["map_50_95"] > report["development"][best_model]["detection"]["map_50_95"] + .01
    selected = candidates[(best_model, use_tiles)]
    options = [{"backend": "iou", "threshold": t} for t in (.05, .1, .2, .25)]
    options += [{"backend": "bytetrack", "track_high_thresh": t,
                 "track_low_thresh": min(.05, t/2), "new_track_thresh": t,
                 "track_buffer": 10, "match_thresh": .8, "fuse_score": True}
                for t in (.05, .1, .2, .25)]
    ranked = []
    for option in options:
        metrics = mot_metrics(dev, tracking_predictions(selected, option))
        ranked.append({"options": option, "metrics": metrics})
    # Use the declared joint objective, not hand-selecting a favorable metric.
    best = max(ranked, key=lambda row: row["metrics"]["mota"] + row["metrics"]["idf1"])
    report["tracking_development"] = ranked
    report["selected_before_holdout"] = {
        "model": best_model, "tiling": use_tiles, "tracking": best["options"]}
    # Persist selection BEFORE computing/reading the holdout results.
    (OUT / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    held_gt = ground_truth["holdout"]
    original, _ = predict("current", False, "holdout")
    chosen, _ = predict(best_model, use_tiles, "holdout")
    report["holdout"] = {
        "baseline": {"detection": detection_metrics(held_gt, original),
                     "tracking": mot_metrics(held_gt, tracking_predictions(original, {"backend": "iou", "threshold": .25}))},
        "selected": {"detection": detection_metrics(held_gt, chosen),
                     "tracking": mot_metrics(held_gt, tracking_predictions(chosen, best["options"]))},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
