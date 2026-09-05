"""Render public development GT versus cached detections; keep outputs local."""
import json
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate_vnbee_tracking import load_ground_truth, iou_xywh


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "output/detection_error_review"
    output.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(str(root / "tmp/2022-04-08-12-30.txt"))
    cache = json.loads((root / "output/onnx_candidates/original_0_development.json").read_text(encoding="utf-8"))
    video = cv2.VideoCapture(str(root / "data/external/vnbee/2022-04-08-12-30.mp4"))
    best_overlaps = []
    try:
        for fid in range(1, 121):
            ok, image = video.read()
            if not ok:
                raise ValueError(f"missing frame {fid}")
            predictions = cache["predictions"][str(fid)][:200]
            for target in gt.get(fid, []):
                best_overlaps.append(max((iou_xywh(target["bbox"], p["bbox"])
                                          for p in predictions), default=0))
            if fid not in (1, 30, 60, 90, 120):
                continue
            panels = []
            for title, rows, color in (("Ground truth", gt[fid], (0, 255, 0)),
                                       ("Predictions conf >= 0.25", [p for p in predictions if p["confidence"] >= .25], (0, 200, 255))):
                canvas = image.copy()
                for row in rows:
                    x, y, w, h = map(round, row["bbox"])
                    cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
                cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
                panels.append(canvas)
            if not cv2.imwrite(str(output / f"frame_{fid}.jpg"), np.hstack(panels)):
                raise OSError("cannot write review image")
    finally:
        video.release()
    overlaps = np.asarray(best_overlaps)
    report = {"gt_count": len(overlaps), "best_candidate_iou_median": float(np.median(overlaps)),
              "fraction_below_iou_0.1": float(np.mean(overlaps < .1)),
              "fraction_at_least_iou_0.5": float(np.mean(overlaps >= .5)),
              "note": "Best overlap allows candidate reuse; diagnostic only, NOT recall or AP."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
