"""Fresh GPU regression for a locked box scale; no further parameter search."""
import json
from pathlib import Path

import cv2

from inference import algorithm_cli as cli
from inference.box_calibration import calibrate_detections
from tools.compare_onnx_candidates import detection_metrics
from tools.evaluate_vnbee_tracking import load_ground_truth


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "output/postprocessing_experiment"
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    factor = selection["locked"]["detection_scale"]
    cfg = cli.load_runtime_config(str(root / "configs/algorithm_config.json"))
    cfg.update(_scene="outside", _device="cuda")
    capture = cv2.VideoCapture(str(root / "data/external/vnbee/2022-04-08-12-30.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 480)
    before, after = {}, {}
    try:
        for fid in range(481, 601):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"missing frame {fid}")
            before[fid], _ = cli.run_detection_array(frame, cfg, conf_override=0, topk=200)
            after[fid] = calibrate_detections(before[fid], factor, frame.shape[1], frame.shape[0])
            if fid % 30 == 0:
                print(f"GPU regression frame {fid}", flush=True)
    finally:
        capture.release()
    gt = load_ground_truth(str(root / "tmp/2022-04-08-12-30.txt"))
    gt = {fid: gt.get(fid, []) for fid in before}
    report = {"scale": factor, "before": detection_metrics(gt, before),
              "after": detection_metrics(gt, after), "frames": 120,
              "device": "cuda", "independent_validation": False}
    (out / "gpu_regression.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
