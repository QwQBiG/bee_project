"""Evaluate an Ultralytics detector on one image at several thresholds."""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO
from utils.common import get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20]
    )
    parser.add_argument("--device", default=get_device())
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    summary = {
        "model": args.model,
        "image": args.image,
        "imgsz": args.imgsz,
        "classes": model.names,
        "thresholds": {},
    }

    for threshold in args.thresholds:
        result = model.predict(
            source=args.image,
            conf=threshold,
            iou=0.45,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        count = len(result.boxes)
        confidences = result.boxes.conf.detach().cpu().tolist() if count else []
        summary["thresholds"][str(threshold)] = {
            "detections": count,
            "mean_confidence": (
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
            "max_confidence": max(confidences) if confidences else 0.0,
        }
        rendered = result.plot()
        cv2.imwrite(
            str(output_dir / f"detections_conf_{threshold:.2f}.jpg"), rendered
        )

    with (output_dir / "frame_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()