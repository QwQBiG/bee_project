"""Export the current inside/outside Ultralytics PT weights to static ONNX."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".runtime" / "ultralytics"))


def export_one(source: Path, destination: Path, imgsz: int, opset: int) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    from ultralytics import YOLO
    model = YOLO(str(source))
    exported = Path(model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=False,
        simplify=False,
        device="cpu",
    )).resolve()
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics did not create ONNX output: {exported}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported != destination.resolve():
        shutil.move(str(exported), str(destination))
    return destination.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outside-pt",
        default=str(PROJECT_ROOT / "artifacts/models/hive_entrance_bee_yolov8n.pt"))
    parser.add_argument(
        "--inside-pt",
        default=str(PROJECT_ROOT / "artifacts/models/honey_bee_detector_yolov8s.pt"))
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "artifacts/models"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    outside = export_one(
        Path(args.outside_pt).resolve(),
        output / "hive_entrance_bee_yolov8n.onnx", 1280, args.opset)
    inside = export_one(
        Path(args.inside_pt).resolve(),
        output / "bee_inside_ft_v1.onnx", 640, args.opset)
    print(f"outside ONNX: {outside}")
    print(f"inside ONNX: {inside}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
