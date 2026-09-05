"""Folder-level entry point required by the September 2026 competition spec."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inference.algorithm_cli import load_runtime_config, match_ious, run_detection
from inference.submission_contract import (
    VALID_SEQUENCES,
    build_detection_result,
    build_tracking_result,
    frame_id_from_path,
    write_result,
)


class StatusParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = StatusParser(prog="bee-competition")
    parser.add_argument("--input", required=True,
                        help="Official JPG image folder.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                        help="Diagnostic runtime override; cuda fails if GPU is unavailable.")
    # These three options support source-level testing. Packaged executables
    # infer the same values from their required filenames and fixed output dir.
    parser.add_argument("--sequence", choices=sorted(VALID_SEQUENCES))
    parser.add_argument("--team-id")
    parser.add_argument("--output-dir", default="C:/TestResults/")
    return parser


def identity_from_executable(executable: str | Path) -> Tuple[str, str]:
    stem = Path(executable).stem
    match = re.fullmatch(
        r"(Inside|Outside)-(detection|tracking)-(\d{6})", stem)
    if not match:
        raise ValueError(
            "executable name must be <Inside|Outside>-"
            "<detection|tracking>-<six-digit team ID>.exe")
    return f"{match.group(1)}-{match.group(2)}", match.group(3)


def list_official_images(input_dir: str | Path) -> List[Tuple[int, Path]]:
    folder = Path(input_dir)
    if not folder.is_dir():
        raise ValueError(f"input directory does not exist: {folder}")
    indexed = [(frame_id_from_path(path), path) for path in folder.iterdir()
               if path.is_file() and path.suffix.lower() == ".jpg"]
    if not indexed:
        raise ValueError(f"no JPG images found in: {folder}")
    indexed.sort(key=lambda pair: pair[0])
    ids = [frame_id for frame_id, _ in indexed]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frame_id values in input directory")
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("official frame IDs must be continuous from 1 to N")
    return indexed


def _tracking_rows(images: Sequence[Tuple[int, Path]], config: Dict[str, Any]) \
        -> Tuple[List[Dict[str, Any]], int]:
    """Run the selected scene tracker in memory without stale file cache."""
    options = config.get("tracking", {})
    backend = options.get("backend", "iou")
    if backend == "bytetrack":
        from tracking.onnx_bytetrack import OnnxByteTracker
        tracker = OnnxByteTracker(options)
        rows, elapsed = [], 0
        for frame_id, image_path in images:
            # Low-score boxes are association candidates, never unconditional
            # MOT output. BYTETracker confirms/filters the actual output tracks.
            detections, frame_ms = run_detection(
                image_path, config, conf_override=tracker.low, topk=600)
            elapsed += frame_ms
            rows.extend({"frame_id": frame_id, **row}
                        for row in tracker.update(detections))
        return rows, elapsed
    if backend != "iou":
        raise ValueError(f"unknown tracking backend: {backend}")
    previous_boxes: List[List[float]] = []
    previous_ids: List[int] = []
    next_id = 1
    rows: List[Dict[str, Any]] = []
    elapsed = 0
    for frame_id, image_path in images:
        detections, frame_ms = run_detection(image_path, config)
        elapsed += frame_ms
        boxes = [item["bbox"] for item in detections]
        matches = match_ious(boxes, previous_boxes)
        current_ids: List[int] = []
        for index, detection in enumerate(detections):
            if index in matches:
                track_id = previous_ids[matches[index]]
            else:
                track_id = next_id
                next_id += 1
            current_ids.append(track_id)
            rows.append({"frame_id": frame_id, "track_id": track_id,
                         "bbox": detection["bbox"]})
        previous_boxes, previous_ids = boxes, current_ids
    return rows, elapsed


def execute(args: argparse.Namespace, executable: str | Path) -> Path:
    started = time.perf_counter()
    if args.sequence and args.team_id:
        sequence, team_id = args.sequence, args.team_id
    elif args.sequence or args.team_id:
        raise ValueError("--sequence and --team-id must be supplied together")
    else:
        sequence, team_id = identity_from_executable(executable)

    if not re.fullmatch(r"[0-9]{6}", team_id):
        raise ValueError("team ID must be six ASCII digits")
    parts = Path(args.input).parts
    path_sequence = None
    if len(parts) >= 3 and parts[-1].lower() == "images":
        candidate = f"{parts[-3].capitalize()}-{parts[-2].lower()}"
        if candidate in VALID_SEQUENCES:
            path_sequence = candidate
    if path_sequence is not None and sequence != path_sequence:
        raise ValueError("input path and executable task disagree")
    if Path(executable).suffix.lower() == ".exe":
        if identity_from_executable(executable) != (sequence, team_id):
            raise ValueError("executable identity cannot be overridden")

    images = list_official_images(args.input)
    config = copy.deepcopy(load_runtime_config(args.config))
    config["_scene"] = sequence.split("-")[0].lower()
    config["_device"] = getattr(args, "device", None) or config.get("runtime", {}).get("device", "auto")
    # Scene-specific options must not silently change the other task domain.
    config["tracking"] = config.get("tracking", {}).get(config["_scene"], {})
    config["tiling"] = config.get("tiling", {}).get(config["_scene"], {})
    if sequence.endswith("-detection"):
        # Detection scoring forbids confidence-based removal below the official
        # per-frame cap. Tracking keeps its configured decision threshold.
        limit = 600 if sequence.startswith("Inside-") else 200
        from inference.box_calibration import calibrate_detections
        scale = float(config.get("detector", {}).get(config["_scene"], {}).get("detection_box_scale", 1.0))
        # Reject invalid calibration before loading a model or writing results.
        calibrate_detections([], scale, 1, 1)
        records: List[Dict[str, Any]] = []
        measured_ms = 0
        for frame_id, image_path in images:
            detections, frame_ms = run_detection(
                image_path, config, conf_override=0.0, topk=limit)
            if scale != 1.0:
                import cv2
                import numpy as np
                raw = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw is None:
                    raise ValueError(f"cannot decode calibration image: {image_path}")
                detections = calibrate_detections(detections, scale, raw.shape[1], raw.shape[0])
            measured_ms += frame_ms
            records.extend({
                "frame_id": frame_id,
                "class_id": item.get("class_id", 0),
                "conf": item["confidence"],
                "bbox": item["bbox"],
            } for item in detections)
        processing_ms = max(
            measured_ms, int((time.perf_counter() - started) * 1000))
        payload = build_detection_result(
            team_id, sequence, len(images), processing_ms, records)
    else:
        records, measured_ms = _tracking_rows(images, config)
        processing_ms = max(
            measured_ms, int((time.perf_counter() - started) * 1000))
        payload = build_tracking_result(
            team_id, sequence, len(images), processing_ms, records)

    output = Path(args.output_dir) / f"{sequence}-{team_id}.json"
    return write_result(output, payload)


def _print_status(status: str, detail: Any) -> None:
    clean_detail = " ".join(str(detail).splitlines())
    sys.stdout.write(f"{status} {clean_detail}\n")


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        output = execute(args, sys.argv[0])
        # The official contract allows one console status-summary line.
        _print_status("OK", f"output={output}")
        return 0
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:
        _print_status("ERROR", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
