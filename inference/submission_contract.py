"""Build result JSON for the 2026 competition submission contract.

This module deliberately contains no model/runtime dependency so the four
packaged executables can share one deterministic output implementation.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DETECTION_FIELDS = [
    "frame_id", "class_id", "conf", "bb_left", "bb_top",
    "bb_width", "bb_height", "ignore",
]
TRACKING_FIELDS = [
    "frame_id", "track_id", "bb_left", "bb_top", "bb_width",
    "bb_height", "conf", "x", "y", "z",
]
VALID_SEQUENCES = {
    "Inside-detection", "Inside-tracking",
    "Outside-detection", "Outside-tracking",
}


def frame_id_from_path(path: str | Path) -> int:
    """Return the 1-based numeric suffix from an official image filename."""
    match = re.search(r"(\d+)$", Path(path).stem)
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"cannot parse positive frame_id from: {path}")
    return int(match.group(1))


def _check_header(team_id: str, sequence: str, task: str,
                  num_frames: int, processing_time_ms: int) -> None:
    if not re.fullmatch(r"[0-9]{6}", team_id):
        raise ValueError("team_id must contain exactly six digits")
    if sequence not in VALID_SEQUENCES or not sequence.endswith(task):
        raise ValueError(f"sequence does not match task={task}: {sequence}")
    if (num_frames < 0 or processing_time_ms < 0 or
            not math.isfinite(float(processing_time_ms))):
        raise ValueError("frame count and processing time must be non-negative")


def _box(row: Mapping[str, Any]) -> List[float]:
    box = row.get("bbox")
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) \
            or len(box) != 4:
        raise ValueError("bbox must be [left, top, width, height]")
    left, top, width, height = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (left, top, width, height)):
        raise ValueError("bbox values must be finite numbers")
    if round(width, 2) <= 0 or round(height, 2) <= 0:
        raise ValueError("bbox width and height must be greater than zero")
    return [round(left, 2), round(top, 2), round(width, 2), round(height, 2)]


def build_detection_result(
    team_id: str,
    sequence: str,
    num_frames: int,
    processing_time_ms: int,
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build HBB detections, sorted and capped exactly as required."""
    _check_header(team_id, sequence, "detection", num_frames,
                  processing_time_ms)
    grouped: Dict[int, List[List[Any]]] = defaultdict(list)
    for item in records:
        frame_id = int(item["frame_id"])
        class_id = int(item.get("class_id", 0))
        conf = float(item["conf"])
        if not 1 <= frame_id <= num_frames:
            raise ValueError(f"frame_id outside 1..{num_frames}: {frame_id}")
        if (class_id != 0 or not math.isfinite(conf) or
                not 0.0 <= conf <= 1.0):
            raise ValueError("class_id must be 0 and conf must be within [0, 1]")
        grouped[frame_id].append(
            [frame_id, 0, round(conf, 6), *_box(item), 1])

    per_frame_limit = 600 if sequence.startswith("Inside-") else 200
    rows: List[List[Any]] = []
    for frame_id in sorted(grouped):
        rows.extend(sorted(grouped[frame_id], key=lambda row: -row[2])[
            :per_frame_limit])
    return {
        "team_id": team_id,
        "sequence": sequence,
        "task": "detection",
        "repr": "HBB",
        "num_frames": num_frames,
        "num_records": len(rows),
        "processing_time_ms": int(processing_time_ms),
        "fields": DETECTION_FIELDS,
        "detections": rows,
    }


def build_tracking_result(
    team_id: str,
    sequence: str,
    num_frames: int,
    processing_time_ms: int,
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build MOTChallenge-compatible HBB tracking records."""
    _check_header(team_id, sequence, "tracking", num_frames,
                  processing_time_ms)
    rows: List[List[Any]] = []
    track_ids = set()
    frame_track_pairs = set()
    for item in records:
        frame_id, track_id = int(item["frame_id"]), int(item["track_id"])
        if not 1 <= frame_id <= num_frames or track_id < 1:
            raise ValueError("frame_id must be in range and track_id positive")
        if (frame_id, track_id) in frame_track_pairs:
            raise ValueError("track_id must occur at most once per frame")
        frame_track_pairs.add((frame_id, track_id))
        track_ids.add(track_id)
        rows.append([frame_id, track_id, *_box(item), 1, -1, -1, -1])
    rows.sort(key=lambda row: (row[0], row[1]))
    return {
        "team_id": team_id,
        "sequence": sequence,
        "task": "tracking",
        "repr": "HBB",
        "num_frames": num_frames,
        "num_tracks": len(track_ids),
        "num_records": len(rows),
        "processing_time_ms": int(processing_time_ms),
        "fields": TRACKING_FIELDS,
        "tracks": rows,
    }


def write_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write UTF-8 compact JSON without a trailing newline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination
