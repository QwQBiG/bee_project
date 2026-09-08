"""Shared validation helpers for the 2026 competition submission bundle."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


DEFAULT_TEAM_ID = "614689"
SEQUENCES = (
    "Inside-detection",
    "Inside-tracking",
    "Outside-detection",
    "Outside-tracking",
)
DETECTION_FIELDS = [
    "frame_id", "class_id", "conf", "bb_left", "bb_top",
    "bb_width", "bb_height", "ignore",
]
TRACKING_FIELDS = [
    "frame_id", "track_id", "bb_left", "bb_top", "bb_width",
    "bb_height", "conf", "x", "y", "z",
]
MAX_ZIP_BYTES = 4 * 1024**3
MAX_UNPACKED_BYTES = 8 * 1024**3


def validate_team_id(team_id: str) -> str:
    if not re.fullmatch(r"[0-9]{6}", team_id):
        raise ValueError("team_id must contain exactly six digits")
    return team_id


def executable_identity(path: str | Path) -> Tuple[str, str]:
    match = re.fullmatch(
        r"(Inside|Outside)-(detection|tracking)-(\d{6})",
        Path(path).stem,
    )
    if not match:
        raise ValueError(f"invalid competition executable name: {Path(path).name}")
    return f"{match.group(1)}-{match.group(2)}", match.group(3)


def parse_status_line(stdout: str) -> Dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise ValueError(f"stdout must contain exactly one line, got {len(lines)}")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("stdout line is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") not in {"ok", "error"}:
        raise ValueError("stdout JSON must contain status=ok or status=error")
    return payload


def validate_result_payload(
    payload: Dict[str, Any], sequence: str, team_id: str
) -> None:
    if payload.get("team_id") != team_id or payload.get("sequence") != sequence:
        raise ValueError("result team_id/sequence does not match executable")
    task = sequence.rsplit("-", 1)[1]
    if payload.get("task") != task or payload.get("repr") != "HBB":
        raise ValueError("result task/repr is invalid")
    if not isinstance(payload.get("num_frames"), int) or payload["num_frames"] < 1:
        raise ValueError("num_frames must be a positive integer")
    if not isinstance(payload.get("processing_time_ms"), int) \
            or payload["processing_time_ms"] < 0:
        raise ValueError("processing_time_ms must be a non-negative integer")
    rows_key = "detections" if task == "detection" else "tracks"
    fields = DETECTION_FIELDS if task == "detection" else TRACKING_FIELDS
    rows = payload.get(rows_key)
    if payload.get("fields") != fields or not isinstance(rows, list):
        raise ValueError(f"invalid {rows_key} fields or records")
    if payload.get("num_records") != len(rows):
        raise ValueError("num_records does not equal record array length")
    if task == "tracking":
        track_ids = {row[1] for row in rows if isinstance(row, list) and len(row) == 10}
        if payload.get("num_tracks") != len(track_ids):
            raise ValueError("num_tracks does not equal unique track_id count")
    expected_width = len(fields)
    if any(not isinstance(row, list) or len(row) != expected_width for row in rows):
        raise ValueError(f"every {rows_key} row must contain {expected_width} values")
    if task == "detection":
        if any(row[1] != 0 or row[7] != 1 for row in rows):
            raise ValueError("detection class_id must be 0 and ignore must be 1")
        if rows != sorted(rows, key=lambda row: (row[0], -row[2])):
            raise ValueError("detections must be sorted by frame_id then confidence")
        cap = 600 if sequence.startswith("Inside-") else 200
        counts: Dict[int, int] = {}
        for row in rows:
            counts[row[0]] = counts.get(row[0], 0) + 1
        if any(count > cap for count in counts.values()):
            raise ValueError("detections exceed the per-frame record limit")
    else:
        if any(row[6:] != [1, -1, -1, -1] for row in rows):
            raise ValueError("tracking conf/x/y/z must equal 1/-1/-1/-1")
        if rows != sorted(rows, key=lambda row: (row[0], row[1])):
            raise ValueError("tracks must be sorted by frame_id then track_id")


def load_and_validate_result(
    path: str | Path, sequence: str, team_id: str
) -> Dict[str, Any]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if "\n" in raw or "\r" in raw:
        raise ValueError(f"result JSON must be compact single-line UTF-8: {source}")
    payload = json.loads(raw)
    validate_result_payload(payload, sequence, team_id)
    return payload


def directory_size(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def zip_bundle(source_dir: str | Path, destination: str | Path) -> Path:
    """Zip the children of source_dir directly, without an outer directory."""
    source = Path(source_dir).resolve()
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as archive:
            directories = sorted(path for path in source.rglob("*") if path.is_dir())
            for directory in directories:
                if not any(directory.iterdir()):
                    archive.writestr(directory.relative_to(source).as_posix() + "/", "")
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source).as_posix())
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            raise ValueError("submission ZIP exceeds the 4 GiB compressed limit")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
