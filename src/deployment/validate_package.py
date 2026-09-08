"""Validate the unpacked EXE submission directory before final zipping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from deployment.package_common import (
    DEFAULT_TEAM_ID,
    MAX_UNPACKED_BYTES,
    SEQUENCES,
    directory_size,
    load_and_validate_result,
    validate_team_id,
)


FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pb"}


def validate_package(team_id: str, exe_dir: str | Path,
                     require_selfcheck: bool = True) -> int:
    validate_team_id(team_id)
    root = Path(exe_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required_dirs = [root / "_internal", root / "weights", root / "selfcheck"]
    for directory in required_dirs:
        if not directory.is_dir():
            raise ValueError(f"missing required directory: {directory.name}")
    for sequence in SEQUENCES:
        exe = root / f"{sequence}-{team_id}.exe"
        if not exe.is_file():
            raise ValueError(f"missing executable: {exe.name}")
    allowed_root_names = {
        "_internal", "weights", "selfcheck",
        *(f"{sequence}-{team_id}.exe" for sequence in SEQUENCES),
    }
    unexpected = [path.name for path in root.iterdir()
                  if path.name not in allowed_root_names]
    if unexpected:
        raise ValueError(f"unexpected item in submission root: {unexpected[0]}")
    onnx_files = list((root / "weights").glob("*.onnx"))
    if len(onnx_files) < 2:
        raise ValueError("weights/ must contain the inside and outside ONNX models")
    forbidden = [path for path in root.rglob("*")
                 if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise ValueError(f"forbidden native model file: {forbidden[0]}")
    config = root / "_internal" / "configs" / "algorithm_config.json"
    if not config.is_file():
        raise ValueError("missing packaged algorithm_config.json")
    if require_selfcheck:
        for sequence in SEQUENCES:
            result = root / "selfcheck" / f"{sequence}-{team_id}.json"
            if not result.is_file():
                raise ValueError(f"missing selfcheck result: {result.name}")
            load_and_validate_result(result, sequence, team_id)
        log = root / "selfcheck" / f"console-{team_id}.log"
        if not log.is_file():
            raise ValueError(f"missing selfcheck log: {log.name}")
    size = directory_size(root.rglob("*"))
    if size > MAX_UNPACKED_BYTES:
        raise ValueError("submission directory exceeds the 8 GiB unpacked limit")
    return size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--exe_dir", default=None)
    parser.add_argument("--allow_empty_selfcheck", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exe_dir = args.exe_dir or f"EXE-{args.team_id}"
    size = validate_package(
        args.team_id, exe_dir,
        require_selfcheck=not args.allow_empty_selfcheck)
    print(f"package valid: {Path(exe_dir).resolve()} ({size / 1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
