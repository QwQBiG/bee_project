"""Simulate the organizer's process call and validate exit/stdout/result JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from deployment.package_common import (
    executable_identity,
    load_and_validate_result,
    parse_status_line,
)


@dataclass
class EvaluationRun:
    sequence: str
    team_id: str
    result_path: Path
    stdout: str
    stderr: str


def evaluate(executable: str | Path, input_dir: str | Path) -> EvaluationRun:
    exe = Path(executable).resolve()
    images = Path(input_dir).resolve()
    sequence, team_id = executable_identity(exe)
    if not exe.is_file():
        raise FileNotFoundError(exe)
    if not images.is_dir():
        raise FileNotFoundError(images)
    completed = subprocess.run(
        [str(exe), "--input", str(images)],
        cwd=exe.parent,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="strict")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise ValueError(f"process exit code must be 0, got {completed.returncode}")
    status = parse_status_line(stdout)
    if status.get("status") != "ok":
        raise ValueError(f"executable returned error status: {status}")
    if status.get("team_id") != team_id or status.get("sequence") != sequence:
        raise ValueError("stdout team_id/sequence does not match executable")
    output_value = status.get("output_path")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("stdout success JSON is missing output_path")
    result_path = Path(output_value)
    if not result_path.is_absolute():
        result_path = (exe.parent / result_path).resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"reported result does not exist: {result_path}")
    load_and_validate_result(result_path, sequence, team_id)
    return EvaluationRun(sequence, team_id, result_path, stdout, stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--input", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        run = evaluate(args.exe, args.input)
        print(json.dumps({
            "status": "ok",
            "sequence": run.sequence,
            "team_id": run.team_id,
            "output_path": run.result_path.as_posix(),
        }, ensure_ascii=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "error", "message": str(exc),
        }, ensure_ascii=False, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
