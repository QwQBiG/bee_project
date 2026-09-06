"""Generate selfcheck files by actually running all four packaged EXEs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional

from deployment.evaluation_call import evaluate
from deployment.package_common import DEFAULT_TEAM_ID, SEQUENCES, validate_team_id


def input_for_sequence(root: Path, sequence: str) -> Path:
    scene, task = sequence.split("-")
    return root / scene / task / "images"


def generate_selfcheck(team_id: str, exe_dir: str | Path,
                       selftest_root: str | Path) -> Path:
    validate_team_id(team_id)
    bundle = Path(exe_dir).resolve()
    inputs = Path(selftest_root).resolve()
    selfcheck = bundle / "selfcheck"
    selfcheck.mkdir(parents=True, exist_ok=True)
    logs = []
    for sequence in SEQUENCES:
        exe = bundle / f"{sequence}-{team_id}.exe"
        image_dir = input_for_sequence(inputs, sequence)
        run = evaluate(exe, image_dir)
        destination = selfcheck / f"{sequence}-{team_id}.json"
        shutil.copy2(run.result_path, destination)
        # When C:/TestResults is unavailable the EXE correctly falls back to
        # its own directory. Keep the copied selfcheck file, but remove that
        # transient root-level result so it is not included in the final ZIP.
        if run.result_path.parent.resolve() == bundle:
            run.result_path.unlink()
        logs.extend([
            f"===== {sequence} =====",
            f"command: {exe.name} --input \"{image_dir}\"",
            f"stdout: {run.stdout.rstrip()}",
            "stderr:",
            run.stderr.rstrip(),
            "",
        ])
    (selfcheck / f"console-{team_id}.log").write_text(
        "\n".join(logs), encoding="utf-8")
    return selfcheck


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--exe_dir", default=None)
    parser.add_argument("--selftest_root", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exe_dir = args.exe_dir or f"EXE-{args.team_id}"
    output = generate_selfcheck(args.team_id, exe_dir, args.selftest_root)
    print(f"selfcheck generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
