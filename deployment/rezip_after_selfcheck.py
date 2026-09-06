"""Validate selfcheck and create the final ZIP without an outer directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from deployment.package_common import DEFAULT_TEAM_ID, zip_bundle
from deployment.validate_package import validate_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--exe_dir", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exe_dir = Path(args.exe_dir or f"EXE-{args.team_id}").resolve()
    validate_package(args.team_id, exe_dir, require_selfcheck=True)
    output = Path(args.output).resolve() if args.output else \
        exe_dir.parent / f"EXE-{args.team_id}.zip"
    result = zip_bundle(exe_dir, output)
    print(f"final ZIP created: {result} ({result.stat().st_size / 1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
