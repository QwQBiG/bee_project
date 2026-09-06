"""Build one PyInstaller onedir bundle and expose four shared-runtime EXEs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import site
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from deployment.package_common import (
    DEFAULT_TEAM_ID, SEQUENCES, validate_team_id, zip_bundle,
)
from deployment.validate_package import validate_package


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_clean(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved.parent != parent or not resolved.name.startswith(("EXE-", "submission-")):
        raise ValueError(f"refusing to clean unsafe build path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _resolve_models(config_path: Path, weights_dir: Path) \
        -> Tuple[Dict, Dict[str, Path]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    packaged = copy.deepcopy(config)
    models: Dict[str, Path] = {}
    for scene in ("inside", "outside"):
        model_value = config.get("detector", {}).get(scene, {}).get("model")
        if not isinstance(model_value, str):
            raise ValueError(f"config is missing detector.{scene}.model")
        basename = Path(model_value).name
        candidates = [
            (config_path.parent / model_value).resolve(),
            (weights_dir / basename).resolve(),
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None or source.suffix.lower() != ".onnx":
            raise FileNotFoundError(
                f"missing {scene} ONNX model; expected one of: {candidates}")
        models[scene] = source
        # Packaged config lives in _internal/configs; weights live beside EXEs.
        packaged["detector"][scene]["model"] = f"../../weights/{source.name}"
    packaged.setdefault("runtime", {})["device"] = "auto"
    return packaged, models


def _nvidia_binary_options() -> List[str]:
    """Collect pip-installed NVIDIA DLLs while preserving package layout."""
    options: List[str] = []
    roots = [Path(path).resolve() for path in site.getsitepackages()]
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for dll in sorted(nvidia.rglob("*.dll")):
            destination = dll.parent.relative_to(root).as_posix()
            options.extend(["--add-binary", f"{dll}{os.pathsep}{destination}"])
    return options


def _copy_extra_dlls(source: Optional[str], internal: Path) -> None:
    if not source:
        return
    dlls = sorted(Path(source).resolve().rglob("*.dll"))
    if not dlls:
        raise ValueError("--runtime_dll_dir contains no DLL files")
    for dll in dlls:
        destination = internal / dll.name
        if destination.exists() and destination.read_bytes() != dll.read_bytes():
            raise ValueError(f"runtime DLL name collision: {dll.name}")
        shutil.copy2(dll, destination)


def build(args: argparse.Namespace) -> Tuple[Path, Path]:
    team_id = validate_team_id(args.team_id)
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise RuntimeError("submission must be built with 64-bit Python on Windows")
    runner = Path(args.runner).resolve()
    config_path = Path(args.config).resolve()
    weights_dir = Path(args.weights).resolve()
    if not runner.is_file() or not config_path.is_file():
        raise FileNotFoundError("runner or config file is missing")

    output_parent = Path(args.output_parent).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    bundle = output_parent / f"EXE-{team_id}"
    build_root = (PROJECT_ROOT / "build" / f"submission-{team_id}").resolve()
    _safe_clean(bundle, output_parent)
    _safe_clean(build_root, build_root.parent)
    build_root.mkdir(parents=True)

    packaged_config, models = _resolve_models(config_path, weights_dir)
    generated_config = build_root / "generated" / "algorithm_config.json"
    generated_config.parent.mkdir(parents=True)
    generated_config.write_text(
        json.dumps(packaged_config, ensure_ascii=False, indent=2),
        encoding="utf-8")

    primary_name = f"Inside-detection-{team_id}"
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir", "--console", "--name", primary_name,
        "--contents-directory", "_internal",
        "--distpath", str(build_root / "dist"),
        "--workpath", str(build_root / "work"),
        "--specpath", str(build_root / "spec"),
        "--paths", str(PROJECT_ROOT),
        # The standard PyInstaller ONNX Runtime hook plus its binary files is
        # sufficient. ``--collect-all`` also drags optional training tools and
        # PyTorch into an inference-only submission.
        "--collect-binaries", "onnxruntime",
        "--hidden-import", "lap",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "ultralytics",
        "--add-data", f"{generated_config}{os.pathsep}configs",
        *_nvidia_binary_options(),
        str(runner),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    generated = build_root / "dist" / primary_name
    shutil.move(str(generated), str(bundle))

    primary_exe = bundle / f"{primary_name}.exe"
    for sequence in SEQUENCES:
        destination = bundle / f"{sequence}-{team_id}.exe"
        if destination != primary_exe:
            shutil.copy2(primary_exe, destination)
    packaged_weights = bundle / "weights"
    packaged_weights.mkdir()
    copied = set()
    for source in models.values():
        if source.name not in copied:
            shutil.copy2(source, packaged_weights / source.name)
            copied.add(source.name)
    (bundle / "selfcheck").mkdir()
    _copy_extra_dlls(args.runtime_dll_dir, bundle / "_internal")
    validate_package(team_id, bundle, require_selfcheck=False)
    zip_path = output_parent / f"EXE-{team_id}.zip"
    zip_bundle(bundle, zip_path)
    return bundle, zip_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_id", default=DEFAULT_TEAM_ID)
    parser.add_argument(
        "--runner", default=str(PROJECT_ROOT / "deployment" / "competition_runner.py"))
    parser.add_argument(
        "--weights", default=str(PROJECT_ROOT / "artifacts" / "models"))
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "algorithm_config.json"))
    parser.add_argument("--runtime_dll_dir")
    parser.add_argument("--output_parent", default=str(PROJECT_ROOT))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle, zip_path = build(args)
    print(f"bundle created: {bundle}")
    print(f"initial ZIP created: {zip_path}")
    print("run make_selfcheck.py, validate_package.py, then rezip_after_selfcheck.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
