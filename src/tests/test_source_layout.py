from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_uses_the_requested_src_layout():
    for name in ("src", "weights", "configs"):
        assert (PROJECT_ROOT / name).is_dir()
    for name in ("README.md", "requirements.txt"):
        assert (PROJECT_ROOT / name).is_file()
    for old_source_dir in (
        "annotation", "behavior", "deployment", "inference", "models",
        "tests", "tools", "tracking", "utils", "visualization",
    ):
        assert not (PROJECT_ROOT / old_source_dir).exists()
        assert (PROJECT_ROOT / "src" / old_source_dir).is_dir()


def test_runtime_config_resolves_models_from_weights():
    config_path = PROJECT_ROOT / "configs" / "algorithm_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for scene in ("inside", "outside"):
        relative = config["detector"][scene]["model"]
        model = (config_path.parent / relative).resolve()
        assert model.parent == (PROJECT_ROOT / "weights").resolve()
        assert model.is_file()
        assert model.suffix == ".onnx"


def test_source_tree_contains_no_native_framework_weight_files():
    forbidden = {".pt", ".pth", ".pb"}
    roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "weights")
    found = [
        path for root in roots for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden
    ]
    assert found == []
