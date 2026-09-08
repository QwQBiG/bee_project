from __future__ import annotations

import json
import zipfile

import pytest

from deployment.build_submission import _resolve_models
from deployment.package_common import (
    DETECTION_FIELDS,
    TRACKING_FIELDS,
    parse_status_line,
    validate_result_payload,
    zip_bundle,
)


def test_status_line_requires_one_json_line():
    assert parse_status_line('{"status":"ok"}\n')["status"] == "ok"
    with pytest.raises(ValueError, match="exactly one"):
        parse_status_line('{"status":"ok"}\nextra\n')
    with pytest.raises(ValueError, match="valid JSON"):
        parse_status_line("OK output=x\n")


def test_packaging_validator_accepts_detection_and_tracking_payloads():
    detection = {
        "team_id": "614689", "sequence": "Inside-detection",
        "task": "detection", "repr": "HBB", "num_frames": 1,
        "num_records": 1, "processing_time_ms": 3,
        "fields": DETECTION_FIELDS,
        "detections": [[1, 0, .9, 1, 2, 3, 4, 1]],
    }
    tracking = {
        "team_id": "614689", "sequence": "Outside-tracking",
        "task": "tracking", "repr": "HBB", "num_frames": 1,
        "num_tracks": 1, "num_records": 1, "processing_time_ms": 4,
        "fields": TRACKING_FIELDS,
        "tracks": [[1, 7, 1, 2, 3, 4, 1, -1, -1, -1]],
    }
    validate_result_payload(detection, "Inside-detection", "614689")
    validate_result_payload(tracking, "Outside-tracking", "614689")


def test_zip_has_no_outer_exe_directory(tmp_path):
    bundle = tmp_path / "EXE-614689"
    (bundle / "_internal").mkdir(parents=True)
    (bundle / "_internal" / "runtime.dll").write_bytes(b"dll")
    (bundle / "Inside-detection-614689.exe").write_bytes(b"exe")
    archive = zip_bundle(bundle, tmp_path / "EXE-614689.zip")
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    assert "Inside-detection-614689.exe" in names
    assert "_internal/runtime.dll" in names
    assert not any(name.startswith("EXE-614689/") for name in names)


def test_packaged_config_repoints_models_to_shared_weights(tmp_path):
    config_dir = tmp_path / "configs"
    weights = tmp_path / "weights"
    config_dir.mkdir()
    weights.mkdir()
    (weights / "inside.onnx").write_bytes(b"inside")
    (weights / "outside.onnx").write_bytes(b"outside")
    config = {
        "detector": {
            "inside": {"model": "inside.onnx"},
            "outside": {"model": "outside.onnx"},
        },
        "runtime": {"device": "cuda"},
    }
    path = config_dir / "algorithm_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    packaged, models = _resolve_models(path, weights)
    assert models["inside"] == (weights / "inside.onnx").resolve()
    assert packaged["detector"]["outside"]["model"] == \
        "../../weights/outside.onnx"
    assert packaged["runtime"]["device"] == "auto"
