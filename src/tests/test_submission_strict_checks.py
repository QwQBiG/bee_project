from types import SimpleNamespace

import pytest

from deployment.package_common import DETECTION_FIELDS, validate_result_payload
from deployment.evaluation_call import evaluate


@pytest.mark.parametrize("index,value", [(0, 2), (0, 1.5), (2, 1.1), (3, float("nan")), (5, 0)])
def test_invalid_detection_record_rejected(index, value):
    payload = dict(team_id="614689", sequence="Outside-detection", task="detection",
                   repr="HBB", num_frames=1, num_records=1, processing_time_ms=1,
                   fields=DETECTION_FIELDS, detections=[[1, 0, .9, 0, 0, 10, 10, 1]])
    payload["detections"][0][index] = value
    with pytest.raises(ValueError):
        validate_result_payload(payload, "Outside-detection", "614689")


def test_evaluation_rejects_wrong_output_directory(monkeypatch, tmp_path):
    import json
    exe = tmp_path / "Outside-detection-614689.exe"
    exe.touch()
    images = tmp_path / "images"
    images.mkdir()
    status = dict(status="ok", team_id="614689", sequence="Outside-detection",
                  output_path=str(tmp_path / "Outside-detection-614689.json"))
    monkeypatch.setattr("deployment.evaluation_call.subprocess.run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=json.dumps(status).encode(), stderr=b""))
    with pytest.raises(ValueError, match="must be written"):
        evaluate(exe, images)


def test_explicit_weights_directory_has_priority(tmp_path):
    import json
    from deployment.build_submission import _resolve_models
    weights = tmp_path / "selected"
    weights.mkdir()
    for name in ("inside.onnx", "outside.onnx"):
        (tmp_path / name).write_bytes(b"old")
        (weights / name).write_bytes(b"selected")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"detector": {
        "inside": {"model": "inside.onnx"}, "outside": {"model": "outside.onnx"}}}), encoding="utf-8")
    _, models = _resolve_models(config, weights)
    assert models["inside"] == (weights / "inside.onnx").resolve()
