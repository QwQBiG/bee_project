import pytest

from tools.tune_postprocessing import resize_predictions
from inference.box_calibration import calibrate_detections


def test_calibration_preserves_input_and_confidence():
    original = {1: [{"bbox": [0, 0, 20, 20], "confidence": .7}]}
    result = resize_predictions(original, 1.1, 100, 100)
    assert original[1][0]["bbox"] == [0, 0, 20, 20]
    assert result[1][0]["bbox"] == [0, 0, 21, 21]
    assert result[1][0]["confidence"] == .7


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_bad_calibration_is_rejected(factor):
    with pytest.raises(ValueError):
        resize_predictions({}, factor, 100, 100)


def test_calibration_preserves_count_order_and_clips_far_edge():
    rows = [{"bbox": [90, 90, 10, 10], "confidence": .9, "class_id": 0},
            {"bbox": [30, 30, 10, 10], "confidence": .1, "class_id": 0}]
    result = calibrate_detections(rows, 1.15, 100, 100)
    assert len(result) == 2
    assert [r["confidence"] for r in result] == [.9, .1]
    assert result[0]["bbox"] == [89.25, 89.25, 10.75, 10.75]
    assert all(r["class_id"] == 0 for r in result)


def test_calibration_rejects_excessive_scale():
    with pytest.raises(ValueError):
        calibrate_detections([], 2, 100, 100)


@pytest.mark.parametrize("box", [[float("nan"), 0, 10, 10],
                                  [0, 0, float("inf"), 10], [0, 0, -1, 10]])
def test_invalid_box_is_rejected(box):
    with pytest.raises(ValueError):
        calibrate_detections([{"bbox": box}], 1.15, 100, 100)


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError):
        calibrate_detections([], 1.15, float("nan"), 100)


def test_batch_detection_applies_scene_scale(monkeypatch, tmp_path):
    import json
    import cv2
    import numpy as np
    from inference import batch_cli
    images = tmp_path / "images"
    images.mkdir()
    assert cv2.imwrite(str(images / "frame_1.jpg"), np.zeros((100, 100, 3), np.uint8))
    monkeypatch.setattr(batch_cli, "load_runtime_config", lambda _: {
        "detector": {"outside": {"detection_box_scale": 1.15}}})
    monkeypatch.setattr(batch_cli, "run_detection", lambda *a, **kw: (
        [{"bbox": [30, 30, 10, 10], "confidence": .8, "class_id": 0}], 0))
    args = batch_cli.build_parser().parse_args([
        "--input", str(images), "--sequence", "Outside-detection",
        "--team-id", "123456", "--output-dir", str(tmp_path / "results")])
    result = json.loads(batch_cli.execute(args, "batch_cli.py").read_text(encoding="utf-8"))
    assert result["num_records"] == 1
    assert result["detections"][0][3:7] == [29.25, 29.25, 11.5, 11.5]
