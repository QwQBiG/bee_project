import numpy as np
import pytest

from inference import tiled_detection


def test_tiles_merge_duplicates_in_original_coordinates(monkeypatch):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    calls = []

    def predict(crop, config, **kwargs):
        calls.append(crop.shape)
        # A duplicate from full frame and top-left crop must be suppressed.
        return [{"bbox": [10, 10, 5, 5], "confidence": .8, "class_id": 0}], 1

    monkeypatch.setattr(tiled_detection, "run_detection_array", predict)
    cfg = {"_scene": "outside", "detector": {"outside": {"iou": .45}},
           "tiling": {"enabled": True, "overlap": .2}}
    rows, _ = tiled_detection.detect_array(image, cfg, conf_override=0, topk=200)
    assert len(calls) == 5
    assert len(rows) == 4
    assert max(row["bbox"][0] for row in rows) == 54
    assert all(0 <= row["bbox"][1] <= 100 for row in rows)


def test_bytetrack_retains_id_across_low_confidence_and_gap():
    pytest.importorskip("lap")
    from tracking.onnx_bytetrack import OnnxByteTracker
    tracker = OnnxByteTracker({"track_high_thresh": .2, "track_low_thresh": .05})
    high = {"bbox": [10, 10, 20, 20], "confidence": .9}
    tid = tracker.update([high])[0]["track_id"]
    assert tracker.update([{**high, "confidence": .1}])[0]["track_id"] == tid
    assert tracker.update([]) == []
    assert tracker.update([high])[0]["track_id"] == tid


def test_low_confidence_does_not_create_new_track():
    pytest.importorskip("lap")
    from tracking.onnx_bytetrack import OnnxByteTracker
    tracker = OnnxByteTracker({"track_high_thresh": .2, "track_low_thresh": .05})
    assert tracker.update([{"bbox": [0, 0, 20, 20], "confidence": .1}]) == []


def test_retired_id_is_not_reused():
    pytest.importorskip("lap")
    from tracking.onnx_bytetrack import OnnxByteTracker
    tracker = OnnxByteTracker({"track_buffer": 1})
    high = {"bbox": [10, 10, 20, 20], "confidence": .9}
    original = tracker.update([high])[0]["track_id"]
    for _ in range(4):
        tracker.update([])
    tracker.update([high])
    assert tracker.update([high])[0]["track_id"] != original
