"""Regression cases that the old mock-only folder tests did not exercise."""
import json
from pathlib import Path

import numpy as np
import pytest

from inference import algorithm_cli as cli
from inference import batch_cli


def test_nms_cap_is_applied_after_suppression():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10],
                      [30, 30, 10, 10]], dtype=np.float32)
    result = cli.multiclass_nms(boxes, np.array([.9, .8, .7]), topk=2)
    assert len(result) == 2
    assert result[1, 0] == 30


def test_yolo_center_boxes_are_converted_before_nms(monkeypatch, tmp_path):
    import cv2
    from types import SimpleNamespace
    image = tmp_path / "im_0001.jpg"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    # Same center: IoU=0.36, but treating centers as origins gives only 0.087.
    output = np.array([[[50, 50], [50, 50], [40, 24], [40, 24], [.9, .8]]],
                      dtype=np.float32)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 100, 100])],
        run=lambda *args: [output])
    monkeypatch.setattr(cli, "_get_session", lambda *args: (session, 100))
    cfg = {"_scene": "outside", "_config_dir": str(tmp_path), "detector": {
        "outside": {"model": "test.onnx", "imgsz": 100, "iou": .3}}}
    rows, _ = cli.run_detection(image, cfg)
    assert len(rows) == 1
    assert rows[0]["bbox"] == [30, 30, 40, 40]


def test_bad_arguments_produce_one_status_line(capsys):
    assert batch_cli.main(["--unknown"]) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["status"] == "error"


def test_official_path_cannot_be_assigned_wrong_task():
    args = batch_cli.build_parser().parse_args([
        "--input", "C:/Test/Inside/detection/images/",
        "--team-id", "123456", "--sequence", "Outside-detection"])
    with pytest.raises(ValueError, match="disagree"):
        batch_cli.execute(args, "batch_cli.py")


def test_invalid_payload_preserves_previous_result(tmp_path):
    from inference.submission_contract import write_result
    destination = tmp_path / "result.json"
    write_result(destination, {"ok": 1})
    with pytest.raises(ValueError):
        write_result(destination, {"bad": float("nan")})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"ok": 1}


def test_multiclass_requires_explicit_mapping_and_excludes_other_classes(monkeypatch):
    from types import SimpleNamespace
    output = np.array([[[50], [50], [20], [20], [.2], [.9]]], dtype=np.float32)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 100, 100])],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={
            "task": "detect", "names": "{0: 'bee', 1: 'mite'}"}),
        run=lambda *args: [output])
    monkeypatch.setattr(cli, "_get_session", lambda *args: (session, 100))
    cfg = {"_scene": "outside", "_config_dir": ".", "detector": {
        "outside": {"model": "test.onnx", "imgsz": 100}}}
    image = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError, match="class_ids"):
        cli.run_detection_array(image, cfg)
    cfg["detector"]["outside"]["class_ids"] = [0]
    rows, _ = cli.run_detection_array(image, cfg, conf_override=0)
    assert rows[0]["confidence"] == pytest.approx(.2)
    assert rows[0]["class_id"] == 0
    assert cli.run_detection_array(image, cfg, conf_override=.25)[0] == []


def test_pose_metadata_cannot_be_used_as_detection(monkeypatch):
    from types import SimpleNamespace
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 100, 100])],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={"task": "pose"}),
        run=lambda *args: [np.zeros((1, 14, 2), dtype=np.float32)])
    monkeypatch.setattr(cli, "_get_session", lambda *args: (session, 100))
    cfg = {"_scene": "outside", "_config_dir": ".", "detector": {
        "outside": {"model": "pose.onnx", "imgsz": 100}}}
    with pytest.raises(RuntimeError, match="separate decoder"):
        cli.run_detection_array(np.zeros((100, 100, 3), np.uint8), cfg)


def test_mot_metric_perfect_identity():
    pytest.importorskip("motmetrics")
    from tools.compare_onnx_candidates import mot_metrics
    gt = {1: [{"track_id": 1, "bbox": [0, 0, 10, 10]}],
          2: [{"track_id": 1, "bbox": [1, 0, 10, 10]}]}
    metrics = mot_metrics(gt, gt)
    assert metrics["mota"] == metrics["idf1"] == 1
    assert metrics["num_switches"] == 0
