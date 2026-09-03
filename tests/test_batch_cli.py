"""Tests for the official folder-level competition entry point."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from inference import batch_cli


def _jpgs(folder, count=2):
    folder.mkdir()
    for frame_id in range(1, count + 1):
        (folder / f"im_{frame_id:04d}.jpg").write_bytes(b"jpg")


def test_identity_is_read_from_required_executable_name():
    assert batch_cli.identity_from_executable(
        "Outside-tracking-438137.exe") == ("Outside-tracking", "438137")
    with pytest.raises(ValueError):
        batch_cli.identity_from_executable("algorithm.exe")


def test_images_are_numeric_and_continuous(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "im_0002.jpg").write_bytes(b"jpg")
    (folder / "im_0001.JPG").write_bytes(b"jpg")
    assert [item[0] for item in batch_cli.list_official_images(folder)] == [1, 2]
    (folder / "im_0004.jpg").write_bytes(b"jpg")
    with pytest.raises(ValueError, match="continuous"):
        batch_cli.list_official_images(folder)


def test_detection_folder_writes_new_contract(monkeypatch, tmp_path):
    images = tmp_path / "images"
    _jpgs(images)
    calls = []

    monkeypatch.setattr(batch_cli, "load_runtime_config", lambda _: {})

    def fake_detection(path, config, **kwargs):
        calls.append((path.name, kwargs))
        return ([{"bbox": [1, 2, 3, 4], "class_id": 0,
                  "confidence": 0.75}], 3)

    monkeypatch.setattr(batch_cli, "run_detection", fake_detection)
    args = SimpleNamespace(
        input=str(images), config=None, sequence="Inside-detection",
        team_id="438137", output_dir=str(tmp_path / "results"),
    )
    output = batch_cli.execute(args, "ignored.exe")
    result = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "Inside-detection-438137.json"
    assert result["num_frames"] == 2 and result["num_records"] == 2
    assert [row[0] for row in result["detections"]] == [1, 2]
    assert all(kwargs == {"conf_override": 0.0, "topk": 600}
               for _, kwargs in calls)


def test_tracking_folder_assigns_non_reused_ids(monkeypatch, tmp_path):
    images = tmp_path / "images"
    _jpgs(images, 3)
    outputs = iter([
        ([{"bbox": [0, 0, 10, 10], "confidence": 0.9}], 1),
        ([], 1),
        ([{"bbox": [0, 0, 10, 10], "confidence": 0.9}], 1),
    ])
    monkeypatch.setattr(batch_cli, "load_runtime_config", lambda _: {})
    monkeypatch.setattr(batch_cli, "run_detection",
                        lambda path, config: next(outputs))
    args = SimpleNamespace(
        input=str(images), config=None, sequence="Outside-tracking",
        team_id="438137", output_dir=str(tmp_path / "results"),
    )
    result = json.loads(
        batch_cli.execute(args, "ignored.exe").read_text(encoding="utf-8"))
    assert [row[1] for row in result["tracks"]] == [1, 2]
    assert result["num_tracks"] == 2


def test_sequence_and_team_id_must_be_overridden_together(tmp_path):
    args = SimpleNamespace(
        input=str(tmp_path), config=None, sequence="Inside-detection",
        team_id=None, output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="together"):
        batch_cli.execute(args, "Inside-detection-438137.exe")


def test_error_status_is_exactly_one_line(capsys, tmp_path):
    code = batch_cli.main([
        "--input", str(tmp_path / "missing\nfolder"),
        "--sequence", "Inside-detection", "--team-id", "438137",
        "--output-dir", str(tmp_path),
    ])
    output = capsys.readouterr().out
    assert code == 1
    assert len(output.splitlines()) == 1
    assert output.startswith("ERROR ValueError:")
