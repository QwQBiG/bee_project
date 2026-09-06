"""Tests for the official folder-level competition entry point."""

from __future__ import annotations

import json
from pathlib import Path
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
        assert config["_scene"] == "inside"
        calls.append((path.name, kwargs))
        return ([{"bbox": [1, 2, 3, 4], "class_id": 0,
                  "confidence": 0.75}], 3)

    monkeypatch.setattr(batch_cli, "run_detection", fake_detection)
    args = SimpleNamespace(
        input=str(images), config=None, sequence="Inside-detection",
        team_id="438137", output_dir=str(tmp_path / "results"),
    )
    output = batch_cli.execute(args, "batch_cli.py")
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
        batch_cli.execute(args, "batch_cli.py").read_text(encoding="utf-8"))
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
    captured = capsys.readouterr()
    output = captured.out
    assert code == 0
    assert len(output.splitlines()) == 1
    payload = json.loads(output)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"


def test_success_status_is_json_and_process_code_is_zero(monkeypatch, capsys, tmp_path):
    result = tmp_path / "Inside-detection-438137.json"
    result.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(batch_cli, "execute", lambda args, executable: result)
    monkeypatch.setattr(batch_cli.sys, "argv", ["Inside-detection-438137.exe"])
    code = batch_cli.main(["--input", str(tmp_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload == {
        "status": "ok",
        "team_id": "438137",
        "sequence": "Inside-detection",
        "output_path": result.resolve().as_posix(),
    }
    assert captured.err == ""


def test_result_write_falls_back_beside_executable(monkeypatch, tmp_path):
    preferred = tmp_path / "blocked"
    executable = tmp_path / "bundle" / "Inside-detection-438137.exe"
    executable.parent.mkdir()
    calls = []

    def fake_write(path, payload):
        calls.append(path)
        if Path(path).parent == preferred:
            raise PermissionError("read only")
        Path(path).write_text("{}", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr(batch_cli, "write_result", fake_write)
    output = batch_cli._write_with_fallback(
        {}, "Inside-detection-438137.json", preferred, executable)
    assert output.parent == executable.parent.resolve()
    assert len(calls) == 2


def test_detection_collapses_internal_model_classes_to_official_bee_class(
    monkeypatch, tmp_path):
    images = tmp_path / "Inside" / "detection" / "images"
    images.parent.mkdir(parents=True)
    _jpgs(images, 1)
    monkeypatch.setattr(batch_cli, "load_runtime_config", lambda _: {
        "runtime": {"device": "cpu"},
        "tracking": {"inside": {}, "outside": {}},
        "tiling": {"inside": {}, "outside": {}},
        "detector": {"inside": {"detection_box_scale": 1.0}},
    })
    monkeypatch.setattr(batch_cli, "run_detection", lambda *_a, **_k: ([{
        "class_id": 2, "confidence": 0.9, "bbox": [1, 2, 3, 4],
    }], 1))
    args = batch_cli.build_parser().parse_args([
        "--input", str(images), "--sequence", "Inside-detection",
        "--team-id", "614689", "--output-dir", str(tmp_path / "results"),
    ])
    payload = json.loads(
        batch_cli.execute(args, "source.py").read_text(encoding="utf-8"))
    assert payload["detections"][0][1] == 0
