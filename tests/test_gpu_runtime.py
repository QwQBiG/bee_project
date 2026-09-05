from types import SimpleNamespace
import sys

import pytest

from inference import algorithm_cli as cli


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runtime_preloads_only_for_gpu(monkeypatch, tmp_path, device, capsys):
    events = []
    model = tmp_path / "model.onnx"
    model.touch()
    session = SimpleNamespace(get_outputs=lambda: [1],
                              get_providers=lambda: ["CUDAExecutionProvider"])
    def preload():
        events.append("preload")
        print("loader diagnostic")
    runtime = SimpleNamespace(SessionOptions=SimpleNamespace,
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=preload, InferenceSession=lambda *a, **k: session)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setattr(cli, "_SESSIONS", {})
    cli._get_session(model, 640, str(tmp_path), device)
    assert events == (["preload"] if device == "cuda" else [])
    assert not capsys.readouterr().out


def test_requested_gpu_rejects_cpu_fallback(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    runtime = SimpleNamespace(SessionOptions=SimpleNamespace,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *a, **k: SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setattr(cli, "_SESSIONS", {})
    with pytest.raises(RuntimeError, match="fell back"):
        cli._get_session(model, 640, str(tmp_path), "cuda")
