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
    with pytest.raises(RuntimeError, match="unavailable"):
        cli._get_session(model, 640, str(tmp_path), "cuda")


def test_auto_device_retries_cpu_when_cuda_session_fails(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    attempts = []

    def make_session(*args, **kwargs):
        providers = kwargs["providers"]
        attempts.append(providers)
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("missing CUDA DLL")
        return SimpleNamespace(
            get_outputs=lambda: [1],
            get_providers=lambda: ["CPUExecutionProvider"])

    runtime = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        get_available_providers=lambda: [
            "CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=make_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setattr(cli, "_SESSIONS", {})
    session, _ = cli._get_session(model, 640, str(tmp_path), "auto")
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
