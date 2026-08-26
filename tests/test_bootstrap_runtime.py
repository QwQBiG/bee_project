from tools.bootstrap_runtime import (
    NvidiaInfo,
    backend_from_torch,
    cuda_candidates,
    install_torch,
    parse_cuda_version,
    requested_backend,
)


def test_parse_cuda_version_supports_old_and_new_nvidia_smi_labels():
    assert parse_cuda_version("CUDA Version: 12.8") == (12, 8)
    assert parse_cuda_version("CUDA UMD Version: 13.3") == (13, 3)
    assert parse_cuda_version("No NVIDIA device") is None


def test_cuda_candidates_use_highest_compatible_stable_build_first():
    assert cuda_candidates((13, 3)) == ["cu128", "cu126", "cu118"]
    assert cuda_candidates((12, 7)) == ["cu126", "cu118"]
    assert cuda_candidates((12, 0)) == ["cu118"]
    assert cuda_candidates((11, 7)) == []


def test_auto_backend_selects_only_the_highest_compatible_channel(monkeypatch):
    monkeypatch.delenv("BEE_TORCH_BACKEND", raising=False)
    assert requested_backend(NvidiaInfo(available=True, cuda_max=(13, 3))) == "cu128"
    assert requested_backend(NvidiaInfo()) == "cpu"


def test_backend_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BEE_TORCH_BACKEND", "cpu")
    assert requested_backend(NvidiaInfo(available=True, cuda_max=(13, 3))) == "cpu"


def test_backend_is_read_from_installed_torch_version():
    assert backend_from_torch({"version": "2.11.0+cu128", "cuda_runtime": "12.8"}) == "cu128"
    assert backend_from_torch({"version": "2.13.0+cpu", "cuda_runtime": None}) == "cpu"


def test_install_validates_once_without_downloading_fallbacks(monkeypatch):
    monkeypatch.delenv("BEE_TORCH_BACKEND", raising=False)
    inspections = iter(
        [
            {"version": "2.13.0+cpu", "cuda_runtime": None, "cuda_available": False},
            {"version": "2.11.0+cu128", "cuda_runtime": "12.8", "cuda_available": True},
        ]
    )
    pip_calls = []
    monkeypatch.setattr("tools.bootstrap_runtime.inspect_torch", lambda: next(inspections))
    monkeypatch.setattr(
        "tools.bootstrap_runtime.run_pip", lambda arguments: pip_calls.append(arguments) or True
    )

    backend, info = install_torch(NvidiaInfo(available=True, cuda_max=(13, 3)))

    assert backend == "cu128"
    assert info["cuda_available"] is True
    assert len(pip_calls) == 1
    assert "https://download.pytorch.org/whl/cu128" in pip_calls[0]
