from pathlib import Path

from tools.bootstrap_runtime import (
    LocalTorchWheel,
    NvidiaInfo,
    backend_from_torch,
    backend_cuda_version,
    compatible_local_torch_wheel,
    cuda_candidates,
    install_torch,
    parse_local_torch_wheel,
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


def test_auto_backend_uses_cpu_without_a_compatible_local_wheel(monkeypatch):
    monkeypatch.delenv("BEE_TORCH_BACKEND", raising=False)
    monkeypatch.setattr("tools.bootstrap_runtime.compatible_local_torch_wheel", lambda _: None)
    assert requested_backend(NvidiaInfo(available=True, cuda_max=(13, 3))) == "cpu"
    assert requested_backend(NvidiaInfo()) == "cpu"


def test_backend_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BEE_TORCH_BACKEND", "cpu")
    assert requested_backend(NvidiaInfo(available=True, cuda_max=(13, 3))) == "cpu"


def test_backend_is_read_from_installed_torch_version():
    assert backend_from_torch({"version": "2.11.0+cu128", "cuda_runtime": "12.8"}) == "cu128"
    assert backend_from_torch({"version": "2.13.0+cpu", "cuda_runtime": None}) == "cpu"


def test_local_wheel_filename_and_cuda_backend_are_parsed():
    wheel = parse_local_torch_wheel(
        Path("torch-2.13.0+cu132-cp313-cp313-win_amd64.whl")
    )
    assert wheel is not None
    assert wheel.backend == "cu132"
    assert wheel.python_tag == "cp313"
    assert backend_cuda_version(wheel.backend) == (13, 2)


def test_compatible_local_wheel_checks_python_platform_and_driver(monkeypatch, tmp_path):
    wheel_path = tmp_path / "torch-2.13.0+cu132-cp313-cp313-win_amd64.whl"
    wheel_path.touch()
    monkeypatch.setattr("tools.bootstrap_runtime.LOCAL_WHEEL_DIR", tmp_path)
    monkeypatch.setattr("tools.bootstrap_runtime.current_python_tag", lambda: "cp313")
    monkeypatch.setattr("tools.bootstrap_runtime.current_platform_tag", lambda: "win_amd64")

    wheel = compatible_local_torch_wheel(
        NvidiaInfo(available=True, cuda_max=(13, 2))
    )
    assert wheel is not None
    assert wheel.path == wheel_path
    assert compatible_local_torch_wheel(
        NvidiaInfo(available=True, cuda_max=(13, 1))
    ) is None


def test_install_validates_once_without_downloading_fallbacks(monkeypatch):
    monkeypatch.setenv("BEE_TORCH_BACKEND", "cu128")
    monkeypatch.setattr("tools.bootstrap_runtime.compatible_local_torch_wheel", lambda _: None)
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


def test_install_prefers_a_compatible_local_wheel(monkeypatch, tmp_path):
    wheel = LocalTorchWheel(
        path=tmp_path / "torch-2.13.0+cu132-cp313-cp313-win_amd64.whl",
        version="2.13.0",
        backend="cu132",
        python_tag="cp313",
        abi_tag="cp313",
        platform_tag="win_amd64",
    )
    monkeypatch.delenv("BEE_TORCH_BACKEND", raising=False)
    monkeypatch.setattr("tools.bootstrap_runtime.compatible_local_torch_wheel", lambda _: wheel)
    inspections = iter(
        [
            None,
            {"version": "2.13.0+cu132", "cuda_runtime": "13.2", "cuda_available": True},
        ]
    )
    pip_calls = []
    monkeypatch.setattr("tools.bootstrap_runtime.inspect_torch", lambda: next(inspections))
    monkeypatch.setattr(
        "tools.bootstrap_runtime.run_pip", lambda arguments: pip_calls.append(arguments) or True
    )

    backend, info = install_torch(NvidiaInfo(available=True, cuda_max=(13, 2)))

    assert backend == "cu132"
    assert info["cuda_available"] is True
    assert str(wheel.path) in pip_calls[0]
