from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_cli_always_uses_project_portable_python():
    launcher = read_project_file("run_cli.bat").lower()
    assert ".runtime\\python313\\python.exe" in launcher
    assert '"%bee_python%" "%bee_project_root%main.py" %*' in launcher


def test_setup_does_not_discover_or_reuse_system_python():
    setup = read_project_file("tools/prepare_portable_runtime.ps1").lower()
    assert "get-command python" not in setup
    assert "get-command py" not in setup
    assert "find-externalpython" not in setup
    assert '$pythonversion = "3.13.15"' in setup


def test_cuda_packages_are_pinned_to_matching_builds():
    requirements = read_project_file("requirements.txt").lower()
    setup = read_project_file("tools/prepare_portable_runtime.ps1").lower()
    assert "torch==2.13.0+cu132" in requirements
    assert "torchvision==0.28.0+cu132" in requirements
    assert "lap==0.5.13" in requirements
    assert "https://download.pytorch.org/whl/cu132" in setup


def test_missing_package_check_does_not_use_pip_show_stderr():
    setup = read_project_file("tools/prepare_portable_runtime.ps1").lower()
    assert "-m pip show" not in setup
    assert "importlib.metadata" in setup
