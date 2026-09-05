from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()


def test_obsolete_portable_runtime_launchers_are_removed():
    assert not (PROJECT_ROOT / "setup_runtime.bat").exists()
    assert not (PROJECT_ROOT / "run_cli.bat").exists()
    assert not (PROJECT_ROOT / "tools" / "prepare_portable_runtime.ps1").exists()


def test_environment_guide_uses_python_313_venv_and_local_torch_wheel():
    guide = read_project_file("packages/README.md")
    assert "py -3.13 -m venv .venv" in guide
    assert "torch-2.13.0+cu132-cp313-cp313-win_amd64.whl" in guide
    assert ".venv\\scripts\\python.exe" in guide


def test_environment_guide_also_documents_system_python():
    guide = read_project_file("packages/README.md")
    assert "方案 b：直接使用系统 python" in guide
    assert 'py -3.13 -m pip install "packages\\torch-2.13.0+cu132-cp313-cp313-win_amd64.whl"' in guide
    assert "py -3.13 -m pip install -r requirements.txt" in guide
    assert "py -3.13 main.py --mode outside" in guide


def test_environment_guide_installs_matching_torchvision_and_dependencies():
    guide = read_project_file("packages/README.md")
    requirements = read_project_file("requirements.txt")
    assert "torchvision==0.28.0+cu132" in guide
    assert "https://download.pytorch.org/whl/cu132" in guide
    assert "-r requirements.txt" in guide
    assert "lap==0.5.13" in requirements
