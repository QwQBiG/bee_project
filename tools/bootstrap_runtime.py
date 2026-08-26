"""Prepare Bee Vision's local Python/PyTorch runtime without repeated downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import struct
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = PROJECT_ROOT / "runtime_environment.json"
LOCK_PATH = PROJECT_ROOT / ".runtime-bootstrap.lock"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-web.txt"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{backend}"
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 14)
BOOTSTRAP_VERSION = 2


@dataclass
class NvidiaInfo:
    available: bool = False
    name: str | None = None
    driver_version: str | None = None
    cuda_max: tuple[int, int] | None = None
    compute_capability: str | None = None


def parse_cuda_version(output: str) -> tuple[int, int] | None:
    match = re.search(r"CUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def detect_nvidia() -> NvidiaInfo:
    try:
        banner = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15, check=False
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return NvidiaInfo()
    if banner.returncode != 0:
        return NvidiaInfo()

    info = NvidiaInfo(available=True, cuda_max=parse_cuda_version(banner.stdout))
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if query.returncode == 0 and query.stdout.strip():
        values = [part.strip() for part in query.stdout.splitlines()[0].split(",")]
        if values:
            info.name = values[0]
        if len(values) > 1:
            info.driver_version = values[1]
        if len(values) > 2:
            info.compute_capability = values[2]
    return info


def cuda_candidates(cuda_max: tuple[int, int] | None) -> list[str]:
    """Return supported wheel channels, highest compatible channel first."""
    if cuda_max is None:
        return []
    candidates: list[str] = []
    if cuda_max >= (12, 8):
        candidates.append("cu128")
    if cuda_max >= (12, 6):
        candidates.append("cu126")
    if cuda_max >= (11, 8):
        candidates.append("cu118")
    return candidates


def requirements_digest() -> str:
    return hashlib.sha256(REQUIREMENTS_PATH.read_bytes()).hexdigest()


def installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_torch() -> dict[str, object] | None:
    """Inspect Torch in a fresh process so an old imported module cannot be cached."""
    code = r'''
import json
try:
    import torch
    cuda_ok = False
    device_name = None
    error = None
    try:
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            torch.zeros(1, device="cuda:0")
            device_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        error = str(exc)
    result = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_ok,
        "device_name": device_name,
        "error": error,
    }
    print("BEE_TORCH_INFO=" + json.dumps(result, ensure_ascii=True))
except (ImportError, OSError) as exc:
    print("BEE_TORCH_ERROR=" + repr(exc))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("BEE_TORCH_INFO="):
            return json.loads(line.removeprefix("BEE_TORCH_INFO="))
    return None


def run_pip(arguments: list[str]) -> bool:
    command = [sys.executable, "-m", "pip", "--disable-pip-version-check", *arguments]
    print("[Bee Vision] " + " ".join(command))
    return subprocess.run(command, check=False).returncode == 0


def requested_backend(nvidia: NvidiaInfo) -> str:
    requested = os.environ.get("BEE_TORCH_BACKEND", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cu"):
        return requested
    candidates = cuda_candidates(nvidia.cuda_max) if nvidia.available else []
    return candidates[0] if candidates else "cpu"


def backend_from_torch(torch_info: dict[str, object]) -> str:
    version = str(torch_info.get("version") or "")
    match = re.search(r"\+(cu\d+|cpu)$", version)
    if match:
        return match.group(1)
    runtime = str(torch_info.get("cuda_runtime") or "")
    return "cu" + runtime.replace(".", "") if runtime else "cpu"


def install_torch(nvidia: NvidiaInfo, force: bool = False) -> tuple[str, dict[str, object]]:
    target = requested_backend(nvidia)
    existing = inspect_torch()
    if existing and not force:
        if target == "cpu" or bool(existing.get("cuda_available")):
            return backend_from_torch(existing), existing

    print(f"[Bee Vision] Installing PyTorch backend once: {target}")
    arguments = ["install", "torch", "torchvision"]
    if existing is not None or force:
        arguments.append("--force-reinstall")
    arguments.extend(["--index-url", PYTORCH_INDEX.format(backend=target)])
    if not run_pip(arguments):
        raise RuntimeError(
            f"PyTorch {target} installation failed. It will not download another backend automatically."
        )

    installed = inspect_torch()
    if installed is None:
        raise RuntimeError("PyTorch was installed but cannot be imported in a fresh process.")
    if target != "cpu" and not installed.get("cuda_available"):
        raise RuntimeError(
            f"PyTorch {target} was installed, but CUDA validation failed: "
            f"{installed.get('error') or 'CUDA is unavailable'}"
        )
    return target, installed


def load_status() -> dict | None:
    try:
        value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def cached_environment_is_current(status: dict | None, nvidia: NvidiaInfo) -> bool:
    if not status or status.get("bootstrap_version") != BOOTSTRAP_VERSION:
        return False
    if status.get("requirements_sha256") != requirements_digest():
        return False
    if os.path.normcase(str(status.get("python_executable", ""))) != os.path.normcase(sys.executable):
        return False
    torch_info = status.get("torch")
    if not isinstance(torch_info, dict):
        return False
    if installed_distribution_version("torch") != torch_info.get("version"):
        return False
    if installed_distribution_version("torchvision") != status.get("torchvision_version"):
        return False
    target = requested_backend(nvidia)
    if target != "cpu" and not torch_info.get("cuda_available"):
        return False
    return True


def write_status(backend: str, torch_info: dict[str, object], nvidia: NvidiaInfo) -> dict:
    status = {
        "bootstrap_version": BOOTSTRAP_VERSION,
        "requirements_sha256": requirements_digest(),
        "device": "cuda:0" if torch_info["cuda_available"] else "cpu",
        "backend": backend,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torchvision_version": installed_distribution_version("torchvision"),
        "torch": torch_info,
        "nvidia": asdict(nvidia),
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)
    return status


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def bootstrap_lock():
    for _ in range(2):
        try:
            descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(LOCK_PATH.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                owner = 0
            if process_exists(owner):
                raise RuntimeError("Another Bee Vision startup is already checking the environment.")
            LOCK_PATH.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
        break
    else:
        raise RuntimeError("Unable to acquire the environment setup lock.")
    try:
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def validate_python() -> None:
    version = sys.version_info[:2]
    if not (SUPPORTED_PYTHON_MIN <= version <= SUPPORTED_PYTHON_MAX):
        raise RuntimeError(
            f"Python {version[0]}.{version[1]} is unsupported; use Python 3.10-3.14."
        )
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("64-bit Python is required.")
    if platform.system() not in {"Windows", "Linux", "Darwin"}:
        raise RuntimeError(f"Unsupported operating system: {platform.system()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-torch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_python()
    with bootstrap_lock():
        nvidia = detect_nvidia()
        print(
            f"[Bee Vision] Python {platform.python_version()} | "
            f"NVIDIA: {nvidia.name or 'not detected'} | "
            f"Driver CUDA: {'.'.join(map(str, nvidia.cuda_max)) if nvidia.cuda_max else 'none'}"
        )
        if args.dry_run:
            print(f"[Bee Vision] Selected backend: {requested_backend(nvidia)}")
            return

        cached = load_status()
        if not args.force_torch and cached_environment_is_current(cached, nvidia):
            torch_info = cached["torch"]
            print(
                f"[Bee Vision] Ready (cached): device={cached['device']} "
                f"torch={torch_info['version']} gpu={torch_info.get('device_name') or 'none'}"
            )
            return

        backend, torch_info = install_torch(nvidia, args.force_torch)
        if not run_pip(["install", "-r", str(REQUIREMENTS_PATH)]):
            raise RuntimeError("Application dependency installation failed.")
        status = write_status(backend, torch_info, nvidia)
        print(
            f"[Bee Vision] Ready: device={status['device']} "
            f"torch={torch_info['version']} gpu={torch_info.get('device_name') or 'none'}"
        )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.SubprocessError, subprocess.TimeoutExpired) as error:
        print(f"[Bee Vision] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
