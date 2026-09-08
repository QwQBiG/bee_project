"""Frozen entry point shared by all four official competition executables."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path


def _configure_console() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _configure_console()
    # Keep third-party settings outside the submission tree and never depend
    # on a writable user profile on the offline evaluation machine.
    settings_dir = Path(tempfile.gettempdir()) / "bee-vision-614689" / "ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
    from inference.batch_cli import main
    raise SystemExit(main())
