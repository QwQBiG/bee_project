"""Sandbox-safe YOLOv8 detection fine-tune launcher.

Works around two TRAE sandbox traps that block plain ``ultralytics.train``:
  1. ``Arial.ttf`` auto-download to the read-only Ultralytics user dir.
  2. ``settings.json`` writes to ``%APPDATA%\\Ultralytics``.

We redirect the config dir to a temp path and stub ``check_font`` so the
trainer can bootstrap offline. The actual training call is plain
``YOLO(weights).train(...)`` with the scene-correct imgsz.

Usage::

    py -3.13 tools/train_detector_ft.py \
        --data datasets/yolo_outside/data.yaml \
        --model artifacts/models/hive_entrance_bee_yolov8n.pt \
        --scene outside --epochs 12 --batch 4 \
        --project runs/outside_ft --name bee_v1
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def _sandbox_proof_bootstrap() -> None:
    """Pre-empt Ultralytics font download + settings writes."""
    # 1) Redirect the Ultralytics config dir to a writable temp folder so
    #    settings.json / Arial.ttf land somewhere we are allowed to write.
    cfg_dir = Path(tempfile.gettempdir()) / "ultra_cfg_bee"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(cfg_dir))

    # 2) Stub check_font everywhere ultralytics might import it, so the
    #    trainer never tries to fetch Arial.ttf from the network.
    try:
        import ultralytics.utils.checks as _checks  # type: ignore
        _checks.check_font = lambda *a, **k: None  # noqa: E731
    except Exception:
        pass
    try:
        import ultralytics.data.utils as _du  # type: ignore
        _du.check_font = lambda *a, **k: None  # noqa: E731
    except Exception:
        pass
    try:
        import ultralytics.utils as _u  # type: ignore
        if hasattr(_u, "check_font"):
            _u.check_font = lambda *a, **k: None  # noqa: E731
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Sandbox-safe detection fine-tune")
    ap.add_argument("--data", required=True, help="data.yaml path")
    ap.add_argument("--model", required=True, help="base .pt weights")
    ap.add_argument("--scene", default="outside",
                    choices=("outside", "inside"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--project", required=True, help="run project dir")
    ap.add_argument("--name", default="bee_ft")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    _sandbox_proof_bootstrap()

    from ultralytics import YOLO  # type: ignore

    imgsz = 1280 if args.scene == "outside" else 640
    # Absolute project dir keeps ultralytics away from any datasets_dir
    # inherited from a poisoned settings.json on the host.
    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=imgsz,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        verbose=True,
        workers=args.workers,
        # Disable AMP on CPU to avoid mixed-precision warnings/no-ops.
        amp=False,
    )
    best = project / args.name / "weights" / "best.pt"
    last = project / args.name / "weights" / "last.pt"
    print("TRAIN_DONE", args.scene)
    print("best:", best, best.exists())
    print("last:", last, last.exists())
    print("save_dir:", getattr(results, "save_dir", "?"))


if __name__ == "__main__":
    main()
