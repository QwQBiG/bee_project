"""Auto-label official bee images with an existing detector → YOLO dataset.

This is the "self-training bootstrap" used by track A: the project already
ships two pretrained detectors (hive-entrance / hive-interior). We run them
over the 300+300 official frames to produce pseudo ground-truth in YOLO
``class cx cy w h`` (normalised) format, split 8:2 into train/val, and emit
a ``data.yaml`` ready for ``YOLOTrainer.train_detection_task``.

Usage::

    py -3.13 tools/auto_label_detection.py \
        --images datasets/official_work/.../outside_300 \
        --model artifacts/models/hive_entrance_bee_yolov8n.pt \
        --out datasets/yolo_outside --imgsz 1280 --scene outside

    py -3.13 tools/auto_label_detection.py \
        --images datasets/official_work/.../inside_300 \
        --model artifacts/models/honey_bee_detector_yolov8s.pt \
        --out datasets/yolo_inside --imgsz 640 --scene inside --conf 0.15
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            out.append(p)
    return out


def write_label(txt_path: Path, rows: List[List[float]]) -> int:
    """Write one YOLO label file. Returns number of boxes written."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        # row = [cx, cy, w, h] all normalised 0..1; single class 0.
        lines.append(f"0 {row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8")
    return len(lines)


def split_indices(n: int, val_ratio: float) -> Tuple[List[int], List[int]]:
    """Deterministic 8:2 split (every k-th frame to val) for reproducibility."""
    step = max(2, int(round(1.0 / max(val_ratio, 1e-6))))
    val_ids = set(i for i in range(n) if i % step == 0)
    train = [i for i in range(n) if i not in val_ids]
    val = sorted(val_ids)
    return train, val


def auto_label(
    images_dir: str,
    model_path: str,
    out_dir: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    val_ratio: float = 0.2,
    copy_images: bool = True,
) -> dict:
    """Run inference on every image, write YOLO labels + data.yaml.

    Returns a summary dict with counts. Empty-detection frames are kept
    (with an empty .txt) so the model learns the background — useful for
    the infrared interior where the teacher detector misses many bees.
    """
    from ultralytics import YOLO  # type: ignore

    images = list_images(Path(images_dir))
    if not images:
        raise FileNotFoundError(f"no images under {images_dir}")

    out_root = Path(out_dir)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = split_indices(len(images), val_ratio)
    split_of = {i: ("train" if i in train_idx else "val")
                for i in range(len(images))}

    model = YOLO(model_path)
    n_total = n_labeled = n_empty = 0
    n_boxes_total = 0
    for i, img in enumerate(images):
        split = split_of[i]
        # Infer on this single frame.
        res = model.predict(
            source=str(img), imgsz=imgsz, conf=conf, iou=iou,
            verbose=False, save=False, save_txt=False, save_conf=False,
        )[0]
        boxes = getattr(res, "boxes", None)
        # xywhn = normalised [cx, cy, w, h] in 0..1.
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy().tolist()
        else:
            xywhn = []
        n_boxes_total += len(xywhn)
        n_total += 1
        if xywhn:
            n_labeled += 1
        else:
            n_empty += 1
        # Dest paths (flat names to keep YOLO happy).
        dst_img = out_root / "images" / split / img.name
        dst_txt = out_root / "labels" / split / (img.stem + ".txt")
        if copy_images:
            shutil.copy2(img, dst_img)
        else:
            try:
                (out_root / "images" / split).mkdir(parents=True,
                                                    exist_ok=True)
                if not dst_img.exists():
                    dst_img.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dst_img)
        write_label(dst_txt, xywhn)

    # data.yaml
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: bee\n",
        encoding="utf-8",
    )
    return {
        "images_dir": images_dir,
        "model": model_path,
        "out_dir": str(out_root),
        "imgsz": imgsz,
        "conf": conf,
        "total_frames": n_total,
        "frames_with_boxes": n_labeled,
        "frames_empty": n_empty,
        "total_boxes": n_boxes_total,
        "train": len(train_idx),
        "val": len(val_idx),
        "data_yaml": str(yaml_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pseudo-label official frames → YOLO dataset")
    ap.add_argument("--images", required=True,
                    help="directory of frames to label")
    ap.add_argument("--model", required=True,
                    help=".pt weights used as the teacher detector")
    ap.add_argument("--out", required=True,
                    help="output dataset root (gets data.yaml)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--scene", default="detect",
                    help="only used for the printed banner")
    args = ap.parse_args()

    summary = auto_label(
        images_dir=args.images,
        model_path=args.model,
        out_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        val_ratio=args.val_ratio,
    )
    banner = f"=== auto-label [{args.scene}] ==="
    print(banner)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    # Heuristic: warn if <40% of frames got any box — the teacher is too
    # weak on this scene and hand-labeling may be required.
    if summary["total_frames"] > 0:
        rate = summary["frames_with_boxes"] / summary["total_frames"]
        print(f"  label_coverage: {rate:.1%}")
        if rate < 0.40:
            print("  WARNING: low coverage — teacher detector is weak here;")
            print("           consider hand-labeling (track B) instead.")


if __name__ == "__main__":
    main()
