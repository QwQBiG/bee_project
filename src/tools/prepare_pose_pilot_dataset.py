"""Build a local YOLO pose pilot dataset from a CVAT annotation-only export."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--skip-outside-keypoints", action="store_true")
    return parser.parse_args()


def corrected_row(row: str, padding: float) -> tuple[str, bool]:
    fields = row.split()
    if len(fields) != 14:
        raise ValueError(f"expected 14 fields, got {len(fields)}")
    values = [float(value) for value in fields]
    keypoints = [(values[i], values[i + 1], values[i + 2]) for i in range(5, 14, 3)]
    if any(not 0.0 <= value <= 1.0 for point in keypoints for value in point[:2]):
        raise ValueError("keypoint coordinate is outside [0, 1]")

    head, _, abdomen_tip = keypoints
    body_length = math.hypot(head[0] - abdomen_tip[0], head[1] - abdomen_tip[1])
    pad = padding * body_length
    xs = [point[0] for point in keypoints]
    ys = [point[1] for point in keypoints]
    left, right = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
    top, bottom = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
    bbox = [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]

    original = values[1:5]
    changed = any(abs(a - b) > 1e-8 for a, b in zip(original, bbox))
    output = [values[0], *bbox, *values[5:]]
    return " ".join(f"{value:.6f}" for value in output), changed


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    images_out = args.output / "images" / "train"
    labels_out = args.output / "labels" / "train"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    total = 0
    changed = 0
    skipped_outside_keypoints = 0
    counts: dict[str, int] = {}
    with zipfile.ZipFile(args.annotations) as archive:
        label_names = sorted(
            name for name in archive.namelist() if name.startswith("labels/train/") and name.endswith(".txt")
        )
        for label_name in label_names:
            stem = Path(label_name).stem
            source_image = args.images / f"{stem}.jpg"
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            shutil.copy2(source_image, images_out / source_image.name)

            rows = archive.read(label_name).decode("utf-8").splitlines()
            converted: list[str] = []
            for row in rows:
                if not row.strip():
                    continue
                try:
                    output_row, was_changed = corrected_row(row, args.padding)
                except ValueError as error:
                    if args.skip_outside_keypoints and "outside [0, 1]" in str(error):
                        skipped_outside_keypoints += 1
                        continue
                    raise
                converted.append(output_row)
                total += 1
                changed += int(was_changed)
            (labels_out / f"{stem}.txt").write_text("\n".join(converted) + "\n", encoding="utf-8")
            counts[stem] = len(converted)

    image_names = sorted(path.name for path in images_out.glob("*.jpg"))
    (args.output / "train.txt").write_text(
        "\n".join(f"images/train/{name}" for name in image_names) + "\n", encoding="utf-8"
    )
    (args.output / "data.yaml").write_text(
        "path: .\ntrain: train.txt\nval: train.txt\nkpt_shape: [3, 3]\nnames:\n  0: bee\n",
        encoding="utf-8",
    )
    metadata = {
        "status": "pilot_only_not_final_training_set",
        "images": len(image_names),
        "instances": total,
        "instances_with_recomputed_bbox": changed,
        "skipped_instances_with_outside_keypoints": skipped_outside_keypoints,
        "keypoints": ["head", "thorax", "abdomen_tip"],
        "bbox_method": "keypoint_extent_plus_body_length_padding",
        "bbox_padding_factor": args.padding,
        "counts": counts,
        "note": "Manual keypoints are unchanged. Bounding boxes were recomputed for YOLO pose training.",
    }
    (args.output / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
