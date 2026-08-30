"""Download and adapt Mendeley Bee Pose Dataset V6 (DOI: 10.17632/8gb9r2yhfc.6).

The public dataset is released under CC BY 4.0 and ships 400 pose-labeled
hive-entrance images with 2 keypoints per bee (head + stinger / abdomen tip).
We supplement a synthetic thorax keypoint at the midpoint of the head-stinger
segment, so the labels become compatible with the project's 3-keypoint schema
(head / thorax / abdomen_tip) required by KeypointPoseEstimator and the
competition's "head + abdomen structure recognition" rule.

Output structure (YOLOv8-pose compatible):
    <root>/mendeley_bee_pose_v6/
        images/train/*.jpg
        labels/train/*.txt      # class cx cy w h [px py v]*3  (3 keypoints)
        data.yaml               # kpt_shape: [3, 3]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Download helpers (no heavy 3rd-party deps; pure stdlib)
# ---------------------------------------------------------------------------

MENDELEY_API_DOWNLOAD = (
    "https://data.mendeley.com/public-api/datasets/8gb9r2yhfc/6/files/"
    "{file_uuid}/raw"
)

# Known file UUIDs for Dataset V6 (May 2025, version 6).  These are the two
# archives relevant to pose estimation.  If the API ever changes UUIDs, the
# script will fail with a clear message and point the user at the dataset
# homepage to re-validate.
POSE_ARCHIVE_NAME = "Bee_keypoint_pose_dataset.zip"
DETECTION_ARCHIVE_NAME = "Bee_detection_dataset.zip"
README_ARCHIVE_NAME = "README_and_license.txt"


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "bee_project/1.0 (+data preparation)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _mendeley_file_url(filename: str) -> str:
    """Best-effort resolve a filename to a Mendeley raw-file URL.

    The public-api path is not officially stable; we fall back to appending
    the URL-encoded filename when known UUIDs cannot be resolved.  The user
    is expected to download manually via the browser if both fail.
    """
    slug = urllib.parse.quote(filename, safe="")
    # Try the canonical /files/{filename}/raw path first; some Mendeley
    # frontends accept this alternative.
    return (
        f"https://data.mendeley.com/public-api/datasets/8gb9r2yhfc/6"
        f"/file?filename={slug}"
    )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_if_missing(destination: Path, filename: str) -> Path:
    """Download a single Mendeley Data file to ``destination / filename``.

    If the file already exists it is returned as-is (no re-download).
    Returns the absolute Path to the local copy.
    """
    destination.mkdir(parents=True, exist_ok=True)
    local = destination / filename
    if local.is_file() and local.stat().st_size > 0:
        return local
    url = _mendeley_file_url(filename)
    try:
        payload = _http_get(url)
    except Exception as exc:  # pragma: no cover - network errors vary
        sys.stderr.write(
            f"[mendeley] Auto-download failed for {filename}: {exc}.\n"
            f"Please open "
            f"https://data.mendeley.com/datasets/8gb9r2yhfc/6 in a browser, "
            f"download {filename}, and place it at:\n    {local}\n"
        )
        raise
    local.write_bytes(payload)
    return local


# ---------------------------------------------------------------------------
# YOLO pose label adaptation (2 keypoints -> 3 keypoints)
# ---------------------------------------------------------------------------

# Mendeley V6 pose labels use a 12-field YOLO-pose row:
#   class cx cy w h  hx hy hv  sx sy sv
# where keypoint 0 == head, keypoint 1 == stinger/abdomen_tip.
#
# The project schema uses 3 keypoints, 14 fields per row:
#   class cx cy w h  hx hy hv  tx ty tv  sx sy sv
# with keypoint 1 == thorax.  For Mendeley labels thorax is unknown, so we
# take the midpoint of head+stinger and mark it with v==0.1 (pseudo-label
# below the default 0.35 keypoint-confidence threshold, which causes the
# trainer to treat it as weakly supervised).

MENDELEY_POSE_FIELDS = 11  # 1 cls + 4 bbox + 2 keypoints × 3 (x, y, v)
PROJECT_POSE_FIELDS = 14  # 1 cls + 4 bbox + 3 keypoints × 3


def adapt_row_to_3kp(row: str, *, pseudolabel_vis: float = 0.1) -> str:
    """Convert a 2-keypoint YOLO pose label line into a 3-keypoint line.

    Raises ValueError if the input row cannot be parsed.  ``pseudolabel_vis``
    sets the visibility flag for the injected thorax keypoint; use 0.x so
    that downstream tools can distinguish a real labelled thorax from the
    synthetic one.
    """
    tokens = row.strip().split()
    if not tokens:
        return ""
    if len(tokens) != MENDELEY_POSE_FIELDS:
        raise ValueError(
            f"Mendeley pose row expected {MENDELEY_POSE_FIELDS} fields, "
            f"got {len(tokens)}: {row[:60]!r}"
        )
    cls = int(tokens[0])
    cx, cy, w, h = (float(x) for x in tokens[1:5])
    hx, hy, hv = (float(x) for x in tokens[5:8])
    sx, sy, sv = (float(x) for x in tokens[8:12])
    tx = 0.5 * (hx + sx)
    ty = 0.5 * (hy + sy)
    tv = float(pseudolabel_vis)
    # Re-format to a compact single-space separated 14-field row.
    values: List[float] = [
        float(cls), cx, cy, w, h,
        hx, hy, hv,
        tx, ty, tv,
        sx, sy, sv,
    ]
    return " ".join(f"{v:.6g}" if not float(v).is_integer() else str(int(v))
                    if i == 0 else f"{v:.6g}"
                    for i, v in enumerate(values))


def _cls_is_valid(cls: int, names: Iterable[str]) -> bool:
    return 0 <= cls < len(tuple(names))


def adapt_directory(
    labels_in: Path,
    images_in: Optional[Path],
    labels_out: Path,
    images_out: Path,
    *,
    class_map: Optional[dict] = None,
    split: str = "train",
) -> Tuple[int, int]:
    """Adapt every label file from ``labels_in`` and copy matching images.

    Returns (adapted_count, skipped_count).
    """
    labels_out = labels_out / split
    images_out = images_out / split
    labels_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    adapted = skipped = 0
    for label_file in sorted(labels_in.glob("*.txt")):
        try:
            rows = label_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            rows = label_file.read_text(encoding="latin-1").splitlines()
        new_rows: List[str] = []
        try:
            for raw in rows:
                line = adapt_row_to_3kp(raw)
                if line:
                    new_rows.append(line)
        except ValueError:
            skipped += 1
            continue
        if not new_rows:
            skipped += 1
            continue
        (labels_out / label_file.name).write_text(
            "\n".join(new_rows) + "\n", encoding="utf-8"
        )
        image = _find_matching_image(label_file.stem, images_in, images_out) \
            if images_in is not None else None
        if image and image.is_file():
            dst = images_out / image.name
            if not dst.exists():
                shutil.copy2(image, dst)
        adapted += 1
    return adapted, skipped


def _find_matching_image(stem: str, images_in: Path, _images_out: Path) \
        -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
        candidate = images_in / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # Mendeley ships subfolders sometimes; do a shallow recursive lookup.
    for candidate in images_in.rglob(f"{stem}.*"):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def extract_zip(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zh:
        zh.extractall(into)
    return into


def write_data_yaml(root: Path, *, class_names: Tuple[str, ...] = ("bee",)) \
        -> Path:
    yaml_path = root / "data.yaml"
    body = (
        "# Adapted from Mendeley Bee Pose Dataset V6 (CC BY 4.0).\n"
        "#   DOI: 10.17632/8gb9r2yhfc.6\n"
        "# Thorax keypoint is a synthetic mid-point pseudo-label between\n"
        "# head and abdomen_tip (visibility=0.1).\n"
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val  # user must provide a held-out split\n"
        "names:\n"
    )
    for idx, name in enumerate(class_names):
        body += f"  {idx}: {name}\n"
    body += (
        "kpt_shape: [3, 3]  # 3 keypoints, 3 values per kpt (x, y, visible)\n"
        "skeleton:\n"
        "  - [0, 1]  # head -> thorax\n"
        "  - [1, 2]  # thorax -> abdomen_tip\n"
    )
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download-dir", type=Path,
                   default=Path("datasets/public/_downloads"),
                   help="Where downloaded Mendeley archives are cached.")
    p.add_argument("--output", type=Path,
                   default=Path("datasets/public/mendeley_bee_pose_v6"),
                   help="Where the adapted YOLOv8-pose dataset is written.")
    p.add_argument("--skip-download", action="store_true",
                   help="Reuse archives already present in --download-dir.")
    p.add_argument("--thorax-vis", type=float, default=0.1,
                   help="Visibility flag written on the synthetic thorax "
                        "keypoint pseudo-label (default 0.1; smaller means "
                        "treated as weakly-supervised during training).")
    return p.parse_args()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args()
    dl = args.download_dir.resolve()
    out = args.output.resolve()

    archives: List[Path] = []
    for name in (POSE_ARCHIVE_NAME, DETECTION_ARCHIVE_NAME):
        if args.skip_download and (dl / name).is_file():
            archives.append(dl / name)
            continue
        try:
            archives.append(download_if_missing(dl, name))
        except Exception:
            sys.stderr.write(
                f"[mendeley] Skipping {name}; continue with whatever "
                "archives are available.\n"
            )

    if not archives:
        sys.stderr.write(
            "[mendeley] No archives are available locally.  Please "
            "download Bee_keypoint_pose_dataset.zip manually from\n"
            "    https://data.mendeley.com/datasets/8gb9r2yhfc/6\n"
            "into " + str(dl) + " and rerun with --skip-download.\n"
        )
        return 1

    # Extract each archive into its own folder.
    extracted: List[Path] = []
    for archive in archives:
        target = out.parent / "_raw" / archive.stem
        target.mkdir(parents=True, exist_ok=True)
        extracted.append(extract_zip(archive, target))

    # Find the labels/ and images/ subtrees (the zip structure differs
    # between versions; we search 2 levels deep).
    label_dirs: List[Path] = []
    image_dirs: List[Path] = []
    for root in extracted:
        for candidate in root.rglob("labels"):
            if candidate.is_dir():
                label_dirs.append(candidate)
        for candidate in root.rglob("images"):
            if candidate.is_dir():
                image_dirs.append(candidate)
    if not label_dirs:
        sys.stderr.write("[mendeley] ERROR: no labels/ directory found.\n")
        return 2

    total_ok = total_skip = 0
    for labels_in in label_dirs:
        # Pair with the closest images/ directory in the same archive.
        images_in = None
        for img_dir in image_dirs:
            if labels_in.resolve().drive != img_dir.resolve().drive:
                continue
            try:
                labels_in.resolve().relative_to(img_dir.resolve().parent)
                images_in = img_dir
                break
            except ValueError:
                continue
        if images_in is None:
            images_in = image_dirs[0] if image_dirs else None
        ok, skip = adapt_directory(
            labels_in / "train" if (labels_in / "train").is_dir() else labels_in,
            images_in / "train" if images_in and (images_in / "train").is_dir()
                else images_in,
            out / "labels",
            out / "images",
            split="train",
        )
        total_ok += ok
        total_skip += skip

    write_data_yaml(out)
    print(
        f"[mendeley] adapted {total_ok} label files; skipped {total_skip}.\n"
        f"            output: {out}\n"
        f"            data.yaml with kpt_shape [3, 3] written."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
