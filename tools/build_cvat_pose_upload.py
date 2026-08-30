"""将多 subset 的 YOLO Pose 数据集整理为单个 CVAT 标注任务。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Dict, Iterable
import zipfile


DATA_YAML = """path: ./
train: train.txt
kpt_shape: [3, 3]
flip_idx: [0, 1, 2]
names:
  0: bee
"""
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _normalize_lf(data: bytes) -> bytes:
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if text and not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def _collect_entries(archive: zipfile.ZipFile, roots: Iterable[str],
                     subsets: set[str]) -> Dict[str, Dict[str, str]]:
    collected = {root: {} for root in roots}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or len(path.parts) < 3:
            continue
        root, subset = path.parts[0], path.parts[1]
        if root not in collected or subset not in subsets:
            continue
        name = path.name
        if root == "images" and path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if root == "labels" and path.suffix.lower() != ".txt":
            continue
        if name in collected[root]:
            raise ValueError(f"合并 subset 后文件名重复: {name}")
        collected[root][name] = info.filename
    return collected


def build_single_task_archive(source: Path, output: Path,
                              subsets: Iterable[str] = ("train", "val"),
                              overwrite: bool = False) -> Dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出已存在: {output}")
    included = set(subsets)
    if not included:
        raise ValueError("至少需要包含一个 subset")

    with zipfile.ZipFile(source) as archive:
        entries = _collect_entries(archive, ("images", "labels"), included)
        images, labels = entries["images"], entries["labels"]
        if not images:
            raise ValueError("没有找到待上传图片")
        image_stems = {Path(name).stem for name in images}
        label_stems = {Path(name).stem for name in labels}
        missing = sorted(image_stems - label_stems)
        extra = sorted(label_stems - image_stems)
        if missing or extra:
            raise ValueError(f"图片与标签不匹配: missing={missing}, extra={extra}")

        rows = 0
        normalized_labels: Dict[str, bytes] = {}
        for name, source_name in labels.items():
            data = _normalize_lf(archive.read(source_name))
            for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) not in {14, 15}:
                    raise ValueError(f"{source_name}:{line_number} 不是 YOLO Pose 标签")
                rows += 1
            normalized_labels[name] = data

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
            target.writestr("data.yaml", DATA_YAML.encode("utf-8"))
            train_lines = "".join(
                f"images/train/{name}\n" for name in sorted(images))
            target.writestr("train.txt", train_lines.encode("utf-8"))
            for name in sorted(images):
                target.writestr(f"images/train/{name}", archive.read(images[name]))
                label_name = f"{Path(name).stem}.txt"
                target.writestr(
                    f"labels/train/{label_name}", normalized_labels[label_name])

    return {
        "source": str(source),
        "output": str(output),
        "included_subsets": sorted(included),
        "cvat_subsets": ["train"],
        "images": len(images),
        "labels": len(labels),
        "annotation_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 train/val YOLO Pose 包合并为一个 CVAT 任务")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-subset", action="append", choices=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = build_single_task_archive(
            args.source, args.output, args.include_subset or ("train", "val"), args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
