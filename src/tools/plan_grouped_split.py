"""按完整视频分组生成场景平衡的数据集划分，防止相邻帧泄漏。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def stable_order(items: Iterable[Dict], seed: str) -> List[Dict]:
    return sorted(items, key=lambda item: hashlib.sha256(
        f"{seed}|{item['scene']}|{item['video_id']}|{item['sha256']}".encode("utf-8")
    ).hexdigest())


def scene_split_counts(count: int) -> Dict[str, int]:
    """小数据集优先保证验证、测试均含独立视频。"""
    if count <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if count == 1:
        return {"train": 1, "val": 0, "test": 0}
    if count == 2:
        return {"train": 1, "val": 1, "test": 0}
    return {"train": count - 2, "val": 1, "test": 1}


def assign_grouped_splits(videos: List[Dict], seed: str = "bee-competition-v1") -> List[Dict]:
    scenes: Dict[str, List[Dict]] = defaultdict(list)
    for video in videos:
        for required in ("video_id", "source_path", "sha256", "scene"):
            if not video.get(required):
                raise ValueError(f"视频记录缺少 {required}")
        scenes[video["scene"]].append(video)
    assigned = []
    for scene in sorted(scenes):
        ordered = stable_order(scenes[scene], seed)
        counts = scene_split_counts(len(ordered))
        splits = (["train"] * counts["train"] + ["val"] * counts["val"] +
                  ["test"] * counts["test"])
        for video, split in zip(ordered, splits):
            assigned.append({**video, "split": split})
    return sorted(assigned, key=lambda item: (item["split"], item["scene"], item["video_id"]))


def audit_split(videos: List[Dict]) -> Dict:
    allowed = {"train", "val", "test"}
    errors = []
    seen_ids, seen_hashes, seen_paths = {}, {}, {}
    scene_counts: Dict[str, Counter] = defaultdict(Counter)
    split_counts = Counter()
    for video in videos:
        split = video.get("split")
        if split not in allowed:
            errors.append(f"无效 split: {split}")
            continue
        split_counts[split] += 1
        scene_counts[video["scene"]][split] += 1
        for label, value, seen in (
            ("video_id", video["video_id"], seen_ids),
            ("sha256", video["sha256"], seen_hashes),
            ("source_path", str(Path(video["source_path"]).resolve()).lower(), seen_paths),
        ):
            previous = seen.get(value)
            if previous is not None and previous != split:
                errors.append(f"{label} 跨集合泄漏: {value}: {previous}/{split}")
            seen[value] = split
    for scene, counts in scene_counts.items():
        total = sum(counts.values())
        if total >= 3 and (not counts["val"] or not counts["test"]):
            errors.append(f"场景 {scene} 未同时覆盖 val 和 test")
    return {
        "valid": not errors,
        "errors": errors,
        "video_count": len(videos),
        "split_counts": dict(split_counts),
        "scene_split_counts": {scene: dict(counts) for scene, counts in scene_counts.items()},
        "leakage_checks": ["video_id", "sha256", "resolved_source_path"],
        "policy": "完整视频分组；同一视频及其派生片段不得跨 train/val/test。",
    }


def write_plan(input_manifest: Path, output_dir: Path, seed: str) -> Dict:
    source = json.loads(input_manifest.read_text(encoding="utf-8"))
    videos = source.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("输入清单不包含 videos")
    assigned = assign_grouped_splits(videos, seed)
    audit = audit_split(assigned)
    if not audit["valid"]:
        raise ValueError("；".join(audit["errors"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "1.0",
        "source_manifest": str(input_manifest.resolve()),
        "seed": seed,
        "split_unit": "whole_video",
        "test_usage": "测试集只用于最终锁定模型后的单次报告，不用于调参或模型选择。",
        "videos": assigned,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("train", "val", "test"):
        lines = [item["video_id"] for item in assigned if item["split"] == split]
        (output_dir / f"{split}_videos.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"manifest": manifest, "audit": audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按完整视频生成场景平衡的数据集划分")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", default="bee-competition-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = write_plan(args.manifest, args.output, args.seed)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(result["audit"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
