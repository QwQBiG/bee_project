"""检查 OIST 蜜蜂检测标注包并生成可复现的数据摘要。

每行格式：offset_x offset_y class position_x position_y angle。
该标注包提供检测位置和朝向，不提供跨帧个体 ID，不能直接计算 IDF1。
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_annotation_file(path: Path) -> list[dict]:
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number} 应有 6 列，实际为 {len(fields)}")
        offset_x, offset_y, class_id, position_x, position_y, angle = fields
        rows.append({
            "offset_x": int(offset_x),
            "offset_y": int(offset_y),
            "class_id": int(class_id),
            "position_x": int(position_x),
            "position_y": int(position_y),
            "angle": float(angle),
        })
    return rows


def summarize(root: Path) -> dict:
    files = sorted(root.glob("frames_txt/*.txt"))
    if not files:
        raise FileNotFoundError(f"未找到 OIST 标注文件：{root / 'frames_txt'}")
    objects = 0
    classes = Counter()
    offsets = Counter()
    angles = []
    for path in files:
        rows = parse_annotation_file(path)
        objects += len(rows)
        classes.update(row["class_id"] for row in rows)
        offsets.update((row["offset_x"], row["offset_y"]) for row in rows)
        angles.extend(row["angle"] for row in rows)

    match = re.search(r"(30|70)fps", root.name)
    return {
        "source": "OIST Honeybee Segmentation and Tracking Datasets",
        "annotation_root": str(root),
        "nominal_fps": int(match.group(1)) if match else None,
        "annotation_frames": len(files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "objects": objects,
        "class_counts": {str(key): value for key, value in sorted(classes.items())},
        "offset_tiles": len(offsets),
        "angle_min": min(angles) if angles else None,
        "angle_max": max(angles) if angles else None,
        "identity_ids_available": False,
        "evaluation_note": "Detection-position/orientation labels only; no cross-frame identity IDs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="解压后的 frame_annotations_*fps 目录")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = summarize(Path(args.root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
