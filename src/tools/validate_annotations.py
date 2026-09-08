"""校验统一标注 JSON，防止预测结果混入金标准集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import VideoAnnotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验蜜蜂视频统一标注文件")
    parser.add_argument("input", help="单个 JSON 或标注目录")
    parser.add_argument("--gold", action="store_true", help="要求全部实例、事件和行为均为人工真值")
    parser.add_argument("--pose-gold", action="store_true",
                        help="要求人工头胸腹关键点完整，且头和腹尖可用于朝向")
    parser.add_argument("--require-track-ids", action="store_true",
                        help="姿态金标准中的每个实例还必须包含 track_id")
    parser.add_argument("--report", help="可选的 JSON 校验报告路径")
    return parser.parse_args()


def annotation_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".json" else []
    if path.is_dir():
        return sorted(path.rglob("*.json"))
    return []


def validate_path(path: Path, require_manual: bool = False, pose_gold: bool = False,
                  require_track_ids: bool = False) -> Dict:
    files = annotation_files(path)
    if not files:
        raise FileNotFoundError(f"没有找到标注 JSON: {path}")
    results = []
    total_frames = total_instances = total_events = total_behaviors = 0
    for file_path in files:
        try:
            annotation = VideoAnnotation.load(file_path)
            errors = (annotation.validate_pose_gold(require_track_ids)
                      if pose_gold else annotation.validate(
                          require_manual=require_manual))
            total_frames += len(annotation.frames)
            total_instances += sum(len(frame.instances) for frame in annotation.frames)
            total_events += len(annotation.events)
            total_behaviors += len(annotation.behaviors)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors = [f"无法读取: {error}"]
        results.append({"file": str(file_path), "valid": not errors, "errors": errors})
    invalid = sum(not item["valid"] for item in results)
    return {
        "valid": invalid == 0,
        "files": len(results),
        "invalid_files": invalid,
        "annotated_frames": total_frames,
        "instances": total_instances,
        "events": total_events,
        "behaviors": total_behaviors,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    try:
        report = validate_path(Path(args.input), require_manual=args.gold or args.pose_gold,
                               pose_gold=args.pose_gold, require_track_ids=args.require_track_ids)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
