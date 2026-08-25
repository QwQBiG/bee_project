"""带数据门禁的 Ultralytics YOLO 训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def validate_dataset(data_yaml: Path, expected_task: str) -> Dict:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_yaml}")
    yaml_text = data_yaml.read_text(encoding="utf-8")
    if expected_task == "pose" and "kpt_shape:" not in yaml_text:
        raise ValueError("姿态数据缺少 kpt_shape")
    dataset_meta_path = data_yaml.parent / "dataset_meta.json"
    if not dataset_meta_path.is_file():
        raise ValueError("缺少 dataset_meta.json，不能确认标注来源与数据划分")
    metadata = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    if metadata.get("task") != expected_task:
        raise ValueError(
            f"数据任务类型为 {metadata.get('task')}，与请求的 {expected_task} 不一致")
    if not metadata.get("pose_labels_ready", expected_task != "pose"):
        raise ValueError("姿态标注尚未达到人工金标准要求")
    if not metadata.get("training_ready", False):
        split_counts = metadata.get("split_counts", {})
        raise ValueError(
            "数据集尚不可训练：必须仅含人工完整标注，且 train/val 均来自已分配数据；"
            f"当前 split_counts={split_counts}")
    return metadata


def training_options(args: argparse.Namespace, device: str) -> Dict:
    if (args.epochs < 1 or args.batch == 0 or args.imgsz < 32
            or args.workers < 0 or args.lr0 <= 0):
        raise ValueError("epochs、batch、imgsz 或 workers 参数无效")
    options = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "device": device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "exist_ok": False,
        "amp": args.amp,
        "plots": True,
        "save": True,
        "verbose": True,
    }
    if args.cache:
        options["cache"] = args.cache
    return options


def run_training(args: argparse.Namespace) -> Dict:
    validate_dataset(Path(args.data), args.task)
    from utils.common import normalize_device
    device = normalize_device(args.device)
    options = training_options(args, device)
    if args.dry_run:
        return {"dry_run": True, "model": args.model, "task": args.task, **options}
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("训练需要安装 ultralytics") from error
    model = YOLO(args.model)
    if model.task != args.task:
        raise ValueError(f"模型任务为 {model.task}，不能用于 {args.task} 训练")
    results = model.train(**options)
    save_dir = Path(str(results.save_dir))
    summary = {
        "dry_run": False,
        "model": args.model,
        "task": args.task,
        "device": device,
        "data": options["data"],
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "metrics": getattr(results, "results_dict", {}),
    }
    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练蜜蜂 YOLO Detect/Pose 模型")
    parser.add_argument("model", help="兼容任务的初始权重或模型配置")
    parser.add_argument("data", help="通过数据门禁的 data.yaml")
    parser.add_argument("--task", choices=["detect", "pose"], default="pose")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1, help="-1 表示自动批量")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--device", default="auto", help="auto、cuda:0、mps 或 cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="bee_pose")
    parser.add_argument("--cache", choices=["ram", "disk"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="只校验并显示配置，不启动训练")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_training(args)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
