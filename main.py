"""
智慧养蜂蜜蜂识别与行为量化研究 - 主程序入口
面向智慧养蜂的巢内外蜜蜂个体识别与行为智能量化研究

功能模块:
1. 数据标注工具
2. 巢外蜜蜂检测与跟踪 (可见光视频)
3. 巢内蜜蜂检测与跟踪 (红外视频)
4. 行为量化与分析
5. 结果可视化
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='智慧养蜂蜜蜂识别与行为量化研究',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理巢外视频
  run_cli.bat --mode outside --video data/outside.mp4 --output results/

  # 处理巢内视频
  run_cli.bat --mode inside --video data/inside.mp4 --output results/

  # 多模态同步处理
  run_cli.bat --mode multi --video data/outside.mp4 --video_inside data/inside.mp4 --output results/

  # 使用标注工具
  run_cli.bat --mode annotate --video data/video.mp4 --output annotations/

  # 模型训练
  run_cli.bat --mode train --config configs/train_config.yaml
        """
    )
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'outside', 'inside', 'multi', 
                               'annotate', 'train', 'export'],
                       help='运行模式')
    
    # 输入输出
    parser.add_argument('--video', type=str,
                       help='输入视频路径（巢外/可见光）')
    parser.add_argument('--video_inside', type=str,
                       help='输入视频路径（巢内/红外）')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--model', type=str,
                       help='训练使用的初始权重或模型配置')
    parser.add_argument('--data', type=str,
                       help='通过数据门禁的 data.yaml')
    parser.add_argument('--task', choices=['detect', 'pose'], default='pose',
                       help='训练任务类型')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='训练图像尺寸')
    parser.add_argument('--workers', type=int, default=4,
                       help='数据加载进程数')
    parser.add_argument('--train_name', type=str, default='bee_pose',
                       help='训练运行名称')
    parser.add_argument('--dry_run', action='store_true',
                       help='只执行训练前检查，不启动训练')
    
    # 其他选项
    parser.add_argument('--show', action='store_true',
                       help='显示结果视频')
    parser.add_argument('--device', type=str, default=None,
                       help='运行设备（默认自动检测：CUDA > MPS > CPU）')
    parser.add_argument('--tracker', type=str, default=None,
                       choices=['botsort', 'bytetrack', 'motion_iou'],
                       help='跟踪后端（默认使用配置文件，竞赛准确性优先 botsort，速度优先 bytetrack）')
    
    return parser.parse_args()


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def apply_device_override(config: dict, device: str = None) -> dict:
    """将命令行设备参数同步到所有可能使用的 tracker 配置。"""
    if device is None:
        return config

    config['device'] = device
    for section_name in ('tracker', 'outside_tracker', 'inside_tracker'):
        section = config.get(section_name)
        if isinstance(section, dict):
            section['device'] = device
    return config


def apply_tracker_override(config: dict, tracker_type: str = None) -> dict:
    """将命令行跟踪后端同步到所有可能使用的 tracker 配置。"""
    if tracker_type is None:
        return config

    for section_name in ('tracker', 'outside_tracker', 'inside_tracker'):
        section = config.get(section_name)
        if isinstance(section, dict):
            section['tracker_type'] = tracker_type
            section['tracker_config'] = (
                f'{tracker_type}.yaml'
                if tracker_type in ('botsort', 'bytetrack')
                else None
            )
    return config


def check_dependencies():
    """检查依赖是否已安装"""
    missing_deps = []
    required_deps = ['cv2', 'numpy', 'torch', 'yaml']
    dep_names = {'cv2': 'opencv-python', 'numpy': 'numpy', 
                 'torch': 'torch', 'yaml': 'pyyaml'}
    
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep_names.get(dep, dep))
    
    return missing_deps


def run_demo():
    """运行演示模式"""
    print("=" * 60)
    print("智慧养蜂蜜蜂识别与行为量化研究 - 演示模式")
    print("=" * 60)
    
    # 检查依赖
    missing_deps = check_dependencies()
    
    print("\n[0] 依赖检查")
    print("-" * 40)
    if missing_deps:
        print(f"  [WARN] 缺少以下依赖: {', '.join(missing_deps)}")
        print("  [INFO] 运行前请执行: pip install -r requirements.txt")
    else:
        print("  [OK] 所有依赖已安装")
    
    print("\n[1] 巢外蜜蜂检测与跟踪模块")
    print("-" * 40)
    if missing_deps:
        print("  [SKIP] 需要安装依赖后测试")
    else:
        try:
            from tracking.outside_tracker import create_outside_tracker
            outside_tracker = create_outside_tracker()
            print("  - YOLOv8检测器加载成功")
            print("  - BoT-SORT跟踪器加载成功")
        except Exception as e:
            print(f"  - 初始化警告: {e}")
    
    print("\n[2] 巢内蜜蜂检测与跟踪模块")
    print("-" * 40)
    if missing_deps:
        print("  [SKIP] 需要安装依赖后测试")
    else:
        try:
            from tracking.inside_tracker import create_inside_tracker
            inside_tracker = create_inside_tracker()
            print("  - 红外图像增强器初始化成功")
            print("  - 姿态估计器初始化成功")
        except Exception as e:
            print(f"  - 初始化警告: {e}")
    
    print("\n[3] 行为量化分析模块")
    print("-" * 40)
    try:
        from behavior.quantifier import create_behavior_quantifier
        quantifier = create_behavior_quantifier()
        print("  - 轨迹片段构建器初始化成功")
        print("  - 行为分类器初始化成功")
        print("  - 活动强度分析器初始化成功")
        print("  - 空间密度分析器初始化成功")
    except Exception as e:
        print(f"  - 初始化警告: {e}")
    
    print("\n[4] 结果可视化模块")
    print("-" * 40)
    if missing_deps:
        print("  [SKIP] 需要安装依赖后测试")
    else:
        try:
            from visualization.visualizer import create_visualizer
            visualizer = create_visualizer()
            print("  - 轨迹可视化器初始化成功")
            print("  - 行为可视化器初始化成功")
            print("  - 密度图可视化器初始化成功")
            print("  - 统计图表绘制器初始化成功")
        except Exception as e:
            print(f"  - 初始化警告: {e}")
    
    print("\n[5] 项目文件结构")
    print("-" * 40)
    from pathlib import Path
    project_root = Path(__file__).parent
    dirs = ['configs', 'annotation', 'tracking', 'behavior', 'visualization', 
            'inference', 'models', 'utils', 'datasets', 'docs']
    for d in dirs:
        status = "[OK]" if (project_root / d).exists() else "[MISSING]"
        print(f"  {status} {d}/")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n使用方法:")
    print("  python main.py --mode outside --video <视频路径> --output <输出目录>")
    print("  python main.py --mode inside --video <视频路径> --output <输出目录>")
    print("  python main.py --mode multi --video <巢外视频> --video_inside <巢内视频> --output <输出目录>")
    print("=" * 60)


def run_outside_mode(args):
    """运行巢外模式"""
    print(f"运行巢外蜜蜂检测与跟踪: {args.video}")
    
    from inference.processor import OutsideHiveProcessor
    
    config = load_config(args.config)
    config = apply_device_override(config, args.device)
    config = apply_tracker_override(config, args.tracker)
    
    processor = OutsideHiveProcessor(config)
    
    output_path = os.path.join(args.output, 'outside_result.mp4') if args.output else None
    stats = processor.process_video(args.video, output_path, args.show)
    
    # 保存统计结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'outside_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    from visualization.outside_pollen_report import create_outside_pollen_report
    report_path = create_outside_pollen_report(
        stats.get('pollen_analysis', {}), output_dir / 'outside_pollen_report.html')
    
    print(f"处理完成！结果已保存到: {output_dir}")
    print(f"巢外采粉报告: {report_path}")
    return stats


def run_inside_mode(args):
    """运行巢内模式"""
    print(f"运行巢内蜜蜂检测与跟踪: {args.video}")
    
    from inference.processor import InsideHiveProcessor
    
    config = load_config(args.config)
    config = apply_device_override(config, args.device)
    config = apply_tracker_override(config, args.tracker)
    
    processor = InsideHiveProcessor(config)
    
    output_path = os.path.join(args.output, 'inside_result.mp4') if args.output else None
    stats = processor.process_video(args.video, output_path, args.show)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'inside_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    # 单文件 HTML 报告：直接双击即可离线查看，无需启动常驻服务。
    from visualization.inside_report import create_inside_report
    report_path = create_inside_report(
        stats.get('inside_metrics', {}), output_dir / 'inside_analysis_report.html')
    
    print(f"处理完成！结果已保存到: {output_dir}")
    print(f"巢内分析报告: {report_path}")
    return stats


def run_multi_mode(args):
    """运行多模态模式"""
    print(f"运行多模态同步处理:")
    print(f"  巢外视频: {args.video}")
    print(f"  巢内视频: {args.video_inside}")
    
    from inference.processor import MultiModalProcessor
    
    config = load_config(args.config)
    config = apply_device_override(config, args.device)
    config = apply_tracker_override(config, args.tracker)
    
    processor = MultiModalProcessor(config, config)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = processor.process_synchronized(
        args.video, args.video_inside, output_dir)
    
    with open(output_dir / 'multi_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"处理完成！结果已保存到: {output_dir}")
    return results


def run_annotate_mode(args):
    """拒绝使用不兼容的早期框标注入口。"""
    print("ERROR: 旧本地标注器不支持统一 JSON、头胸腹骨架和来源门禁。",
          file=sys.stderr)
    print("请按 docs/annotation/CVAT姿态标注与回导流程.md 建立和回导标注任务。", file=sys.stderr)
    raise SystemExit(2)


def run_train_mode(args):
    """运行带数据来源门禁的 YOLO 训练。"""
    if not args.model or not args.data:
        print("ERROR: train 模式必须提供 --model 和 --data", file=sys.stderr)
        raise SystemExit(2)

    from tools.train_yolo import run_training
    training_args = argparse.Namespace(
        model=args.model,
        data=args.data,
        task=args.task,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device or "auto",
        workers=args.workers,
        seed=42,
        patience=30,
        project=args.output,
        name=args.train_name,
        cache=None,
        amp=True,
        dry_run=args.dry_run,
    )
    try:
        summary = run_training(training_args)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def main():
    """主函数"""
    args = parse_args()
    
    # 确保输出目录存在
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # 根据模式运行
    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'outside':
        run_outside_mode(args)
    elif args.mode == 'inside':
        run_inside_mode(args)
    elif args.mode == 'multi':
        run_multi_mode(args)
    elif args.mode == 'annotate':
        run_annotate_mode(args)
    elif args.mode == 'train':
        run_train_mode(args)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
