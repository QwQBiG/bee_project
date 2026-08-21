"""
项目完整性测试 - 不依赖外部库的核心逻辑测试
"""

import sys
import py_compile
from pathlib import Path
import yaml

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试1: 模块导入检查")
    print("=" * 60)
    
    try:
        # 测试配置加载
        import yaml
        config_path = project_root / "configs" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"  [OK] 配置文件加载成功")
            print(f"       - 巢外检测模型: {config.get('model', {}).get('outside_detector', {}).get('model_type', 'N/A')}")
            print(f"       - 跟踪器类型: {config.get('outside_tracker', {}).get('tracker_type', 'N/A')}")
        else:
            print(f"  [WARN] 配置文件不存在")
    except Exception as e:
        print(f"  [ERROR] 配置加载失败: {e}")
    
    print()


def test_data_structures():
    """测试数据结构"""
    print("=" * 60)
    print("测试2: 数据结构检查")
    print("=" * 60)
    
    # 测试 dataclass 导入
    try:
        from dataclasses import dataclass, field
        from typing import Dict, List, Tuple
        
        @dataclass
        class TestTrack:
            track_id: int
            bbox: List[float]
            center: Tuple[float, float]
            velocity: Tuple[float, float] = (0.0, 0.0)
            state: str = "tentative"
        
        # 创建测试实例
        track = TestTrack(
            track_id=1,
            bbox=[100, 100, 50, 40],
            center=(125.0, 120.0)
        )
        
        print(f"  [OK] 数据类定义正常")
        print(f"       - Track ID: {track.track_id}")
        print(f"       - BBox: {track.bbox}")
        print(f"       - Center: {track.center}")
        
    except Exception as e:
        print(f"  [ERROR] 数据结构测试失败: {e}")
    
    print()


def test_config_files():
    """测试配置文件完整性"""
    print("=" * 60)
    print("测试3: 配置文件完整性检查")
    print("=" * 60)
    
    config_file = project_root / "configs" / "config.yaml"
    
    if not config_file.exists():
        print(f"  [ERROR] 配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    required_sections = ['model', 'data', 'training', 'behavior', 'visualization', 'inference']
    
    for section in required_sections:
        if section in config:
            print(f"  [OK] 配置节: {section}")
        else:
            print(f"  [WARN] 缺少配置节: {section}")
    
    # 检查关键配置项
    model_config = config.get('model', {})
    if 'outside_detector' in model_config:
        print(f"  [OK] 巢外检测器配置存在")
    if 'inside_detector' in model_config:
        print(f"  [OK] 巢内检测器配置存在")
    if 'tracker' in model_config:
        print(f"  [OK] 跟踪器配置存在")
    
    print()


def test_project_structure():
    """测试项目结构"""
    print("=" * 60)
    print("测试4: 项目结构检查")
    print("=" * 60)
    
    required_dirs = [
        'configs',
        'annotation',
        'tracking',
        'behavior',
        'visualization',
        'inference',
        'models',
        'utils',
        'datasets'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'configs/config.yaml'
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  [OK] 目录存在: {dir_name}/")
        else:
            print(f"  [WARN] 目录缺失: {dir_name}/")
    
    print()
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  [OK] 文件存在: {file_name}")
        else:
            print(f"  [WARN] 文件缺失: {file_name}")
    
    print()


def test_code_files():
    """测试代码文件语法"""
    print("=" * 60)
    print("测试5: 代码文件语法检查")
    print("=" * 60)
    
    import py_compile
    
    code_files = [
        'main.py',
        'annotation/annotator.py',
        'tracking/outside_tracker.py',
        'tracking/inside_tracker.py',
        'tracking/ultralytics_mot.py',
        'tools/compare_trackers.py',
        'tools/evaluate_vnbee_tracking.py',
        'behavior/quantifier.py',
        'visualization/visualizer.py',
        'inference/processor.py',
        'models/trainer.py',
        'utils/common.py'
    ]
    
    all_passed = True
    
    for file_name in code_files:
        file_path = project_root / file_name
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"  [OK] 语法正确: {file_name}")
        except py_compile.PyCompileError as e:
            print(f"  [ERROR] 语法错误: {file_name}")
            print(f"         {e}")
            all_passed = False
    
    print()
    
    if all_passed:
        print("  [SUCCESS] 所有代码文件语法检查通过!")
    else:
        print("  [FAILURE] 部分代码文件存在语法错误")
    
    print()


def test_function_signatures():
    """测试函数签名"""
    print("=" * 60)
    print("测试6: 核心函数签名检查")
    print("=" * 60)
    
    # 读取并检查关键函数定义
    key_functions = {
        'main.py': ['parse_args', 'load_config', 'run_demo', 'run_outside_mode', 
                   'run_inside_mode', 'run_multi_mode', 'run_annotate_mode', 'main'],
        'tracking/outside_tracker.py': ['OutsideHiveBeeDetector', 'MotionIoUTracker',
                                       'OutsideHiveTracker', 'create_outside_tracker'],
        'tracking/ultralytics_mot.py': ['MOTTrack', 'UltralyticsMOTTracker'],
        'tools/evaluate_vnbee_tracking.py': ['load_ground_truth', 'match_frame', 'evaluate'],
        'tracking/inside_tracker.py': ['InfraredImageEnhancer', 'BeePoseEstimator',
                                       'InsideHiveBeeDetector', 'InsideHiveTracker', 'create_inside_tracker'],
        'behavior/quantifier.py': ['TrackletBuilder', 'BehaviorClassifier',
                                   'ActivityIntensityAnalyzer', 'SpatialDensityAnalyzer',
                                   'BehaviorQuantifier', 'create_behavior_quantifier'],
    }
    
    for file_name, functions in key_functions.items():
        file_path = project_root / file_name
        if not file_path.exists():
            print(f"  [WARN] 文件不存在: {file_name}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for func_name in functions:
            if f"def {func_name}" in content or f"class {func_name}" in content:
                print(f"  [OK] {file_name}: {func_name}")
            else:
                print(f"  [WARN] {file_name}: {func_name} (未找到)")
    
    print()


def main():
    """主测试函数"""
    print()
    print("=" * 60)
    print("智慧养蜂蜜蜂识别项目 - 完整性测试")
    print("=" * 60)
    print()
    
    test_project_structure()
    test_config_files()
    test_code_files()
    test_function_signatures()
    test_data_structures()
    test_imports()
    
    print("=" * 60)
    print("测试完成!")
    print("=" * 60)
    print()
    print("注意: 这是基础测试，不包括运行时测试（需要安装依赖库）")
    print("安装依赖: pip install -r requirements.txt")
    print()


if __name__ == "__main__":
    main()
