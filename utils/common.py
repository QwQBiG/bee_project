"""
工具函数模块
包含通用工具函数、日志配置、性能监控等
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import wraps
import numpy as np


def get_device() -> str:
    """自动检测最佳可用设备（CUDA > MPS > CPU）"""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """设置日志配置"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"bee_project_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def timing_decorator(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        logging.info(f"{func.__name__} 耗时: {elapsed:.3f}秒")
        return result
    return wrapper


def memory_monitor():
    """内存监控"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,  # MB
            'vms': mem_info.vms / 1024 / 1024   # MB
        }
    except ImportError:
        return {'rss': 0, 'vms': 0}


class FPSCounter:
    """FPS计数器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
        
    def update(self):
        """更新FPS计数"""
        self.timestamps.append(time.time())
        
        # 清理旧时间戳
        current_time = time.time()
        self.timestamps = [t for t in self.timestamps 
                          if current_time - t < 1.0]
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff > 0:
            return (len(self.timestamps) - 1) / time_diff
        return 0.0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.counters = {}
        
    def start(self, name: str):
        """开始计时"""
        self.metrics[name] = {'start': time.time()}
    
    def stop(self, name: str) -> float:
        """停止计时并返回耗时"""
        if name not in self.metrics:
            return 0.0
        
        elapsed = time.time() - self.metrics[name]['start']
        self.metrics[name]['elapsed'] = elapsed
        
        # 更新计数器统计
        if name not in self.counters:
            self.counters[name] = []
        self.counters[name].append(elapsed)
        
        return elapsed
    
    def get_stats(self, name: str) -> Dict:
        """获取统计信息"""
        if name not in self.counters or len(self.counters[name]) == 0:
            return {}
        
        values = self.counters[name]
        return {
            'count': len(values),
            'total': sum(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values)
        }
    
    def get_all_stats(self) -> Dict:
        """获取所有统计信息"""
        return {name: self.get_stats(name) for name in self.counters.keys()}


def save_json(data: Dict, path: str):
    """保存JSON文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_video_info(video_path: str) -> Dict:
    """获取视频信息"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    info = {}
    if cap.isOpened():
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 
                          cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
    
    cap.release()
    return info


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """计算IoU"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """非极大值抑制"""
    if len(detections) == 0:
        return []
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x.get('confidence', 1.0), 
                      reverse=True)
    
    keep = []
    
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        detections = [
            d for d in detections
            if compute_iou(current['bbox'], d['bbox']) < iou_threshold
        ]
    
    return keep


class AverageMeter:
    """平均值计量器"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"