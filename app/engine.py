"""Web 层推理引擎：配置加载、处理器单例、任务前状态重置。

设计原则：
- 模型权重只在服务启动时加载一次（常驻内存），由本模块持有全局单例。
- 每个任务开始前调用 reset_processor 重置内部的累积状态
  （行为量化器 / 花粉分析 / 巢内指标 / 跟踪器轨迹），
  保证多个视频之间互不串扰，且不重复加载模型。
- 轨迹质量模块未开发，占位数据由 adapters.py 提供。
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import yaml

from behavior.inside_metrics import InsideHiveMetricsAnalyzer
from behavior.outside_pollen import OutsidePollenAnalyzer
from behavior.quantifier import create_behavior_quantifier
from inference.processor import InsideHiveProcessor, OutsideHiveProcessor
from tracking.inside_tracker import InsideHiveAnalyzer

logger = logging.getLogger("app.engine")

# 项目根目录（app/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# config.yaml 未配置花粉入口区域时的默认区域 [x, y, w, h]（归一化），
# 需按实际画面中的蜂箱巢口位置标定。
DEFAULT_ENTRANCE_REGION = [0.30, 0.30, 0.40, 0.40]

_lock = threading.Lock()
_processors: dict = {"outside": None, "inside": None}
_config: Optional[dict] = None


def load_config() -> dict:
    """加载项目配置文件。

    - 模型权重路径解析为绝对路径（以项目根为基准），避免依赖启动目录。
    - pollen_analysis.entrance_region 缺失时使用默认入口区域并告警。
    """
    global _config
    if _config is not None:
        return _config

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 模型路径绝对化
    for key in ("outside_tracker", "inside_tracker"):
        sub = config.get(key) or {}
        model_path = sub.get("model_path")
        if model_path:
            sub["model_path"] = str((PROJECT_ROOT / str(model_path)).resolve())

    # 花粉入口区域：未配置时用默认区域（需按实际画面标定）
    pollen = config.setdefault("pollen_analysis", {})
    if not pollen.get("entrance_region"):
        pollen["entrance_region"] = list(DEFAULT_ENTRANCE_REGION)
        logger.warning(
            "config.yaml 未配置 pollen_analysis.entrance_region，"
            "已使用默认入口区域 %s，请按实际巢口画面标定",
            DEFAULT_ENTRANCE_REGION,
        )

    _config = config
    return config


def init_engine() -> dict:
    """初始化推理处理器单例（模型常驻内存）。

    模型加载失败（如权重文件缺失）时不会中断服务启动，
    处理器保持 None，任务执行时会以失败状态返回具体错误。
    """
    with _lock:
        if _processors["outside"] is None:
            try:
                config = load_config()
                _processors["outside"] = OutsideHiveProcessor(config)
                _processors["inside"] = InsideHiveProcessor(config)
                logger.info("推理引擎已初始化：巢外 + 巢内处理器就绪")
            except Exception as exc:  # noqa: BLE001
                _processors["outside"] = None
                _processors["inside"] = None
                logger.error("推理引擎初始化失败: %s", exc)
                raise
        return _processors


def get_processors() -> dict:
    """获取处理器单例（未初始化时先初始化）。"""
    with _lock:
        if _processors["outside"] is None:
            return init_engine()
        return _processors


def _reset_tracker_state(tracker) -> None:
    """浅重置跟踪器内部状态（不重新加载模型）。"""
    for attr, is_dict in (("tracks", True), ("previous_bboxes", True),
                          ("next_track_id", False), ("frame_count", False)):
        if hasattr(tracker, attr):
            setattr(tracker, attr, {} if is_dict else 0)
    # 内部 tracker（UltralyticsMOTTracker / MotionIoUTracker）
    inner = getattr(tracker, "tracker", None)
    if inner is not None and inner is not tracker:
        for attr in ("tracks", "frame_count"):
            if hasattr(inner, attr):
                setattr(inner, attr, {} if attr == "tracks" else 0)


def _reset_ultralytics_track_id() -> None:
    """重置 ultralytics 内置跟踪器的全局轨迹 ID 计数（兼容不同版本）。"""
    try:
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack.reset_id()
    except Exception:  # noqa: BLE001  (版本差异，非关键路径)
        pass


def reset_processor(proc) -> None:
    """重置处理器内部累积状态，供每个任务开始前调用。

    重建行为量化器 / 花粉分析 / 巢内指标等累积器（开销极小），
    跟踪器仅重置状态属性，模型对象保持不变。
    """
    proc.quantifier = create_behavior_quantifier(proc.config.get("behavior", {}))
    if hasattr(proc, "pollen_analyzer"):  # 巢外处理器
        proc.pollen_analyzer = OutsidePollenAnalyzer(proc.config.get("pollen_analysis", {}))
        proc.stats = {
            "total_frames": 0,
            "total_tracks": 0,
            "detection_history": [],
            "track_history": [],
            "entry_events": 0,
            "exit_events": 0,
        }
    if hasattr(proc, "inside_metrics"):  # 巢内处理器
        proc.inside_metrics = InsideHiveMetricsAnalyzer(
            proc.config.get("inside_metrics",
                            proc.config.get("behavior", {}).get("inside_metrics", {})))
        proc.stats = {
            "total_frames": 0,
            "total_tracks": 0,
            "track_history": [],
            "pose_distribution": [],
        }
    if hasattr(proc, "analyzer"):  # 巢内姿态/活动分析器
        proc.analyzer = InsideHiveAnalyzer()
    _reset_tracker_state(proc.tracker)
    _reset_ultralytics_track_id()
