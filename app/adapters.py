"""真实推理 stats → 前端展示结构适配层。

前端页面按 demo_data.py 定义的结构渲染，本模块把
inference/processor.py 返回的真实 stats 转换为该结构：
- 巢外：summary / track_quality(占位) / pollen_analysis / anomalies
- 巢内：summary / metrics(五类指标) / alerts

轨迹质量模块未开发，track_quality 使用占位数据（implemented=False）。
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2

from .demo_data import _outside_track_quality_placeholder

logger = logging.getLogger("app.adapters")

MODE_LABELS = {
    "outside": "巢外视频分析",
    "inside": "巢内视频分析",
    "multi": "双路同步分析",
}

# 前端行为分布展示的六类（与巢外行为分类键名一致）
BEHAVIOR_KEYS = ("entering", "exiting", "foraging", "resting", "wandering", "moving")

# 巢内指标 values 中需排除的元数据键
_METRIC_SKIP_KEYS = {
    "name", "status", "description", "limitations",
    "orientation_samples", "stationary_speed_threshold",
}
# 巢内指标字段名映射（真实 → 前端）
_METRIC_KEY_MAP = {
    "mean_speed_pixels_per_frame": "mean_speed",
    "minimum_stationary_frames": "threshold_frames",
}
# 巢内活跃趋势（真实为中文，前端期望英文键）
_TREND_MAP = {"上升": "up", "下降": "down", "稳定": "stable"}

_SEVERITY_TEXT = {"warning": "需复核", "danger": "严重", "info": "提示"}


def _read_fps(video_path: Union[str, Path]) -> Optional[float]:
    """读取视频帧率（处理失败时返回 None）。"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
    except Exception:  # noqa: BLE001
        return None
    return round(float(fps), 2) if fps else None


def _base_summary(raw: dict, video_path: Union[str, Path]) -> dict:
    return {
        "total_frames": int(raw.get("total_frames", 0)),
        "fps": _read_fps(video_path) or 0.0,
        "total_tracks": int(raw.get("total_tracks", 0)),
        "processing_time_s": round(float(raw.get("processing_time", 0)), 1),
    }


def adapt_outside(raw: dict, video_path: Union[str, Path],
                  annotated_url: Optional[str] = None,
                  report_url: Optional[str] = None) -> dict:
    """巢外真实 stats → 前端结构。"""
    behavior_analysis = raw.get("behavior_analysis") or {}
    individual = behavior_analysis.get("individual_summary") or {}
    counts = individual.get("behavior_counts") or {}
    summary = _base_summary(raw, video_path)
    summary["behavior"] = {k: int(counts.get(k, 0) or 0) for k in BEHAVIOR_KEYS}

    pollen_raw = raw.get("pollen_analysis") or {}
    nutrition = pollen_raw.get("nutrition_assessment") or {}
    ratio = pollen_raw.get("pollen_inbound_ratio")
    if ratio is None:
        ratio = pollen_raw.get("incoming_with_pollen_ratio", 0.0)
    pollen = {
        "status": nutrition.get("status", "unknown"),
        "incoming_with_pollen_ratio": round(float(ratio or 0.0), 4),
        "pollen_carrying_tracks": int(pollen_raw.get("pollen_inbound_events", 0) or 0),
        "assessment": nutrition.get("message", "未生成花粉营养评估。"),
    }

    anomalies = []
    for item in behavior_analysis.get("anomalies") or []:
        anomalies.append({
            "type": item.get("type", "unknown"),
            "frame": item.get("frame"),
            "detail": item.get("description", item.get("detail", "")),
            "severity": item.get("severity", "warning"),
        })

    return {
        "mode": "outside",
        "mode_label": MODE_LABELS["outside"],
        "status": "done",
        "summary": summary,
        "track_quality": _outside_track_quality_placeholder(),
        "pollen_analysis": pollen,
        "anomalies": anomalies,
        "report_html": report_url,
        "annotated_video": annotated_url,
    }


def _adapt_metric(m: dict) -> dict:
    """巢内单个指标 → 前端结构（values 包装 + 字段映射）。"""
    values = {}
    for key, value in m.items():
        if key in _METRIC_SKIP_KEYS:
            continue
        out_key = _METRIC_KEY_MAP.get(key, key)
        if out_key == "trend":
            value = _TREND_MAP.get(value, value)
        elif out_key == "peak_grid_cell" and isinstance(value, dict):
            value = [int(value.get("row", 0)), int(value.get("column", 0))]
        values[out_key] = value
    return {
        "name": m.get("name", ""),
        "status": m.get("status", "unknown"),
        "description": m.get("description", ""),
        "values": values,
    }


def adapt_inside(raw: dict, video_path: Union[str, Path],
                 annotated_url: Optional[str] = None,
                 report_url: Optional[str] = None) -> dict:
    """巢内真实 stats → 前端结构。"""
    metrics_raw = raw.get("inside_metrics") or {}
    metrics = [_adapt_metric(m) for m in metrics_raw.get("metrics") or []]

    alerts = []
    for a in metrics_raw.get("alerts") or []:
        severity = a.get("severity", "warning")
        metric = a.get("metric", "某项指标")
        alerts.append({
            "level": severity,
            "text": "{0}：{1}，建议人工复核。".format(
                metric, _SEVERITY_TEXT.get(severity, severity)),
        })

    return {
        "mode": "inside",
        "mode_label": MODE_LABELS["inside"],
        "status": "done",
        "summary": _base_summary(raw, video_path),
        "metrics": metrics,
        "alerts": alerts,
        "report_html": report_url,
        "annotated_video": annotated_url,
    }


def adapt_multi(outside_result: dict, inside_result: dict) -> dict:
    """双路同步：合并巢外 + 巢内两个适配结果。"""
    return {
        "mode": "multi",
        "mode_label": MODE_LABELS["multi"],
        "status": "done",
        "summary": {
            "total_frames": outside_result["summary"]["total_frames"],
            "fps": outside_result["summary"]["fps"],
            "total_tracks": outside_result["summary"]["total_tracks"]
                            + inside_result["summary"]["total_tracks"],
            "processing_time_s": round(
                outside_result["summary"]["processing_time_s"]
                + inside_result["summary"]["processing_time_s"], 1),
        },
        "outside": outside_result,
        "inside": inside_result,
    }
