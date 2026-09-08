"""巢内红外视频的可解释指标与预警。

本模块只使用检测器、跟踪器及当前轻量姿态估计已经提供的信息；它不把
“翅膀振动”等需要关键点/高帧率模型的现象伪装成已确认的识别结果。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _as_number(value: object) -> float:
    """把 numpy 标量安全地变成 JSON 友好的 float。"""
    return float(value) if value is not None else 0.0


def _angle_distance(first: float, second: float) -> float:
    """返回两个角度的最小夹角（0 到 180 度）。"""
    return abs((first - second + 180.0) % 360.0 - 180.0)


@dataclass
class _Observation:
    frame: int
    center: Tuple[float, float]
    velocity: Tuple[float, float]
    orientation: float
    aspect_ratio: float


class InsideHiveMetricsAnalyzer:
    """将巢内轨迹转为朝向、姿态、活跃度、聚集和静止五类指标。"""

    def __init__(self, config: Dict | None = None):
        config = config or {}
        self.grid_rows, self.grid_cols = config.get("density_grid", [10, 10])
        self.stationary_speed = float(config.get("stationary_speed", 0.6))
        self.stationary_min_frames = int(config.get("stationary_min_frames", 30))
        self.pose_mismatch_degrees = float(config.get("pose_mismatch_degrees", 75))
        self.pose_min_speed = float(config.get("pose_min_speed", 1.0))
        self.cluster_share_threshold = float(config.get("cluster_share_threshold", 0.20))
        self.frame_shape: Tuple[int, int] | None = None
        self.track_observations: Dict[int, List[_Observation]] = defaultdict(list)
        self.frame_counts: List[int] = []
        self.frame_speeds: List[float] = []
        self.frame_density: List[np.ndarray] = []
        self.pose_mismatch_frames = 0
        self.pose_observed_frames = 0

    def update(self, tracks: Iterable, frame_id: int, frame_shape: Tuple[int, int]) -> None:
        self.frame_shape = frame_shape
        tracks = list(tracks)
        self.frame_counts.append(len(tracks))
        speeds: List[float] = []
        density = np.zeros((self.grid_rows, self.grid_cols), dtype=float)
        height, width = frame_shape

        for track in tracks:
            vx, vy = getattr(track, "velocity", (0.0, 0.0))
            speed = hypot(vx, vy)
            speeds.append(speed)
            x, y, box_width, box_height = getattr(track, "bbox", (0, 0, 1, 1))
            cx, cy = getattr(track, "center", (x + box_width / 2, y + box_height / 2))
            row = min(int(cy / max(height, 1) * self.grid_rows), self.grid_rows - 1)
            col = min(int(cx / max(width, 1) * self.grid_cols), self.grid_cols - 1)
            density[row, col] += 1
            pose = getattr(track, "pose", None)
            orientation = _as_number(getattr(pose, "orientation", 0.0))
            aspect_ratio = float(box_width) / max(float(box_height), 1.0)
            observation = _Observation(frame_id, (float(cx), float(cy)),
                                       (float(vx), float(vy)), orientation, aspect_ratio)
            self.track_observations[int(track.track_id)].append(observation)

            if speed >= self.pose_min_speed and orientation:
                travel_angle = degrees(atan2(vy, vx)) % 360.0
                self.pose_observed_frames += 1
                if _angle_distance(orientation, travel_angle) >= self.pose_mismatch_degrees:
                    self.pose_mismatch_frames += 1

        self.frame_speeds.append(float(np.mean(speeds)) if speeds else 0.0)
        if density.sum():
            density /= density.sum()
        self.frame_density.append(density)

    def _orientation_metric(self) -> Dict:
        valid = [item.orientation for values in self.track_observations.values()
                 for item in values if item.orientation]
        if not self.track_observations:
            return {
                "name": "个体朝向与特定运动轨迹", "status": "unknown",
                "description": "未形成确认轨迹，无法评估个体朝向与运动轨迹。",
                "limitations": "请检查视频清晰度、检测模型类别和置信度阈值。",
            }
        consistency = 1.0 - self.pose_mismatch_frames / max(self.pose_observed_frames, 1)
        return {
            "name": "个体朝向与特定运动轨迹",
            "status": "normal" if consistency >= 0.55 else "warning",
            "description": "基于头腹部轴线近似朝向，并与轨迹运动方向进行一致性比对。",
            "mean_orientation_degrees": round(float(np.mean(valid)), 1) if valid else None,
            "orientation_samples": len(valid),
            "motion_alignment": round(consistency, 3),
            "limitations": "当前头腹部位置来自检测框几何近似；需关键点标注模型后才能作为精确头部识别结论。",
        }

    def _posture_metric(self) -> Dict:
        ratios = [item.aspect_ratio for values in self.track_observations.values() for item in values]
        if not ratios:
            return {
                "name": "个体异常姿态", "status": "unknown",
                "description": "未形成确认轨迹，无法进行姿态候选筛查。",
                "limitations": "腹部上翘、翅膀高频振动需姿态关键点或高帧率翅膀分类模型。",
            }
        median = float(np.median(ratios)) if ratios else 0.0
        deviations = [abs(ratio - median) / max(median, 0.01) for ratio in ratios]
        abnormal_ratio = sum(value > 0.55 for value in deviations) / max(len(deviations), 1)
        return {
            "name": "个体异常姿态",
            "status": "warning" if abnormal_ratio > 0.15 else "normal",
            "description": "以身体框长宽比的异常变化和朝向/运动方向偏离作为候选姿态异常筛查。",
            "candidate_ratio": round(abnormal_ratio, 3),
            "median_body_aspect_ratio": round(median, 3),
            "limitations": "腹部上翘、翅膀高频振动不能由普通检测框可靠确认；需姿态关键点或高帧率翅膀分类模型。",
        }

    def _activity_metric(self) -> Dict:
        if not self.track_observations:
            return {
                "name": "群体运动速度与活跃度", "status": "unknown",
                "description": "未形成确认轨迹，无法区分检测失败与蜂群低活跃。",
                "limitations": "请先确认模型类别、红外画面对比度和检测置信度阈值。",
            }
        mean_count = float(np.mean(self.frame_counts)) if self.frame_counts else 0.0
        mean_speed = float(np.mean(self.frame_speeds)) if self.frame_speeds else 0.0
        if len(self.frame_speeds) >= 10:
            split = len(self.frame_speeds) // 2
            trend_value = float(np.mean(self.frame_speeds[split:]) - np.mean(self.frame_speeds[:split]))
        else:
            trend_value = 0.0
        status = "warning" if mean_count < 1 or mean_speed < 0.1 else "normal"
        return {
            "name": "群体运动速度与活跃度",
            "status": status,
            "description": "统计每帧确认轨迹数量和平均运动速度，用于评估蜂群活跃度。",
            "mean_active_tracks": round(mean_count, 2),
            "mean_speed_pixels_per_frame": round(mean_speed, 3),
            "trend": "上升" if trend_value > 0.15 else "下降" if trend_value < -0.15 else "稳定",
        }

    def _cluster_metric(self) -> Dict:
        if not self.frame_density or not self.track_observations:
            return {"name": "局部空间异常高密度聚集", "status": "unknown", "description": "暂无足够轨迹数据。"}
        average = np.mean(self.frame_density, axis=0)
        peak = float(average.max())
        row, col = np.unravel_index(int(average.argmax()), average.shape)
        status = "warning" if peak >= self.cluster_share_threshold else "normal"
        return {
            "name": "局部空间异常高密度聚集",
            "status": status,
            "description": "将画面划分网格，寻找长期占比最高的蜂群活动区域。",
            "peak_cell_share": round(peak, 3),
            "peak_grid_cell": {"row": int(row), "column": int(col)},
            "grid": average.round(4).tolist(),
        }

    def _stationary_metric(self) -> Dict:
        if not self.track_observations:
            return {
                "name": "个体静止时间与掉落轨迹", "status": "unknown",
                "description": "未形成确认轨迹，无法筛选静止或掉落候选对象。",
            }
        candidates = []
        for track_id, values in self.track_observations.items():
            longest = current = 0
            for item in values:
                if hypot(*item.velocity) <= self.stationary_speed:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            if longest >= self.stationary_min_frames:
                candidates.append({"track_id": track_id, "stationary_frames": longest})
        return {
            "name": "个体静止时间与掉落轨迹",
            "status": "warning" if candidates else "normal",
            "description": "筛选持续低速或静止的轨迹，作为掉落、死亡或受困的待复核对象。",
            "candidate_tracks": candidates,
            "stationary_speed_threshold": self.stationary_speed,
            "minimum_stationary_frames": self.stationary_min_frames,
        }

    def build_report(self) -> Dict:
        metrics = [self._orientation_metric(), self._posture_metric(), self._activity_metric(),
                   self._cluster_metric(), self._stationary_metric()]
        alerts = [{"metric": metric["name"], "severity": metric["status"]}
                  for metric in metrics if metric.get("status") == "warning"]
        return {
            "report_type": "巢内红外视频行为分析",
            "frames_analyzed": len(self.frame_counts),
            "tracked_individuals": len(self.track_observations),
            "metrics": metrics,
            "alerts": alerts,
        }
