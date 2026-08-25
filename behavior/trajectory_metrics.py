"""轨迹质量与可解释运动指标。

这里仅计算视频直接支持的描述性结果。没有真实空间尺度时不输出物理速度，
没有人工身份真值时也不把短时 track_id 解释为蜜蜂个体身份。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class _Point:
    frame_id: int
    x: float
    y: float
    confidence: float


class TrajectoryMetricsAnalyzer:
    """累计确认轨迹，并报告轨迹稳定性和运动统计。"""

    def __init__(self, config: Dict | None = None):
        config = config or {}
        self.min_track_frames = int(config.get("min_track_frames", 10))
        self.max_gap_frames = int(config.get("max_gap_frames", 2))
        self.stationary_speed = float(config.get("stationary_speed_normalized_per_second", 0.01))
        self.pixels_per_mm = config.get("pixels_per_mm")
        if self.min_track_frames < 2 or self.max_gap_frames < 1:
            raise ValueError("轨迹质量帧参数无效")
        if self.stationary_speed < 0:
            raise ValueError("静止速度阈值不能为负数")
        if self.pixels_per_mm is not None and float(self.pixels_per_mm) <= 0:
            raise ValueError("pixels_per_mm 必须为正数")
        self.pixels_per_mm = float(self.pixels_per_mm) if self.pixels_per_mm is not None else None
        self.fps: float | None = None
        self.frame_count = 0
        self.frame_shape: Tuple[int, int] | None = None
        self.observations: Dict[int, List[_Point]] = defaultdict(list)

    def set_video_fps(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("视频 FPS 必须为正数")
        self.fps = float(fps)

    def update(self, tracks: Iterable, frame_id: int,
               frame_shape: Tuple[int, int]) -> None:
        height, width = frame_shape
        if height <= 0 or width <= 0:
            raise ValueError("frame_shape 必须为正数")
        self.frame_count = max(self.frame_count, frame_id + 1)
        self.frame_shape = frame_shape
        for track in tracks:
            if getattr(track, "state", "confirmed") != "confirmed":
                continue
            center = getattr(track, "center", None)
            if center is None:
                x, y, box_width, box_height = track.bbox
                center = (x + box_width / 2, y + box_height / 2)
            self.observations[int(track.track_id)].append(_Point(
                frame_id=frame_id,
                x=float(center[0]),
                y=float(center[1]),
                confidence=float(getattr(track, "confidence", 1.0)),
            ))

    def _track_features(self, points: List[_Point]) -> Dict:
        assert self.frame_shape is not None
        height, width = self.frame_shape
        normalized_distance = 0.0
        pixel_distance = 0.0
        speeds: List[float] = []
        gap_count = 0
        observed_intervals = 0
        for previous, current in zip(points, points[1:]):
            delta_frames = current.frame_id - previous.frame_id
            if delta_frames <= 0:
                continue
            if delta_frames > self.max_gap_frames:
                gap_count += 1
            dx = current.x - previous.x
            dy = current.y - previous.y
            pixel_step = hypot(dx, dy)
            normalized_step = hypot(dx / width, dy / height)
            pixel_distance += pixel_step
            normalized_distance += normalized_step
            if self.fps:
                speeds.append(normalized_step * self.fps / delta_frames)
            observed_intervals += 1
        span = points[-1].frame_id - points[0].frame_id + 1
        result = {
            "observations": len(points),
            "span_frames": span,
            "coverage": round(len(points) / max(span, 1), 4),
            "gap_count": gap_count,
            "mean_confidence": round(mean(item.confidence for item in points), 4),
            "path_length_normalized": round(normalized_distance, 6),
            "path_length_pixels": round(pixel_distance, 3),
            "mean_speed_normalized_per_second": round(mean(speeds), 6) if speeds else None,
            "stationary_interval_fraction": (
                round(sum(speed <= self.stationary_speed for speed in speeds) / len(speeds), 4)
                if speeds else None
            ),
            "observed_intervals": observed_intervals,
        }
        if self.pixels_per_mm is not None:
            result["path_length_mm"] = round(pixel_distance / self.pixels_per_mm, 3)
            result["mean_speed_mm_per_second"] = (
                round((pixel_distance / self.pixels_per_mm) * self.fps /
                      max(points[-1].frame_id - points[0].frame_id, 1), 3)
                if self.fps and len(points) > 1 else None
            )
        return result

    def build_report(self) -> Dict:
        if not self.observations:
            return {
                "status": "unknown",
                "frames_analyzed": self.frame_count,
                "unique_track_ids": 0,
                "quality": {},
                "motion": {},
                "limitations": ["未形成确认轨迹，无法计算轨迹质量和运动指标。"],
                "available_signals": ["video_fps"] if self.fps else [],
            }

        features = {track_id: self._track_features(points)
                    for track_id, points in self.observations.items()}
        lengths = [item["observations"] for item in features.values()]
        coverages = [item["coverage"] for item in features.values()]
        short_fraction = sum(length < self.min_track_frames for length in lengths) / len(lengths)
        one_frame_fraction = sum(length == 1 for length in lengths) / len(lengths)
        quality_status = "warning" if short_fraction > 0.5 or median(coverages) < 0.7 else "descriptive_only"
        valid_speeds = [item["mean_speed_normalized_per_second"] for item in features.values()
                        if item["mean_speed_normalized_per_second"] is not None]
        valid_stationary = [item["stationary_interval_fraction"] for item in features.values()
                            if item["stationary_interval_fraction"] is not None]
        limitations = [
            "track_id 仅代表本视频中的短时轨迹，不能作为跨视频个体身份。",
            "轨迹质量是无真值诊断，不等同于 MOTA、HOTA 或 IDF1。",
        ]
        if self.pixels_per_mm is None:
            limitations.append("未提供 pixels_per_mm，不能输出毫米/秒等物理速度。")
        return {
            "status": quality_status,
            "frames_analyzed": self.frame_count,
            "unique_track_ids": len(features),
            "quality": {
                "minimum_recommended_track_frames": self.min_track_frames,
                "median_observations_per_track": median(lengths),
                "short_track_fraction": round(short_fraction, 4),
                "one_frame_track_fraction": round(one_frame_fraction, 4),
                "median_track_coverage": round(median(coverages), 4),
                "total_gap_count": sum(item["gap_count"] for item in features.values()),
            },
            "motion": {
                "mean_track_speed_normalized_per_second": round(mean(valid_speeds), 6)
                if valid_speeds else None,
                "mean_stationary_interval_fraction": round(mean(valid_stationary), 4)
                if valid_stationary else None,
                "speed_unit": "normalized_frame_distance/second",
                "physical_scale_calibrated": self.pixels_per_mm is not None,
            },
            "tracks": {str(track_id): item for track_id, item in sorted(features.items())},
            "available_signals": ["tracked_trajectories"] + (["video_fps"] if self.fps else [])
                                 + (["spatial_scale_calibration"] if self.pixels_per_mm else []),
            "limitations": limitations,
        }
