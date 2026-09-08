"""将项目统一轨迹对象接入巢门进出状态机。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from behavior.entrance_events import EntranceEvent, EntranceEventDetector, EntranceGeometry


class TrackEntranceAnalyzer:
    """管理活跃轨迹、轨迹消失和结构化进出统计。"""

    def __init__(self, config: Dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.missing_tolerance = int(config.get("missing_tolerance", 10))
        if self.missing_tolerance < 0:
            raise ValueError("missing_tolerance 不能为负数")
        self.detector = None
        if self.enabled:
            geometry_config = config.get("geometry")
            if not geometry_config:
                raise ValueError("启用进出事件分析时必须配置 geometry")
            geometry = EntranceGeometry(
                origin=tuple(geometry_config["origin"]),
                inward=tuple(geometry_config["inward"]),
                outside_depth=float(geometry_config["outside_depth"]),
                inside_depth=float(geometry_config["inside_depth"]),
                half_width=float(geometry_config["half_width"])
                if geometry_config.get("half_width") is not None else None,
            )
            self.detector = EntranceEventDetector(
                geometry=geometry,
                min_confirm_frames=int(config.get("min_confirm_frames", 2)),
                max_transition_frames=int(config.get("max_transition_frames", 60)),
                cooldown_frames=int(config.get("cooldown_frames", 15)),
                min_displacement=float(config.get("min_displacement", 0.02)),
                min_observation_confidence=float(config.get("min_observation_confidence", 0.0)),
            )
        self.last_seen: Dict[int, int] = {}

    @staticmethod
    def _normalized_center(track, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        height, width = frame_shape
        if width <= 0 or height <= 0:
            raise ValueError("frame_shape 必须为正数")
        center = getattr(track, "center", None)
        if center is None:
            x, y, box_width, box_height = track.bbox
            center = (x + box_width / 2, y + box_height / 2)
        return float(center[0]) / width, float(center[1]) / height

    def update(self, tracks: Iterable, frame_id: int,
               frame_shape: Tuple[int, int]) -> List[EntranceEvent]:
        if not self.enabled or self.detector is None:
            return []
        events = []
        active_ids = set()
        for track in tracks:
            if getattr(track, "state", "confirmed") != "confirmed":
                continue
            track_id = int(track.track_id)
            active_ids.add(track_id)
            self.last_seen[track_id] = frame_id
            event = self.detector.update(
                track_id, self._normalized_center(track, frame_shape), frame_id,
                float(getattr(track, "confidence", 1.0)))
            if event is not None:
                events.append(event)
        expired = [track_id for track_id, last_frame in self.last_seen.items()
                   if track_id not in active_ids and frame_id - last_frame > self.missing_tolerance]
        for track_id in expired:
            event = self.detector.finish_track(track_id)
            if event is not None:
                events.append(event)
            del self.last_seen[track_id]
        return events

    def finalize(self) -> List[EntranceEvent]:
        if not self.enabled or self.detector is None:
            return []
        events = []
        for track_id in list(self.last_seen):
            event = self.detector.finish_track(track_id)
            if event is not None:
                events.append(event)
            del self.last_seen[track_id]
        return events

    def build_report(self) -> Dict:
        if not self.enabled or self.detector is None:
            return {
                "enabled": False,
                "calibrated": False,
                "events": [],
                "counts": {"entering": 0, "leaving": 0, "uncertain": 0, "net_flow": 0},
                "limitations": "未配置巢门几何参数，进出事件分析未启用。",
            }
        return {
            "enabled": True,
            "calibrated": True,
            "coordinate_system": "normalized_frame",
            "events": [event.to_dict() for event in self.detector.events],
            "counts": self.detector.counts(),
            "limitations": "结果依赖人工标定的巢门方向与范围；正式指标须与人工事件真值评测。",
        }
