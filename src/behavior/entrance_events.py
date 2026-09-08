"""基于轨迹穿越巢门缓冲区的进出事件状态机。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Dict, List, Optional, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class EntranceGeometry:
    """用入口原点、向内单位方向和两个深度阈值描述巢门。"""

    origin: Point
    inward: Point
    outside_depth: float
    inside_depth: float
    half_width: Optional[float] = None

    def __post_init__(self) -> None:
        length = hypot(*self.inward)
        if length <= 0:
            raise ValueError("inward 方向向量不能为零")
        if self.outside_depth >= self.inside_depth:
            raise ValueError("outside_depth 必须小于 inside_depth")
        if self.half_width is not None and self.half_width <= 0:
            raise ValueError("half_width 必须为正数")
        object.__setattr__(self, "inward", (self.inward[0] / length, self.inward[1] / length))

    def depth(self, point: Point) -> float:
        return ((point[0] - self.origin[0]) * self.inward[0]
                + (point[1] - self.origin[1]) * self.inward[1])


    def lateral(self, point: Point) -> float:
        return (point[0] - self.origin[0]) * -self.inward[1] + (point[1] - self.origin[1]) * self.inward[0]

    def zone(self, point: Point) -> str:
        depth = self.depth(point)
        if depth <= self.outside_depth:
            return "outside"
        if self.half_width is not None and abs(self.lateral(point)) > self.half_width:
            return "outside_corridor"
        if depth >= self.inside_depth:
            return "inside"
        return "buffer"


@dataclass
class EntranceEvent:
    track_id: int
    event_type: str
    start_frame: int
    crossing_frame: int
    end_frame: int
    displacement: float
    confidence: float
    status: str = "confirmed"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class _TrackTransition:
    stable_zone: Optional[str] = None
    last_stable_point: Optional[Point] = None
    candidate_from: Optional[str] = None
    candidate_start_frame: int = -1
    candidate_start_point: Optional[Point] = None
    buffer_frame: int = -1
    target_zone: Optional[str] = None
    target_streak: int = 0
    last_event_frame: int = -1
    last_frame: int = -1
    last_point: Optional[Point] = None


class EntranceEventDetector:
    """把 outside→buffer→inside 与反向轨迹转换为去重事件。"""

    def __init__(self, geometry: EntranceGeometry, min_confirm_frames: int = 2,
                 max_transition_frames: int = 60, cooldown_frames: int = 15,
                 min_displacement: float = 8.0, min_observation_confidence: float = 0.0):
        if min_confirm_frames < 1 or max_transition_frames < 1 or cooldown_frames < 0:
            raise ValueError("状态机帧参数无效")
        self.geometry = geometry
        self.min_confirm_frames = min_confirm_frames
        self.max_transition_frames = max_transition_frames
        self.cooldown_frames = cooldown_frames
        self.min_displacement = min_displacement
        self.min_observation_confidence = min_observation_confidence
        self.tracks: Dict[int, _TrackTransition] = {}
        self.events: List[EntranceEvent] = []

    def update(self, track_id: int, point: Point, frame_id: int,
               confidence: float = 1.0) -> Optional[EntranceEvent]:
        if frame_id < 0:
            raise ValueError("frame_id 不能为负数")
        state = self.tracks.setdefault(track_id, _TrackTransition())
        if frame_id <= state.last_frame:
            raise ValueError(f"track {track_id} 的帧号必须严格递增")
        state.last_frame = frame_id
        if confidence < self.min_observation_confidence:
            return None
        state.last_point = point

        zone = self.geometry.zone(point)
        if zone == "outside_corridor":
            if state.candidate_from is not None:
                self._cancel_candidate(state)
            return None

        if state.stable_zone is None:
            if zone != "buffer":
                state.stable_zone = zone
                state.last_stable_point = point
            return None

        if state.candidate_from is None:
            if zone == "buffer":
                self._start_candidate(state, frame_id, point)
            elif zone != state.stable_zone:
                self._start_candidate(state, frame_id, point)
                state.target_zone, state.target_streak = zone, 1
            else:
                state.last_stable_point = point
            return None

        if frame_id - state.candidate_start_frame > self.max_transition_frames:
            self._cancel_candidate(state, stable_zone=zone if zone != "buffer" else None)
            return None
        if zone == state.candidate_from:
            self._cancel_candidate(state, stable_zone=zone)
            return None
        if zone == "buffer":
            state.target_zone, state.target_streak = None, 0
            return None

        state.target_streak = state.target_streak + 1 if state.target_zone == zone else 1
        state.target_zone = zone
        if state.target_streak < self.min_confirm_frames:
            return None
        return self._confirm(track_id, state, point, frame_id, confidence)

    def _start_candidate(self, state: _TrackTransition, frame_id: int, point: Point) -> None:
        state.candidate_from = state.stable_zone
        state.candidate_start_frame = frame_id
        state.candidate_start_point = state.last_stable_point or point
        state.buffer_frame = frame_id

    @staticmethod
    def _cancel_candidate(state: _TrackTransition, stable_zone: Optional[str] = None) -> None:
        if stable_zone is not None:
            state.stable_zone = stable_zone
        state.candidate_from = state.target_zone = None
        state.candidate_start_point = None
        state.candidate_start_frame = state.buffer_frame = -1
        state.target_streak = 0

    def _confirm(self, track_id: int, state: _TrackTransition, point: Point,
                 frame_id: int, confidence: float) -> Optional[EntranceEvent]:
        start = state.candidate_start_point or point
        displacement = hypot(point[0] - start[0], point[1] - start[1])
        from_zone, target_zone = state.candidate_from, state.target_zone
        event_type = "entering" if (from_zone, target_zone) == ("outside", "inside") else "leaving"
        allowed = (from_zone, target_zone) in {("outside", "inside"), ("inside", "outside")}
        allowed &= displacement >= self.min_displacement
        allowed &= state.last_event_frame < 0 or frame_id - state.last_event_frame > self.cooldown_frames
        event = None
        if allowed:
            event = EntranceEvent(track_id, event_type, state.candidate_start_frame,
                                  state.buffer_frame, frame_id, round(displacement, 3),
                                  max(0.0, min(float(confidence), 1.0)))
            self.events.append(event)
            state.last_event_frame = frame_id
            state.last_stable_point = point
        self._cancel_candidate(state, stable_zone=target_zone)
        return event

    def finish_track(self, track_id: int) -> Optional[EntranceEvent]:
        state = self.tracks.pop(track_id, None)
        if state is None or state.candidate_from is None:
            return None
        start = state.candidate_start_point or state.last_point or (0.0, 0.0)
        end = state.last_point or start
        event = EntranceEvent(
            track_id, "uncertain", state.candidate_start_frame, state.buffer_frame,
            state.last_frame, round(hypot(end[0] - start[0], end[1] - start[1]), 3),
            0.0, status="uncertain")
        self.events.append(event)
        return event


    def counts(self) -> Dict[str, int]:
        entering = sum(item.event_type == "entering" for item in self.events)
        leaving = sum(item.event_type == "leaving" for item in self.events)
        uncertain = sum(item.event_type == "uncertain" for item in self.events)
        return {"entering": entering, "leaving": leaving, "uncertain": uncertain,
                "net_flow": entering - leaving}
