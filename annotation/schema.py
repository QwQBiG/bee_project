"""统一的蜜蜂视频标注数据结构与基础校验。

坐标使用像素值；关键点可见性遵循 0=未标注、1=遮挡、2=可见。
模型预测可以写入 ``source=prediction``，但不能作为人工真值使用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
SCENES = {"unknown", "inside_ir", "inside_visible", "outside_entrance"}
INSTANCE_SOURCES = {"manual", "prediction", "interpolated"}
EVENT_TYPES = {"entering", "leaving", "uncertain"}
KEYPOINT_NAMES = ("head", "thorax", "abdomen_tip")


@dataclass
class Keypoint:
    name: str
    x: float
    y: float
    visibility: int = 2

    def validate(self, width: int, height: int) -> List[str]:
        errors: List[str] = []
        if self.name not in KEYPOINT_NAMES:
            errors.append(f"未知关键点: {self.name}")
        if self.visibility not in (0, 1, 2):
            errors.append(f"关键点 {self.name} visibility 必须为 0、1 或 2")
        if self.visibility and not (0 <= self.x < width and 0 <= self.y < height):
            errors.append(f"关键点 {self.name} 超出画面")
        return errors


@dataclass
class BeeInstance:
    instance_id: str
    bbox: List[float]
    category: str = "bee"
    track_id: Optional[int] = None
    keypoints: List[Keypoint] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    confidence: Optional[float] = None

    def validate(self, width: int, height: int) -> List[str]:
        errors: List[str] = []
        if len(self.bbox) != 4:
            return [f"实例 {self.instance_id} bbox 必须为 [x,y,w,h]"]
        x, y, box_width, box_height = self.bbox
        if box_width <= 0 or box_height <= 0:
            errors.append(f"实例 {self.instance_id} bbox 宽高必须为正数")
        if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
            errors.append(f"实例 {self.instance_id} bbox 超出画面")
        if self.source not in INSTANCE_SOURCES:
            errors.append(f"实例 {self.instance_id} source 无效")
        if self.source == "prediction" and self.confidence is None:
            errors.append(f"预测实例 {self.instance_id} 缺少 confidence")
        names = [item.name for item in self.keypoints]
        if len(names) != len(set(names)):
            errors.append(f"实例 {self.instance_id} 存在重复关键点")
        for keypoint in self.keypoints:
            errors.extend(keypoint.validate(width, height))
        return errors


@dataclass
class FrameAnnotation:
    frame_index: int
    timestamp_ms: float
    instances: List[BeeInstance] = field(default_factory=list)


@dataclass
class TemporalEvent:
    event_id: str
    event_type: str
    start_frame: int
    end_frame: int
    track_ids: List[int] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.event_type not in EVENT_TYPES:
            errors.append(f"事件 {self.event_id} 类型无效: {self.event_type}")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            errors.append(f"事件 {self.event_id} 帧范围无效")
        if self.source not in INSTANCE_SOURCES:
            errors.append(f"事件 {self.event_id} source 无效")
        return errors


@dataclass
class BehaviorSegment:
    segment_id: str
    label: str
    start_frame: int
    end_frame: int
    track_ids: List[int] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    reviewer: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.label.strip():
            errors.append(f"行为片段 {self.segment_id} 缺少 label")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            errors.append(f"行为片段 {self.segment_id} 帧范围无效")
        if self.source not in INSTANCE_SOURCES:
            errors.append(f"行为片段 {self.segment_id} source 无效")
        return errors



@dataclass
class VideoAnnotation:
    video_id: str
    source_path: str
    scene: str
    width: int
    height: int
    fps: float
    frame_count: int
    frames: List[FrameAnnotation] = field(default_factory=list)
    events: List[TemporalEvent] = field(default_factory=list)
    behaviors: List[BehaviorSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self, require_manual: bool = False) -> List[str]:
        errors: List[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"不支持的 schema_version: {self.schema_version}")
        if self.scene not in SCENES:
            errors.append(f"scene 无效: {self.scene}")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0 or self.frame_count <= 0:
            errors.append("视频宽高、fps 和 frame_count 必须为正数")
        seen_frames = set()
        for frame in self.frames:
            if frame.frame_index in seen_frames:
                errors.append(f"重复帧标注: {frame.frame_index}")
            seen_frames.add(frame.frame_index)
            if not 0 <= frame.frame_index < self.frame_count:
                errors.append(f"帧号超出范围: {frame.frame_index}")
            if frame.timestamp_ms < 0:
                errors.append(f"帧 {frame.frame_index} timestamp_ms 不能为负数")
            instance_ids, track_ids = set(), set()

            for instance in frame.instances:
                errors.extend(instance.validate(self.width, self.height))
                if instance.instance_id in instance_ids:
                    errors.append(f"帧 {frame.frame_index} 存在重复实例ID: {instance.instance_id}")
                instance_ids.add(instance.instance_id)
                if instance.track_id is not None:
                    if instance.track_id < 0:
                        errors.append(f"实例 {instance.instance_id} track_id 不能为负数")
                    if instance.track_id in track_ids:
                        errors.append(f"帧 {frame.frame_index} 存在重复 track_id: {instance.track_id}")
                    track_ids.add(instance.track_id)

                if require_manual and instance.source != "manual":
                    errors.append(f"金标准集包含非人工实例: {instance.instance_id}")
        for event in self.events:
            errors.extend(event.validate())
            if event.end_frame >= self.frame_count:
                errors.append(f"事件 {event.event_id} 超出视频帧范围")
            if require_manual and event.source != "manual":
                errors.append(f"金标准集包含非人工事件: {event.event_id}")
        for behavior in self.behaviors:
            errors.extend(behavior.validate())
            if behavior.end_frame >= self.frame_count:
                errors.append(f"行为片段 {behavior.segment_id} 超出视频帧范围")
            if require_manual and behavior.source != "manual":
                errors.append(f"金标准集包含非人工行为: {behavior.segment_id}")

        return errors

    def validate_pose_gold(self, require_track_ids: bool = False) -> List[str]:
        """校验可用于姿态与朝向评测的人工金标准。"""
        errors = self.validate(require_manual=True)
        required = set(KEYPOINT_NAMES)
        for frame in self.frames:
            for instance in frame.instances:
                points = {point.name: point for point in instance.keypoints}
                missing = sorted(required - set(points))
                if missing:
                    errors.append(
                        f"实例 {instance.instance_id} 缺少姿态关键点: {', '.join(missing)}")
                    continue
                for name in ("head", "abdomen_tip"):
                    if points[name].visibility == 0:
                        errors.append(
                            f"实例 {instance.instance_id} 的 {name} 不可见，无法作为朝向真值")
                if require_track_ids and instance.track_id is None:
                    errors.append(f"实例 {instance.instance_id} 缺少 track_id")
        return errors

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, input_path: str | Path) -> "VideoAnnotation":
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        frames = []
        for raw_frame in data.pop("frames", []):
            instances = []
            for raw_instance in raw_frame.pop("instances", []):
                keypoints = [Keypoint(**item) for item in raw_instance.pop("keypoints", [])]
                instances.append(BeeInstance(keypoints=keypoints, **raw_instance))
            frames.append(FrameAnnotation(instances=instances, **raw_frame))
        events = [TemporalEvent(**item) for item in data.pop("events", [])]
        behaviors = [BehaviorSegment(**item) for item in data.pop("behaviors", [])]
        return cls(frames=frames, events=events, behaviors=behaviors, **data)
