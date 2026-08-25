"""蜜蜂视频标注、转换与质量校验。"""

from .schema import (
    BehaviorSegment,
    BeeInstance,
    FrameAnnotation,
    Keypoint,
    TemporalEvent,
    VideoAnnotation,
)

__all__ = [
    "BehaviorSegment",
    "BeeInstance",
    "FrameAnnotation",
    "Keypoint",
    "TemporalEvent",
    "VideoAnnotation",
]
