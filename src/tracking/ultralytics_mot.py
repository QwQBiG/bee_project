"""Ultralytics 官方多目标跟踪适配器。

将 YOLO 的 ByteTrack/BoT-SORT 输出转换为项目统一的轨迹状态。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MOTTrack:
    """项目内部统一的跟踪结果。"""

    track_id: int
    bbox: List[float]
    confidence: float
    class_id: int
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "confirmed"
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


class UltralyticsMOTTracker:
    """调用 Ultralytics 内置的 ByteTrack 或 BoT-SORT。"""

    def __init__(self, model, tracker: str = "botsort.yaml",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 imgsz: int = 640,
                 device: str = "cpu",
                 max_age: int = 30):
        self.model = model
        self.tracker = tracker
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.max_age = max_age
        self.tracks: Dict[int, MOTTrack] = {}
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> Tuple[List[MOTTrack], List[Dict]]:
        """处理一帧并返回当前检测与已分配 ID 的轨迹。"""
        self.frame_count += 1
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return [], []

        result = results[0]
        boxes = result.boxes
        detections: List[Dict] = []
        if boxes is None or len(boxes) == 0:
            self._age_missing_tracks(set())
            return [], []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id
        track_ids = ids.int().cpu().numpy() if ids is not None else None
        active_ids = set()
        tracks: List[MOTTrack] = []

        for index, coords in enumerate(xyxy):
            x1, y1, x2, y2 = [float(value) for value in coords]
            bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            detection = {
                "bbox": bbox,
                "confidence": float(confidences[index]),
                "class_id": int(classes[index]),
            }
            detections.append(detection)
            if track_ids is None:
                continue

            track_id = int(track_ids[index])
            active_ids.add(track_id)
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            previous = self.tracks.get(track_id)
            if previous is None:
                track = MOTTrack(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=detection["confidence"],
                    class_id=detection["class_id"],
                    center=center,
                    trajectory=[center],
                )
            else:
                velocity = (center[0] - previous.center[0],
                            center[1] - previous.center[1])
                previous.bbox = bbox
                previous.confidence = detection["confidence"]
                previous.class_id = detection["class_id"]
                previous.velocity = velocity
                previous.center = center
                previous.age += 1
                previous.hits += 1
                previous.time_since_update = 0
                previous.state = "confirmed"
                previous.trajectory.append(center)
                if len(previous.trajectory) > 100:
                    previous.trajectory.pop(0)
                track = previous
            self.tracks[track_id] = track
            tracks.append(track)

        self._age_missing_tracks(active_ids)
        return tracks, detections

    def _age_missing_tracks(self, active_ids: set) -> None:
        """清理本适配器中已经不再由当前帧返回的轨迹。"""
        for track_id in list(self.tracks):
            if track_id in active_ids:
                continue
            track = self.tracks[track_id]
            track.time_since_update += 1
            if track.time_since_update > self.max_age:
                del self.tracks[track_id]
