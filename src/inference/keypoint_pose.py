"""关键点姿态推理与轨迹匹配。

默认关闭。只有加载经过验证的 YOLO Pose 权重并同时得到 head、thorax、
abdomen_tip 三点时，才把头尾朝向写入轨迹对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple


PointScore = Tuple[float, float, float]


@dataclass(frozen=True)
class PoseDetection:
    bbox: Tuple[float, float, float, float]
    keypoints: Dict[str, PointScore]
    confidence: float


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """计算两个 [x, y, width, height] 框的 IoU。"""
    ax, ay, aw, ah = [float(value) for value in first]
    bx, by, bw, bh = [float(value) for value in second]
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - intersection
    return intersection / union if union > 0 else 0.0


def _clear_unverified_pose(track) -> None:
    pose = getattr(track, "pose", None)
    if pose is None:
        return
    if not bool(getattr(pose, "head_tail_known", False)):
        pose.head_bbox = None
        pose.abdomen_bbox = None
        pose.head_tail_known = False
        pose.orientation_kind = "body_axis_unoriented" if getattr(pose, "orientation", 0.0) else "unknown"
        pose.source = "image_geometry_heuristic" if getattr(pose, "orientation", 0.0) else "none"


def _apply_detection(track, detection: PoseDetection, min_keypoint_confidence: float) -> bool:
    required = ("head", "thorax", "abdomen_tip")
    if any(name not in detection.keypoints for name in required):
        _clear_unverified_pose(track)
        return False
    if any(detection.keypoints[name][2] < min_keypoint_confidence for name in required):
        _clear_unverified_pose(track)
        return False
    head = detection.keypoints["head"]
    abdomen = detection.keypoints["abdomen_tip"]
    pose = getattr(track, "pose", None) or SimpleNamespace()
    pose.keypoints = detection.keypoints
    pose.orientation = degrees(atan2(head[1] - abdomen[1], head[0] - abdomen[0])) % 360.0
    pose.orientation_kind = "head_direction"
    pose.head_tail_known = True
    pose.source = "validated_pose_model"
    pose.confidence = min(detection.confidence, *(detection.keypoints[name][2] for name in required))
    radius = max(min(float(track.bbox[2]), float(track.bbox[3])) * 0.12, 1.0)
    pose.head_bbox = [head[0] - radius, head[1] - radius, radius * 2, radius * 2]
    pose.abdomen_bbox = [abdomen[0] - radius, abdomen[1] - radius, radius * 2, radius * 2]
    track.pose = pose
    return True


def match_pose_to_tracks(tracks: Iterable, detections: Iterable[PoseDetection],
                         min_iou: float = 0.3,
                         min_keypoint_confidence: float = 0.35) -> Dict[str, int]:
    """按 IoU 一对一匹配姿态检测与轨迹，并清除所有未验证伪头尾结果。"""
    tracks = list(tracks)
    detections = sorted(detections, key=lambda item: item.confidence, reverse=True)
    for track in tracks:
        _clear_unverified_pose(track)
    available = set(range(len(tracks)))
    matched = 0
    rejected = 0
    for detection in detections:
        candidates = [(bbox_iou(tracks[index].bbox, detection.bbox), index) for index in available]
        if not candidates:
            rejected += 1
            continue
        overlap, index = max(candidates)
        if overlap < min_iou:
            rejected += 1
            continue
        available.remove(index)
        if _apply_detection(tracks[index], detection, min_keypoint_confidence):
            matched += 1
        else:
            rejected += 1
    return {"matched_tracks": matched, "unmatched_tracks": len(tracks) - matched,
            "rejected_pose_detections": rejected}


class KeypointPoseEstimator:
    """可选 YOLO Pose 推理器；禁用时只执行姿态安全清理。"""

    def __init__(self, config: Dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.model_path = config.get("model_path")
        self.confidence = float(config.get("confidence", 0.25))
        self.keypoint_confidence = float(config.get("keypoint_confidence", 0.35))
        self.min_iou = float(config.get("min_track_iou", 0.3))
        self.imgsz = int(config.get("imgsz", 640))
        self.device = config.get("device")
        self.model = None
        self.frames = self.matched = self.rejected = 0
        if self.enabled:
            if not self.model_path or not Path(self.model_path).is_file():
                raise FileNotFoundError("启用关键点姿态推理时必须提供存在的 model_path")
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            if getattr(self.model, "task", None) != "pose":
                raise ValueError("姿态权重必须是 Ultralytics YOLO Pose 模型")
            from utils.common import normalize_device
            self.device = normalize_device(self.device)

    def update(self, frame, tracks: Iterable) -> Dict[str, int]:
        tracks = list(tracks)
        self.frames += 1
        if not self.enabled or self.model is None:
            result = match_pose_to_tracks(tracks, [], self.min_iou, self.keypoint_confidence)
            return result
        prediction = self.model.predict(
            frame, conf=self.confidence, imgsz=self.imgsz,
            device=self.device, verbose=False)[0]
        detections: List[PoseDetection] = []
        if prediction.boxes is not None and prediction.keypoints is not None:
            boxes = prediction.boxes.xywh.cpu().tolist()
            scores = prediction.boxes.conf.cpu().tolist()
            points = prediction.keypoints.xy.cpu().tolist()
            point_scores = (prediction.keypoints.conf.cpu().tolist()
                            if prediction.keypoints.conf is not None else None)
            for index, (box, score, coords) in enumerate(zip(boxes, scores, points)):
                if len(coords) < 3:
                    continue
                confidences = point_scores[index] if point_scores is not None else [1.0] * len(coords)
                names = ("head", "thorax", "abdomen_tip")
                keypoints = {name: (float(coords[k][0]), float(coords[k][1]), float(confidences[k]))
                             for k, name in enumerate(names)}
                detections.append(PoseDetection(tuple(map(float, box)), keypoints, float(score)))
        result = match_pose_to_tracks(tracks, detections, self.min_iou, self.keypoint_confidence)
        self.matched += result["matched_tracks"]
        self.rejected += result["rejected_pose_detections"]
        return result

    def build_report(self) -> Dict:
        return {
            "enabled": self.enabled,
            "head_tail_supported": self.enabled,
            "frames_analyzed": self.frames,
            "matched_track_observations": self.matched,
            "rejected_pose_detections": self.rejected,
            "source": "validated_pose_model" if self.enabled else "none",
            "limitations": ("关键点模型未启用，头、胸、腹端及头部朝向输出为未知。"
                            if not self.enabled else
                            "模型输出仍须在人工关键点真值上报告 PCK、OKS 和头尾翻转率。"),
        }
