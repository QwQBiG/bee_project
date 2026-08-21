"""
巢外蜜蜂检测与跟踪模块
基于YOLOv8检测器和DeepSORT跟踪器
针对高密度、频繁进出场景优化
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import time

# 尝试导入ultralytics，如果不可用则使用替代方案
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


@dataclass
class TrackState:
    """跟踪状态"""
    track_id: int
    bbox: List[float]  # [x, y, w, h]
    confidence: float
    class_id: int
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    features: np.ndarray = None
    
    
class OutsideHiveBeeDetector:
    """巢外蜜蜂检测器 - 基于YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8m.pt", 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = None,
                 imgsz: int = 640):
        from utils.common import get_device
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device is not None else get_device()
        self.imgsz = imgsz
        
        if HAS_YOLO:
            self.model = YOLO(model_path)
            self.model.to(device)
        else:
            self.model = None
            print("Warning: ultralytics not available, using mock detector")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """检测蜜蜂
        
        Args:
            frame: 输入图像 (H, W, C)
            
        Returns:
            检测结果列表 [{'bbox': [x,y,w,h], 'confidence': float, 'class_id': int}, ...]
        """
        if self.model is None:
            # 返回模拟检测结果用于测试
            return self._mock_detect(frame)
        
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'confidence': conf,
                    'class_id': cls
                })
        
        return detections
    
    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        """模拟检测结果（用于测试）"""
        h, w = frame.shape[:2]
        detections = []
        
        # 生成随机检测模拟真实场景
        num_detections = np.random.randint(3, 12)
        for i in range(num_detections):
            x = np.random.randint(50, w - 100)
            y = np.random.randint(50, h - 100)
            detections.append({
                'bbox': [x, y, 50, 40],
                'confidence': np.random.uniform(0.5, 0.95),
                'class_id': 0
            })
        
        return detections


class DeepSORTTracker:
    """DeepSORT跟踪器 - 针对蜜蜂跟踪优化"""
    
    def __init__(self, 
                 max_age: int = 30,
                 min_confidence: float = 0.3,
                 nms_max_overlap: float = 0.7,
                 max_iou_distance: float = 0.7,
                 max_feature_distance: float = 0.2,
                 nn_budget: int = 100):
        self.max_age = max_age
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_feature_distance = max_feature_distance
        self.nn_budget = nn_budget
        
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # 特征提取器（简化版本，实际应使用ReID模型）
        self.feature_cache = {}
        
    def update(self, detections: List[Dict], 
               frame: np.ndarray = None) -> List[TrackState]:
        """更新跟踪状态
        
        Args:
            detections: 检测结果列表
            frame: 当前帧图像
            
        Returns:
            活跃的跟踪状态列表
        """
        self.frame_count += 1
        
        # 过滤过低置信度框，避免它们反复创建短暂轨迹。
        detections = [
            det for det in detections
            if det.get('confidence', 0.0) >= self.min_confidence
        ]

        # 提取确定性的几何特征。当前轻量 Demo 没有蜜蜂 ReID 网络，
        # 因此匹配主要使用 IoU、中心距离和上一帧速度。
        detection_features = []
        for det in detections:
            feat = self._extract_feature(det['bbox'], frame)
            detection_features.append(feat)
        
        # 匹配跟踪和检测
        matched, unmatched_detections, unmatched_tracks = \
            self._match_detections(detection_features, detections)
        
        # 更新匹配的跟踪
        for track_idx, det_idx in matched:
            track_id = list(self.tracks.keys())[track_idx]
            det = detections[det_idx]
            
            self._update_track(track_id, det, detection_features[det_idx])
        
        # 初始化新的跟踪
        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx], detection_features[det_idx])
        
        # 标记丢失的跟踪
        for track_idx in unmatched_tracks:
            track_id = list(self.tracks.keys())[track_idx]
            self.tracks[track_id].time_since_update += 1
            
            if self.tracks[track_id].time_since_update > self.max_age:
                self.tracks[track_id].state = "deleted"
        
        # 移除删除的跟踪
        self.tracks = {k: v for k, v in self.tracks.items() 
                      if v.state != "deleted"}
        
        return self._get_active_tracks()
    
    def _extract_feature(self, bbox: List[float], 
                        frame: np.ndarray = None) -> np.ndarray:
        """提取确定性的几何特征（不伪造外观/ReID特征）。"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2

        if frame is not None:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_h, frame_w = 1080, 1920

        feature = np.array([
            cx / max(frame_w, 1),
            cy / max(frame_h, 1),
            w / max(frame_w, 1),
            h / max(frame_h, 1),
        ], dtype=np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature
    
    def _match_detections(self, features: List[np.ndarray],
                         detections: List[Dict]) -> Tuple[List, List, List]:
        """匈牙利算法匹配检测和跟踪"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 轻量运动匹配：IoU + 相对中心距离。上一帧速度用于预测本帧中心。
        # 这比原先的随机“外观特征”稳定，也明确不冒充真正的 DeepSORT ReID。
        cost_matrix = np.full(
            (len(self.tracks), len(detections)), np.inf, dtype=np.float32)
        
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_center = (
                track.center[0] + track.velocity[0],
                track.center[1] + track.velocity[1],
            )
            
            for j, det in enumerate(detections):
                det_center = (det['bbox'][0] + det['bbox'][2]/2,
                             det['bbox'][1] + det['bbox'][3]/2)
                
                iou_dist = 1.0 - self._compute_iou(track.bbox, det['bbox'])
                center_distance = np.hypot(
                    det_center[0] - predicted_center[0],
                    det_center[1] - predicted_center[1],
                )
                track_diag = np.hypot(track.bbox[2], track.bbox[3])
                det_diag = np.hypot(det['bbox'][2], det['bbox'][3])
                normalized_center_distance = center_distance / max(
                    track_diag, det_diag, 1.0)

                # 超过约两个框对角线的候选不可能是 60 FPS 下的相邻位置。
                if normalized_center_distance <= 2.0:
                    center_cost = min(normalized_center_distance / 2.0, 1.0)
                    cost_matrix[i, j] = 0.65 * iou_dist + 0.35 * center_cost
        
        # 匈牙利算法（简化实现）
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # 贪心匹配
        while True:
            min_val = np.inf
            min_idx = None
            
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if cost_matrix[i, j] < min_val:
                        min_val = cost_matrix[i, j]
                        min_idx = (i, j)
            
            if min_idx is None or min_val > 0.82:
                break
                
            matched.append(min_idx)
            unmatched_tracks.remove(min_idx[0])
            unmatched_detections.remove(min_idx[1])
        
        return matched, unmatched_detections, unmatched_tracks
    
    def _compute_iou(self, bbox1: List[float], 
                    bbox2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 转换为角点坐标
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _update_track(self, track_id: int, detection: Dict, feature: np.ndarray):
        """更新跟踪状态"""
        track = self.tracks[track_id]
        
        # 计算速度
        old_center = track.center
        new_center = (detection['bbox'][0] + detection['bbox'][2]/2,
                     detection['bbox'][1] + detection['bbox'][3]/2)
        velocity = (new_center[0] - old_center[0],
                  new_center[1] - old_center[1])
        
        # 更新状态
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.center = new_center
        track.velocity = velocity
        track.age += 1
        track.hits += 1
        track.time_since_update = 0
        track.features = feature
        
        # 更新轨迹
        track.trajectory.append(new_center)
        if len(track.trajectory) > 100:  # 限制轨迹长度
            track.trajectory.pop(0)
        
        # 更新状态
        if track.hits >= 3:
            track.state = "confirmed"
    
    def _initiate_track(self, detection: Dict, feature: np.ndarray):
        """初始化新跟踪"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        center = (detection['bbox'][0] + detection['bbox'][2]/2,
                 detection['bbox'][1] + detection['bbox'][3]/2)
        
        new_track = TrackState(
            track_id=track_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            class_id=detection['class_id'],
            center=center,
            age=1,
            hits=1,
            features=feature,
            trajectory=[center]
        )
        
        self.tracks[track_id] = new_track
    
    def _get_active_tracks(self) -> List[TrackState]:
        """获取活跃跟踪"""
        return [track for track in self.tracks.values()
                if track.state == "confirmed" and track.time_since_update == 0]


class ByteTracker:
    """ByteTrack跟踪器 - 适用于高密度场景"""
    
    def __init__(self, track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 second_match_thresh: float = 0.5):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.next_track_id = 0
        self.frame_count = 0
        
    def update(self, detections: List[Dict]) -> List[TrackState]:
        """更新跟踪"""
        self.frame_count += 1
        
        # 分离高低置信度检测
        high_det = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_det = [d for d in detections if d['confidence'] < self.track_thresh]
        
        # 第一阶段匹配：高置信度检测与活跃跟踪
        matched, unmatched_high = self._match(
            high_det, self.tracked_tracks, self.match_thresh)
        
        # 第二阶段匹配：低置信度检测与丢失跟踪
        matched_lost, unmatched_low = self._match(
            low_det, self.lost_tracks, self.second_match_thresh)
        
        # 更新匹配到的跟踪
        for det_idx, track in matched:
            self._update_track(track, high_det[det_idx])
            
        for det_idx, track in matched_lost:
            self._update_track(track, low_det[det_idx])
            track.state = "tracked"
            self.tracked_tracks.append(track)
        
        # 标记未匹配的跟踪为丢失
        for idx in unmatched_high:
            track = self.tracked_tracks[idx]
            track.state = "lost"
            self.lost_tracks.append(track)
        
        # 初始化新的跟踪
        for idx in unmatched_low:
            self._initiate_track(low_det[idx])
        
        # 移除超时丢失跟踪
        self.lost_tracks = [t for t in self.lost_tracks 
                           if self.frame_count - t.last_frame < self.track_buffer]
        
        return [t for t in self.tracked_tracks if t.state == "tracked"]
    
    def _match(self, detections: List[Dict], 
               tracks: List[TrackState], 
               thresh: float) -> Tuple[List, List]:
        """匹配检测和跟踪"""
        if len(tracks) == 0:
            return [], list(range(len(detections)))
        
        # 计算距离矩阵
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self._compute_iou(track.bbox, det['bbox'])
        
        # 贪心匹配
        matched = []
        unmatched_det = list(range(len(detections)))
        unmatched_track = list(range(len(tracks)))
        
        while True:
            min_val = np.inf
            min_idx = None
            for i in unmatched_track:
                for j in unmatched_det:
                    if cost_matrix[i, j] < min_val:
                        min_val = cost_matrix[i, j]
                        min_idx = (i, j)
            
            if min_val > thresh:
                break
                
            matched.append(min_idx)
            unmatched_det.remove(min_idx[1])
            unmatched_track.remove(min_idx[0])
        
        return matched, unmatched_det
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _update_track(self, track: TrackState, detection: Dict):
        """更新跟踪状态"""
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.center = (detection['bbox'][0] + detection['bbox'][2]/2,
                       detection['bbox'][1] + detection['bbox'][3]/2)
        track.last_frame = self.frame_count
        track.hits += 1
        
        track.trajectory.append(track.center)
        if len(track.trajectory) > 100:
            track.trajectory.pop(0)
    
    def _initiate_track(self, detection: Dict):
        """初始化新跟踪"""
        track = TrackState(
            track_id=self.next_track_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            class_id=detection['class_id'],
            center=(detection['bbox'][0] + detection['bbox'][2]/2,
                   detection['bbox'][1] + detection['bbox'][3]/2),
            trajectory=[(detection['bbox'][0] + detection['bbox'][2]/2,
                        detection['bbox'][1] + detection['bbox'][3]/2)]
        )
        self.next_track_id += 1
        self.tracked_tracks.append(track)


class OutsideHiveTracker:
    """巢外蜜蜂跟踪主类 - 整合检测和跟踪"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化检测器
        self.detector = OutsideHiveBeeDetector(
            model_path=config.get('model_path', 'yolov8m.pt'),
            conf_threshold=config.get('conf_threshold', 0.25),
            iou_threshold=config.get('iou_threshold', 0.45),
            device=config.get('device', None),
            imgsz=config.get('imgsz', 640)
        )
        
        # 初始化跟踪器
        tracker_type = config.get('tracker_type', 'deepsort')
        if tracker_type == 'bytetrack':
            self.tracker = ByteTracker(
                track_thresh=config.get('track_thresh', 0.5),
                track_buffer=config.get('track_buffer', 30)
            )
        else:
            self.tracker = DeepSORTTracker(
                max_age=config.get('max_age', 30),
                min_confidence=config.get('min_confidence', 0.3)
            )
        
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[List[TrackState], List[Dict]]:
        """处理单帧
        
        Args:
            frame: 输入帧
            
        Returns:
            (跟踪结果列表, 检测结果列表)
        """
        # 检测
        detections = self.detector.detect(frame)
        
        # 跟踪
        tracks = self.tracker.update(detections, frame)
        
        self.frame_count += 1
        
        return tracks, detections
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """处理视频
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
            
        Returns:
            跟踪统计结果
        """
        cap = cv2.VideoCapture(video_path)
        
        results = {
            'total_frames': 0,
            'total_tracks': 0,
            'track_history': [],
            'entry_events': 0,
            'exit_events': 0
        }
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            tracks, detections = self.process_frame(frame)
            
            results['total_frames'] += 1
            results['total_tracks'] = max(results['total_tracks'], 
                                          max([t.track_id for t in tracks]) + 1 if tracks else 0)
            results['track_history'].append(len(tracks))
            
            # 绘制结果
            annotated_frame = self._draw_results(frame, tracks, detections)
            
            if output_path:
                out.write(annotated_frame)
        
        cap.release()
        if output_path:
            out.release()
            
        return results
    
    def _draw_results(self, frame: np.ndarray, 
                     tracks: List[TrackState],
                     detections: List[Dict]) -> np.ndarray:
        """绘制跟踪结果"""
        output = frame.copy()
        
        # 绘制检测框
        for det in detections:
            x, y, w, h = [int(v) for v in det['bbox']]
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        # 绘制跟踪轨迹
        colors = {}
        for track in tracks:
            if track.track_id not in colors:
                colors[track.track_id] = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255)
                )
            
            color = colors[track.track_id]
            
            # 绘制边界框
            x, y, w, h = [int(v) for v in track.bbox]
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            
            # 绘制跟踪ID
            cv2.putText(output, f"ID:{track.track_id}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制轨迹
            if len(track.trajectory) > 1:
                for i in range(1, len(track.trajectory)):
                    pt1 = (int(track.trajectory[i-1][0]), 
                          int(track.trajectory[i-1][1]))
                    pt2 = (int(track.trajectory[i][0]), 
                          int(track.trajectory[i][1]))
                    cv2.line(output, pt1, pt2, color, 1)
        
        # 绘制统计信息
        cv2.putText(output, f"Tracks: {len(tracks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output


def create_outside_tracker(config: Dict = None) -> OutsideHiveTracker:
    """创建巢外跟踪器"""
    if config is None:
        config = {
            'model_path': 'yolov8m.pt',
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'device': None,
            'tracker_type': 'deepsort',
            'max_age': 30,
            'min_confidence': 0.3
        }
    
    return OutsideHiveTracker(config)


if __name__ == "__main__":
    # 测试代码
    tracker = create_outside_tracker()
    print("巢外蜜蜂跟踪器创建成功")