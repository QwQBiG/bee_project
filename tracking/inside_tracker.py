"""
巢内蜜蜂检测与跟踪模块
针对红外视频特性优化：低对比度、模糊、频繁遮挡
包含图像增强、姿态识别、方向判别功能
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoseState:
    """姿态状态"""
    head_bbox: List[float] = None  # 头部边界框
    abdomen_bbox: List[float] = None  # 腹部边界框
    orientation: float = 0.0  # 朝向角度 [0, 360)
    direction: Tuple[float, float] = (0.0, 0.0)  # 运动方向
    confidence: float = 0.0


@dataclass
class InsideTrackState:
    """巢内跟踪状态"""
    track_id: int
    bbox: List[float]
    confidence: float
    class_id: int
    center: Tuple[float, float]
    pose: PoseState = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


class InfraredImageEnhancer:
    """红外图像增强器 - 改善低对比度模糊图像"""
    
    def __init__(self, clip_limit: float = 2.0, 
                 tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """增强红外图像
        
        Args:
            image: 输入红外图像 (H, W) 或 (H, W, C)
            
        Returns:
            增强后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 自适应直方图均衡化 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)
        
        # 锐化处理
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 边缘增强
        blurred = cv2.GaussianBlur(sharpened, (0, 0), 3)
        enhanced = cv2.addWeighted(sharpened, 1.5, blurred, -0.5, 0)
        
        # 恢复通道
        if len(image.shape) == 3:
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            result = enhanced
            
        return result
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """降噪处理（先缩放到小尺寸降噪，再恢复原尺寸以提升速度）"""
        h, w = image.shape[:2]
        max_side = 480
        
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            small = cv2.resize(image, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        else:
            small = image
        
        if len(small.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(small, None,
                                                       h=10,
                                                       hColor=10,
                                                       templateWindowSize=7,
                                                       searchWindowSize=21)
        else:
            denoised = cv2.fastNlMeansDenoising(small, None,
                                                h=10,
                                                templateWindowSize=7,
                                                searchWindowSize=21)
        
        if max(h, w) > max_side:
            denoised = cv2.resize(denoised, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return denoised
    
    def enhance_contrast(self, image: np.ndarray, 
                        clip_limit: float = 2.0) -> np.ndarray:
        """对比度增强"""
        # 自适应对比度增强
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result


class BeePoseEstimator:
    """蜜蜂姿态估计器 - 头部/腹部结构识别与方向判别"""
    
    def __init__(self):
        self.keypoint_mapping = {
            'head': 0,
            'thorax': 1,
            'abdomen': 2,
            'left_wing': 3,
            'right_wing': 4,
            'left_antenna': 5,
            'right_antenna': 6
        }
        
    def estimate_pose(self, bbox: List[float], 
                     frame: np.ndarray) -> PoseState:
        """估计蜜蜂姿态
        
        Args:
            bbox: 边界框 [x, y, w, h]
            frame: 当前帧图像
            
        Returns:
            PoseState: 姿态估计结果
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # 提取目标区域
        roi = frame[max(0, y):min(frame.shape[0], y+h),
                   max(0, x):min(frame.shape[1], x+w)]
        
        if roi.size == 0:
            return PoseState()
        
        pose = PoseState()
        
        # 估计头部和腹部位置
        head_ratio = 0.35  # 头部占身体的比例
        head_end = int(w * head_ratio)
        
        pose.head_bbox = [x, y, head_end, h]
        pose.abdomen_bbox = [x + head_end, y, w - head_end, h]
        
        # 估计朝向
        pose.orientation = self._estimate_orientation(roi)
        
        # 估计运动方向（需要历史信息，这里简化处理）
        pose.direction = (0.0, 0.0)
        
        return pose
    
    def _estimate_orientation(self, roi: np.ndarray) -> float:
        """估计蜜蜂朝向角度"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # 使用边缘方向估计朝向
        edges = cv2.Canny(gray, 50, 150)
        
        # 计算霍夫变换
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:10]:  # 取前10条线
                rho, theta = line[0]
                angle = np.degrees(theta)
                angles.append(angle)
            
            # 取中位数角度
            orientation = np.median(angles)
        else:
            orientation = 0.0
            
        return orientation
    
    def estimate_direction(self, current_bbox: List[float],
                          previous_bbox: List[float]) -> Tuple[float, float]:
        """估计运动方向
        
        Args:
            current_bbox: 当前帧边界框
            previous_bbox: 上一帧边界框
            
        Returns:
            (dx, dy): 运动方向向量
        """
        cx1, cy1 = current_bbox[0] + current_bbox[2]/2, current_bbox[1] + current_bbox[3]/2
        cx2, cy2 = previous_bbox[0] + previous_bbox[2]/2, previous_bbox[1] + previous_bbox[3]/2
        
        dx = cx1 - cx2
        dy = cy1 - cy2
        
        # 归一化
        magnitude = np.sqrt(dx*dx + dy*dy)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
            
        return (dx, dy)


class InsideHiveBeeDetector:
    """巢内蜜蜂检测器 - 针对红外图像优化"""
    
    def __init__(self, model_path: str = "yolov8m.pt",
                 conf_threshold: float = 0.2,
                 iou_threshold: float = 0.45,
                 device: str = None,
                 use_enhancement: bool = True):
        from utils.common import get_device
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device is not None else get_device()
        self.use_enhancement = use_enhancement
        
        self.enhancer = InfraredImageEnhancer()
        
        # 尝试加载YOLO模型
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except:
            self.model = None
            print("Warning: Using mock detector for inside hive")
        
        self.pose_estimator = BeePoseEstimator()
        
    def detect(self, frame: np.ndarray, 
              enhance: bool = True) -> List[Dict]:
        """检测蜜蜂
        
        Args:
            frame: 输入红外图像
            enhance: 是否进行图像增强
            
        Returns:
            检测结果列表
        """
        # 图像增强
        if enhance and self.use_enhancement:
            processed = self.enhancer.enhance(frame)
            processed = self.enhancer.denoise(processed)
        else:
            processed = frame
            
        if self.model:
            results = self.model(processed, conf=self.conf_threshold,
                                iou=self.iou_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), 
                                float(x2-x1), float(y2-y1)],
                        'confidence': conf,
                        'class_id': cls
                    })
        else:
            detections = self._mock_detect(processed)
            
        return detections
    
    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        """模拟检测结果"""
        h, w = frame.shape[:2]
        detections = []
        
        num = np.random.randint(5, 20)
        for i in range(num):
            x = np.random.randint(50, w - 100)
            y = np.random.randint(50, h - 100)
            detections.append({
                'bbox': [x, y, 40, 35],
                'confidence': np.random.uniform(0.4, 0.9),
                'class_id': 0
            })
            
        return detections


class InsideHiveTracker:
    """巢内蜜蜂跟踪器 - 整合检测、跟踪和姿态估计"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化检测器
        self.detector = InsideHiveBeeDetector(
            model_path=config.get('model_path', 'yolov8m.pt'),
            conf_threshold=config.get('conf_threshold', 0.2),
            iou_threshold=config.get('iou_threshold', 0.45),
            device=config.get('device', None),
            use_enhancement=config.get('use_enhancement', True)
        )
        
        # 跟踪器参数
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 2)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        self.tracks: Dict[int, InsideTrackState] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        self.pose_estimator = BeePoseEstimator()
        self.previous_bboxes = {}  # 用于方向估计
        
    def update(self, detections: List[Dict], 
               frame: np.ndarray = None) -> List[InsideTrackState]:
        """更新跟踪状态"""
        self.frame_count += 1
        
        # 姿态估计
        for det in detections:
            pose = self.pose_estimator.estimate_pose(det['bbox'], frame)
            det['pose'] = pose
        
        # IoU匹配
        matched, unmatched_det, unmatched_track = self._match(detections)
        
        # 更新匹配的跟踪
        for det_idx, track_id in matched:
            self._update_track(track_id, detections[det_idx])
        
        # 初始化新跟踪
        for det_idx in unmatched_det:
            self._initiate_track(detections[det_idx])
        
        # 处理未匹配的跟踪
        for track_id in unmatched_track:
            self.tracks[track_id].time_since_update += 1
            if self.tracks[track_id].time_since_update > self.max_age:
                self.tracks[track_id].state = "deleted"
        
        # 移除删除的跟踪
        self.tracks = {k: v for k, v in self.tracks.items()
                      if v.state != "deleted"}
        
        # 更新前一帧边界框
        for track in self.tracks.values():
            self.previous_bboxes[track.track_id] = track.bbox.copy()
        
        return self._get_active_tracks()
    
    def _match(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """IoU匹配检测和跟踪"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, det['bbox'])
        
        # 贪心匹配
        matched = []
        used_det = set()
        used_track = set()
        
        while True:
            max_iou = 0.3  # 阈值
            max_idx = None
            
            for i in range(len(self.tracks)):
                if i in used_track:
                    continue
                for j in range(len(detections)):
                    if j in used_det:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_idx = (i, j)
            
            if max_idx is None:
                break
                
            matched.append((max_idx[1], track_ids[max_idx[0]]))
            used_det.add(max_idx[1])
            used_track.add(max_idx[0])
        
        unmatched_det = [i for i in range(len(detections)) if i not in used_det]
        unmatched_track = [
            track_ids[i] for i in range(len(track_ids))
            if i not in used_track
        ]
        
        return matched, unmatched_det, unmatched_track
    
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
    
    def _update_track(self, track_id: int, detection: Dict):
        """更新跟踪状态"""
        track = self.tracks[track_id]
        previous_bbox = track.bbox.copy()
        
        # 计算速度
        old_center = track.center
        new_center = (detection['bbox'][0] + detection['bbox'][2]/2,
                      detection['bbox'][1] + detection['bbox'][3]/2)
        velocity = (new_center[0] - old_center[0],
                   new_center[1] - old_center[1])

        pose = detection.get('pose')
        if pose is not None:
            pose.direction = self.pose_estimator.estimate_direction(
                detection['bbox'], previous_bbox)
        
        # 更新状态
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.center = new_center
        track.velocity = velocity
        track.pose = pose
        track.age += 1
        track.hits += 1
        track.time_since_update = 0
        
        # 更新轨迹
        track.trajectory.append(new_center)
        if len(track.trajectory) > 100:
            track.trajectory.pop(0)
        
        if track.hits >= self.min_hits:
            track.state = "confirmed"
    
    def _initiate_track(self, detection: Dict):
        """初始化新跟踪"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        center = (detection['bbox'][0] + detection['bbox'][2]/2,
                 detection['bbox'][1] + detection['bbox'][3]/2)
        
        new_track = InsideTrackState(
            track_id=track_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            class_id=detection['class_id'],
            center=center,
            pose=detection.get('pose'),
            trajectory=[center]
        )
        
        self.tracks[track_id] = new_track
    
    def _get_active_tracks(self) -> List[InsideTrackState]:
        """获取活跃跟踪"""
        return [track for track in self.tracks.values()
                if track.state == "confirmed" and track.time_since_update <= self.max_age]
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[InsideTrackState], List[Dict]]:
        """处理单帧"""
        detections = self.detector.detect(frame)
        tracks = self.update(detections, frame)
        return tracks, detections


class InsideHiveAnalyzer:
    """巢内行为分析器"""
    
    def __init__(self):
        self.track_history = {}
        
    def analyze_pose_distribution(self, tracks: List[InsideTrackState]) -> Dict:
        """分析姿态分布"""
        orientations = []
        directions = []
        
        for track in tracks:
            if track.pose and track.pose.orientation != 0:
                orientations.append(track.pose.orientation)
            if track.pose:
                directions.append(track.pose.direction)
        
        return {
            'mean_orientation': np.mean(orientations) if orientations else 0,
            'orientation_std': np.std(orientations) if orientations else 0,
            'direction_histogram': self._compute_direction_histogram(directions)
        }
    
    def _compute_direction_histogram(self, directions: List[Tuple],
                                     num_bins: int = 8) -> np.ndarray:
        """计算方向直方图"""
        hist = np.zeros(num_bins)
        bin_size = 360 / num_bins
        
        for dx, dy in directions:
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            bin_idx = int(angle / bin_size) % num_bins
            hist[bin_idx] += 1
        
        if np.sum(hist) > 0:
            hist /= np.sum(hist)
            
        return hist
    
    def analyze_activity_patterns(self, tracks: List[InsideTrackState],
                                  frame_idx: int) -> Dict:
        """分析活动模式"""
        velocities = [np.sqrt(t.velocity[0]**2 + t.velocity[1]**2) 
                     for t in tracks if t.velocity]
        
        return {
            'num_active_bees': len([t for t in tracks if t.state == "confirmed"]),
            'mean_velocity': np.mean(velocities) if velocities else 0,
            'velocity_std': np.std(velocities) if velocities else 0,
            'high_activity_ratio': sum(1 for v in velocities if v > 5) / max(len(velocities), 1)
        }


def create_inside_tracker(config: Dict = None) -> InsideHiveTracker:
    """创建巢内跟踪器"""
    if config is None:
        config = {
            'model_path': 'yolov8m.pt',
            'conf_threshold': 0.2,
            'iou_threshold': 0.45,
            'device': None,
            'use_enhancement': True,
            'max_age': 30,
            'min_hits': 2
        }
    
    return InsideHiveTracker(config)


if __name__ == "__main__":
    tracker = create_inside_tracker()
    print("巢内蜜蜂跟踪器创建成功")