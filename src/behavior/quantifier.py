"""
行为量化与分析模块
从个体和群体层面量化蜜蜂行为指标
支持进出巢行为、采集行为、聚集行为等的分析与检测
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path


@dataclass
class IndividualBehavior:
    """个体行为数据"""
    track_id: int
    behavior_type: str  # entering, exiting, foraging, resting, etc.
    start_frame: int
    end_frame: int
    duration: int = 0
    confidence: float = 0.0
    features: Dict = field(default_factory=dict)


@dataclass
class GroupBehavior:
    """群体行为数据"""
    frame_range: Tuple[int, int]
    num_bees: int
    activity_intensity: float  # 0-1, 0=安静, 1=活跃
    density_map: np.ndarray = None
    aggregation_score: float = 0.0
    dominant_direction: float = 0.0


class TrackletBuilder:
    """轨迹片段构建器 - 从连续跟踪生成行为轨迹"""
    
    def __init__(self, min_length: int = 10):
        self.min_length = min_length
        self.tracklets = defaultdict(list)
        
    def add_detection(self, track_id: int, frame_id: int, 
                     bbox: List[float], center: Tuple[float, float],
                     velocity: Tuple[float, float] = (0, 0)):
        """添加检测到轨迹"""
        self.tracklets[track_id].append({
            'frame_id': frame_id,
            'bbox': bbox,
            'center': center,
            'velocity': velocity
        })
    
    def get_tracklet(self, track_id: int) -> List[Dict]:
        """获取完整轨迹"""
        return self.tracklets.get(track_id, [])
    
    def compute_tracklet_features(self, track_id: int) -> Dict:
        """计算轨迹特征"""
        tracklet = self.get_tracklet(track_id)
        if len(tracklet) < self.min_length:
            return {}
        
        centers = np.array([t['center'] for t in tracklet])
        velocities = np.array([t['velocity'] for t in tracklet])
        
        # 位置统计
        mean_center = np.mean(centers, axis=0)
        std_center = np.std(centers, axis=0)
        
        # 速度统计
        mean_velocity = np.mean(velocities, axis=0)
        speed = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speed)
        speed_std = np.std(speed)
        max_speed = np.max(speed)
        
        # 移动距离
        total_distance = np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))
        
        # 方向变化
        if len(velocities) > 1:
            direction_changes = np.sum(np.abs(np.diff(
                np.arctan2(velocities[:, 1], velocities[:, 0]))))
        else:
            direction_changes = 0
        
        return {
            'track_id': track_id,
            'length': len(tracklet),
            'mean_position': mean_center.tolist(),
            'position_std': std_center.tolist(),
            'mean_velocity': mean_velocity.tolist(),
            'mean_speed': float(mean_speed),
            'speed_std': float(speed_std),
            'max_speed': float(max_speed),
            'total_distance': float(total_distance),
            'direction_changes': float(direction_changes)
        }


class BehaviorClassifier:
    """行为分类器 - 识别不同类型的行为"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 行为阈值配置
        self.entering_threshold = self.config.get('entering_threshold', 0.8)
        self.foraging_threshold = self.config.get('foraging_threshold', 2.0)
        self.resting_threshold = self.config.get('resting_threshold', 0.5)
        self.wandering_threshold = self.config.get('wandering_threshold', 3.0)
        
        # 蜂箱入口区域（需要根据实际标定）
        self.hive_entrance = {
            'x': 0.45,  # 相对于画面宽度的比例
            'y': 0.5,
            'width': 0.1,
            'height': 0.2
        }
        
    def classify_individual_behavior(self, tracklet: List[Dict],
                                   frame_shape: Tuple[int, int],
                                   is_inside_hive: bool = False) -> str:
        """分类个体行为
        
        Args:
            tracklet: 轨迹片段
            frame_shape: 帧尺寸 (H, W)
            is_inside_hive: 是否为巢内场景
            
        Returns:
            行为类型: entering, exiting, foraging, resting, wandering, grooming
        """
        if len(tracklet) < 5:
            return "unknown"
        
        h, w = frame_shape
        centers = np.array([t['center'] for t in tracklet])
        velocities = np.array([t['velocity'] for t in tracklet])
        
        # 计算特征
        speed = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speed)
        max_speed = np.max(speed)
        
        # 位置方差（判断是否静止）
        position_std = np.std(centers, axis=0)
        position_variance = np.sum(position_std)
        
        # 方向一致性
        if len(velocities) > 1:
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angle_diff = np.abs(np.diff(angles))
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            direction_consistency = np.mean(angle_diff < 0.5)
        else:
            direction_consistency = 0
        
        # 入口区域检测
        entrance_region = self._get_entrance_region(w, h)
        starts_in_entrance = self._is_in_region(centers[0], entrance_region)
        ends_in_entrance = self._is_in_region(centers[-1], entrance_region)
        
        # 行为分类逻辑
        if not is_inside_hive:
            # 巢外场景
            if starts_in_entrance and not ends_in_entrance:
                return "entering"
            elif not starts_in_entrance and ends_in_entrance:
                return "exiting"
            elif mean_speed > self.foraging_threshold and direction_consistency > 0.7:
                return "foraging"
            elif position_variance < self.resting_threshold:
                return "resting"
            elif position_variance > self.wandering_threshold:
                return "wandering"
            else:
                return "moving"
        else:
            # 巢内场景
            if mean_speed > self.foraging_threshold:
                return "working"
            elif position_variance < self.resting_threshold:
                return "resting"
            elif direction_consistency > 0.8:
                return "grooming"
            else:
                return "moving"
    
    def _get_entrance_region(self, w: int, h: int) -> Dict:
        """获取蜂箱入口区域"""
        return {
            'x': int(self.hive_entrance['x'] * w),
            'y': int(self.hive_entrance['y'] * h),
            'width': int(self.hive_entrance['width'] * w),
            'height': int(self.hive_entrance['height'] * h)
        }
    
    def _is_in_region(self, point: Tuple[float, float], 
                     region: Dict) -> bool:
        """判断点是否在区域内"""
        px, py = point
        return (region['x'] <= px <= region['x'] + region['width'] and
                region['y'] <= py <= region['y'] + region['height'])


class ActivityIntensityAnalyzer:
    """活动强度分析器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.velocity_history = deque(maxlen=window_size)
        self.count_history = deque(maxlen=window_size)
        
    def update(self, tracks: List, frame_id: int):
        """更新活动强度统计"""
        velocities = []
        for track in tracks:
            if hasattr(track, 'velocity'):
                v = np.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
                velocities.append(v)
        
        mean_velocity = np.mean(velocities) if velocities else 0
        num_bees = len(tracks)
        
        self.velocity_history.append(mean_velocity)
        self.count_history.append(num_bees)
    
    def compute_intensity(self) -> float:
        """计算当前活动强度 [0, 1]"""
        if len(self.velocity_history) < 5:
            return 0.5  # 默认中等强度
        
        # 归一化的速度和数量
        velocity_score = np.mean(self.velocity_history) / 10.0  # 假设最大速度10
        count_score = np.mean(self.count_history) / 50.0  # 假设最大50只蜜蜂
        
        intensity = np.clip((velocity_score + count_score) / 2, 0, 1)
        return float(intensity)
    
    def get_activity_trend(self) -> str:
        """获取活动趋势"""
        if len(self.velocity_history) < 10:
            return "stable"
        
        recent = np.mean(list(self.velocity_history)[-5:])
        older = np.mean(list(self.velocity_history)[-10:-5])
        
        diff = recent - older
        if diff > 0.5:
            return "increasing"
        elif diff < -0.5:
            return "decreasing"
        else:
            return "stable"


class SpatialDensityAnalyzer:
    """空间密度分析器"""
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10),
                 frame_size: Tuple[int, int] = (1920, 1080)):
        self.grid_size = grid_size
        self.frame_size = frame_size
        self.density_map = np.zeros(grid_size)
        
    def compute_density(self, tracks: List) -> np.ndarray:
        """计算空间密度图
        
        Args:
            tracks: 跟踪列表
            
        Returns:
            密度图 (grid_h, grid_w)
        """
        self.density_map = np.zeros(self.grid_size)
        
        h, w = self.frame_size
        grid_h, grid_w = self.grid_size
        cell_h = h / grid_h
        cell_w = w / grid_w
        
        for track in tracks:
            if hasattr(track, 'center'):
                cx, cy = track.center
                grid_x = min(int(cx / cell_w), grid_w - 1)
                grid_y = min(int(cy / cell_h), grid_h - 1)
                self.density_map[grid_y, grid_x] += 1
        
        # 归一化
        if np.sum(self.density_map) > 0:
            self.density_map /= np.sum(self.density_map)
            
        return self.density_map
    
    def detect_aggregation(self, threshold: float = 0.1) -> List[Tuple[int, int]]:
        """检测聚集区域
        
        Args:
            threshold: 聚集阈值（相对于平均密度的倍数）
            
        Returns:
            聚集区域坐标列表 [(grid_x, grid_y), ...]
        """
        mean_density = np.mean(self.density_map)
        high_density_threshold = mean_density * 2
        
        aggregation_regions = []
        grid_h, grid_w = self.grid_size
        
        for y in range(grid_h):
            for x in range(grid_w):
                if self.density_map[y, x] > high_density_threshold:
                    aggregation_regions.append((x, y))
        
        return aggregation_regions
    
    def compute_aggregation_score(self) -> float:
        """计算聚集分数"""
        if np.sum(self.density_map) == 0:
            return 0.0
            
        # 使用熵作为聚集度量的简化版本
        # 低熵表示高聚集
        p = self.density_map.flatten()
        p = p[p > 0]  # 移除零
        
        if len(p) == 0:
            return 0.0
            
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(p))
        
        aggregation_score = 1 - entropy / max_entropy if max_entropy > 0 else 0
        return float(aggregation_score)


class BehaviorQuantifier:
    """行为量化主类"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 子模块初始化
        self.tracklet_builder = TrackletBuilder(
            min_length=self.config.get('min_tracklet_length', 10))
        self.behavior_classifier = BehaviorClassifier(config.get('behavior', {}))
        self.intensity_analyzer = ActivityIntensityAnalyzer(
            window_size=self.config.get('intensity_window', 30))
        self.density_analyzer = SpatialDensityAnalyzer(
            grid_size=tuple(self.config.get('density_grid', [10, 10])),
            frame_size=tuple(self.config.get('frame_size', [1080, 1920]))
        )
        
        # 历史数据
        self.behavior_history = []
        self.group_behavior_history = []
        
    def update(self, tracks: List, frame_id: int, 
               frame_shape: Tuple[int, int],
               is_inside_hive: bool = False):
        """更新行为分析
        
        Args:
            tracks: 当前帧跟踪列表
            frame_id: 帧ID
            frame_shape: 帧尺寸 (H, W)
            is_inside_hive: 是否为巢内场景
        """
        # 更新轨迹片段
        for track in tracks:
            self.tracklet_builder.add_detection(
                track_id=track.track_id,
                frame_id=frame_id,
                bbox=track.bbox,
                center=track.center,
                velocity=getattr(track, 'velocity', (0, 0))
            )
        
        # 更新活动强度
        self.intensity_analyzer.update(tracks, frame_id)
        
        # 更新空间密度
        self.density_analyzer.compute_density(tracks)
        
        # 分析当前帧行为
        current_behaviors = self._analyze_current_frame(
            tracks, frame_id, frame_shape, is_inside_hive)
        
        self.behavior_history.extend(current_behaviors)
        
        # 计算群体行为
        group_behavior = self._compute_group_behavior(frame_id, tracks)
        self.group_behavior_history.append(group_behavior)
    
    def _analyze_current_frame(self, tracks: List, frame_id: int,
                              frame_shape: Tuple[int, int],
                              is_inside_hive: bool) -> List[IndividualBehavior]:
        """分析当前帧个体行为"""
        behaviors = []
        
        for track in tracks:
            tracklet = self.tracklet_builder.get_tracklet(track.track_id)
            if len(tracklet) < 5:
                continue
            
            behavior_type = self.behavior_classifier.classify_individual_behavior(
                tracklet, frame_shape, is_inside_hive)
            
            # 计算持续时间
            start_frame = tracklet[0]['frame_id']
            end_frame = tracklet[-1]['frame_id']
            duration = end_frame - start_frame
            
            behavior = IndividualBehavior(
                track_id=track.track_id,
                behavior_type=behavior_type,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration,
                confidence=track.confidence if hasattr(track, 'confidence') else 1.0
            )
            
            behaviors.append(behavior)
        
        return behaviors
    
    def _compute_group_behavior(self, frame_id: int, 
                               tracks: List) -> GroupBehavior:
        """计算群体行为"""
        activity_intensity = self.intensity_analyzer.compute_intensity()
        aggregation_score = self.density_analyzer.compute_aggregation_score()
        density_map = self.density_analyzer.density_map
        
        # 计算主方向
        velocities = []
        for track in tracks:
            if hasattr(track, 'velocity'):
                velocities.append(track.velocity)
        
        if velocities:
            mean_vx = np.mean([v[0] for v in velocities])
            mean_vy = np.mean([v[1] for v in velocities])
            dominant_direction = np.degrees(np.arctan2(mean_vy, mean_vx))
        else:
            dominant_direction = 0
        
        return GroupBehavior(
            frame_range=(frame_id, frame_id),
            num_bees=len(tracks),
            activity_intensity=activity_intensity,
            density_map=density_map,
            aggregation_score=aggregation_score,
            dominant_direction=dominant_direction
        )
    
    def get_individual_summary(self) -> Dict:
        """获取个体行为汇总"""
        if not self.behavior_history:
            return {}
        
        # 按行为类型统计
        behavior_counts = defaultdict(int)
        behavior_durations = defaultdict(list)
        
        for behavior in self.behavior_history:
            behavior_counts[behavior.behavior_type] += 1
            behavior_durations[behavior.behavior_type].append(behavior.duration)
        
        summary = {
            'total_behaviors': len(self.behavior_history),
            'behavior_counts': dict(behavior_counts),
            'behavior_durations': {
                k: {
                    'mean': np.mean(v),
                    'std': np.std(v),
                    'min': np.min(v),
                    'max': np.max(v)
                }
                for k, v in behavior_durations.items()
            }
        }
        
        return summary
    
    def get_group_summary(self) -> Dict:
        """获取群体行为汇总"""
        if not self.group_behavior_history:
            return {}
        
        intensities = [gb.activity_intensity for gb in self.group_behavior_history]
        aggregations = [gb.aggregation_score for gb in self.group_behavior_history]
        num_bees = [gb.num_bees for gb in self.group_behavior_history]
        
        return {
            'activity_intensity': {
                'mean': np.mean(intensities),
                'std': np.std(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities),
                'trend': self.intensity_analyzer.get_activity_trend()
            },
            'aggregation_score': {
                'mean': np.mean(aggregations),
                'std': np.std(aggregations),
                'min': np.min(aggregations),
                'max': np.max(aggregations)
            },
            'bee_count': {
                'mean': np.mean(num_bees),
                'std': np.std(num_bees),
                'min': np.min(num_bees),
                'max': np.max(num_bees)
            }
        }
    
    def detect_anomalies(self) -> List[Dict]:
        """检测异常行为"""
        anomalies = []
        
        group_summary = self.get_group_summary()
        
        # 检测异常高活动强度
        if group_summary.get('activity_intensity', {}).get('max', 0) > 0.9:
            anomalies.append({
                'type': 'high_activity',
                'description': '检测到异常高的蜂群活动强度',
                'severity': 'warning'
            })
        
        # 检测异常低活动强度
        if group_summary.get('activity_intensity', {}).get('min', 1) < 0.1:
            anomalies.append({
                'type': 'low_activity',
                'description': '检测到异常低的蜂群活动强度，可能存在健康问题',
                'severity': 'critical'
            })
        
        # 检测异常聚集
        if group_summary.get('aggregation_score', {}).get('max', 0) > 0.8:
            anomalies.append({
                'type': 'abnormal_aggregation',
                'description': '检测到异常聚集行为',
                'severity': 'warning'
            })
        
        return anomalies
    
    def save_results(self, output_path: str):
        """保存分析结果"""
        results = {
            'individual_summary': self.get_individual_summary(),
            'group_summary': self.get_group_summary(),
            'anomalies': self.detect_anomalies()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


class HiveEntranceAnalyzer:
    """蜂箱入口分析器 - 专门用于分析进出巢行为"""
    
    def __init__(self, entrance_region: Dict):
        self.entrance_region = entrance_region
        self.entry_count = 0
        self.exit_count = 0
        self.trajectory_history = defaultdict(list)
        
    def analyze_crossing(self, tracklet: List[Dict]) -> Tuple[Optional[str], int]:
        """分析穿越行为
        
        Returns:
            (行为类型, 穿越帧ID): entering, exiting, 或 None
        """
        if len(tracklet) < 2:
            return None, -1
        
        entrance = self.entrance_region
        centers = [t['center'] for t in tracklet]
        
        # 检查是否穿越入口区域
        crossed = False
        crossing_frame = -1
        
        for i, center in enumerate(centers):
            if self._is_in_entrance(center):
                if i > 0:
                    crossed = True
                    crossing_frame = tracklet[i]['frame_id']
                    break
        
        if not crossed:
            return None, -1
        
        # 判断进入还是离开
        # 基于运动方向判断
        velocities = [t['velocity'] for t in tracklet 
                     if t['velocity'] != (0, 0)]
        
        if velocities:
            mean_velocity = np.mean(velocities, axis=0)
            
            # 如果向入口方向运动，判断为离开
            # 如果从入口向外运动，判断为进入
            # 这里需要根据实际坐标系调整
            direction = np.arctan2(mean_velocity[1], mean_velocity[0])
            
            # 简化判断：基于入口Y坐标和运动方向
            start_in = self._is_in_entrance(centers[0])
            end_in = self._is_in_entrance(centers[-1])
            
            if start_in and not end_in:
                return "exiting", crossing_frame
            elif not start_in and end_in:
                return "entering", crossing_frame
        
        return None, -1
    
    def _is_in_entrance(self, point: Tuple[float, float]) -> bool:
        """判断点是否在入口区域"""
        px, py = point
        e = self.entrance_region
        return (e['x'] <= px <= e['x'] + e['width'] and
                e['y'] <= py <= e['y'] + e['height'])
    
    def get_traffic_stats(self) -> Dict:
        """获取交通统计"""
        return {
            'total_entries': self.entry_count,
            'total_exits': self.exit_count,
            'net_flow': self.entry_count - self.exit_count
        }


def create_behavior_quantifier(config: Dict = None) -> BehaviorQuantifier:
    """创建行为量化器"""
    if config is None:
        config = {
            'min_tracklet_length': 10,
            'intensity_window': 30,
            'density_grid': [10, 10],
            'frame_size': [1080, 1920],
            'behavior': {
                'entering_threshold': 0.8,
                'foraging_threshold': 2.0,
                'resting_threshold': 0.5,
                'wandering_threshold': 3.0
            }
        }
    
    return BehaviorQuantifier(config)


if __name__ == "__main__":
    quantifier = create_behavior_quantifier()
    print("行为量化器创建成功")
