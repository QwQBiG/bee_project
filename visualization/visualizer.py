"""
可视化模块 - 蜜蜂检测、跟踪、行为分析结果展示
支持实时视频标注、轨迹可视化、统计图表生成
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import colorsys


class ColorGenerator:
    """颜色生成器 - 为不同跟踪ID生成区分度高的颜色"""
    
    def __init__(self, num_colors: int = 1000):
        self.num_colors = num_colors
        self.colors = self._generate_distinct_colors(num_colors)
        self.cache = {}
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """生成n个区分度高的颜色"""
        colors = []
        for i in range(n):
            # 使用HSV色彩空间均匀分布
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 3) / 2
            value = 0.9 - 0.2 * (i % 2)
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(rgb[0] * 255), 
                         int(rgb[1] * 255), 
                         int(rgb[2] * 255)))
        return colors
    
    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """获取指定ID对应的颜色"""
        if track_id not in self.cache:
            self.cache[track_id] = self.colors[track_id % self.num_colors]
        return self.cache[track_id]


class TrackVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.color_generator = ColorGenerator()
        
        # 配置参数
        self.track_line_width = self.config.get('track_line_width', 2)
        self.bbox_line_width = self.config.get('bbox_line_width', 2)
        self.font_scale = self.config.get('font_scale', 0.5)
        self.show_labels = self.config.get('show_labels', True)
        self.show_confidence = self.config.get('show_confidence', True)
        self.show_velocity = self.config.get('show_velocity', True)
        self.max_trail_length = self.config.get('max_trail_length', 50)
        
    def draw_track(self, frame: np.ndarray, track, 
                  color: Tuple[int, int, int] = None) -> np.ndarray:
        """绘制单个跟踪目标
        
        Args:
            frame: 输入帧
            track: 跟踪状态对象
            color: 颜色，None时自动分配
            
        Returns:
            绘制后的帧
        """
        output = frame.copy()
        
        # 获取颜色
        if color is None:
            color = self.color_generator.get_color(track.track_id)
        
        # 绘制边界框
        x, y, w, h = [int(v) for v in track.bbox]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, self.bbox_line_width)
        
        # 绘制标签
        if self.show_labels:
            label = f"ID:{track.track_id}"
            
            if self.show_confidence and hasattr(track, 'confidence'):
                label += f" {track.confidence:.2f}"
            
            # 绘制背景
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            cv2.rectangle(output, 
                         (x, y - label_h - baseline - 5),
                         (x + label_w, y),
                         color, -1)
            
            # 绘制文字
            cv2.putText(output, label, (x, y - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (255, 255, 255), 1)
        
        # 绘制速度向量
        if self.show_velocity and hasattr(track, 'velocity'):
            vx, vy = track.velocity
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                center_x = x + w // 2
                center_y = y + h // 2
                end_x = int(center_x + vx * 5)
                end_y = int(center_y + vy * 5)
                cv2.arrowedLine(output, (center_x, center_y),
                              (end_x, end_y), color, 2, tipLength=0.3)
        
        # 绘制轨迹
        if hasattr(track, 'trajectory') and len(track.trajectory) > 1:
            trajectory = track.trajectory[-self.max_trail_length:]
            
            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                
                # 渐变透明度
                alpha = i / len(trajectory)
                line_color = tuple([int(c * alpha) for c in color])
                
                cv2.line(output, pt1, pt2, line_color, 1)
        
        return output
    
    def draw_tracks(self, frame: np.ndarray, 
                   tracks: List, color_map: Dict = None) -> np.ndarray:
        """绘制多个跟踪目标"""
        output = frame.copy()
        
        for track in tracks:
            color = None
            if color_map and track.track_id in color_map:
                color = color_map[track.track_id]
            output = self.draw_track(output, track, color)
        
        return output


class BehaviorVisualizer:
    """行为可视化器"""
    
    def __init__(self):
        self.behavior_colors = {
            'entering': (0, 255, 0),      # 绿色 - 进入
            'exiting': (0, 0, 255),       # 红色 - 离开
            'foraging': (255, 255, 0),    # 青色 - 采集
            'resting': (128, 128, 128),  # 灰色 - 休息
            'wandering': (255, 0, 255),  # 紫色 - 游荡
            'working': (255, 255, 0),    # 青色 - 工作
            'grooming': (0, 255, 255),   # 黄色 - 梳洗
            'moving': (128, 128, 255),   # 浅蓝色 - 移动
        }
    
    def draw_behavior_label(self, frame: np.ndarray, 
                           track, behavior_type: str) -> np.ndarray:
        """在跟踪目标上绘制行为标签"""
        output = frame.copy()
        
        color = self.behavior_colors.get(behavior_type, (255, 255, 255))
        
        x, y, w, h = [int(v) for v in track.bbox]
        
        # 在边界框下方绘制行为标签
        behavior_label = behavior_type.upper()
        (label_w, label_h), baseline = cv2.getTextSize(
            behavior_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        label_y = y + h + label_h + baseline + 5
        
        if label_y < frame.shape[0]:
            cv2.rectangle(output,
                         (x, y + h),
                         (x + label_w, label_y),
                         color, -1)
            cv2.putText(output, behavior_label,
                      (x, label_y - 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                      (255, 255, 255), 1)
        
        return output


class DensityMapVisualizer:
    """密度图可视化器"""
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        self.grid_size = grid_size
        
    def draw_density_map(self, frame: np.ndarray, 
                         density_map: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
        """在帧上叠加密度热力图
        
        Args:
            frame: 输入帧
            density_map: 密度图
            alpha: 叠加透明度
            
        Returns:
            带密度图的帧
        """
        h, w = frame.shape[:2]
        grid_h, grid_w = self.grid_size
        
        # 创建热力图
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        cell_h = h / grid_h
        cell_w = w / grid_w
        
        # 归一化密度图
        if np.max(density_map) > 0:
            normalized = density_map / np.max(density_map)
        else:
            normalized = density_map
        
        # 绘制热力图
        for i in range(grid_h):
            for j in range(grid_w):
                value = normalized[i, j]
                if value > 0.1:
                    # 颜色从蓝到红
                    color = self._value_to_color(value)
                    
                    y1 = int(i * cell_h)
                    y2 = int((i + 1) * cell_h)
                    x1 = int(j * cell_w)
                    x2 = int((j + 1) * cell_w)
                    
                    cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, -1)
        
        # 叠加到原图
        output = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return output
    
    def _value_to_color(self, value: float) -> Tuple[int, int, int]:
        """将密度值转换为颜色"""
        # 蓝->绿->黄->红
        if value < 0.25:
            r = 0
            g = int(255 * value * 4)
            b = 255
        elif value < 0.5:
            r = 0
            g = 255
            b = int(255 * (0.5 - value) * 4)
        elif value < 0.75:
            r = int(255 * (value - 0.5) * 4)
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - value) * 4)
            b = 0
        
        return (b, g, r)


class StatisticsPlotter:
    """统计图表绘制器"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        
    def plot_track_count(self, track_history: List[int],
                        output_path: str = None) -> np.ndarray:
        """绘制跟踪数量变化图"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        frames = range(len(track_history))
        ax.plot(frames, track_history, 'b-', linewidth=1)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of Tracks')
        ax.set_title('Track Count Over Time')
        ax.grid(True)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # 转换为numpy数组
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data
    
    def plot_behavior_distribution(self, behavior_counts: Dict,
                                  output_path: str = None) -> np.ndarray:
        """绘制行为分布饼图"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = list(behavior_counts.keys())
        values = list(behavior_counts.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax.pie(values, labels=labels, autopct='%1.1f%%',
              colors=colors, startangle=90)
        ax.set_title('Behavior Distribution')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data
    
    def plot_activity_intensity(self, intensities: List[float],
                                output_path: str = None) -> np.ndarray:
        """绘制活动强度曲线"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        frames = range(len(intensities))
        ax.fill_between(frames, intensities, alpha=0.3)
        ax.plot(frames, intensities, 'r-', linewidth=1)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Activity Intensity')
        ax.set_title('Group Activity Intensity Over Time')
        ax.set_ylim([0, 1])
        ax.grid(True)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data
    
    def plot_velocity_distribution(self, velocities: List[float],
                                   output_path: str = None) -> np.ndarray:
        """绘制速度分布直方图"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.hist(velocities, bins=30, color='blue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Velocity (pixels/frame)')
        ax.set_ylabel('Count')
        ax.set_title('Velocity Distribution')
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data
    
    def plot_direction_histogram(self, directions: List[float],
                                num_bins: int = 8,
                                output_path: str = None) -> np.ndarray:
        """绘制方向直方图（极坐标）"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # 创建极坐标直方图
        angles = np.deg2rad(directions)
        bins = np.linspace(0, 2*np.pi, num_bins + 1)
        
        ax.hist(angles, bins=bins, color='green', alpha=0.7, edgecolor='black')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        ax.set_title('Movement Direction Distribution')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return data


class VideoAnnotator:
    """视频标注器 - 整合所有可视化功能"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.track_visualizer = TrackVisualizer(
            self.config.get('visualization', self.config))
        self.behavior_visualizer = BehaviorVisualizer()
        self.density_visualizer = DensityMapVisualizer(
            grid_size=tuple(self.config.get('density_grid', [10, 10])))
        self.plotter = StatisticsPlotter()
        
        self.show_tracks = self.config.get('show_tracks', True)
        self.show_detections = self.config.get('show_detections', True)
        self.show_stats = self.config.get('show_stats', True)
        self.show_density = self.config.get('show_density', False)
        
    def annotate_frame(self, frame: np.ndarray,
                      tracks: List,
                      detections: List = None,
                      behaviors: Dict = None,
                      density_map: np.ndarray = None,
                      stats: Dict = None) -> np.ndarray:
        """标注单帧
        
        Args:
            frame: 输入帧
            tracks: 跟踪列表
            detections: 检测列表
            behaviors: 行为字典 {track_id: behavior_type}
            density_map: 密度图
            stats: 统计信息
            
        Returns:
            标注后的帧
        """
        output = frame.copy()
        
        # 绘制密度热力图
        if density_map is not None and self.show_density:
            output = self.density_visualizer.draw_density_map(output, density_map)

        # 先画检测框（青色、细线），再叠加已确认轨迹及 ID。
        if detections and self.show_detections:
            for det in detections:
                x, y, w, h = [int(v) for v in det['bbox']]
                cv2.rectangle(output, (x, y), (x + w, y + h),
                              (255, 255, 0), 1)
                confidence = det.get('confidence')
                if confidence is not None:
                    cv2.putText(
                        output, f"det {confidence:.2f}", (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        
        # 绘制跟踪
        if self.show_tracks:
            output = self.track_visualizer.draw_tracks(output, tracks)
        
        # 绘制行为标签
        if behaviors:
            for track in tracks:
                if track.track_id in behaviors:
                    output = self.behavior_visualizer.draw_behavior_label(
                        output, track, behaviors[track.track_id])
        
        # 绘制统计信息
        if stats and self.show_stats:
            output = self._draw_stats(output, stats)
        
        return output
    
    def _draw_stats(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """绘制统计信息"""
        output = frame.copy()
        
        y_offset = 30
        line_height = 25
        
        for key, value in stats.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(output, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
        
        return output
    
    def create_tracking_video(self, video_path: str,
                            output_path: str,
                            process_func,
                            fps: int = 30):
        """创建跟踪结果视频
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            process_func: 处理函数，接受帧返回(tracks, detections, stats)
            fps: 输出视频帧率
        """
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            tracks, detections, stats = process_func(frame, frame_idx)
            
            annotated = self.annotate_frame(frame, tracks, detections,
                                           stats=stats)
            
            out.write(annotated)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames")
        
        cap.release()
        out.release()
        print(f"Video saved to {output_path}")


def create_visualizer(config: Dict = None) -> VideoAnnotator:
    """创建可视化器"""
    return VideoAnnotator(config)


if __name__ == "__main__":
    visualizer = create_visualizer()
    print("可视化器创建成功")
