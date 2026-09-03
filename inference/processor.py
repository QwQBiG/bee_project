"""
推理模块 - 整合检测、跟踪和行为分析
支持视频文件处理和实时流处理
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import time

def verified_pose_distribution(analyzer, tracks, pose_enabled: bool):
    """仅在关键点姿态模型启用时生成方向分布，防止几何启发式冒充头尾。"""
    if not pose_enabled:
        return None
    return analyzer.analyze_pose_distribution(tracks)


class OutsideHiveProcessor:
    """巢外视频处理器"""
    
    def __init__(self, config: Dict):
        from behavior.entrance_adapter import TrackEntranceAnalyzer
        from behavior.outside_pollen import OutsidePollenAnalyzer
        from behavior.quantifier import create_behavior_quantifier
        from behavior.trajectory_metrics import TrajectoryMetricsAnalyzer
        from tracking.outside_tracker import create_outside_tracker
        from visualization.visualizer import create_visualizer

        self.config = config
        
        # 初始化跟踪器（优先使用 outside_tracker 配置，兼容单 tracker 配置）
        self.tracker = create_outside_tracker(
            config.get('outside_tracker', config.get('tracker', {})))
        
        # 初始化行为量化器
        self.quantifier = create_behavior_quantifier(config.get('behavior', {}))
        self.pollen_analyzer = OutsidePollenAnalyzer(config.get('pollen_analysis', {}))
        self.entrance_analyzer = TrackEntranceAnalyzer(config.get('entrance_events', {}))
        self.trajectory_metrics = TrajectoryMetricsAnalyzer(config.get('trajectory_metrics', {}))
        self._seen_track_ids = set()
        
        # 初始化可视化器
        self.visualizer = create_visualizer(config.get('visualization', {}))
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_tracks': 0,
            'detection_history': [],
            'track_history': [],
            'entry_events': 0,
            'exit_events': 0
        }
        
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple:
        """处理单帧
        
        Returns:
            (annotated_frame, tracks, detections, stats)
        """
        # 检测和跟踪
        tracks, detections = self.tracker.process_frame(frame)
        
        # 行为量化
        h, w = frame.shape[:2]
        self.quantifier.update(tracks, frame_idx, (h, w), is_inside_hive=False)
        self.pollen_analyzer.update(frame, tracks, frame_idx)
        entrance_events = self.entrance_analyzer.update(tracks, frame_idx, (h, w))
        self.trajectory_metrics.update(tracks, frame_idx, (h, w))
        self._seen_track_ids.update(int(track.track_id) for track in tracks)
        
        # 获取统计
        individual_stats = self.quantifier.get_individual_summary()
        group_stats = self.quantifier.get_group_summary()
        
        stats = {
            'frame': frame_idx,
            'num_tracks': len(tracks),
            'num_detections': len(detections),
            'entrance_events': [event.to_dict() for event in entrance_events],
            **group_stats
        }
        
        # 可视化
        annotated = self.visualizer.annotate_frame(
            frame, tracks, detections, stats={
                'frame': frame_idx,
                'detections': len(detections),
                'confirmed_tracks': len(tracks),
            })
        
        return annotated, tracks, detections, stats
    
    def process_video(self, video_path: str,
                     output_path: str = None,
                     show_video: bool = False,
                     progress_callback=None) -> Dict:
        """处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            show_video: 是否显示视频
            
        Returns:
            处理统计结果
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0:
            self.trajectory_metrics.set_video_fps(fps)
        if progress_callback:
            progress_callback(0, total_frames)
        
        print(f"Video info: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # 创建视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            annotated, tracks, detections, stats = self.process_frame(frame, frame_idx)
            
            # 写入输出视频
            if writer:
                writer.write(annotated)
            
            # 显示视频
            if show_video:
                cv2.imshow('Outside Hive Processing', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 更新统计
            self.stats['total_frames'] = frame_idx + 1
            self.stats['detection_history'].append(len(detections))
            self.stats['track_history'].append(len(tracks))
            self.stats['total_tracks'] = len(self._seen_track_ids)
            
            frame_idx += 1

            if progress_callback and (
                    frame_idx == total_frames or frame_idx % max(total_frames // 100, 1) == 0):
                progress_callback(frame_idx, total_frames)
            
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_idx / elapsed
                print(f"Processed {frame_idx}/{total_frames} frames, "
                     f"{fps_processed:.1f} fps")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # 保存行为分析结果
        behavior_results = {
            'individual_summary': self.quantifier.get_individual_summary(),
            'group_summary': self.quantifier.get_group_summary(),
            'anomalies': self.quantifier.detect_anomalies()
        }
        
        self.entrance_analyzer.finalize()
        entrance_report = self.entrance_analyzer.build_report()
        entrance_counts = entrance_report['counts']
        self.stats['entry_events'] = entrance_counts['entering']
        self.stats['exit_events'] = entrance_counts['leaving']
        self.stats['entrance_analysis'] = entrance_report
        self.stats['behavior_analysis'] = behavior_results
        self.stats['pollen_analysis'] = self.pollen_analyzer.build_report()
        self.stats['trajectory_analysis'] = self.trajectory_metrics.build_report()
        self.stats['processing_time'] = time.time() - start_time
        if progress_callback:
            progress_callback(frame_idx, total_frames)
        
        return self.stats


class InsideHiveProcessor:
    """巢内视频处理器"""
    
    def __init__(self, config: Dict):
        from behavior.inside_metrics import InsideHiveMetricsAnalyzer
        from behavior.quantifier import create_behavior_quantifier
        from behavior.trajectory_metrics import TrajectoryMetricsAnalyzer
        from inference.keypoint_pose import KeypointPoseEstimator
        from tracking.inside_tracker import InsideHiveAnalyzer, create_inside_tracker
        from visualization.visualizer import create_visualizer

        self.config = config
        
        # 初始化跟踪器
        self.tracker = create_inside_tracker(
            config.get('inside_tracker', config.get('tracker', {})))
        
        # 初始化分析器
        self.analyzer = InsideHiveAnalyzer()
        
        # 初始化行为量化器
        self.quantifier = create_behavior_quantifier(config.get('behavior', {}))
        self.inside_metrics = InsideHiveMetricsAnalyzer(
            config.get('inside_metrics', config.get('behavior', {}).get('inside_metrics', {})))
        self.trajectory_metrics = TrajectoryMetricsAnalyzer(config.get('trajectory_metrics', {}))
        self.keypoint_pose = KeypointPoseEstimator(config.get('pose_model', {}))
        self._seen_track_ids = set()
        
        # 初始化可视化器
        self.visualizer = create_visualizer(config.get('visualization', {}))
        
        self.stats = {
            'total_frames': 0,
            'total_tracks': 0,
            'track_history': [],
            'pose_distribution': []
        }
        
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Tuple:
        """处理单帧"""
        # 检测和跟踪
        tracks, detections = self.tracker.process_frame(frame)
        pose_model_stats = self.keypoint_pose.update(frame, tracks)
        
        # 姿态分析
        pose_stats = verified_pose_distribution(
            self.analyzer, tracks, self.keypoint_pose.enabled)
        activity_stats = self.analyzer.analyze_activity_patterns(tracks, frame_idx)
        
        # 行为量化
        h, w = frame.shape[:2]
        self.quantifier.update(tracks, frame_idx, (h, w), is_inside_hive=True)
        self.inside_metrics.update(tracks, frame_idx, (h, w))
        self.trajectory_metrics.update(tracks, frame_idx, (h, w))
        self._seen_track_ids.update(int(track.track_id) for track in tracks)
        
        stats = {
            'frame': frame_idx,
            'num_tracks': len(tracks),
            'pose_distribution': pose_stats,
            'pose_model': pose_model_stats,
            'activity': activity_stats
        }
        
        # 获取密度图
        density_map = self.quantifier.density_analyzer.density_map if hasattr(
            self.quantifier, 'density_analyzer') else None
        
        # 可视化
        annotated = self.visualizer.annotate_frame(
            frame, tracks, detections, density_map=density_map)
        
        return annotated, tracks, detections, stats
    
    def process_video(self, video_path: str,
                     output_path: str = None,
                     show_video: bool = False,
                     progress_callback=None) -> Dict:
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0:
            self.trajectory_metrics.set_video_fps(fps)
        if progress_callback:
            progress_callback(0, total_frames)
        
        print(f"Video info: {width}x{height}, {fps}fps, {total_frames} frames")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, tracks, detections, stats = self.process_frame(frame, frame_idx)
            
            if writer:
                writer.write(annotated)
            
            if show_video:
                cv2.imshow('Inside Hive Processing', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.stats['total_frames'] = frame_idx + 1
            self.stats['track_history'].append(len(tracks))
            self.stats['total_tracks'] = len(self._seen_track_ids)
            pose_distribution = stats.get('pose_distribution')
            if pose_distribution is not None:
                self.stats['pose_distribution'].append(pose_distribution)
            
            frame_idx += 1

            if progress_callback and (
                    frame_idx == total_frames or frame_idx % max(total_frames // 100, 1) == 0):
                progress_callback(frame_idx, total_frames)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        self.stats['behavior_analysis'] = {
            'individual_summary': self.quantifier.get_individual_summary(),
            'group_summary': self.quantifier.get_group_summary(),
            'anomalies': self.quantifier.detect_anomalies()
        }
        inside_report = self.inside_metrics.build_report()
        if not self.keypoint_pose.enabled:
            guarded_names = set()
            for metric in inside_report.get('metrics', []):
                if 'mean_orientation_degrees' in metric or 'motion_alignment' in metric:
                    metric.update(status='unknown', mean_orientation_degrees=None,
                                  motion_alignment=None,
                                  description='Pose model disabled; head-tail direction is unknown.')
                    guarded_names.add(metric.get('name'))
                if 'candidate_ratio' in metric or 'median_body_aspect_ratio' in metric:
                    metric.update(status='candidate_only',
                                  description='Bounding-box shape candidate only; not a head-tail or disease conclusion.')
                    guarded_names.add(metric.get('name'))
            inside_report['alerts'] = [item for item in inside_report.get('alerts', [])
                                       if item.get('metric') not in guarded_names]
        self.stats['inside_metrics'] = inside_report
        self.stats['pose_model'] = self.keypoint_pose.build_report()
        self.stats['trajectory_analysis'] = self.trajectory_metrics.build_report()
        self.stats['processing_time'] = time.time() - start_time
        if progress_callback:
            progress_callback(frame_idx, total_frames)
        
        return self.stats


class MultiModalProcessor:
    """多模态处理器 - 同时处理巢内和巢外视频"""
    
    def __init__(self, outside_config: Dict, inside_config: Dict):
        self.outside_processor = OutsideHiveProcessor(outside_config)
        self.inside_processor = InsideHiveProcessor(inside_config)
        
    def process_synchronized(self, outside_video: str, inside_video: str,
                            output_dir: str = None) -> Dict:
        """同步处理巢内和巢外视频
        
        Args:
            outside_video: 巢外视频路径
            inside_video: 巢内视频路径
            output_dir: 输出目录
            
        Returns:
            综合分析结果
        """
        outside_cap = cv2.VideoCapture(outside_video)
        inside_cap = cv2.VideoCapture(inside_video)
        
        if not outside_cap.isOpened() or not inside_cap.isOpened():
            raise ValueError("Cannot open one or both videos")
        
        # 获取视频信息
        fps = int(outside_cap.get(cv2.CAP_PROP_FPS))
        width_out = int(outside_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_out = int(outside_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_in = int(inside_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_in = int(inside_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        combined_height = min(height_out, height_in)
        combined_width = int(width_out * combined_height / height_out) + int(width_in * combined_height / height_in)
        
        # 创建视频写入器
        writer = None
        combined_writer = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_dir / "outside_result.mp4"), fourcc, fps, (width_out, height_out))
            combined_writer = cv2.VideoWriter(
                str(output_dir / "combined_result.mp4"), fourcc, fps, (combined_width, combined_height))
        
        frame_idx = 0
        start_time = time.time()
        
        results = {
            'outside': {'track_history': [], 'behavior_summary': {}},
            'inside': {'track_history': [], 'behavior_summary': {}},
            'synchronized_events': []
        }
        
        while True:
            ret_out, frame_out = outside_cap.read()
            ret_in, frame_in = inside_cap.read()
            
            if not ret_out or not ret_in:
                break
            
            # 处理巢外帧
            annotated_out, tracks_out, _, stats_out = \
                self.outside_processor.process_frame(frame_out, frame_idx)
            
            # 处理巢内帧
            annotated_in, tracks_in, _, stats_in = \
                self.inside_processor.process_frame(frame_in, frame_idx)
            
            # 同步事件检测
            sync_events = self._detect_synchronized_events(
                tracks_out, tracks_in, frame_idx)
            if sync_events:
                results['synchronized_events'].extend(sync_events)
            
            # 保存结果
            if writer:
                writer.write(annotated_out)
            if combined_writer:
                h_out, w_out = annotated_out.shape[:2]
                h_in, w_in = annotated_in.shape[:2]
                if h_out != h_in:
                    target_h = min(h_out, h_in)
                    if h_out > target_h:
                        new_w = int(w_out * target_h / h_out)
                        annotated_out = cv2.resize(annotated_out, (new_w, target_h))
                    if h_in > target_h:
                        new_w = int(w_in * target_h / h_in)
                        annotated_in = cv2.resize(annotated_in, (new_w, target_h))
                combined = np.hstack([annotated_out, annotated_in])
                combined_writer.write(combined)
            
            # 更新统计
            results['outside']['track_history'].append(len(tracks_out))
            results['inside']['track_history'].append(len(tracks_in))
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} synchronized frames")
        
        outside_cap.release()
        inside_cap.release()
        if writer:
            writer.release()
        if combined_writer:
            combined_writer.release()
        
        # 获取最终分析结果
        results['outside']['behavior_summary'] = \
            self.outside_processor.quantifier.get_individual_summary()
        results['inside']['behavior_summary'] = \
            self.inside_processor.quantifier.get_individual_summary()
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def _detect_synchronized_events(self, outside_tracks: List,
                                   inside_tracks: List,
                                   frame_id: int) -> List[Dict]:
        """检测同步事件（如进入后紧接着在巢内检测到）"""
        events = []
        
        # 简化实现：检测进出巢的时间相关性
        # 实际应用中需要更复杂的轨迹关联算法
        
        return events


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='蜜蜂识别与行为分析')
    parser.add_argument('--mode', type=str, default='outside',
                       choices=['outside', 'inside', 'multi'],
                       help='处理模式')
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--video_inside', type=str,
                       help='巢内视频路径（多模态模式）')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='输出视频路径')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--show', action='store_true',
                       help='显示视频')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except:
        config = {}
    
    if args.mode == 'outside':
        processor = OutsideHiveProcessor(config)
        processor.process_video(args.video, args.output, args.show)
        
    elif args.mode == 'inside':
        processor = InsideHiveProcessor(config)
        processor.process_video(args.video, args.output, args.show)
        
    elif args.mode == 'multi':
        if not args.video_inside:
            print("Error: video_inside required for multi-modal mode")
            return
        
        processor = MultiModalProcessor(config, config)
        results = processor.process_synchronized(
            args.video, args.video_inside, Path(args.output).parent)
        
        # 保存综合结果
        output_dir = Path(args.output).parent
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
