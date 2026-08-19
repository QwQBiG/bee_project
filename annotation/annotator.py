"""
数据标注模块 - 支持视频帧标注和批量处理
提供交互式标注工具和数据格式转换功能
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET


class BeeAnnotationConfig:
    """标注配置类"""
    
    # 巢外场景标注类别
    OUTSIDE_CLASSES = {
        0: "bee_entering",    # 进入蜂箱的蜜蜂
        1: "bee_exiting",     # 飞出蜂箱的蜜蜂
        2: "bee_foraging",    # 采集花蜜的蜜蜂
        3: "bee_resting",     # 停留休息的蜜蜂
    }
    
    # 巢内场景标注类别
    INSIDE_CLASSES = {
        0: "bee_normal",      # 正常蜜蜂
        1: "bee_queen",       # 蜂王
        2: "bee_working",     # 工作状态蜜蜂
        3: "bee_moving",      # 移动中蜜蜂
    }
    
    # 行为标注类型
    BEHAVIOR_TYPES = {
        0: "entering",        # 进入行为
        1: "exiting",         # 离开行为
        2: "foraging",        # 采集行为
        3: "resting",         # 休息行为
        4: "grooming",        # 梳洗行为
        5: "trophallaxis",     # 交哺行为
        6: "wandering",       # 游荡行为
    }


class VideoAnnotationExtractor:
    """视频帧提取与预标注工具"""
    
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
    def open_video(self) -> bool:
        """打开视频文件"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True
    
    def extract_frames(self, frame_indices: List[int], prefix: str = "frame") -> List[str]:
        """提取指定帧"""
        saved_paths = []
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                output_path = self.output_dir / f"{prefix}_{idx:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_paths.append(str(output_path))
        return saved_paths
    
    def extract_key_frames(self, interval: int = 30) -> List[str]:
        """按间隔提取关键帧"""
        frame_indices = list(range(0, self.total_frames, interval))
        return self.extract_frames(frame_indices, "keyframe")
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def close(self):
        """关闭视频"""
        if self.cap:
            self.cap.release()


class COCOAnnotationConverter:
    """COCO格式标注转换器"""
    
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.image_id = 0
        self.annotation_id = 0
        
    def add_category(self, category_id: int, name: str, supercategory: str = "bee"):
        """添加类别"""
        self.categories.append({
            "id": category_id,
            "name": name,
            "supercategory": supercategory
        })
    
    def add_image(self, file_name: str, width: int, height: int, 
                  frame_id: int = None) -> int:
        """添加图像信息"""
        image_info = {
            "id": self.image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "frame_id": frame_id
        }
        self.images.append(image_info)
        self.image_id += 1
        return self.image_id - 1
    
    def add_annotation(self, image_id: int, category_id: int, 
                       bbox: List[float], area: float = None,
                       track_id: int = None) -> int:
        """添加标注"""
        x, y, w, h = bbox
        if area is None:
            area = w * h
            
        annotation = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0
        }
        
        if track_id is not None:
            annotation["track_id"] = track_id
            
        self.annotations.append(annotation)
        self.annotation_id += 1
        return self.annotation_id - 1
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
    
    def save(self, output_path: str):
        """保存为JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, json_path: str) -> 'COCOAnnotationConverter':
        """从JSON文件加载"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converter = cls()
        converter.images = data.get('images', [])
        converter.annotations = data.get('annotations', [])
        converter.categories = data.get('categories', [])
        converter.image_id = max([img['id'] for img in converter.images] or [0]) + 1
        converter.annotation_id = max([ann['id'] for ann in converter.annotations] or [0]) + 1
        return converter


class MOTAnnotationConverter:
    """MOT格式标注转换器（用于跟踪任务）"""
    
    def __init__(self):
        self.tracks = []
        
    def add_detection(self, frame_id: int, track_id: int, x: float, y: float,
                      w: float, h: float, confidence: float = 1.0,
                      class_id: int = 0, visibility: float = 1.0):
        """添加检测结果
        
        MOT格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        self.tracks.append([
            frame_id, track_id, x, y, w, h, confidence, class_id, visibility
        ])
    
    def save(self, output_path: str):
        """保存为MOT格式文本文件"""
        with open(output_path, 'w') as f:
            for track in sorted(self.tracks, key=lambda x: (x[0], x[1])):
                f.write(f"{int(track[0])},{int(track[1])},{track[2]:.2f},{track[3]:.2f},"
                       f"{track[4]:.2f},{track[5]:.2f},{track[6]:.2f},{int(track[7])},"
                       f"{track[8]:.2f}\n")
    
    @classmethod
    def load(cls, mot_path: str) -> 'MOTAnnotationConverter':
        """从MOT文件加载"""
        converter = cls()
        with open(mot_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    converter.tracks.append([float(x) for x in parts[:9]])
        return converter


class ManualAnnotationTool:
    """手动标注工具（提供标注接口）"""
    
    def __init__(self, window_name: str = "Bee Annotation Tool"):
        self.window_name = window_name
        self.current_frame = None
        self.bboxes = []
        self.current_bbox = None
        self.is_drawing = False
        self.class_id = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.current_bbox = [x, y, 0, 0]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.current_bbox[2] = x - self.current_bbox[0]
                self.current_bbox[3] = y - self.current_bbox[1]
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            if self.current_bbox[2] > 0 and self.current_bbox[3] > 0:
                self.bboxes.append(self.current_bbox.copy())
            self.current_bbox = None
    
    def set_frame(self, frame: np.ndarray):
        """设置当前帧"""
        self.current_frame = frame.copy()
        self.bboxes = []
        
    def get_bboxes(self) -> List[List[float]]:
        """获取当前帧的所有标注框"""
        return self.bboxes.copy()
    
    def display(self) -> np.ndarray:
        """显示当前帧和标注"""
        display_frame = self.current_frame.copy()
        
        # 绘制已标注的框
        for i, bbox in enumerate(self.bboxes):
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID:{i}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制正在绘制的框
        if self.current_bbox:
            x, y, w, h = [int(v) for v in self.current_bbox]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        return display_frame
    
    def save_annotations(self, output_path: str, frame_id: int):
        """保存标注结果"""
        annotations = {
            "frame_id": frame_id,
            "bboxes": self.bboxes,
            "class_id": self.class_id
        }
        with open(output_path, 'w') as f:
            json.dump(annotations, f)


class YOLOAnnotationConverter:
    """YOLO格式标注转换器"""
    
    def __init__(self, class_names: Dict[int, str]):
        self.class_names = class_names
        self.annotations = []
        
    def add_annotation(self, bbox: List[float], class_id: int, 
                       img_width: int, img_height: int):
        """添加YOLO格式标注
        
        YOLO格式: <class_id> <x_center> <y_center> <width> <height>
        所有值都归一化到[0, 1]
        """
        x, y, w, h = bbox
        
        # 转换为中心点和宽高，并归一化
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        self.annotations.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": norm_w,
            "height": norm_h
        })
        
    def save(self, output_path: str):
        """保存为YOLO格式文本文件"""
        with open(output_path, 'w') as f:
            for ann in self.annotations:
                f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                       f"{ann['width']:.6f} {ann['height']:.6f}\n")
                
    def save_dataset_structure(self, images_dir: str, output_dir: str, 
                               train_split: float = 0.7,
                               val_split: float = 0.15):
        """生成YOLO数据集目录结构"""
        output_dir = Path(output_dir)
        
        # 创建目录
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
        
        # 创建数据集配置文件
        dataset_config = {
            "path": str(output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.class_names
        }
        
        return dataset_config


def batch_convert_to_coco(video_paths: List[str], output_dir: str,
                          annotation_config: BeeAnnotationConfig,
                          frame_interval: int = 30) -> COCOAnnotationConverter:
    """批量转换视频为COCO格式标注
    
    Args:
        video_paths: 视频文件路径列表
        output_dir: 输出目录
        annotation_config: 标注配置
        frame_interval: 帧采样间隔
    
    Returns:
        COCOAnnotationConverter对象
    """
    converter = COCOAnnotationConverter()
    
    # 添加类别
    for class_id, class_name in annotation_config.OUTSIDE_CLASSES.items():
        converter.add_category(class_id, class_name, "outside_bee")
    for class_id, class_name in annotation_config.INSIDE_CLASSES.items():
        converter.add_category(class_id + 10, class_name, "inside_bee")
    
    for video_path in video_paths:
        extractor = VideoAnnotationExtractor(video_path, output_dir)
        if not extractor.open_video():
            continue
            
        frame_idx = 0
        while True:
            frame = extractor.get_frame(frame_idx)
            if frame is None:
                break
                
            image_id = converter.add_image(
                file_name=f"{Path(video_path).stem}_{frame_idx:06d}.jpg",
                width=extractor.width,
                height=extractor.height,
                frame_id=frame_idx
            )
            
            # 这里需要接入实际的检测模型或手动标注
            # 目前为占位符，实际使用时需要替换为检测结果
            
            frame_idx += frame_interval
            
        extractor.close()
        
    return converter


def create_sample_annotations(output_dir: str):
    """创建示例标注数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建COCO格式示例
    coco_converter = COCOAnnotationConverter()
    coco_converter.add_category(0, "bee", "insect")
    
    for i in range(10):
        coco_converter.add_image(
            file_name=f"sample_{i:04d}.jpg",
            width=1920,
            height=1080,
            frame_id=i
        )
        # 添加一些示例标注
        for j in range(5):
            x = np.random.randint(100, 800)
            y = np.random.randint(100, 600)
            w = np.random.randint(30, 80)
            h = np.random.randint(30, 80)
            coco_converter.add_annotation(
                image_id=i,
                category_id=0,
                bbox=[x, y, w, h],
                track_id=j
            )
    
    coco_converter.save(str(output_dir / "sample_coco.json"))
    
    # 创建MOT格式示例
    mot_converter = MOTAnnotationConverter()
    for frame_id in range(100):
        for track_id in range(5):
            x = 400 + track_id * 50 + np.random.randint(-10, 10)
            y = 300 + np.random.randint(-5, 5)
            mot_converter.add_detection(
                frame_id=frame_id,
                track_id=track_id,
                x=x, y=y, w=50, h=40,
                confidence=0.9
            )
    
    mot_converter.save(str(output_dir / "sample_mot.txt"))
    
    print(f"示例标注已保存到: {output_dir}")


if __name__ == "__main__":
    # 测试标注工具
    create_sample_annotations("e:/杨东英/bee_project/datasets")
