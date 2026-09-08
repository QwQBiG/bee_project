"""
模型训练模块
支持YOLOv8模型的微调和迁移学习
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


class BeeDataset(Dataset):
    """蜜蜂数据集"""
    
    def __init__(self, data_root: str, split: str = 'train',
                 transform=None, img_size: Tuple[int, int] = (640, 640)):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # 加载图像路径和标签
        self.images = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """加载数据路径"""
        img_dir = self.data_root / 'images' / self.split
        label_dir = self.data_root / 'labels' / self.split
        
        if not img_dir.exists():
            return
        
        for img_path in img_dir.glob('*.jpg'):
            label_path = label_dir / (img_path.stem + '.txt')
            
            self.images.append(str(img_path))
            self.labels.append(str(label_path) if label_path.exists() else None)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple:
        """获取数据"""
        img_path = self.images[idx]
        label_path = self.labels[idx]
        
        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        img = cv2.resize(img, self.img_size)
        
        # 加载标签
        boxes = []
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        boxes.append([cls, x, y, w, h])
        
        return img, boxes, img_path
    
    @staticmethod
    def collate_fn(batch):
        """批处理整理函数"""
        imgs, labels, paths = zip(*batch)
        
        # 填充图像到相同大小
        max_h, max_w = 0, 0
        for img in imgs:
            h, w = img.shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        
        # 填充
        padded_imgs = []
        for img in imgs:
            h, w = img.shape[:2]
            padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            padded[:h, :w] = img
            padded_imgs.append(padded)
        
        return torch.from_numpy(np.stack(padded_imgs)), labels, paths


class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 训练参数
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0005)
        
        # 设备
        self.device = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 模型
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def prepare_data(self, data_yaml: str):
        """准备数据配置"""
        # 创建数据配置文件
        data_config = {
            'path': self.config.get('data_root', './datasets'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'bee',
                1: 'bee_entering',
                2: 'bee_exiting'
            }
        }
        
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f)
    
    def train(self, model_config: str, data_yaml: str, 
             checkpoint_dir: str = 'checkpoints'):
        """训练模型
        
        Args:
            model_config: 模型配置文件路径
            data_yaml: 数据配置文件路径
            checkpoint_dir: 检查点保存目录
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Error: ultralytics not installed. Please install with: pip install ultralytics")
            return
        
        # 创建模型
        self.model = YOLO(model_config)
        
        # 创建检查点目录
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 开始训练
        results = self.model.train(
            data=data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            lr0=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            project=str(checkpoint_dir),
            name='bee_detection',
            exist_ok=True,
            verbose=True
        )
        
        # 保存最佳模型
        best_model_path = checkpoint_dir / 'bee_detection' / 'weights' / 'best.pt'
        if best_model_path.exists():
            print(f"训练完成！最佳模型保存到: {best_model_path}")
        
        return results
    
    def validate(self, weights_path: str, data_yaml: str,
                 task: str = "detect") -> Dict:
        """验证模型 (检测或姿态).

        ``task`` must be ``detect`` or ``pose``; for pose the returned
        metrics additionally include ``pose_mAP50`` / ``pose_mAP50-95``.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return {}

        model = YOLO(weights_path, task=task)
        results = model.val(data=data_yaml, device=self.device, verbose=False)

        out = {
            'mAP50': float(getattr(results.box, "map50", 0.0) or 0.0),
            'mAP50-95': float(getattr(results.box, "map", 0.0) or 0.0),
            'precision': float(getattr(results.box, "mp", 0.0) or 0.0),
            'recall': float(getattr(results.box, "mr", 0.0) or 0.0),
        }
        if task == "pose" and hasattr(results, "keypoints"):
            out['pose_mAP50'] = float(getattr(results.keypoints, "map50", 0.0) or 0.0)
            out['pose_mAP50-95'] = float(getattr(results.keypoints, "map", 0.0) or 0.0)
        return out

    def train_pose(self,
                   data_yaml: str,
                   base_model: str = "yolov8n-pose.pt",
                   checkpoint_dir: str = "checkpoints",
                   run_name: str = "bee_pose",
                   imgsz: int = 640):
        """Fine-tune an Ultralytics YOLO pose model (kpt_shape in data.yaml).

        The caller must feed a YOLO-pose ``data.yaml`` whose ``kpt_shape``
        matches the task (for this project it is ``[3, 3]``: head / thorax
        / abdomen_tip).  The method picks imgsz=1280 for hive-entrance
        (outside) and imgsz=640 for inside IR by convention; pass a value
        to override.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            print(f"Error: ultralytics not installed ({exc}).")
            return None

        model = YOLO(base_model)
        if getattr(model, "task", None) != "pose":
            raise ValueError(
                f"{base_model} is not a YOLO pose backbone (task="
                f"{getattr(model, 'task', None)})")

        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        results = model.train(
            data=data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            lr0=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            imgsz=imgsz,
            project=str(ckpt),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )
        best = ckpt / run_name / "weights" / "best.pt"
        if best.exists():
            print(f"[train_pose] best weights saved to {best}")
        return results

    def train_detection_task(self,
                             data_yaml: str,
                             scene: str = "outside",
                             checkpoint_dir: str = "checkpoints",
                             base_model: str = "yolov8m.pt") -> object:
        """Train a scene-aware detector with the right imgsz default.

        ``scene``:
          - ``outside`` → hive entrance / visible light, imgsz=1280
          - ``inside``  → hive interior / infrared,       imgsz=640
        """
        scene = scene.lower()
        if scene not in {"outside", "inside"}:
            raise ValueError(f"scene must be 'outside' or 'inside', got {scene!r}")
        imgsz = 1280 if scene == "outside" else 640
        run_name = f"bee_det_{scene}"
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            print(f"Error: ultralytics not installed ({exc}).")
            return None

        model = YOLO(base_model)
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        results = model.train(
            data=data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            lr0=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            imgsz=imgsz,
            project=str(ckpt),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )
        best = ckpt / run_name / "weights" / "best.pt"
        if best.exists():
            print(f"[train_detection_task] scene={scene} saved {best}")
        return results

    def export_model(self, weights_path: str, 
                    export_format: str = 'onnx',
                    output_dir: str = 'exports',
                    imgsz: int = 640,
                    opset: int = 12,
                    simplify: bool = True,
                    dynamic: bool = False):
        """导出模型（带正确 imgsz/opset/simplify 参数，默认 match 蜂巢场景）。

        The ``imgsz`` must match the training size; for hive-entrance
        detectors pass 1280, inside-IR pass 640.  Leaving it at the 640
        default used to produce mismatched ONNX input shapes — this was
        the key reason the old export helper could not be reused directly.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return
        
        model = YOLO(weights_path)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_path = model.export(
            format=export_format,
            project=str(output_dir),
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
        )
        
        print(f"模型已导出到: {exported_path}")
        return exported_path


class TransferLearning:
    """迁移学习辅助类"""
    
    @staticmethod
    def load_pretrained_weights(model, pretrained_path: str,
                               freeze_layers: int = 0):
        """加载预训练权重
        
        Args:
            model: 模型
            pretrained_path: 预训练权重路径
            freeze_layers: 冻结前N层
        """
        if not os.path.exists(pretrained_path):
            print(f"预训练权重不存在: {pretrained_path}")
            return
        
        try:
            from ultralytics import YOLO
            yolo = YOLO(pretrained_path)
            
            # 复制权重
            model.model.load_state_dict(yolo.model.state_dict())
            
            # 冻结层
            if freeze_layers > 0:
                for i, param in enumerate(model.model.parameters()):
                    if i < freeze_layers:
                        param.requires_grad = False
            
            print(f"已加载预训练权重: {pretrained_path}")
            if freeze_layers > 0:
                print(f"已冻结前{freeze_layers}层")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    @staticmethod
    def fine_tune_config(base_lr: float = 0.001,
                        last_lr: float = 0.0001) -> Dict:
        """生成微调配置
        
        使用分层学习率：骨干网络使用低学习率，检测头使用高学习率
        """
        return {
            'base_lr': base_lr,
            'head_lr': last_lr,
            'optimizer': 'AdamW',
            'weight_decay': 0.0001,
            'scheduler': 'cosine',
            'warmup_epochs': 3
        }


def create_training_config(task: str = 'outside') -> Dict:
    """创建训练配置
    
    Args:
        task: 任务类型 ('outside' 或 'inside')
    """
    
    if task == 'outside':
        # 巢外可见光检测配置
        config = {
            'model_type': 'yolov8m',
            'data_root': './datasets/outside',
            'img_size': 640,
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.1
            },
            'device': 'cuda:0'
        }
    else:
        # 巢内红外检测配置
        config = {
            'model_type': 'yolov8m',
            'data_root': './datasets/inside',
            'img_size': 640,
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'augmentation': {
                'hsv_h': 0.01,
                'hsv_s': 0.5,
                'hsv_v': 0.3,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.05
            },
            'preprocessing': {
                'clahe': True,
                'denoise': True
            },
            'device': 'cuda:0'
        }
    
    return config


if __name__ == "__main__":
    # 演示训练配置
    config = create_training_config('outside')
    print("巢外训练配置:")
    print(yaml.dump(config, default_flow_style=False))
    
    config = create_training_config('inside')
    print("\n巢内训练配置:")
    print(yaml.dump(config, default_flow_style=False))
