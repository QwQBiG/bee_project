"""模型训练与迁移学习模块。"""

from .trainer import (
    BeeDataset,
    YOLOTrainer,
    TransferLearning,
    create_training_config,
)

__all__ = [
    "BeeDataset",
    "YOLOTrainer",
    "TransferLearning",
    "create_training_config",
]

__version__ = "1.0.0"
