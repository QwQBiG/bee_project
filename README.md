# 智慧养蜂蜜蜂识别与行为量化研究项目配置

## 项目概述
本项目针对智慧养蜂场景，实现巢内外蜜蜂个体识别与行为智能量化分析。

## 目录结构
```
bee_project/
├── configs/         # 配置文件
├── data/            # 数据集目录
├── models/          # 模型定义（五角星）
├── utils/           # 工具函数
├── annotation/      # 数据标注模块
├── tracking/        # 跟踪算法模块（五角星）
├── behavior/        # 行为分析模块 （五角星）
├── visualization/   # 可视化模块 （五角星）
├── inference/       # 推理模块（五角星）
├── datasets/        # 数据集处理
└── main.py          # 主程序入口
```

## 技术方案

### 1. 巢外蜜蜂检测与跟踪（可见光视频）
- **检测模型**: 蜜蜂专用 YOLOv8
- **跟踪算法**: Ultralytics 官方 BoT-SORT / ByteTrack
- **关键技术**: 
  - 高密度场景下的目标检测
  - 长时间轨迹连续性维护
  - 身份切换抑制

### 2. 巢内蜜蜂识别与跟踪（红外视频）
- **图像增强**: 自适应直方图均衡化 + 去噪
- **检测模型**: YOLOv8 + 红外图像专用头
- **跟踪算法**: Ultralytics 官方 BoT-SORT / ByteTrack + 几何方向估计
- **关键技术**:
  - 低对比度图像增强
  - 蜜蜂头部/腹部结构识别
  - 个体朝向判别

### 3. 行为量化指标
- **个体行为**: 进出巢频率、停留时间、运动速度、方向变化
- **群体行为**: 蜂群活跃度、采集高峰期、异常聚集检测

## 依赖环境
```
torch >= 2.0.0
opencv-python >= 4.8.0
ultralytics >= 8.0.0
numpy >= 1.24.0
pandas >= 2.0.0
pillow >= 10.0.0
pyyaml >= 6.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

## 数据标注格式
采用COCO格式和MOT格式结合：
- 检测框: [x, y, w, h, confidence, class_id]
- 跟踪ID: [frame_id, track_id, x, y, w, h, confidence, class_id, visible]
- 行为标注: [track_id, behavior_type, start_frame, end_frame]
