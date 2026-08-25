# CVAT 姿态标注与回导流程

## 1. 适用范围

本流程用于将巢内红外和巢外入口抽帧转换为可训练、可评测的整蜂检测与头胸腹姿态金标准。

现有权重只提供待校正的整蜂候选框。候选框不是人工真值，头、胸和腹尖均需人工标注。

## 2. 标签定义

姿态任务只使用一个骨架类别：

- 类别：`bee`
- 关键点顺序：`head`、`thorax`、`abdomen_tip`
- 骨架连线：`head → thorax → abdomen_tip`

首轮姿态任务不区分工蜂、雄蜂、蜂王或携粉蜂。只有画面证据和独立标注规范充分时，才建立对应的专用分类数据集。

关键点可见性：

- `2`：清晰可见；
- `1`：被遮挡，但位置可以合理确定；
- `0`：无法标注或位于画面外。

## 3. 任务包

任务包由 `tools/export_yolo_pose.py` 生成，格式为 **Ultralytics YOLO Pose 1.0**。

```powershell
python tools/export_yolo_pose.py <统一标注目录> <抽帧目录> <任务目录> `
  --include-non-manual --split train --scene inside_ir `
  --collapse-to-bee --archive <任务包.zip>
```

任务包包含：

- `images/train/`：待标注帧；
- `labels/train/`：候选框和空关键点；
- `data.yaml`：单类骨架数据集配置；
- `annotation_map.json`：图像与原视频、原帧号的回溯映射；
- `train.txt`：图像列表。

`annotation_map.json` 需在本地保留，不应由标注人员修改。

## 4. CVAT 项目设置

1. 在 CVAT 中创建用于蜜蜂姿态标注的项目。
2. 创建名称为 `bee` 的 Skeleton 标签。
3. 按顺序添加 `head`、`thorax`、`abdomen_tip` 三个点。
4. 添加 `head-thorax` 和 `thorax-abdomen_tip` 两条连线。
5. 将任务包按 `Ultralytics YOLO Pose 1.0` 格式导入。

CVAT 要求任务中的骨架标签与导入数据兼容，项目创建后不应随意修改关键点顺序。

参考：

- [Ultralytics YOLO Pose 格式](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
- [CVAT 骨架标注](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/)

## 5. 人工校正规则

每一帧都应执行以下检查：

1. 删除落在背景、线缆、木板或蜂巢纹理上的错误候选。
2. 为所有可辨认且满足任务范围的蜜蜂补充漏标框。
3. 调整整蜂框，使其紧贴完整身体，不只框头部或腹部。
4. 标注头部中心、胸部中心和腹部末端。
5. 无法判断头腹方向时，不根据运动方向猜测。
6. 严重重叠且无法分离的个体不强行拆分，应记录为困难样本。
7. 同一连续片段需要跟踪评测时，使用 Track 模式保持同一个 `track_id`。
8. 完全遮挡后无法确认身份时结束旧轨迹，不猜测重识别结果。

需要在 CVAT 与统一 JSON 之间交换既有轨迹 ID 时，任务导出命令增加
`--include-track-ids`。该选项只用于标注交换，不用于生成正式训练标签。

检测金标准要求任务范围内的目标尽量完整标注；只修改已有候选而不补漏标，会导致召回率评测失真。

## 6. 第一批质量控制

第一批不直接扩大到全部视频，先完成试标：

- 巢外5帧；
- 巢内5帧；
- 每种场景至少1帧由第二人独立复核；
- 统一处理遮挡、画面边缘、极小目标和密集重叠规则；
- 发现标签定义歧义时先修订规范，再继续剩余帧。

试标通过后完成各20帧，并固定其中一部分作为测试集。相邻帧和同源视频片段不得跨训练、验证、测试集合。

## 7. 从 CVAT 回导

在 CVAT 中以 `Ultralytics YOLO Pose 1.0` 导出标注 ZIP，然后执行：

```powershell
python tools/import_yolo_pose.py <CVAT导出.zip> `
  <原任务目录/annotation_map.json> <统一JSON输出目录> --reviewed
```

只有在任务已完成人工标注和复核后才能使用 `--reviewed`。未添加该参数时，
回导实例保持非金标准来源，防止未修改的预标注任务误通过 `--gold` 校验。

导入器会恢复：

- 原始 `video_id` 和帧号；
- 像素坐标整蜂框；
- 头、胸、腹尖关键点及可见性；
- 可选 `track_id`；
- 原视频路径、帧率、场景和哈希元数据。

## 8. 金标准校验

普通人工检测标注：

```powershell
python tools/validate_annotations.py <统一JSON目录> --gold
```

姿态与朝向金标准：

```powershell
python tools/validate_annotations.py <统一JSON目录> --pose-gold
```

需要连续跟踪 ID 的姿态金标准：

```powershell
python tools/validate_annotations.py <统一JSON目录> --pose-gold --require-track-ids
```

姿态校验要求：

- 全部实例为人工来源；
- 每个实例具有头、胸、腹尖三个定义；
- 头和腹尖可用于确定身体朝向；
- 启用跟踪校验时每个实例必须包含 `track_id`。

只有通过相应校验的文件才能进入正式评测集。

## 9. 训练前检查与正式训练

回导并按原始视频划分 train、val、test 后，重新导出正式数据集：

```powershell
python tools/export_yolo_pose.py <统一JSON目录> <抽帧目录> <正式数据集目录>
```

正式训练导出不得添加 `--include-track-ids`。CVAT 可以在标签末尾保存轨迹 ID，
但 Ultralytics 姿态训练要求每行严格为类别、边界框和三组关键点字段。
导出器默认生成训练兼容标签，并同时写入 `dataset_meta.json`。

`training_ready` 只有在以下条件全部满足时才为 `true`：

- 实例全部为人工来源；
- 头、胸、腹尖标注满足姿态金标准；
- 不包含 prediction 或 interpolated 实例；
- train 和 val 均包含来自已分配源视频的帧。

先执行 dry-run：

```powershell
python main.py --mode train --task pose `
  --model <兼容的pose初始权重.pt> --data <正式数据集/data.yaml> `
  --epochs 100 --batch_size -1 --imgsz 640 --device auto `
  --output runs/train --train_name bee_pose --dry_run
```

dry-run 通过后移除 `--dry_run` 启动训练。设备选择顺序为 CUDA、MPS、CPU；
使用 CPU 时程序会明确提示。训练结果目录会保存 best、last 权重和
`training_summary.json`，正式结果应以固定测试集复评，而不是使用训练集指标。
