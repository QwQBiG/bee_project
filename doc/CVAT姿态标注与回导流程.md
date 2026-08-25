# CVAT Online 蜜蜂姿态标注与回导流程

## 1. 这项工作要完成什么

本流程使用 [CVAT Online](https://app.cvat.ai/) 标注巢内红外和巢外入口图片，不需要在电脑上部署标注服务。

人工需要为每只有效蜜蜂完成四项内容：

1. 校正整只蜜蜂的矩形框；
2. 标出头部中心 `head`；
3. 标出胸部中心 `thorax`；
4. 标出腹部末端 `abdomen_tip`。

现有模型生成的框只是候选结果，不是正确答案。人工仍需删除误检、补充漏检并校正位置。

## 2. 第一次使用：先选择工作空间

打开 CVAT Online 并登录。页面右上角显示当前账号。

- 只有一人试用：可以保留 `Personal workspace`。
- 两人共同标注：先点击右上角账号，选择 `Organization` → `+ Create`，创建组织并邀请另一名成员。

需要协作时，创建项目之前必须切换到对应组织。个人空间中的项目不会自动共享给同学。免费方案的团队人数和任务数量有限，首批只安排少量试标。

## 3. 创建姿态项目

1. 进入顶部 `Projects` 页面。
2. 点击右侧蓝色 `+`，选择 `Create new project`。
3. 项目名称填写 `bee_pose_official`。
4. 点击 `Add label`，标签名称填写 `bee`。
5. 标签形状选择 `Skeleton`。
6. 按顺序创建三个点：`head`、`thorax`、`abdomen_tip`。
7. 添加两条连线：`head-thorax`、`thorax-abdomen_tip`。
8. 点击 `Submit` 保存项目。

三个点的顺序属于训练格式的一部分。项目创建后不得交换名称或顺序。

## 4. 导入首批试标任务

首批只导入两个小任务，不上传原始完整长视频，也不上传封存测试视频：

- 巢外：`datasets/official_work/diverse_60/online/outside_pilot5.zip`
- 巢内红外：`datasets/official_work/diverse_60/online/inside_pilot5_enhanced.zip`

每个任务包含5张图片、候选框和回溯映射。候选框必须人工复核。

导入步骤：

1. 打开刚创建的 `bee_pose_official` 项目。
2. 点击项目右上角 `Actions`。
3. 选择 `Import dataset`。
4. 格式选择 `Ultralytics YOLO Pose 1.0`。
5. 选择一个 ZIP 试标包并提交。
6. 进入顶部 `Requests`，等待状态变为 `Finished`。
7. 返回项目，打开自动创建的任务和 Job。
8. 第一个任务确认正常后，再按相同步骤导入另一个任务。

若导入失败，不要重复创建项目或修改标签；记录页面错误信息并交由项目维护人员检查任务包。

## 5. 每张图片怎么标

打开 Job 后，按以下顺序处理当前图片：

1. 检查候选框是否真的对应蜜蜂，误检框直接删除。
2. 调整保留的框，使其包住完整蜜蜂，不只框头部或腹部。
3. 找到漏掉的蜜蜂，使用 `Skeleton` 工具补充一个 `bee`。
4. 将 `head` 放在头部中心，将 `thorax` 放在胸部中心，将 `abdomen_tip` 放在腹部末端。
5. 清晰可见的点保持可见；被遮挡但位置可以合理确定时标记为遮挡；完全无法判断时标记为不可见。
6. 按 `Ctrl+S` 保存，再切换到下一张图片。

标注时遵守以下规则：

- 当前任务范围内可辨认的蜜蜂应尽量全部标注；
- 多只蜜蜂重叠时，每只分别标注；
- 不把阴影、木板、线缆或蜂巢纹理当作蜜蜂；
- 无法判断头尾时不根据运动方向猜测；
- 严重重叠且无法分离的目标记录为困难样本，交给复核人员处理。

关键点可见性对应关系：

- `2`：清晰可见；
- `1`：被遮挡，但位置可以合理确定；
- `0`：无法标注或位于画面外。

## 6. 第一批如何分工

第一批先完成巢外5帧和巢内5帧，确认规则后再扩大数量。

- 标注者甲：巢外任务；
- 标注者乙：巢内红外任务；
- 每种场景至少抽取1帧交叉复核；
- 发现标签定义不一致时先停止扩充，统一规则后再继续。

首轮只做独立图片中的整蜂框和三个关键点。连续轨迹、固定个体 ID 和进出巢事件另建短视频任务，不与首轮姿态标注混在一起。

## 7. 完成、复核与导出

1. 标注人员确认当前 Job 已保存，将状态设置为 `Completed`。
2. 复核人员检查漏标、误标、边界框和三个关键点。
3. 在任务的 `Actions` 中选择 `Export task dataset`。
4. 格式选择 `Ultralytics YOLO Pose 1.0`。
5. 下载标注 ZIP，并按任务名称和日期保存，不要解压后手工修改标签文件。

Online 账号、密码和官方原始数据均不提交到 Git 仓库。导出的标注包应先本地备份，再交由项目工具回导和校验。

参考：

- [CVAT 创建任务](https://docs.cvat.ai/docs/manual/basics/create-annotation-task/)
- [CVAT 导入数据集和标注](https://docs.cvat.ai/docs/manual/advanced/import-datasets/)
- [Ultralytics YOLO Pose 格式](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
- [CVAT 骨架标注](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/)

## 8. 从 CVAT 回导

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

## 9. 金标准校验

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

## 10. 训练前检查与正式训练

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
