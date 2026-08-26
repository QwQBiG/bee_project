# CVAT Online 蜜蜂姿态标注与回导流程

## 1. 这项工作要完成什么

本流程使用 [CVAT Online](https://app.cvat.ai/) 标注巢内红外和巢外入口图片，不需要在电脑上部署标注服务。

人工只为每只有效蜜蜂标出三个关键点：

1. 头部中心 `head`；
2. 胸部中心 `thorax`；
3. 腹部末端 `abdomen_tip`。

CVAT 的骨架会显示连线和导出矩形框，但人工不需要另外画框。回导时会依据三个关键点重新生成适合姿态训练的蜜蜂框。

## 2. 第一次使用：先选择工作空间

打开 CVAT Online 并登录。页面右上角显示当前账号。

- 只有一人试用：可以保留 `Personal workspace`。
- 两人共同标注：先点击右上角账号，选择 `Organization` → `+ Create`，创建组织并邀请另一名成员。

需要协作时，创建项目之前必须切换到对应组织。个人空间中的项目不会自动共享给同学。使用 Online 免费方案时，优先只上传小批量图片任务和导出标注；不要依赖免费方案导出原图。

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

首批试标任务为巢外和巢内红外各 6 张局部图，不上传原始完整长视频：

- 巢外：`datasets/official_work/diverse_60/online/outside_manual_pose_pilot.zip`
- 巢内红外：`datasets/official_work/diverse_60/online/inside_manual_pose_pilot.zip`

任务包只包含图片；蜜蜂和关键点均由人工创建。

导入步骤：

1. 打开刚创建的 `bee_pose_official` 项目。
2. 点击任务区域右侧蓝色 `+`，创建 Task。
3. 填写任务名称，例如 `outside_manual_pose_pilot` 或 `inside_manual_pose_pilot`。
4. 上传对应的图片 ZIP，提交任务。
5. 任务创建完成后，点击 `Open` 进入 Job。
6. 第一个任务确认正常后，再按相同步骤创建另一个任务。

若导入失败，不要重复创建项目或修改标签；记录页面错误信息并交由项目维护人员检查任务包。

## 5. 每张图片怎么标

打开 Job 后，按以下顺序处理当前图片：

1. 在左侧选择 `Draw new skeleton`，选择 `Shape`，标签选择 `bee`。
2. 对每只符合条件的蜜蜂创建一个骨架。
3. 将 `head` 放在头部中心，`thorax` 放在胸部中心，`abdomen_tip` 放在腹部最末端。
4. 三点连线应沿同一只蜜蜂的身体长轴排列，不能跨到相邻蜜蜂。
5. 点击 `Save` 保存，再切换到下一张图片。

标注时遵守以下规则：

- 当前任务范围内身体完整、头尾可辨认的蜜蜂应尽量全部标注；
- 多只蜜蜂重叠时，只有能够明确分离且能判断头尾的个体才分别标注；
- 不把阴影、木板、线缆或蜂巢纹理当作蜜蜂；
- 无法判断头尾时不根据运动方向猜测；
- 严重模糊、严重遮挡、身体不完整或头部、腹部末端位于画面外的个体不标；
- 三个关键点都必须落在图片内部，不能为了表达画外部位把点拖到图片外。

首轮试标只保留三个点均可明确判断的完整个体，不使用猜测性关键点。

## 6. 第一批如何分工

第一批已完成巢外 6 帧和巢内 6 帧，确认规则后再扩大数量。

- 标注者甲、乙、丙按图片或视频片段分配任务；
- 每种场景至少抽取 10% 图片交叉复核；
- 发现标签定义不一致时先停止扩充，统一规则后再继续。

首轮只做独立图片中的三个关键点。连续轨迹、固定个体 ID 和进出巢事件另建短视频任务，不与首轮姿态标注混在一起。

## 7. 当前试验数据状态

截至首轮试标完成：

- 巢外 6 张图，87 个有效三点姿态实例；
- 巢内红外 6 张图，80 个有效三点姿态实例；
- 合计 12 张图、167 个实例。

这些数据已用于验证 CVAT 标注、回导和项目姿态处理链路。它们尚不足以支持正式准确率结论；下一阶段需从不同视频、时段、密度和光照条件中继续抽取清晰帧，并保留独立视频作为验证与测试数据。

## 8. 完成、复核与导出

1. 标注人员确认当前 Job 已保存，将状态设置为 `Completed`。
2. 复核人员检查漏标、误标、个体完整性和三个关键点。
3. 在任务的 `Actions` 中选择 `Export task dataset`。
4. 格式选择 `Ultralytics YOLO Pose 1.0`。
5. 免费方案导出时不要勾选原图；下载标注 ZIP，并按任务名称和日期保存，不要解压后手工修改标签文件。

Online 账号、密码和官方原始数据均不提交到 Git 仓库。导出的标注包应先本地备份，再交由项目工具回导和校验。

参考：

- [CVAT 创建任务](https://docs.cvat.ai/docs/manual/basics/create-annotation-task/)
- [CVAT 导入数据集和标注](https://docs.cvat.ai/docs/manual/advanced/import-datasets/)
- [Ultralytics YOLO Pose 格式](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)
- [CVAT 骨架标注](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/)

## 9. 从 CVAT 回导

在 CVAT 中以 `Ultralytics YOLO Pose 1.0` 导出标注 ZIP。原图已保留在本地任务包中，因此只需将标注 ZIP 交给数据整理人员。数据整理时执行：

```powershell
python tools/prepare_pose_pilot_dataset.py `
  --annotations <CVAT导出.zip> --images <原任务图片目录> `
  --output <本地姿态数据集目录> --padding 0.25
```

如果导出中包含画外关键点，按标注规则应先在 CVAT 中删除对应的截断个体；历史试标数据可在整理时加 `--skip-outside-keypoints`，仅排除这些无法训练的实例，原始导出 ZIP 保持不变。

整理工具会：

- 将标注 ZIP 与本地原图合并；
- 保留人工标出的头、胸、腹尖关键点；
- 重新生成适合 YOLO Pose 训练的整蜂框；
- 输出逐图数量和数据集元信息。

## 10. 金标准校验

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

## 11. 训练前检查与正式训练

当样本量达到要求并完成复核后，按原始视频划分 train、val、test，再生成正式数据集。首批 12 张、167 个实例仅用于检查流程，不能作为正式精度结论。

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

先执行训练前检查：

```powershell
python tools/train_yolo.py <兼容的pose初始权重.pt> <正式数据集/data.yaml> `
  --task pose --epochs 100 --batch -1 --imgsz 640 --device auto --dry-run
```

dry-run 通过后移除 `--dry-run` 启动训练。设备选择顺序为 CUDA、MPS、CPU；
使用 CPU 时程序会明确提示。训练结果目录会保存 best、last 权重和
`training_summary.json`，正式结果应以固定测试集复评，而不是使用训练集指标。
