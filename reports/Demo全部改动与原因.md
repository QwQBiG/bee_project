# Bee Project Demo 全部改动与原因

整理日期：2026-07-23  
项目目录：`D:\bee_project\bee_project`  
整理范围：本轮巢内红外与巢外蜂箱入口 Demo 测试期间，对项目源码、配置、模型、测试数据、分析工具及报告所做的改动。

## 1. 总览

本轮工作的目标不是重写整个项目，而是使用真实公开数据和蜜蜂专用 YOLOv8 权重，把原有 Demo 的巢内、巢外处理链路实际运行起来，定位阻断问题，并产出可复现的短视频测试、统计和报告。

改动分为以下六类：

1. 修复巢内跟踪器中会导致程序报错或轨迹状态错误的问题。
2. 修复巢外跟踪器的配置、推理参数和首次匹配问题。
3. 删除巢外跟踪器伪造的随机 ReID 特征，换成可解释的轻量运动匹配。
4. 增加检测框、轨迹和帧级统计可视化。
5. 增加独立测试配置、评估脚本、数据分析脚本和综合报告。
6. 引入公开测试视频和蜜蜂专用 YOLOv8 权重。

## 2. 源码修改

### 2.1 `tracking/inside_tracker.py`

#### 改动 1：修复 OpenCV 降噪参数

- 修改位置：`InfraredImageEnhancer.denoise`
- 原问题：调用 `cv2.fastNlMeansDenoisingColored` 时使用了当前 OpenCV 不接受的参数名 `hForColorComponents`。
- 修改内容：改为 OpenCV 当前接口接受的 `hColor=10`。
- 修改原因：原参数会直接抛出异常，巢内视频无法进入后续检测和跟踪流程。
- 影响：仅修复 API 兼容性，不改变设定的降噪强度。

#### 改动 2：修复首次匹配返回值顺序

- 修改位置：`InsideHiveTracker._match`
- 原问题：没有历史轨迹时，函数把“未匹配检测”和“未匹配轨迹”的返回位置弄反。
- 修改内容：首次收到检测时返回：
  - 已匹配：空列表
  - 未匹配检测：全部检测下标
  - 未匹配轨迹：空列表
- 修改原因：原逻辑会把检测下标当作轨迹 ID 使用，导致错误更新或异常。
- 影响：首帧检测现在可以正确初始化为新轨迹。

#### 改动 3：修复未匹配轨迹下标与轨迹 ID 混用

- 修改位置：`InsideHiveTracker._match`
- 原问题：匹配过程内部使用列表下标，但后续代码需要的是真实 `track_id`。轨迹删除后，列表下标与轨迹 ID 不再必然相同。
- 修改内容：根据 `track_ids` 显式把未匹配的列表位置转换为真实轨迹 ID。
- 修改原因：防止错误轨迹被增加 `time_since_update`，或访问不存在的字典键。
- 影响：轨迹生命周期更新对象正确。

#### 改动 4：修复方向估计的历史框使用方式

- 修改位置：`InsideHiveTracker._update_track`
- 原问题：
  - 旧逻辑曾尝试用边界框列表作为字典键，列表不可哈希。
  - 方向估计没有稳定使用“同一轨迹的上一帧框”和“当前帧框”。
- 修改内容：
  - 更新轨迹前先复制该轨迹的旧边界框。
  - 在匹配成功后，用当前检测框与该轨迹旧框计算方向。
  - 把结果写入当前 `pose.direction`。
- 修改原因：避免不可哈希异常，并确保方向来自同一条已匹配轨迹的帧间位移。
- 影响：方向计算链路可运行；但传统图像姿态方法本身仍未达到可靠头尾识别精度。

### 2.2 `tracking/outside_tracker.py`

#### 改动 1：修复配置字典被当作文件路径

- 修改位置：`create_outside_tracker`
- 原问题：调用方已经传入解析后的配置字典，旧工厂函数却按 YAML 文件路径处理。
- 修改内容：工厂函数直接接受 `Dict`；只有参数为 `None` 时才使用默认配置。
- 修改原因：否则巢外处理器初始化时会因类型错误而失败。
- 影响：`inference/processor.py` 传入的 `config['tracker']` 可以正确生效。

#### 改动 2：修复首次匹配返回顺序

- 修改位置：`DeepSORTTracker._match_detections`
- 原问题：没有历史轨迹时，未匹配检测和未匹配轨迹的返回位置颠倒。
- 修改内容：返回 `([], 全部检测下标, [])`。
- 修改原因：确保首帧检测用于创建轨迹，而不是被当作失配轨迹处理。
- 影响：巢外轨迹能够正常开始建立。

#### 改动 3：增加 `imgsz` 推理参数

- 修改位置：
  - `OutsideHiveBeeDetector.__init__`
  - `OutsideHiveBeeDetector.detect`
  - `OutsideHiveTracker.__init__`
- 原问题：巢外蜜蜂在 1080×720 画面中较小，固定 640 输入会损失小目标信息；配置也无法控制推理尺寸。
- 修改内容：
  - 检测器新增 `imgsz` 参数。
  - YOLO 推理时显式传入 `imgsz=self.imgsz`。
  - 从 `tracker.imgsz` 配置读取该值。
- 修改原因：通过 1280 输入提高蜂箱入口小蜜蜂的可检测性。
- 影响：召回能力改善，但显存占用和单帧耗时增加。

#### 改动 4：显式传入推理设备

- 修改位置：`OutsideHiveBeeDetector.detect`
- 原问题：模型虽然在初始化阶段移动到 GPU，但单次推理没有显式携带配置设备。
- 修改内容：YOLO 推理调用增加 `device=self.device`。
- 修改原因：保证实际推理使用 `cuda:0`，避免环境或 Ultralytics 默认值变化导致静默回到 CPU。
- 影响：巢外测试在 RTX 5060 Laptop GPU 上运行。

#### 改动 5：增加跟踪前置信度过滤

- 修改位置：`DeepSORTTracker.update`
- 原问题：低于跟踪置信度的检测仍会不断创建短暂轨迹。
- 修改内容：进入匹配前按 `min_confidence` 过滤检测。
- 修改原因：降低低置信度噪声引起的短轨迹和 ID 膨胀。
- 影响：当前 Demo 检测阈值和跟踪阈值均设置为 0.10，以优先保留小目标。

#### 改动 6：删除随机生成的伪 ReID 特征

- 修改位置：`DeepSORTTracker._extract_feature`
- 原问题：旧实现生成 128 维向量，其中后 124 维是根据框位置重新生成的随机数。蜜蜂稍微移动，随机向量就会改变，相邻帧中的同一目标反而被判为不同个体。
- 修改内容：
  - 删除所有随机数生成逻辑。
  - 只生成确定性的归一化几何特征：中心 x、中心 y、框宽、框高。
  - 归一化尺度根据当前帧宽高计算，不再固定假设 1920×1080。
- 修改原因：随机特征不是外观特征，也不能冒充 ReID；它是此前“有检测但始终没有确认轨迹”的主要原因。
- 影响：匹配结果可复现，连续轨迹能够形成。
- 说明：这仍不是真正的蜜蜂外观 ReID。

#### 改动 7：将匹配规则改为轻量运动匹配

- 修改位置：`DeepSORTTracker._match_detections`
- 原问题：原成本由 IoU 和随机特征距离各占 50%，随机特征使成本失真；只依赖 IoU 又难以跟踪快速飞行或模糊蜜蜂。
- 修改内容：
  - 使用上一帧速度预测本帧中心。
  - 计算上一框与检测框的 IoU 距离。
  - 计算预测中心与检测中心的距离，并按两框最大对角线归一化。
  - 超过约两个框对角线的候选直接拒绝。
  - 综合成本：`0.65 × IoU距离 + 0.35 × 中心距离成本`。
  - 贪心接受阈值设置为 0.82。
- 修改原因：VnBee 视频为 60 FPS，相邻帧运动通常连续，几何和短时速度比随机特征更适合简单 Demo。
- 影响：10 帧冒烟测试中可以形成确认轨迹；完整 2 秒测试平均得到 9.76 条确认轨迹/帧。
- 限制：快速交叉、严重遮挡和重新入画仍可能导致 ID 切换。

#### 改动 8：调整新轨迹初始命中计数

- 修改位置：`DeepSORTTracker._initiate_track`
- 原问题：新轨迹初始 `hits=0`，实际需要四帧才达到“命中三帧”的确认条件。
- 修改内容：新轨迹设置为 `age=1`、`hits=1`。
- 修改原因：创建轨迹的当前检测本身就是第一次有效命中。
- 影响：连续三帧检测后即可确认轨迹。

#### 改动 9：只输出本帧实际更新的确认轨迹

- 修改位置：`DeepSORTTracker._get_active_tracks`
- 原问题：旧实现会把最长 `max_age` 帧内没有新检测匹配的旧框继续输出，但当前轻量跟踪器没有卡尔曼预测，旧框停留在原位置会误导可视化和统计。
- 修改内容：只返回 `state == confirmed` 且 `time_since_update == 0` 的轨迹。
- 修改原因：输出画面应代表当前帧真实匹配结果。
- 影响：失配轨迹仍在内部保留等待重新匹配，但不会以静止旧框显示。

#### 命名说明

项目类名和配置值仍保留 `DeepSORTTracker` / `tracker_type: deepsort`，以减少对原项目接口的破坏；但修改后的实现没有真正的 ReID 网络和完整卡尔曼滤波，因此应称为“轻量 IoU/中心距离运动匹配器”，目前不是完整 DeepSORT。

### 2.3 `visualization/visualizer.py`

#### 改动 1：修复可视化配置层级

- 修改位置：`VideoAnnotator.__init__`
- 原问题：`OutsideHiveProcessor` 已经传入 `visualization` 子字典，但 `VideoAnnotator` 又尝试读取其中的第二层 `visualization`，导致线宽、标签和置信度等配置被忽略。
- 修改内容：优先兼容嵌套配置，否则直接把当前配置传给 `TrackVisualizer`。
- 修改原因：让 `configs/demo_outside_test.yaml` 中的显示参数真正生效。
- 影响：轨迹线宽、框线宽、字体和标签设置可控。

#### 改动 2：增加检测框显示

- 修改位置：
  - `VideoAnnotator.__init__`
  - `VideoAnnotator.annotate_frame`
- 原问题：处理器虽然把 `detections` 传给可视化器，但原函数完全忽略检测结果，只画确认轨迹。跟踪未确认或失败时，用户会误以为模型没有检测。
- 修改内容：
  - 新增 `show_detections` 配置。
  - 先用青色细框绘制所有检测。
  - 显示 `det + 置信度`。
  - 再叠加彩色轨迹框、ID、置信度和轨迹线。
- 修改原因：区分“YOLO 已检测”和“跟踪器已确认”两个阶段，便于定位问题。
- 影响：巢外结果视频可同时核对检测与跟踪。

### 2.4 `inference/processor.py`

#### 改动 1：增加检测数量历史

- 修改位置：`OutsideHiveProcessor.stats` 和 `process_video`
- 原问题：JSON 只有 `track_history`，无法分析检测器每帧输出，也无法判断问题出在检测还是跟踪。
- 修改内容：
  - 增加 `detection_history`。
  - 每帧记录 `len(detections)`。
- 修改原因：为 pandas 帧级对比检测数和确认轨迹数提供原始数据。
- 影响：最终 `outside_stats.json` 可以独立分析检测和跟踪变化。

#### 改动 2：把帧级统计传入视频标注器

- 修改位置：`OutsideHiveProcessor.process_frame`
- 原问题：可视化器支持 `stats`，处理器却没有传入。
- 修改内容：画面左上角显示：
  - 当前帧号
  - 当前检测数
  - 当前确认轨迹数
- 修改原因：方便直接从结果视频判断流程状态。
- 影响：结果视频具备基本 Demo 展示和排错能力。

## 3. 新增配置文件

### 3.1 `configs/demo_inside_test.yaml`

新增原因：不修改项目通用配置，给本次 OIST 红外测试建立可复现的独立参数。

关键设置：

- 模型：`artifacts/models/honey_bee_detector_yolov8s.pt`
- 检测置信度：0.15
- NMS IoU：0.45
- 设备：`cuda:0`
- 红外增强：开启
- `max_age`：30
- `min_hits`：2
- 行为最短轨迹长度：5
- 输出帧率：10 FPS

### 3.2 `configs/demo_outside_test.yaml`

新增原因：为 VnBee 蜂箱入口小目标建立独立测试配置，并避免覆盖项目默认配置。

关键设置：

- 模型：`artifacts/models/hive_entrance_bee_yolov8n.pt`
- 检测置信度：0.10
- 跟踪置信度：0.10
- NMS IoU：0.45
- 输入尺寸：1280
- 设备：`cuda:0`
- 跟踪器接口：`deepsort`，实际为本文所述轻量运动匹配
- `max_age`：10
- 显示检测框和确认轨迹
- 输出帧率：60 FPS

## 4. 新增测试与分析工具

### 4.1 `tools/evaluate_detector_frame.py`

- 用途：在单张巢外图像上加载指定 YOLOv8 权重，以多个置信度阈值执行检测。
- 输出：
  - 各阈值检测数量
  - 平均/最大置信度
  - 类别统计
  - 对应标注图片
- 新增原因：在完整视频运行前快速判断候选权重是否真正适合蜂箱入口画面，并选择置信度阈值。

### 4.2 `tools/analyze_demo_results.py`

- 用途：使用 pandas/matplotlib 分析巢内 `inside_stats.json`。
- 输出：
  - `analysis_summary.json`
  - `frame_track_counts.csv`
  - `behavior_distribution.csv`
  - `track_count_over_time.png`
  - `behavior_distribution.png`
- 新增原因：把原始 JSON 转成可检查的统计指标、表格和图，而不是仅凭结果视频判断 Demo。

### 4.3 `tools/analyze_outside_results.py`

- 用途：使用 pandas/matplotlib 分析巢外 `outside_stats.json`。
- 输出：
  - `analysis_summary.json`
  - `frame_counts.csv`
  - `counts_over_time.png`
- 计算内容：
  - 每帧检测数和确认轨迹数
  - 非空帧轨迹确认比例
  - 处理吞吐率和实时系数
  - 行为标签记录数
- 新增原因：量化本次巢外短测试的功能和性能结果，并明确区分检测数量与轨迹数量。

## 5. 新增模型

### 5.1 巢内模型

- 文件：`artifacts/models/honey_bee_detector_yolov8s.pt`
- 大小：22,521,386 字节
- 来源：Apiarist Honey Bee Detector
- 类型：蜜蜂专用 YOLOv8s
- 新增原因：项目原来的通用 COCO `yolov8m.pt` 不是蜜蜂专用权重，无法作为合理的巢内测试模型。
- 限制：该权重主要基于可见光蜜蜂图像，不是 OIST 近红外专用模型。

### 5.2 巢外模型

- 文件：`artifacts/models/hive_entrance_bee_yolov8n.pt`
- 大小：6,250,979 字节
- SHA-256：`429F13221BA6146676566C8250CA99CB069C0BE2AD5E16564BD9A9AE606525FA`
- 来源：`GuanranPei/yolov8_honeybee`
- 类型：原生 YOLOv8n
- 类别：bee、drone、pollenbee、queen
- 新增原因：用最合适的 YOLOv8 蜂箱入口相关权重；该权重能在 VnBee 抽样帧中实际检出蜜蜂。
- 安全说明：用户已明确授权加载和运行该第三方 PT 文件。

## 6. 新增测试数据

### 6.1 OIST 巢内红外

- `data/external/oist/oist_M13_waggle_dance_ir.mp4`
  - 公开原始近红外视频。
- `data/external/oist/oist_M13_ir_test_10s.mp4`
  - 从原视频截取的 10 秒、100 帧测试片段。
- `data/external/oist/preview/*.jpg`
  - 用于测试前视觉检查的抽帧预览。

新增原因：项目原目录没有适合本轮真实红外端到端测试的短视频；使用公开数据避免随机生成输入。

### 6.2 VnBeeTracking 巢外入口

- `data/external/vnbee/2022-04-08-12-30.mp4`
  - VnBeeTracking 公开原始视频。
- `data/external/vnbee/vnbee_outside_test_2s.mp4`
  - 完整 Demo 使用的 2 秒、120 帧片段。
- `data/external/vnbee/vnbee_outside_smoke_10f.mp4`
  - 修改跟踪器后先运行的 10 帧冒烟测试。
- `data/external/vnbee/preview/outside_frame_05s.jpg`
  - 模型选择与视觉检查用抽帧。

新增原因：用户要求增加一个小型巢外测试；该数据是蜂箱入口真实场景，并提供公开跟踪数据背景。

## 7. 新增测试输出

### 7.1 巢内

目录：`output/demo_test`

主要产物：

- `inside_result.mp4`
- `inside_stats.json`
- `analysis/analysis_summary.json`
- `analysis/frame_track_counts.csv`
- `analysis/behavior_distribution.csv`
- `analysis/track_count_over_time.png`
- `analysis/behavior_distribution.png`
- `preview/result_01s.jpg`
- `preview/result_05s.jpg`
- `preview/result_09s.jpg`

### 7.2 巢外

目录：`output/demo_outside_test`

主要产物：

- `outside_result.mp4`
- `outside_stats.json`
- `analysis/analysis_summary.json`
- `analysis/frame_counts.csv`
- `analysis/counts_over_time.png`
- `preview/result_v8_fixed_05s.jpg`
- `model_check/frame_evaluation.json`
- `model_check/detections_conf_0.05.jpg`
- `model_check/detections_conf_0.10.jpg`
- `model_check/detections_conf_0.20.jpg`

### 7.3 巢外冒烟测试

目录：`output/demo_outside_smoke`

用途：在运行完整 120 帧前，先验证 10 帧内是否能形成检测框和确认轨迹。

### 7.4 失败基线留档

目录：`output/demo_outside_test/baseline_frame_model_failure`

- 记录了不匹配的巢内帧模型用于巢外入口时检测为零的结果。
- 保留原因：作为模型选择过程和失败原因的证据。
- 不属于最终巢外结果。

## 8. 新增和更新的报告

### 8.1 `reports/巢内外Demo综合测试报告.md`

- 把巢内红外与巢外入口测试整合为一份报告。
- 对比两种场景的模型、数据、吞吐率和主要问题。
- 明确区分“工程链路通过”和“算法精度达到科研/比赛要求”。

### 8.2 `reports/Demo全部改动与原因.md`

- 即当前文件。
- 用于集中说明本轮对 Demo 的所有修改、添加、删除及原因。

## 9. 删除和明确未采用的内容

### 9.1 已删除 YOLOv5 权重

- 删除文件：`artifacts/models/hive_entrance_bee_yolov5.pt`
- 删除原因：明确要求 YOLOv8 相关权重。
- 当前状态：已检查，文件不存在。

### 9.2 没有创建新的项目虚拟环境

- 最终使用：
  - 我的本地py环境，即 `C:\Users\111\AppData\Local\Programs\Python\Python313\python.exe`
  - PyTorch 2.13.0+cu132
  - Ultralytics 8.4.96
  - RTX 5060 Laptop GPU

### 9.3 没有修改的核心内容

- 没有修改 `main.py` 的命令行接口。
- 没有训练或微调任何模型。
- 没有把 VnBee 官方真值转换为正式评测格式。
- 没有把入口穿越分析器接入 `entry_events` / `exit_events`。
- 没有把巢外轻量匹配升级为真正的 ReID DeepSORT。
- 没有修正巢内 `total_tracks=0` 的统计问题。
- 没有把现有规则行为标签改造成经过生物学验证的行为识别模型。

这些未修改项是报告中保留限制说明的原因。

## 10. 自动生成但不属于人工源码改动的文件

运行 Python 后产生了以下缓存：

- `tracking/__pycache__/*.pyc`
- `inference/__pycache__/*.pyc`
- `visualization/__pycache__/*.pyc`
- `tools/__pycache__/*.pyc`

这些文件只是解释器字节码缓存，不包含独立功能，不需要纳入代码审查或交付说明；删除后运行 Python 会自动重新生成。

## 11. 修改后的实际效果

巢内测试：

- 10 秒、100 帧近红外视频完整运行。
- 蜜蜂专用 YOLOv8s 能在红外画面中检出大量蜜蜂。
- 追踪、姿态和行为结果暴露出明显问题，已在报告中如实标记为不可靠。

巢外测试：

- 2 秒、120 帧蜂箱入口视频完整运行。
- 平均检测 10.67 个/帧。
- 平均确认轨迹 9.76 条/帧。
- 非空检测帧平均确认比例 89.79%。
- 处理吞吐率 9.81 FPS。
- 结果视频能区分检测框和确认轨迹。
- 删除随机伪 ReID 后，原先“有检测但零确认轨迹”的问题消失。

## 12. 使用与复现

巢内：

```powershell
$env:YOLO_CONFIG_DIR='D:\bee_project\bee_project\.ultralytics'
& 'C:\Users\111\AppData\Local\Programs\Python\Python313\python.exe' main.py `
  --mode inside `
  --video data\external\oist\oist_M13_ir_test_10s.mp4 `
  --output output\demo_test `
  --config configs\demo_inside_test.yaml `
  --device cuda:0
```

巢外：

```powershell
$env:YOLO_CONFIG_DIR='D:\bee_project\bee_project\.ultralytics'
& 'C:\Users\111\AppData\Local\Programs\Python\Python313\python.exe' main.py `
  --mode outside `
  --video data\external\vnbee\vnbee_outside_test_2s.mp4 `
  --output output\demo_outside_test `
  --config configs\demo_outside_test.yaml `
  --device cuda:0
```

巢外数据分析：

```powershell
& 'C:\Users\111\AppData\Local\Programs\Python\Python313\python.exe' `
  tools\analyze_outside_results.py `
  --stats output\demo_outside_test\outside_stats.json `
  --output output\demo_outside_test\analysis `
  --fps 60
```

## 13. 最终说明

本轮修改使原 Demo 能够在本人现有 GPU 环境中，使用真实公开巢内红外和巢外入口视频、蜜蜂专用 YOLOv8 权重，完整生成检测、跟踪、行为字段、可视化及数据分析结果。

这些修改主要解决“程序能否正确运行、检测与跟踪阶段能否被区分、结果能否被量化分析”的工程问题。它们没有证明现有系统已达到比赛指标或科学行为量化精度！！！后续若继续开发，最重要的是接入带身份真值的正式评测、红外微调模型、真实蜜蜂 ReID/运动模型以及经过标注验证的头尾姿态和行为识别。
