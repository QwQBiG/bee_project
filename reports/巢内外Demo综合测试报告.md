# 智慧养蜂巢内外 Demo 综合测试报告

测试日期：2026-07-23  
项目目录：`D:\bee_project\bee_project`  
测试范围：公开短视频上的巢内红外与巢外蜂箱入口检测、跟踪、行为量化、可视化及数据分析

## 1. 综合结论

巢内 10 秒红外测试和巢外 2 秒可见光测试均已完成端到端运行，并成功生成标注视频、JSON、CSV 和统计图。

总体判定：**经过多次改进方使工程 Demo 链路通过；检测器具备明显的蜜蜂场景适应性，但现有个体跟踪、姿态和规则行为量化还不能作为比赛精度。**

| 项目 | 巢内红外 | 巢外入口 |
|---|---|---|
| 输入 | OIST 近红外，10 秒/100 帧 | VnBeeTracking，2 秒/120 帧 |
| 专用检测模型 | Apiarist YOLOv8s | Honeybee YOLOv8n |
| 检测 | 定性通过，无本地真值 | 定性通过，无本次逐帧真值评测 |
| 跟踪 | 不通过，ID 膨胀严重 | 功能通过，身份稳定性未验证且存在碎片化 |
| 行为量化 | 不可信，受跟踪误差影响 | 仅验证程序输出，不代表真实行为事件 |
| GPU | RTX 5060 Laptop GPU | RTX 5060 Laptop GPU |
| 吞吐率 | 0.122 FPS | 9.81 FPS |
| 实时性 | 约慢 82.09 倍 | 约为源帧率的 16.35%，慢 6.12 倍 |

## 2. 测试环境

- Python：3.13
- PyTorch：2.13.0+cu132
- Ultralytics：8.4.96
- CUDA：可用
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- 推理设备：`cuda:0`
- 数据分析：pandas、matplotlib

（曾误定位到 ComfyUI 的 PyTorch 开发版环境，该环境因 Triton 无法识别 Windows SDK 而不能导入完整 Ultralytics。随后根据原巢内报告恢复到上述 Python 3.13 环境，测试正常继续。该问题与 YOLOv8 权重无关。）

## 3. 测试数据

### 3.1 巢内红外

- 来源：[OIST Honeybee Tracking Datasets](https://www.oist.jp/research/research-units/bptu/honeybee_tracking_datasets)
- 原始文件：`data/external/oist/oist_M13_waggle_dance_ir.mp4`
- 测试片段：`data/external/oist/oist_M13_ir_test_10s.mp4`
- 片段：原视频约 20–30 秒
- 规格：884 × 1170，10 FPS，100 帧，10 秒

这是高密度、遮挡明显的巢内近红外场景。本地测试片段未配置对应逐帧检测框及身份真值，因此不计算 mAP、Precision、Recall、MOTA 或 IDF1。（就是说现有内容不完善，无法通过常规的评估值、方法做评估。）

### 3.2 巢外入口

- 来源：[VnBeeTracking Dataset](https://sigmlab.com/datasets/VnBeeTracking/)
- 原始文件：`data/external/vnbee/2022-04-08-12-30.mp4`
- 测试片段：`data/external/vnbee/vnbee_outside_test_2s.mp4`
- 规格：1080 × 720，60 FPS，120 帧，2 秒
- 冒烟片段：`data/external/vnbee/vnbee_outside_smoke_10f.mp4`

VnBeeTracking 是真实蜂箱入口视频数据，公开页面同时提供跟踪标注与数据集统计。本次按“简单 Demo 测试”要求只运行了很短片段，没有将官方逐帧真值转换进本项目评测格式，因此报告只给出功能、性能和定性视觉结论。

## 4. 模型选择

### 4.1 巢内模型

- 文件：`artifacts/models/honey_bee_detector_yolov8s.pt`
- 来源：[Apiarist Honey Bee Detector](https://huggingface.co/maryammeda/apiarist-honey-bee-detector)
- 架构：YOLOv8s
- 配置：`configs/demo_inside_test.yaml`

该模型不是红外专用模型，但在 OIST 近红外抽样帧上能检出大量真实蜜蜂，证明具备一定跨域迁移能力。低阈值、密集遮挡及可见光到红外的域差异仍可能造成漏检、重复框和误检。

### 4.2 巢外模型

- 文件：`artifacts/models/hive_entrance_bee_yolov8n.pt`
- 来源：[GuanranPei/yolov8_honeybee](https://github.com/GuanranPei/yolov8_honeybee)
- 架构：原生 YOLOv8n
- SHA-256：`429F13221BA6146676566C8250CA99CB069C0BE2AD5E16564BD9A9AE606525FA`
- 训练类别：bee、drone、pollenbee、queen
- 配置：`configs/demo_outside_test.yaml`

加载并运行该第三方 PT 权重。测试前使用单帧在 `imgsz=1280` 下检查：

| 置信度阈值 | 检测数 | 平均置信度 | 最大置信度 |
|---:|---:|---:|---:|
| 0.05 | 19 | 0.3631 | 0.8860 |
| 0.10 | 18 | 0.3779 | 0.8860 |
| 0.20 | 13 | 0.4716 | 0.8860 |

为保留小目标召回，本次 Demo 采用 0.10。视觉检查显示多数明显蜜蜂被框出，同时仍存在低置信度模糊目标，不能把全部框都视为真阳性。

## 5. 代码问题及修复

### 5.1 巢内

1. 修复 OpenCV 参数兼容问题。
2. 删除使用不可哈希边界框列表查询历史方向的错误。
3. 修复未匹配轨迹下标与真实轨迹 ID 混用。
4. 修复首次检测时未匹配检测和未匹配轨迹返回顺序颠倒。

涉及文件：`tracking/inside_tracker.py`

### 5.2 巢外

1. 修复 `create_outside_tracker` 将配置字典误当作 YAML 路径的问题。
2. 修复首次检测时未匹配列表返回顺序颠倒。
3. 将 `imgsz` 和推理设备正确传入 YOLOv8。
4. 删除随机生成的 124 维“外观特征”。该随机特征会让相邻帧中的同一只蜜蜂无法匹配。
5. 改用适合 60 FPS 短 Demo 的轻量匹配：IoU、速度预测中心和相对中心距离。
6. 新轨迹连续命中 3 帧后确认；只输出本帧实际更新的确认轨迹，避免把无预测能力的旧框继续画在画面上。
7. 增加检测框、置信度、确认轨迹和帧级统计可视化。
8. 增加 `detection_history`，使检测和跟踪数量可以用 pandas 逐帧分析。

涉及文件：

- `tracking/outside_tracker.py`
- `visualization/visualizer.py`
- `inference/processor.py`
- `configs/demo_outside_test.yaml`

当前实现应描述为“轻量运动匹配器”，不是带真实蜜蜂 ReID 网络和卡尔曼滤波的完整 DeepSORT。

## 6. 巢内结果

| 指标 | 结果 |
|---|---:|
| 总帧数 | 100 |
| 处理时间 | 820.925 秒 |
| 吞吐率 | 0.122 FPS |
| 源帧率 | 10 FPS |
| 相对实时减速 | 82.09 倍 |
| 每帧活动轨迹均值 | 320.58 |
| 中位数 | 351 |
| 最大值 | 375 |
| 最后一帧 | 368 |
| JSON `total_tracks` | 0（统计错误） |

模型能够在红外画面中检出大量蜜蜂，但当前跟踪器出现轨迹残留和 ID 膨胀。逐框传统线特征姿态估计在任意旋转、密集遮挡场景中也不可靠。行为统计共记录 29,658 条逐帧标签，其中 `working` 占 96.60%；这些是受错误轨迹影响的重复帧记录，不是独立行为事件。

巢内主要性能瓶颈包括 CPU 非局部均值降噪、逐框姿态线特征计算，以及高密度目标下效率较低的匹配过程。

## 7. 巢外结果

### 7.1 输出完整性

- 输出视频：2.000 秒、120 帧、1080 × 720、60 FPS
- 编码：MPEG-4 Part 2（`mp4v`）
- 输出文件大小：3,197,502 字节
- 10 帧冒烟测试：通过
- 完整 120 帧测试：通过
- 0.5 秒结果帧：13 个检测、12 条确认轨迹，检测框和 ID 均正常显示

### 7.2 pandas 统计

| 指标 | 结果 |
|---|---:|
| 帧级检测总记录 | 1,280 |
| 每帧检测均值 | 10.67 |
| 每帧检测中位数 | 11 |
| 每帧检测范围 | 0–19 |
| 每帧确认轨迹均值 | 9.76 |
| 每帧确认轨迹中位数 | 10 |
| 每帧确认轨迹范围 | 0–19 |
| 非空检测帧平均确认比例 | 89.79% |
| 创建的轨迹 ID | 58 |
| 处理时间 | 12.232 秒 |
| 推理吞吐率 | 9.81 FPS |
| 相对 60 FPS 实时系数 | 0.1635 |

“确认比例”只表示检测框中有多少在连续帧中形成确认轨迹，不是跟踪准确率。2 秒内创建 58 个 ID，而平均同时轨迹约 9.76，提示仍可能存在 ID 切换或短轨迹碎片；没有身份真值时不能计算 IDF1 或断言 58 只不同蜜蜂。

### 7.3 行为字段

巢外输出包含 111 条 `moving`、886 条 `wandering`、5 条 `foraging` 逐帧标签。这些标签由固定速度和位置方差规则产生，是程序功能输出，不是已验证的生物学事件。

`entry_events=0`、`exit_events=0` 不能解释为这段视频确实没有进出行为：当前主处理器尚未把入口穿越分析器接入这两个累计字段，而且 2 秒片段也不足以可靠验证进出巢事件。

## 8. 视觉检查

巢内：

- 红外画面中大量蜜蜂能被检测。
- 框、轨迹和方向箭头过密，轨迹残留明显。
- 朝向及行为结果不可靠。

巢外：

- YOLOv8 蜂箱入口权重能检出飞行、停落和模糊蜜蜂。
- 青色细框表示所有检测，彩色粗框和 `ID` 表示连续命中后确认的轨迹。
- 低至 0.10 的阈值保留了模糊小目标，也带来可疑低置信度框。
- 轻量运动匹配已能稳定产生短时轨迹，但遮挡、快速交叉和出入画仍可能导致 ID 切换。

## 9. 输出文件

巢内：

- `output/demo_test/inside_result.mp4`
- `output/demo_test/inside_stats.json`
- `output/demo_test/analysis/analysis_summary.json`
- `output/demo_test/analysis/frame_track_counts.csv`
- `output/demo_test/analysis/track_count_over_time.png`
- `output/demo_test/analysis/behavior_distribution.csv`
- `output/demo_test/analysis/behavior_distribution.png`

巢外：

- `output/demo_outside_test/outside_result.mp4`
- `output/demo_outside_test/outside_stats.json`
- `output/demo_outside_test/analysis/analysis_summary.json`
- `output/demo_outside_test/analysis/frame_counts.csv`
- `output/demo_outside_test/analysis/counts_over_time.png`
- `output/demo_outside_test/preview/result_v8_fixed_05s.jpg`
- `output/demo_outside_test/model_check/frame_evaluation.json`
- `tools/analyze_outside_results.py`

失败基线被单独保留在 `output/demo_outside_test/baseline_frame_model_failure`，用于说明通用/不匹配模型在该巢外场景中的失败，不会与最终结果混淆。

## 10. 限制与下一步

1. 把 OIST 与 VnBeeTracking 的官方标注转换成统一评测格式，至少报告检测 Precision/Recall/mAP 和跟踪 HOTA/IDF1。
2. 巢内优先采用红外微调检测器，并使用适合高密度遮挡的 ByteTrack、BoT-SORT 或专用蜜蜂跟踪方法。
3. 巢外用真实蜜蜂 ReID 特征与运动模型替换当前轻量贪心匹配，减少快速交叉时的 ID 切换。
4. 入口区域应按相机几何人工标定，再将穿越判定真正接入累计进出巢字段。
5. 姿态应改为带头尾关键点标注的模型，不应继续依赖固定头部区域或 Hough 线特征。
6. 在可靠轨迹建立前，不应把规则行为统计用于健康诊断、蜂群状态判断或比赛精度宣传。
7. 如果要求接近实时，需降低输入尺寸、批量/并行处理并分析 CPU 预处理和逐框计算瓶颈。

## 11. 最终判定

本次综合测试证明该项目能够在本人现有 GPU 环境中，分别加载巢内和巢外蜂类专用 YOLOv8 权重，处理真实公开红外及蜂箱入口视频，并完整生成可视化与数据分析结果。

目前最可靠的结论是“检测与工程处理链路已接通”。个体身份连续性、进出巢计数、头尾姿态和行为量化仍需要带真值的正式评测与算法替换，需要改进。
