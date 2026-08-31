# SY-202601 · 比赛标准 JSON 输出字段与评分线对照总表

> **权威依据**：腾讯文档《数据集说明及模型评测机制 §3.3 数据模型交付与自动评测规范》
> **实现文件**：`inference/algorithm_cli.py`（自动评测入口程序，兼容 onnxruntime GPU/CPU）
> **默认配置**：`configs/algorithm_config.json`（指定 onnx 权重路径、imgsz、conf 阈值、场景自动识别正则）

---

## 一、标准调用方式（CLI 接口）

| 项 | 规范 |
|----|------|
| **可执行文件** | `algorithm.exe`（PyInstaller `--onedir` 模式编译，严禁单文件 `-F`） |
| **命令行** | `algorithm.exe --image_path "<图片绝对或相对路径>"` |
| **路径兼容** | 支持空格长路径（如 `"C:\Program Files\test pic.jpg"`）、支持中文字符路径 |
| **硬件依赖** | 仅依赖宿主机 NVIDIA Driver；cuda/cudnn/onnxruntime_cuda provider DLL 由队伍自带进 onedir |
| **超时强杀** | 单张图片推理 ≤ 10 s，否则评测主进程强制 kill，该图计 0 分 |
| **日志纯净** | onnxruntime `session_options.log_severity_level = 3`（屏蔽 Info/Warning，保证 stdout 首行只有 JSON） |

---

## 二、标准 JSON 输出（两种任务二选一，由程序在首行或且仅在首行打印一行单行 JSON）

### 2.1 目标检测任务（对应评测 §1.1 / §1.2 · 巢外/巢内 mAP）

```json
{
  "image_id": "A-5-1-3922c3762a68_frame_00000076",
  "code": 1,
  "detections": [
    {
      "bbox": [201.97, 661.71, 29.09, 23.62],
      "label": "bee",
      "class_id": 0,
      "confidence": 0.6842
    }
  ],
  "processing_time_ms": 14023,
  "message": "ok"
}
```

**字段硬约束**（评测机按这些字段判分，缺字段或改名直接 0 分）：

| 字段 | 类型 | 必填 | 说明 / 像素坐标约定 |
|------|------|------|--------------------|
| `image_id` | string | ✅ | 图片 id，用于评测机对齐答案。本工程取**图片文件名去掉扩展名**（如 `A-5-1-xxxx_frame_00000076.jpg` → `A-5-1-xxxx_frame_00000076`）。 |
| `code` | int | ✅ | **状态码**：<br>• `1` = 正常识别成功，`Exit Code = 0`（评测机捕获）<br>• `0` = 失败（路径不存在 / 文件损坏 / 非法格式），`Exit Code = 0` 并附 `message` 字段说明原因 |
| `detections` | list[object] | ✅（可空数组） | 每张图的检测框。`[]` 代表图中没检测到蜜蜂。 |
| `detections[].bbox` | number[4] | ✅ | `[x, y, width, height]`，**左上角原点**，单位像素。**不是 [x1,y1,x2,y2]**，评测时按 COCO VOC 规则计算 IoU。 |
| `detections[].label` | string | ✅ | 比赛只有一类，填 `"bee"`。 |
| `detections[].class_id` | int | ✅ | 类别编号，填 `0`。 |
| `detections[].confidence` | float | ✅ | 置信度（0–1）。评测机按 confidence 阈值做 TP/FP/FN 排序，用于 mAP@0.5:0.95。**不要全填 1.0**，会把 PR 曲线拉成一条竖线，mAP 会低。 |
| `processing_time_ms` | int | ✅ | 单张推理耗时（毫秒）。评测系统拿这个算 FPS / Latency，用于 §1.5 模型复杂度 6 分排名。 |
| `message` | string | ❌ | code=0 时说明原因：`file_not_found` / `invalid_image` / `model_init_failed`。code=1 时可为 `"ok"` 或省略。 |
| `error` | string | ❌ | 旧字段，保留兼容；同 `message`，二选一就行。 |

### 2.2 多目标跟踪任务（对应评测 §1.3 / §1.4 · MOTA + IDF1 + MT/ML）

```json
{
  "frame_id": 12,
  "sequence_id": "seq_A-5-1",
  "code": 1,
  "tracks": [
    {
      "track_id": 7,
      "bbox": [120, 330, 38, 28],
      "label": "bee",
      "class_id": 0,
      "confidence": 0.9341
    }
  ],
  "processing_time_ms": 31
}
```

**tracking 新增字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | int | 视频帧序号（1-based），评测用对齐 GT 帧 |
| `sequence_id` | string | 所属视频序列 id，用于跨帧 id 连续性检查（ID Switch 的基础） |
| `tracks` | list[object] | 当前帧上的跟踪结果；替代 `detections`。跟踪评测时若返回 `tracks` 则优先读取，否则会降级用 `detections` 仅判检测 mAP（跟踪分会丢） |
| `tracks[].track_id` | int | 跨帧一致的跟踪身份 id。同一只蜜蜂在整个 sequence_id 里 **track_id 必须不变**。**评测机按这个算 IDSW / IDF1 / MT / ML**。一旦切帧被切 id，IDF1 直接暴跌。 |

---

## 三、自动 Exit Code（评测机侧捕获用，比赛 §3.3 第 5 条）

| 情形 | 进程 Exit Code（shell） | JSON 内 `code` 字段 | JSON 是否仍输出单行 |
|------|------------------------|--------------------|-------------------|
| 正常识别成功 | `0`（Windows 进程成功惯例） | `1`（比赛 §3.3 规范：成功=code=1） | ✅ 是，第一行就是 JSON |
| 图片路径不存在（`--image_path nope.jpg`） | `0` | `0` | ✅ 是，附 `"message":"file_not_found"` |
| 图片损坏 / 非法格式（无法 imdecode） | `0` | `0` | ✅ 是，附 `"message":"invalid_image"` |
| onnx 权重缺失 / 初始化失败 | `0` | `0` | ✅ 是，附 `"message":"model_init_failed"` |
| 运行时异常（空指针等） | `0` | `0` | ✅ 是，被 `try/except` 兜底捕获，不会弹 Windows 错误框 |
| 推理 > 10s 被评测机强杀 | —（评测机 SIGKILL，进程消失） | — | 该张图片计 0 分（本工程内加了 `conf` 降敏 + imgsz 场景化控制，巢内 640 典型 <500ms，巢外 1280 在 GPU 下 <2s，<10s 充足） |

---

## 四、评分映射（自动评测系统打分对照表，比赛 §3.2（1）满分 50 分）

### §1.1 巢外可见光检测精度 — 12 分
评测指标：**mAP@0.5:0.95**（10 个 IoU 阈值 0.50–0.95 步长 0.05 取平均）

| mAP50-95 区间 | 得分 | 本工程当前值（Stage1 伪标签 bootstrap 微调后，证据：`runs/outside_ft/bee_v1/results.csv`） |
|--------------|------|------|
| ≥ 0.80 | 10 – 12 | — |
| [0.70, 0.80) | 7 – 9 | — |
| [0.60, 0.70) | 4 – 6 | **0.661 → 4-6 分**（目标：人工精标 50 张 bbox → 冲 ≥0.78） |
| [0.50, 0.60) | 1 – 3 | — |
| < 0.50 | 0 | — |

### §1.2 巢内红外检测精度 — 12 分

| mAP50-95 区间 | 得分 | 本工程当前值（Stage1，证据：`runs/inside_ft/bee_v1/results.csv`） |
|--------------|------|------|
| ≥ 0.70 | 10 – 12 | — |
| [0.60, 0.70) | 7 – 9 | **0.617 → 7-9 分**（目标：人工精标 50 张 bbox → 冲 ≥0.70 入满分档） |
| [0.50, 0.60) | 4 – 6 | — |
| [0.40, 0.50) | 1 – 3 | — |
| < 0.40 | 0 | — |

### §1.3 巢外多目标跟踪 — 10 分
评测指标：**MOTA + IDF1**（跟踪部分），另算 MT/ML/IDSW 细节

| 条件 | 得分 |
|------|------|
| MOTA ≥ 0.60 **且** IDF1 ≥ 0.65 | 8 – 10 |
| 0.50 ≤ MOTA < 0.60（IDF1 不达标但 MOTA 不错） | 5 – 7 |
| 0.40 ≤ MOTA < 0.50 | 3 – 4 |
| 0.30 ≤ MOTA < 0.40 | 1 – 2 |
| MOTA < 0.30 | 0 |

### §1.4 巢内红外跟踪性能 — 10 分

| 条件 | 得分 |
|------|------|
| MOTA ≥ 0.65 **且** IDF1 ≥ 0.70 | 8 – 10 |
| 0.55 ≤ MOTA < 0.65 | 5 – 7 |
| 0.45 ≤ MOTA < 0.55 | 3 – 4 |
| 0.35 ≤ MOTA < 0.45 | 1 – 2 |
| MOTA < 0.35 | 0 |

### §1.5 模型复杂度 — 6 分
评测：**权重数越少 + 参数量越小 + inference `processing_time_ms` 越快 → 排名越前**（队伍排名百分位制）

| 排名百分位 | 得分 |
|-----------|------|
| 前 5 % | 6 |
| 5 – 10 % | 4 |
| 10 – 30 % | 2 |
| 其他 | 0 – 1 |

> **本工程的轻量化策略**：
> - 巢外使用 yolov8n（6.2M 参数）而非 yolov8s — 参数量比 baseline 小 3x
> - 巢内在不损失 mAP 前提下继续导出 yolov8s 为 onnx opset 12 + dynamic=False（固定输入尺寸，TensorRT 部署更快）
> - imgsz 场景化：巢外 1280（精度优先）、巢内 640（速度优先，且红外图目标天然分辨率低）
> - 推理端 onnxruntime CUDA provider（需自带 cudnn/cublas/cudart DLL 进 onedir）

**自动评测 50 分总分汇总**：12 + 12 + 10 + 10 + 6 = **50**

---

## 五、行为量化 20 分（数据产物输出规范，比赛 §3.2（2），不通过 CLI JSON 输出）

由 `tools/` 下离线脚本产出，不进入自动评测系统的 CLI 流水线，但须在提交时作为附件交付：

| 评测子项 | 产出物格式要求 | 本工程对应工具入口 |
|---------|--------------|------------------|
| §2.1 个体行为指标（8 分） | CSV：`frame_id,track_id,x,y,w,h,vx,vy,angular_speed,wiggle_amp,dwell_time_s` 每行一只蜜蜂一帧；至少 ≥3 类运动学指标（瞬时速度 / 角速度 / 摆尾振幅 / 停留时长） | `tools/behavior_quantify_individual.py`（计划 Stage 3 后接入，需 pose 关键点 + tracks 两输入） |
| §2.2 群体行为指标（7 分） | ① 巢口通量 CSV：`window_1s, enter_count, exit_count, net_flow`；② 群体密度热力图 PNG（按 1s/帧滑窗统计每平方米蜜蜂数）；③ 交互网络 GEXF/CSV（蜜蜂之间 50px 内视为接触）；④ 时序变化曲线 PNG | `tools/behavior_quantify_colony.py`（计划中） |
| §2.3 生物学含义（5 分） | 文档段落：指标 → 养蜂生产决策（分蜂预警 / 农药危害 / 天气影响 / 蜜源节律） | —（文档类） |

---

## 六、工程交付合规（比赛 §3.2（3）20 分 · 本工程 Checklist）

| 子项 | 要求 | 当前状态 | 证据位置 |
|------|------|---------|---------|
| §3.1 程序运行 & 绝对解耦（6 分） | onedir PyInstaller，零依赖；中文字符路径不崩；无弹窗/死锁 | ✅ 代码层满足；打包脚本待补（`tools/package_onedir_algorithm.ps1` 计划中） | `inference/algorithm_cli.py`（路径用 `pathlib`/UTF-8，try/except 全包） |
| §3.2 Exit Code 规范（5 分） | 成功 0；路径缺失 0；文件损坏 0（JSON code=0/1 承载语义）；无阻断弹窗 | ✅ | `inference/algorithm_cli.py` `_emit()` + `main()` 各 except 分支；`tests/test_algorithm_cli.py` |
| §3.3 Stdout 单行 JSON（5 分） | 首行有且仅有一行 JSON；onnxruntime log_severity_level=3；≤10s | ✅ | `inference/algorithm_cli.py` `run_onnx()` 内 sess_opts.log_severity_level = 3；`test_algorithm_cli.py` 断言 `json.loads(out.splitlines()[0])` |
| §3.4 标注成果与文档（4 分） | COCO/YOLO 标准格式；坐标轴边界；多人多轮校验机制文档 | ⏳ 等你标完生成；README.txt 里已写明标注规则；`index.csv` 可用于 2 轮交叉复核 | `datasets/label_tasks_final/easy_label_*/*/README.txt`；`tools/pick_easy_label_images.py` 2 轮标注差异比对计划中 |

---

## 七、评测工具调用链（从 CLI 输出到分数）

```
评测机每帧调用：
  algorithm.exe --image_path "<abs_path>.jpg"
        │
        ▼
  stdout 第一行 JSON（本工程 algorithm_cli.py 产出）
        │
        ├─ detections[].bbox / detections[].confidence
        │        │
        │        ▼
        │   tools/evaluate_vnbee_tracking.py
        │     └─ 计算 mAP@0.5:0.95、mAP50、mAP75（§1.1 / 1.2）
        │
        └─ tracks[].track_id / bbox（若跟踪任务）
                 │
                 ▼
            evaluate_vnbee_tracking.py
              ├─ MOTA / IDF1 / IDSW
              └─ MT(≥80%) / PT(20-80%) / ML(<20%)（§1.3 / 1.4）
        │
        ▼
  processing_time_ms → 排名百分位（§1.5 6 分）
```

**工具测试入口**：`tests/test_evaluation_metrics.py` — 覆盖 mAP 插值 AP、MT/ML 三档、空边界、完美匹配、噪声等情况；`tests/test_algorithm_cli.py` — 覆盖 CLI JSON、Exit Code、不存在/损坏分支。

---

*文档版本：v0.1（Stage 1 伪标签微调后）· 下一次修订：你完成 Stage 2 bbox 精标并回传 XML 后，更新 §4 表格当前值列，补 §3.1 打包脚本 + onedir 清单*
