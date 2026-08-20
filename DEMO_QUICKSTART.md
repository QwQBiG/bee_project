# 蜜蜂识别 Demo 部署与运行说明

> 文档版本：2026-08-19　代码基线：`076b3b626fc069abd89d03819aee584e3334fc32`

## 一、最简运行流程

以下流程面向 Windows 10/11、NVIDIA RTX 30/40 系显卡。首次部署需要联网下载依赖和两个模型权重。

### 1. 获取代码并创建独立环境

推荐使用 Python 3.11；项目也已在 Python 3.13.7 环境完成运行验证。

```powershell
git clone https://github.com/QwQBiG/bee_project.git
Set-Location -LiteralPath ".\bee_project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. 安装 RTX 30/40 系 GPU 环境

先更新 [NVIDIA 显卡驱动](https://www.nvidia.com/Download/index.aspx)，然后安装 PyTorch CUDA 12.6 版本和项目依赖：

```powershell
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

检查 GPU：

```powershell
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

最后必须显示 `可用: True` 和实际显卡名称。

### 3. 放入权重和短视频

创建模型目录，并按下列路径放置文件：

```text
artifacts/models/honey_bee_detector_yolov8s.pt
artifacts/models/hive_entrance_bee_yolov8n.pt
data/external/oist/oist_M13_ir_test_10s.mp4
data/external/vnbee/vnbee_outside_test_2s.mp4
```

- [下载巢内 YOLOv8s 权重](https://huggingface.co/maryammeda/apiarist-honey-bee-detector/blob/main/honey_bee_detector.pt)，下载后改为上述巢内文件名。
- [下载巢外 YOLOv8n 权重](https://github.com/GuanranPei/yolov8_honeybee/blob/main/runs/detect/yolov8n-honeybee/weights/best.pt)，下载后改为上述巢外文件名。
- 红外视频来源：[OIST Honeybee Tracking Datasets](https://www.oist.jp/research/research-units/bptu/honeybee_tracking_datasets)。
- 巢外视频来源：[VnBeeTracking](https://sigmlab.com/datasets/VnBeeTracking/)。

测试视频可以替换为其他短 MP4，只需同步修改运行命令中的 `--video` 路径。

### 4. 自检并运行

```powershell
python test_project.py

python main.py --mode inside --video ".\data\external\oist\oist_M13_ir_test_10s.mp4" --output ".\output\demo_inside" --config ".\configs\demo_inside_test.yaml" --device cuda:0

python main.py --mode outside --video ".\data\external\vnbee\vnbee_outside_test_2s.mp4" --output ".\output\demo_outside" --config ".\configs\demo_outside_test.yaml" --device cuda:0
```

成功后应得到：

```text
output/demo_inside/inside_result.mp4
output/demo_inside/inside_stats.json
output/demo_outside/outside_result.mp4
output/demo_outside/outside_stats.json
```

到此即完成最低 Demo 的部署和运行。

---

## 二、环境选择详细说明

### 1. RTX 30/40 系应选择什么版本

| 显卡系列 | 架构 | 官方起始 CUDA 支持 | 本项目建议 |
|---|---|---:|---|
| RTX 30 系 | Ampere | CUDA 11.0 | PyTorch 2.12.0 + CUDA 12.6 |
| RTX 40 系 | Ada | CUDA 11.8 | PyTorch 2.12.0 + CUDA 12.6 |

CUDA 12.6 同时支持这两代显卡，便于团队统一环境。Windows 驱动建议更新至 560.76 或更高版本；直接安装当前最新的 NVIDIA 驱动更省事。架构兼容关系可查阅 [NVIDIA CUDA 架构矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)，PyTorch 版本命令来自 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)。

通过官方 `pip` 命令安装的 PyTorch 已携带所需 CUDA 运行库。普通运行本项目通常只需 NVIDIA 驱动，不需要另外安装完整 CUDA Toolkit 或 cuDNN；只有编译 CUDA 扩展时才需要 Toolkit。

### 2. Python 版本

- 团队统一环境推荐 Python 3.11，第三方库兼容范围更稳。
- 当前 Demo 已在 Python 3.13.7、PyTorch 2.13.0+cu132、Ultralytics 8.4.96 环境完成验证。
- 上述已验证环境使用 RTX 5060 Laptop GPU；30/40 系推荐安装命令采用兼容范围更广的 CUDA 12.6，而不是照搬 CUDA 13.2。

### 3. 为什么先安装 PyTorch，再安装 requirements

`requirements.txt` 只规定了最低 PyTorch 版本，没有指定 CUDA 构建。直接安装全部依赖可能得到 CPU 版 PyTorch。先用 PyTorch 官方 CUDA 12.6 地址安装，可明确获得 GPU 版；之后安装其余依赖不会替换已满足要求的版本。

### 4. CPU 备用方式

没有 NVIDIA GPU 时可以使用 CPU。安装 CPU 版 PyTorch，并把运行命令中的 `cuda:0` 改为 `cpu`：

```powershell
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
```

CPU 模式功能不变，但视频处理速度会明显降低。

## 三、模型与数据文件说明

GitHub 仓库包含核心代码、Demo 配置、分析工具和测试报告，但不包含以下运行资产：

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| `honey_bee_detector_yolov8s.pt` | 22,521,386 | `894B7A41C9AD05FAB487158C66F49EA521FF1543B55948BFC0487BCE1AD7C2B9` |
| `hive_entrance_bee_yolov8n.pt` | 6,250,979 | `429F13221BA6146676566C8250CA99CB069C0BE2AD5E16564BD9A9AE606525FA` |

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath ".\artifacts\models\honey_bee_detector_yolov8s.pt", ".\artifacts\models\hive_entrance_bee_yolov8n.pt"
```

`.pt` 文件会被 PyTorch 反序列化，只应从可信来源下载。测试视频和数据集还应遵守原数据来源的授权条件，不应直接提交到公开仓库。

## 四、常见问题

- **`python` 不是命令**：重新安装 Python 并勾选加入 `PATH`，或使用 `python.exe` 完整路径。
- **虚拟环境无法激活**：执行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，仅影响当前 PowerShell 窗口。
- **`nvidia-smi` 不存在**：NVIDIA 驱动未正确安装。
- **`torch.cuda.is_available()` 为 `False`**：常见原因是装成 CPU 版、环境未激活或驱动过旧。激活 `.venv` 后重新执行 CUDA 12.6 安装命令。
- **找不到模型或视频**：检查路径、扩展名和文件名；GitHub 中的 JPG 是预览图，不是可运行的 MP4。
- **网络下载较慢**：可使用本机已有的 HTTP/HTTPS 代理，或通过浏览器下载后复制到指定目录；不要从不明网盘获取 `.pt`。
- **没有真实检测框**：确认控制台加载的是两个专用权重，而不是随机检测框兜底逻辑。

## 五、Demo 的客观说明口径

该 Demo 已完成巢内红外视频和巢外入口视频的最小端到端流程，包括视频读取、蜜蜂专用 YOLOv8 检测、轻量级轨迹关联、行为统计以及结果可视化输出。两类场景均能生成处理后视频和结构化统计文件，可用于展示算法流程和继续集成。

“Demo 跑通”表示工程链路能够执行，不代表已经达到最终比赛精度。正式参赛前仍需在统一标注数据上报告检测指标、跟踪指标、行为量化指标、运行速度和失败案例，并继续处理遮挡、密集蜂群、红外域差异及身份切换问题。

模型权重可后续放入 GitHub Release、Git LFS 或团队共享盘；生成结果和大视频不应直接提交到普通 Git 历史。参见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 六、演示


### 1. 进入项目并确认环境

```powershell
Set-Location -LiteralPath "D:\bee_project\bee_project"
$py = "C:\Users\111\AppData\Local\Programs\Python\Python313\python.exe"
$env:YOLO_CONFIG_DIR = "D:\bee_project\bee_project\.ultralytics"
& $py test_project.py
```

看到最后的成功提示后，说明代码、配置和主要接口检查通过。这个脚本只做自检，不会处理视频。

### 2. 现场只实时运行巢外 Demo

巢外片段只有 2 秒，易于测试；复制下面整段命令即可：

```powershell
& $py main.py `
  --mode outside `
  --video "data\external\vnbee\vnbee_outside_test_2s.mp4" `
  --output "output\presentation_outside" `
  --config "configs\demo_outside_test.yaml" `
  --device cuda:0
```

“处理完成”后，打开结果视频：

```powershell
Invoke-Item ".\output\presentation_outside\outside_result.mp4"
```

可以打开统计图：

```powershell
Invoke-Item ".\output\presentation_outside\analysis\counts_over_time.png"
```

重点：

1. 青色框是检测器找到的蜜蜂。
2. 彩色框和 `ID` 是连续帧匹配后形成的临时轨迹。
3. 右侧或 JSON 中的统计用于说明程序输出，不等同于正式精度指标。

### 3. 巢内结果采用“播放已有结果 + 说明”

巢内完整视频处理时间较长，可以自行尝试复现。直接播放已经完成的结果：

```powershell
Invoke-Item ".\output\demo_test\inside_result.mp4"
```

同时可以打开：

```powershell
Invoke-Item ".\output\demo_test\analysis\track_count_over_time.png"
Invoke-Item ".\output\demo_test\analysis\behavior_distribution.png"
```

解释为：巢内红外流程已经完成检测、跟踪和行为统计，但高密度遮挡导致当前跟踪结果和行为标签仍属于 Demo 级输出，不能宣称已经达到比赛精度。

### 4. 大概

> 今天展示的是一个可运行的端到端 Demo。输入一段巢内红外或巢外视频，系统会依次完成蜜蜂检测、轨迹关联、行为字段生成、结果视频保存和数据分析。

> 本次演示证明的是工程链路已经接通，并且可以输出可视化和结构化结果。检测、跟踪和行为统计的正式准确率，还需要带标注真值的数据集进行后续评估。
