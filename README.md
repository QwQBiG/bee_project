# 面向智慧养蜂的巢内外蜜蜂识别与行为量化

队伍编号：`614689`。

项目采用标准 `src` 布局，顶层只保留源码、ONNX 权重、配置和依赖说明：

```text
bee_project/
├─ src/                 训练、推理、评测、打包代码及项目资料
├─ weights/             正式 ONNX 权重
├─ configs/             算法与运行配置
├─ requirements.txt     开发依赖
└─ README.md            使用说明
```

详细文档见 [src/docs/README.md](src/docs/README.md)，环境配置见
[src/packages/README.md](src/packages/README.md)。本机放置于 `packages/` 的大型
Torch wheel 不属于源码，继续由 Git 忽略。

## 配置开发环境

安装 64 位 Python 3.13 后，可创建项目虚拟环境：

```bat
cd /d C:\你的路径\bee_project
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

也可以直接用系统 Python 3.13 安装依赖，详细命令见环境配置文档。

## 运行视频分析

```bat
# 巢外视频
.venv\Scripts\python.exe src\main.py --mode outside --video "data\outside.mp4" --output "output\outside"

# 巢内视频
.venv\Scripts\python.exe src\main.py --mode inside --video "data\inside.mp4" --output "output\inside"
```

直接使用系统 Python 时，将 `.venv\Scripts\python.exe` 换成 `py -3.13`。默认配置
位于 `configs/config.yaml`，正式推理模型位于 `weights/`。如需明确指定配置，可在
命令末尾加 `--config "configs\config.yaml"`。

## 运行四个比赛 EXE

四个 EXE 接收的是连续编号的 JPG 图片目录，而不是视频文件：

```bat
cd /d C:\你的路径\bee_project\EXE-614689
Inside-detection-614689.exe --input "C:\Test\Inside\detection\images"
Inside-tracking-614689.exe  --input "C:\Test\Inside\tracking\images"
Outside-detection-614689.exe --input "C:\Test\Outside\detection\images"
Outside-tracking-614689.exe  --input "C:\Test\Outside\tracking\images"
```

检测程序生成逐帧目标框，跟踪程序在检测基础上生成个体 ID 和轨迹。比赛格式 JSON
写入 `C:\TestResults\`；图表、标注视频、逐帧 CSV 和 HTML 分析预警报告写入
`EXE-614689\output\` 的 `figures\`、`videos\`、`data\`、`reports\` 子目录。
巢外报告给出携粉候选与营养趋势，巢内报告给出数量和活动趋势。报告中的自动判断是
辅助筛查，并会标注置信度与局限，不能替代养蜂人员现场诊断。

## 生成队伍 614689 比赛 EXE

正式 EXE 仅使用 ONNX 推理，不收集 Torch、Torchvision 或 Ultralytics：

```bat
py -3.10 -m venv .venv-build
.venv-build\Scripts\python.exe -m pip install -r src\deployment\requirements-submission.txt
.venv-build\Scripts\python.exe src\build_submission.py --team_id 614689
```

打包后生成 `EXE-614689/` 和 `EXE-614689.zip`。生成 selfcheck、校验和最终重新
压缩的方法见 [队伍 614689 可执行程序打包说明](src/docs/competition/614689可执行程序打包说明.md)。

## 运行测试

```bat
.venv\Scripts\python.exe -m pytest -q src\tests src\test_project.py
```
