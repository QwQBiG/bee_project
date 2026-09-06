# 面向智慧养蜂的巢内外蜜蜂识别与行为量化

本项目面向真实蜂箱生产与监测场景，处理巢外可见光视频和巢内红外视频，研究蜜蜂个体检测、多目标跟踪、姿态特征以及个体与群体行为量化。项目通过命令行提供研究与评测工具。

完整的使用、标注和评测文档见 [docs/README.md](docs/README.md)。

## 配置运行环境（Windows）

项目不再下载或解压便携 Python，也不再提供双击式安装、启动脚本。请先安装
64 位 Python 3.13，并将
`torch-2.13.0+cu132-cp313-cp313-win_amd64.whl` 放入 `packages/`，然后按照
[packages/README.md](packages/README.md) 安装 CUDA 版 PyTorch 和项目依赖。文档同时提供：

1. `.venv` 项目虚拟环境方案（推荐，不影响其他项目）；
2. 系统 Python 3.13 直接安装方案（命令较短，但可能产生依赖冲突）。

运行 PyTorch CUDA wheel 通常只需要兼容的 NVIDIA 显卡驱动，不要求另外安装
CUDA Toolkit；如需编译 CUDA 扩展，再安装与 `cu132` 对应的 CUDA Toolkit 13.2。

## 命令行用法

使用 `.venv`：

```bat
# 巢外视频
.venv\Scripts\python.exe main.py --mode outside --video data\outside.mp4 --output output\outside

# 巢内视频
.venv\Scripts\python.exe main.py --mode inside --video data\inside.mp4 --output output\inside
```

直接使用系统 Python 3.13：

```bat
# 巢外视频
py -3.13 main.py --mode outside --video data\outside.mp4 --output output\outside

# 巢内视频
py -3.13 main.py --mode inside --video data\inside.mp4 --output output\inside
```

默认配置位于 `configs/config.yaml`，模型文件位于 `artifacts/models/`。结果写入 `--output` 指定的目录，包括标注视频、统计 JSON 和离线 HTML 分析报告。CPU 可以运行，但长视频会比较慢；正式评测建议使用具备 CUDA 能力的 NVIDIA GPU。

## 队伍614689正式打包

正式评测采用四个文件夹级EXE，不使用上面的开发视频命令。程序通过`--input`
接收连续JPG目录，汇总JSON写入`C:/TestResults/`。完整流程见
[队伍614689可执行程序打包说明](docs/competition/614689可执行程序打包说明.md)。

```bat
py -3.13 tools\export_submission_onnx.py
py -3.10 -m venv .venv-build
.venv-build\Scripts\python.exe -m pip install -r deployment\requirements-submission.txt
.venv-build\Scripts\python.exe build_submission.py --team_id 614689
```

打包后必须使用本队自建连续图片生成`selfcheck/`，再执行目录校验和最终重压缩。
提交打包环境与日常训练环境分开：最终EXE仅使用ONNX权重，不收集Torch、
Torchvision或Ultralytics；CUDA 运行组件固定为评测机支持的13.2系列。

## 测试

```bat
.venv\Scripts\python.exe -m pytest -q

# 使用系统 Python 时
py -3.13 -m pytest -q
```
