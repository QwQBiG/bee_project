# 面向智慧养蜂的巢内外蜜蜂识别与行为量化

本项目面向真实蜂箱生产与监测场景，处理巢外可见光视频和巢内红外视频，研究蜜蜂个体检测、多目标跟踪、姿态特征以及个体与群体行为量化。项目通过命令行提供研究与评测工具。

完整的使用、标注和评测文档见 [docs/README.md](docs/README.md)。

## 准备便携运行环境（Windows）

项目不依赖评测机或用户电脑预先安装的 Python。首次从 GitHub 拉取代码后：

1. 将 `torch-2.13.0+cu132-cp313-cp313-win_amd64.whl` 放入 `packages/`；
2. 双击或在命令行执行 `setup_runtime.bat`；
3. 脚本自动下载并校验 Python 3.13.15 嵌入版、安装本地 PyTorch wheel、安装其余依赖并完成运行检查。

```bat
setup_runtime.bat
run_cli.bat --help
```

Python、pip 和全部依赖安装在项目内的 `.runtime/python313/`。该目录不提交到 Git。初始化完成后，可以将整个项目目录压缩并复制到另一台 64 位 Windows 电脑；解压后直接使用 `run_cli.bat`，不需要安装 Python 或重新下载依赖。目标电脑仍需具备与 CUDA 13.2 wheel 兼容的 NVIDIA 驱动。

## 命令行用法

```bash
# 巢外视频
run_cli.bat --mode outside --video data/outside.mp4 --output output/outside

# 巢内视频
run_cli.bat --mode inside --video data/inside.mp4 --output output/inside
```

默认配置位于 `configs/config.yaml`，模型文件位于 `artifacts/models/`。结果写入 `--output` 指定的目录，包括标注视频、统计 JSON 和离线 HTML 分析报告。CPU 可以运行，但长视频会比较慢；正式评测建议使用具备 CUDA 能力的 NVIDIA GPU。

## 测试

```bash
.runtime\python313\python.exe -m pytest -q
```
