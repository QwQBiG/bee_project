# 蜂群视频智能分析

这是一个面向养蜂与蜂群行为研究的本地视频分析工具。现在既保留了命令行能力，也提供了普通用户可以直接使用的中文网页。

完整的使用、标注和评测文档见 [docs/README.md](docs/README.md)。

网页基本流程：

1. 选择“巢内视频”或“巢外视频”
2. 从电脑中选择并上传视频
3. 等待分析完成
4. 在线查看标注视频，并下载 HTML 报告、JSON 数据或完整结果包

## 最简单的启动方式（Windows）

双击项目根目录下的 `start_web.bat`。启动脚本会优先查找 64 位 Python 3.13，并用它创建项目环境。找不到时，会从 Python 官方站点下载经过 SHA-256 校验的 Python 3.13.15 便携包，解压到 `.runtime/python313/` 并初始化 pip。

随后脚本检测显卡和 NVIDIA 驱动，并安装 PyTorch。可将预先下载的 CUDA 版 PyTorch wheel 放在项目的 `packages/` 目录中，例如：

```text
packages/torch-2.13.0+cu132-cp313-cp313-win_amd64.whl
```

- 本地 wheel 的操作系统、Python 标签、CUDA 版本和 NVIDIA 驱动全部兼容：直接使用本地文件安装
- 没有 NVIDIA 显卡：联网安装 CPU 版本
- 有 NVIDIA 显卡但本地 wheel 不兼容或不存在：联网安装 CPU 版本
- 已经安装并验证通过的 PyTorch：直接复用，不会重复安装

`packages/*.whl` 默认不会提交到 Git。该 Windows wheel 不能用于 macOS、Linux 或不同 Python 版本。

Python选择顺序为：已有的 Python 3.13 项目环境、外部 Python 3.13、项目内便携 Python 3.13。若原 `.venv` 使用其他 Python 版本，脚本会保留它，并新建 `.venv-py313`，不会删除旧环境。

CUDA 版 PyTorch 通常需要下载 1～3 GB，第一次启动耗时会较长；后续启动会复用已经验证过的环境。浏览器会打开：

```text
http://127.0.0.1:8000
```

停止服务时，点击网页右上角的“关闭程序”，或回到黑色命令窗口按 `Ctrl+C`。

## 最简单的启动方式（macOS）

在 Finder 中双击项目根目录下的 `start_web.command`。脚本会定位项目目录、检查 `8000` 端口、创建或复用 `.venv-macos`，安装并校验依赖，启动成功后自动打开默认浏览器。

Mac 需要 64 位 Python 3.10～3.13，推荐 Python 3.12。脚本会优先查找 Python 3.12；如果没有兼容 Python 但已经安装 Homebrew，会通过 Homebrew 安装 `python@3.12`。如果没有 Homebrew，脚本会打开 Python 官方下载页面，安装 Python 后再次双击即可。

Apple Silicon Mac 会安装标准 macOS 版 PyTorch，并在可用时自动使用 Metal/MPS GPU 加速，不需要也不会下载 CUDA 版 PyTorch。Intel Mac 或 MPS 不可用时自动使用 CPU。首次安装依赖需要联网且耗时较长，后续启动会复用环境。

如果 macOS 首次阻止脚本运行，可在 Finder 中右键 `start_web.command`，选择“打开”并确认。停止服务时，可点击网页右上角的“关闭程序”，或在启动窗口按 `Ctrl+C`。

## Docker 部署

已安装 Docker Desktop 的电脑或服务器可以直接运行：

```bash
docker compose up --build
```

然后访问 `http://服务器地址:8000`。分析结果保存在 Docker 数据卷 `bee-results` 中，容器重建后仍会保留。

默认使用 CPU。如果服务器已经正确安装 NVIDIA Container Toolkit，可在部署时按服务器环境调整 `compose.yaml`，并把 `BEE_DEVICE` 改为 `cuda:0`。

## 手动启动（Web 界面）

```bash
python tools/bootstrap_runtime.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动后在浏览器访问 `http://127.0.0.1:8000`。Linux/macOS 也可以在终端运行：

```bash
chmod +x start_web.sh
./start_web.sh
```

## 网页输出

- 巢外：标注视频、进出事件、采粉分析报告、轨迹与行为统计
- 巢内：标注视频、密度/活动/聚集分析报告、轨迹与行为统计
- 所有任务：任务记录、结构化分析结果和统计 JSON 下载接口

运行产生的上传视频、标注视频和报告默认位于 `app/uploads/`，任务记录保存在 `app/task_store.json`。推理任务会串行运行，避免多个任务同时争抢显卡或内存。

可通过环境变量调整：

- `BEE_DEVICE`：推理设备，例如 `cpu`、`mps`、`cuda:0`
- `BEE_TORCH_BACKEND`：覆盖自动选择，例如 `cpu`、`mps`、`cu118`、`cu126`、`cu128`；手动指定 CUDA 后端时允许从官方源联网安装

检测与安装结果会写入本地 `runtime_environment.json`，网页侧栏会显示当前使用“GPU 加速”还是“CPU 模式”。Windows 的 PyTorch CUDA wheel 自带运行库，普通用户只需要兼容的 NVIDIA 驱动，不需要另外安装完整 CUDA Toolkit；macOS 使用标准 PyTorch 包内置的 MPS 支持。

## 原命令行用法

```bash
# 巢外视频
python main.py --mode outside --video data/outside.mp4 --output output/outside

# 巢内视频
python main.py --mode inside --video data/inside.mp4 --output output/inside
```

默认配置位于 `configs/config.yaml`，模型文件位于 `artifacts/models/`。CPU 可以运行，但长视频会比较慢；正式部署更建议使用具备 CUDA 能力的 NVIDIA GPU。

## 测试

```bash
python -m pytest -q
```
