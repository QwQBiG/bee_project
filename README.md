# 蜂群视频智能分析

这是一个面向养蜂与蜂群行为研究的本地视频分析工具。现在既保留了命令行能力，也提供了普通用户可以直接使用的中文网页：

1. 选择“巢内视频”或“巢外视频”
2. 从电脑中选择并上传视频
3. 等待分析完成
4. 在线查看标注视频，并下载 HTML 报告、JSON 数据或完整结果包

## 最简单的启动方式（Windows）

双击项目根目录下的 `start_web.bat`。第一次启动会自动创建独立的 Python 环境、检测显卡和 NVIDIA 驱动，并安装匹配的 PyTorch：

- 驱动支持 CUDA 12.8 及以上：优先安装官方 `cu128`
- 驱动支持 CUDA 12.6：优先安装官方 `cu126`
- 驱动支持 CUDA 11.8：安装官方 `cu118`
- 没有兼容的 NVIDIA 显卡：自动安装 CPU 版本
- CUDA 安装或实际运算验证失败：尝试较低兼容版本，最后回退 CPU

CUDA 版 PyTorch 通常需要下载 1～3 GB，第一次启动耗时会较长；后续启动会复用已经验证过的环境。浏览器会打开：

```text
http://127.0.0.1:8000
```

停止服务时，点击网页右上角的“关闭程序”，或回到黑色命令窗口按 `Ctrl+C`。

## Docker 部署

已安装 Docker Desktop 的电脑或服务器可以直接运行：

```bash
docker compose up --build
```

然后访问 `http://服务器地址:8000`。分析结果保存在 Docker 数据卷 `bee-results` 中，容器重建后仍会保留。

默认使用 CPU。如果服务器已经正确安装 NVIDIA Container Toolkit，可在部署时按服务器环境调整 `compose.yaml`，并把 `BEE_DEVICE` 改为 `cuda:0`。

## 手动启动（Web 界面）

```bash
python -m pip install -r requirements.txt
python app/run.py
```

启动后自动打开浏览器访问 `http://127.0.0.1:8000`（本质是 `uvicorn app.main:app --reload`）。
Linux/macOS 也可以运行：

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

- `BEE_DEVICE`：推理设备，例如 `cpu`、`cuda:0`
- `BEE_TORCH_BACKEND`：覆盖自动选择，例如 `cpu`、`cu118`、`cu126`、`cu128`

检测与安装结果会写入本地 `runtime_environment.json`，网页侧栏会显示当前使用“GPU 加速”还是“CPU 模式”。PyTorch 的 CUDA wheel 自带运行库，普通用户只需要兼容的 NVIDIA 驱动，不需要另外安装完整 CUDA Toolkit。

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
