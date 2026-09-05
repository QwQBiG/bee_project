# Windows 环境配置说明

> 温馨提示：以下环境面向 64 位 Windows 10/11 和 NVIDIA RTX 显卡，当前固定使用
> Python 3.13、PyTorch 2.13.0 + CUDA 13.2。不同 Python 或 CUDA 构建的 wheel
> 不能混用。

本项目不再自动下载、解压 Python，也不再使用 `setup_runtime.bat` 或
`run_cli.bat`。环境需要按下面步骤手动配置一次。

## 1. 安装 Python 3.13（64 位）

从 [Python 官方 Windows 下载页](https://www.python.org/downloads/windows/) 下载
Python 3.13 的 **Windows installer (64-bit)**。安装时启用 `pip`；如果安装程序提供
`Add python.exe to PATH` 选项，也建议勾选。

打开 CMD，检查版本：

```bat
py -3.13 --version
```

然后从下面两种安装方式中选择一种即可。

## 2. 选择 Python 环境

### 方案 A：使用 `.venv`（推荐）

`.venv` 是只属于本项目的 Python 环境，安装在项目的 `.venv` 文件夹中，不会影响
电脑上的其他 Python 项目。进入项目根目录并创建一次：

```bat
cd /d C:\你的路径\bee_project
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

后续命令直接使用 `.venv\Scripts\python.exe`，无需激活虚拟环境，也不会把依赖
安装到系统 Python 中。`.venv` 创建一次即可，以后不用重复创建。

### 方案 B：直接使用系统 Python

如果电脑主要用于这一个 Python 项目，也可以不创建 `.venv`，直接使用系统安装的
Python 3.13：

```bat
cd /d C:\你的路径\bee_project
py -3.13 -m pip install --upgrade pip
```

这种方式会把依赖安装到系统 Python 3.13，可能与其他项目需要的版本发生冲突。
下文中的系统环境命令统一使用 `py -3.13`，确保不会误用其他 Python 版本；如果
`python --version` 明确显示为 `3.13.x`，也可以把 `py -3.13` 换成 `python`。

## 3. 检查 NVIDIA 驱动

先在 CMD 中执行：

```bat
nvidia-smi
```

如果命令不存在、显卡未显示或驱动过旧，请从
[NVIDIA 官方驱动下载页](https://www.nvidia.com/Download/index.aspx) 更新驱动。

`cu132` wheel 已携带推理所需的 CUDA 运行库，普通运行本项目不需要额外安装
CUDA Toolkit。只有在需要使用 NVCC 编译 CUDA 扩展时，才需要另行安装
[CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-13-2-0-download-archive)，并应与
wheel 的 `cu132` 保持一致，不要改装 CUDA 13.3 来替代它。

## 4. 安装 PyTorch 和 torchvision

Git 仓库不提交大型 wheel。请将下面这个文件放到当前 `packages` 目录：

```text
packages\torch-2.13.0+cu132-cp313-cp313-win_amd64.whl
```

方案 A（`.venv`）执行：

```bat
.venv\Scripts\python.exe -m pip install "packages\torch-2.13.0+cu132-cp313-cp313-win_amd64.whl"
.venv\Scripts\python.exe -m pip install --no-deps torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
```

方案 B（系统 Python）执行：

```bat
py -3.13 -m pip install "packages\torch-2.13.0+cu132-cp313-cp313-win_amd64.whl"
py -3.13 -m pip install --no-deps torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
```

第一条命令只使用你放入 `packages` 的本地 Torch wheel；第二条命令从 PyTorch
CUDA 13.2 官方源下载与其匹配的 torchvision。

## 5. 安装项目的其余 Python 依赖

方案 A（`.venv`）执行：

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

方案 B（系统 Python）执行：

```bat
py -3.13 -m pip install -r requirements.txt
```

这一步会安装 Ultralytics、OpenCV、NumPy、SciPy、Pandas、PyYAML、Matplotlib、
`lap` 和 pytest。`requirements.txt` 不再重复安装 Torch 和 torchvision。

## 6. 验证环境

方案 A（`.venv`）：

```bat
.venv\Scripts\python.exe -c "import torch, torchvision, cv2, lap, ultralytics; print('Python/Torch imports: OK'); print('Torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

方案 B（系统 Python）：

```bat
py -3.13 -c "import torch, torchvision, cv2, lap, ultralytics; print('Python/Torch imports: OK'); print('Torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

正常情况下应看到：

```text
Torch: 2.13.0+cu132
torchvision: 0.28.0+cu132
CUDA: 13.2
CUDA available: True
GPU: 你的 NVIDIA 显卡名称
```

若 `CUDA available` 为 `False`，优先更新 NVIDIA 驱动，然后重启电脑再次检查；
不要先去反复安装 CUDA Toolkit。

## 7. 运行项目

方案 A（`.venv`）：

```bat
.venv\Scripts\python.exe main.py --help
.venv\Scripts\python.exe main.py --mode outside --video "data\outside.mp4" --output "output\outside"
.venv\Scripts\python.exe main.py --mode inside --video "data\inside.mp4" --output "output\inside"
```

方案 B（系统 Python）：

```bat
py -3.13 main.py --help
py -3.13 main.py --mode outside --video "data\outside.mp4" --output "output\outside"
py -3.13 main.py --mode inside --video "data\inside.mp4" --output "output\inside"
```

## 目录中需要保留的文件

```text
packages\
├─ README.md
└─ torch-2.13.0+cu132-cp313-cp313-win_amd64.whl   （本地放入，不提交 Git）
```

不再需要 `python-*-embed-amd64.zip` 或 `get-pip.py`。
