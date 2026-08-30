# 本地运行安装包

Git 仓库不提交 Python、PyTorch wheel 或其他大型二进制文件。首次初始化前，将下列文件放在本目录：

```text
torch-2.13.0+cu132-cp313-cp313-win_amd64.whl
```

然后在项目根目录运行 `setup_runtime.bat`。脚本会：

1. 下载并校验官方 Python 3.13.15 Windows 嵌入版；
2. 从本目录安装指定的 CUDA 13.2 PyTorch wheel；
3. 从 PyTorch CUDA 13.2 官方源安装匹配的 `torchvision 0.28.0+cu132`；
4. 安装 `requirements.txt` 中的全部其余依赖；
5. 验证 Python、PyTorch、CUDA 和项目依赖是否可以导入。

Python 压缩包、`get-pip.py` 和 wheel 均由 `.gitignore` 排除。初始化后的 `.runtime/` 也不提交到 Git，但可以随整个项目目录一起压缩并复制到另一台 64 位 Windows 电脑。
