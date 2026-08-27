# 本地离线安装包

将预先下载的 PyTorch CUDA `.whl` 文件放在此目录。启动脚本会检查 wheel 的
Python、Windows 平台和 CUDA 标签，并确认 NVIDIA 驱动兼容后优先使用本地文件。

例如：

```text
torch-2.13.0+cu132-cp313-cp313-win_amd64.whl
```

上述文件仅适用于 64 位 Windows、CPython 3.13，以及支持 CUDA 13.2 或更高版本
的 NVIDIA 驱动。不兼容或没有本地 wheel 时，自动模式会安装 CPU 版 PyTorch。

此目录中的 `.whl` 已由 `.gitignore` 排除，不会提交到 Git。

如果系统中找不到 64 位 Python 3.13，Windows 启动脚本还会把经过 SHA-256
校验的官方 Python 3.13.15 便携包缓存到此目录，并解压到项目的
`.runtime/python313/`。下载的 Python 压缩包和 pip 引导文件同样不会提交到 Git。
