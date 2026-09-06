# ONNX 文件夹推理

配置入口为 `configs/algorithm_config.json`。模型路径相对于配置文件解析。
本仓库不分发权重、视频或本地数据集；运行前自行准备相应文件。

- 巢外检测：`artifacts/models/hive_entrance_bee_yolov8n.onnx`，输入 1280。
- 巢内检测：`artifacts/models/bee_inside_ft_v1.onnx`，输入 640。
- 巢外跟踪：ByteTrack；巢内跟踪：相邻帧 IoU。
- 多类别检测权重必须通过 `class_ids` 明确选择蜜蜂类别，输出统一为 class_id=0。
- 姿态权重不能用于这个 HBB 解码入口。
- 巢外文件夹检测使用 `detection_box_scale=1.15`，在 NMS 与数量截断后
  围绕框中心调整宽高并裁剪到图像边界；不修改置信度、类别或记录排序。
  这是绑定当前巢外权重的可逆校准，设为 `1.0` 恢复原框。
  不作用于跟踪、巢内或旧单图入口。更换权重或标注口径时应重新验证，
  不应将该系数直接迁移到其他模型。

ByteTrack 适配器验证版本为 Ultralytics 8.4.96、lap 0.5.13。
可选评测依赖见 `requirements-evaluation.txt`；ONNX Runtime 需要单独安装，
CPU/GPU 发行包二选一，GPU 还需匹配其运行库。禁止依赖启动时联网安装。

源码运行示例（队伍 ID 为示例）：

```powershell
python -m inference.batch_cli --input C:/Test/Outside/tracking/images/ --sequence Outside-tracking --team-id 123456 --device cpu
```

结果默认写入 `C:/TestResults/`。输入须为从 1 开始连续编号的 JPG。
检测保留 NMS 后前 200/600 条，不使用跟踪阈值过滤检测结果。
跟踪低置信度候选仅用于关联，最终 JSON 的 conf=1 是提交约定，不是概率。

`--device cpu` 用于 CPU 检查；`--device cuda` 用于严格GPU诊断，无法启用CUDA时
直接报错。正式打包配置固定使用`auto`：优先CUDA，GPU初始化、推理或显存失败时
自动回退CPU，并把回退原因写入stderr，不能污染stdout状态JSON。
创建会话前调用 ONNX Runtime 的 preload_dlls，发现当前环境已安装的
兼容 PyTorch/NVIDIA CUDA 与 cuDNN 运行库，不要求导入训练模块。
源码 CUDA 编译缓存默认保存在 `.runtime/cuda-cache`；冻结程序使用系统临时目录。
可用进程环境变量 `CUDA_CACHE_PATH` 指定其他可写缓存目录。
新显卡首次运行可能需要编译内核，不能把首次耗时排除在正式计时之外。
执行 `python -m tools.verify_onnx_gpu` 可检查模型实际 CUDA 节点执行记录；
本地报告放在 `output/gpu_verification`，不是仅检查 provider 名称。
执行 `python -m tools.smoke_batch_onnx --device cpu` 可检查四个源码入口。
切片实验选项默认为关闭；`tools.compare_onnx_candidates` 是本地对照工具，
不属于正式提交入口。修改输入、配置或算法后应使用新的预测缓存。

正式队伍号为`614689`，完整打包、自检、校验和重压缩命令见
[`614689可执行程序打包说明.md`](614689可执行程序打包说明.md)。
