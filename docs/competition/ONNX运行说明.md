# ONNX 文件夹推理

配置入口为 `configs/algorithm_config.json`。模型路径相对于配置文件解析。
本仓库不分发权重、视频或本地数据集；运行前自行准备相应文件。

- 巢外检测：`artifacts/models/hive_entrance_bee_yolov8n.onnx`，输入 1280。
- 巢内检测：`artifacts/models/bee_inside_ft_v1.onnx`，输入 640。
- 巢外跟踪：ByteTrack；巢内跟踪：相邻帧 IoU。
- 多类别检测权重必须通过 `class_ids` 明确选择蜜蜂类别，输出统一为 class_id=0。
- 姿态权重不能用于这个 HBB 解码入口。

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

`--device cpu` 用于 CPU 检查；`--device cuda` 无法启用 CUDA 时直接报错；
默认 auto 允许 CPU 回退，不可据此判断 GPU 已启用。
执行 `python -m tools.smoke_batch_onnx --device cpu` 可检查四个源码入口。
切片实验选项默认为关闭；`tools.compare_onnx_candidates` 是本地对照工具，
不属于正式提交入口。修改输入、配置或算法后应使用新的预测缓存。
