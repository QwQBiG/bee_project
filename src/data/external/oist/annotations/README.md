# OIST 巢内蜜蜂标注

本目录用于存放 OIST Honeybee Segmentation and Tracking Datasets 的本地标注资源。

官方来源：

- 数据说明页：<https://www.oist.jp/research/research-units/bptu/honeybee_tracking_datasets>
- 30 FPS 标注包：<https://beepositions.unit.oist.jp/frame_annotations_30fps.tgz>
- 70 FPS 标注包：<https://beepositions.unit.oist.jp/frame_annotations_70fps.tgz>

已下载文件的 SHA-256：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `frame_annotations_30fps.tgz` | 1,233,666 bytes | `abc830e2fc3bf1d0792c03ab9f8057b55439ac3145630ce993132e81705776e8` |
| `frame_annotations_70fps.tgz` | 951,571 bytes | `839a1bf1eb0930c2f392b852e87b0410896dbb5c5fc185645bc71a0ed7da6f86` |

解压后目录中包含 `frames_txt/*.txt`，每行格式为：

```text
offset_x offset_y class position_x position_y angle
```

这批标注提供目标位置、类别和朝向，但不提供跨帧个体 ID。因此它可以用于检测位置/朝向评测和红外域适配，不能单独计算 IDF1、MOTA 或 HOTA。

当前项目的 `data/external/oist/oist_M13_ir_test_10s.mp4` 是 OIST 的 M13 演示片段，与这两个检测标注包不是同一段录像；项目不会将它们强行配对计算指标。若要评测当前 M13，仍需对应 M13 帧级标注，或改用 OIST 提供的带检测/轨迹数据的 S1–S5 录制片段。

运行摘要检查：

```powershell
python tools/inspect_oist_annotations.py `
  --root data/external/oist/annotations/frame_annotations_30fps `
  --output output/evaluation/oist_30fps_summary.json
```

压缩包和解压后的原始标注默认保留在本地，不提交到 GitHub；提交分支只包含来源清单和解析工具，避免把第三方数据重新分发到代码仓库。
