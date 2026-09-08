from pathlib import Path

import cv2
import numpy as np

from inference.visual_outputs import generate_visual_outputs


def test_generates_four_visual_output_categories(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    indexed = []
    for frame_id in (1, 2):
        path = images / f"im_{frame_id:04d}.jpg"
        assert cv2.imwrite(str(path), np.full((48, 64, 3), 30, np.uint8))
        indexed.append((frame_id, path))
    executable = tmp_path / "EXE-614689" / "Outside-detection-614689.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"exe")
    payload = {
        "team_id": "614689", "sequence": "Outside-detection",
        "task": "detection", "num_records": 2,
        "detections": [
            [1, 0, .9, 5, 6, 10, 12, 1],
            [2, 0, .8, 7, 8, 10, 12, 1],
        ],
    }
    root = generate_visual_outputs(payload, indexed, executable, fps=5)
    basename = "Outside-detection-614689"
    assert (root / "figures" / f"{basename}-counts.png").is_file()
    assert (root / "figures" / f"{basename}-heatmap.png").is_file()
    assert (root / "videos" / f"{basename}.mp4").stat().st_size > 0
    assert (root / "data" / f"{basename}.csv").is_file()
    report = (root / "reports" / f"{basename}.html").read_text("utf-8")
    assert "平均每帧目标数" in report
    assert "自动分析与预警" in report
    assert "预测置信度" in report
    assert "携粉候选比例" in report


def test_inside_tracking_report_contains_activity_prediction(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    indexed = []
    for frame_id in range(1, 11):
        path = images / f"im_{frame_id:04d}.jpg"
        assert cv2.imwrite(str(path), np.full((48, 64, 3), 30, np.uint8))
        indexed.append((frame_id, path))
    executable = tmp_path / "EXE-614689" / "Inside-tracking-614689.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"exe")
    tracks = []
    for frame_id in range(1, 11):
        tracks.extend([
            [frame_id, 1, 5 + frame_id, 6, 10, 12, 1],
            [frame_id, 2, 20 + frame_id, 8, 10, 12, 1],
        ])
    payload = {
        "team_id": "614689", "sequence": "Inside-tracking",
        "task": "tracking", "num_frames": 10,
        "num_records": len(tracks), "tracks": tracks,
    }
    root = generate_visual_outputs(payload, indexed, executable, fps=5)
    report = (root / "reports" / "Inside-tracking-614689.html").read_text("utf-8")
    assert "目标数量趋势" in report
    assert "平均位移速度" in report
    assert "处理建议" in report
