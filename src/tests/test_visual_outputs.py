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
