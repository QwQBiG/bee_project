"""Exercise all four batch modes with real ONNX weights and local video frames.

Run from repository root: python -m tools.smoke_batch_onnx
Outputs stay under ignored output/. This is not an accuracy benchmark.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / "output" / "onnx_smoke_20260905"
    sources = {
        "Inside": root / "data/external/oist/oist_M13_ir_test_10s.mp4",
        "Outside": root / "data/external/vnbee/vnbee_outside_test_2s.mp4",
    }
    report = []
    for scene, video in sources.items():
        capture = cv2.VideoCapture(str(video))
        frames = []
        try:
            for _ in range(3):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot read three frames: {video}")
                frames.append(frame)
        finally:
            capture.release()
        for task in ("detection", "tracking"):
            folder = work / scene / task / "images"
            folder.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames, 1):
                if not cv2.imwrite(str(folder / f"frame_{index:04d}.jpg"), frame):
                    raise RuntimeError("cannot write smoke input")
            sequence = f"{scene}-{task}"
            run = subprocess.run([
                sys.executable, "-m", "inference.batch_cli", "--input", str(folder),
                "--sequence", sequence, "--team-id", "123456",
                "--output-dir", str(work / "results"),
            ], cwd=root, capture_output=True, text=True, timeout=180)
            if run.returncode or len(run.stdout.splitlines()) != 1:
                raise RuntimeError(f"{sequence}: {run.stdout} {run.stderr}")
            result_file = work / "results" / f"{sequence}-123456.json"
            raw = result_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
            rows = payload["detections" if task == "detection" else "tracks"]
            assert "\n" not in raw and payload["num_frames"] == 3
            assert payload["num_records"] == len(rows)
            for row in rows:
                offset = 3 if task == "detection" else 2
                x, y, w, h = row[offset:offset + 4]
                height, width = frames[row[0] - 1].shape[:2]
                assert 0 <= x <= width and 0 <= y <= height
                assert w > 0 and h > 0 and x + w <= width + .02
                assert y + h <= height + .02
            report.append({"sequence": sequence, "records": len(rows),
                           "processing_time_ms": payload["processing_time_ms"],
                           "stderr": run.stderr.strip()})
    (work / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
