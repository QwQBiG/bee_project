"""在相同视频和检测权重下对照评测官方跟踪器。

该脚本不计算 HOTA/IDF1；没有身份真值时只报告工程指标，避免把轨迹数量误称为准确率。
"""

import argparse
import copy
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from tracking.inside_tracker import InsideHiveTracker
from tracking.outside_tracker import OutsideHiveTracker


def build_config(config_path: str, mode: str, tracker_type: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    section_name = "inside_tracker" if mode == "inside" else "outside_tracker"
    section = config.get(section_name) or config.get("tracker") or {}
    section = copy.deepcopy(section)
    section["tracker_type"] = tracker_type
    section["tracker_config"] = f"{tracker_type}.yaml"
    return section


def run_one(video_path: str, config: dict, mode: str, max_frames: int) -> dict:
    tracker = InsideHiveTracker(config) if mode == "inside" else OutsideHiveTracker(config)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    detection_counts = []
    track_counts = []
    unique_ids = set()
    frame_count = 0
    started = time.perf_counter()
    while frame_count < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        tracks, detections = tracker.process_frame(frame)
        detection_counts.append(len(detections))
        track_counts.append(len(tracks))
        unique_ids.update(track.track_id for track in tracks)
        frame_count += 1
    capture.release()

    elapsed = time.perf_counter() - started
    return {
        "tracker": config["tracker_type"],
        "frames": frame_count,
        "source_fps": source_fps,
        "mean_detections": sum(detection_counts) / max(frame_count, 1),
        "mean_tracks": sum(track_counts) / max(frame_count, 1),
        "max_tracks": max(track_counts, default=0),
        "unique_ids_seen": len(unique_ids),
        "processing_seconds": elapsed,
        "throughput_fps": frame_count / elapsed if elapsed else None,
        "ground_truth_metrics": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inside", "outside"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    results = []
    for tracker_type in ("botsort", "bytetrack"):
        result = run_one(args.video, build_config(args.config, args.mode, tracker_type),
                         args.mode, args.frames)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
