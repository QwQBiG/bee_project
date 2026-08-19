"""Analyze the short outside-hive demo result with pandas and matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    stats_path = Path(args.stats)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    detections = pd.Series(stats["detection_history"], dtype="int64")
    tracks = pd.Series(stats["track_history"], dtype="int64")
    frame_data = pd.DataFrame(
        {
            "frame": range(len(detections)),
            "time_s": pd.Series(range(len(detections))) / args.fps,
            "detections": detections,
            "confirmed_tracks": tracks,
        }
    )
    frame_data["confirmation_ratio"] = (
        frame_data["confirmed_tracks"]
        / frame_data["detections"].replace(0, pd.NA)
    )
    frame_data.to_csv(output_dir / "frame_counts.csv", index=False)

    processing_time = float(stats.get("processing_time", 0.0))
    summary = {
        "frames": int(len(frame_data)),
        "duration_s": float(len(frame_data) / args.fps),
        "detections": {
            "mean_per_frame": float(detections.mean()),
            "median_per_frame": float(detections.median()),
            "min_per_frame": int(detections.min()),
            "max_per_frame": int(detections.max()),
            "total_frame_detections": int(detections.sum()),
        },
        "confirmed_tracks": {
            "mean_per_frame": float(tracks.mean()),
            "median_per_frame": float(tracks.median()),
            "min_per_frame": int(tracks.min()),
            "max_per_frame": int(tracks.max()),
            "unique_track_ids_created": int(stats.get("total_tracks", 0)),
        },
        "mean_confirmation_ratio_nonempty_frames": float(
            frame_data["confirmation_ratio"].dropna().mean()
        ),
        "processing": {
            "time_s": processing_time,
            "throughput_fps": (
                float(len(frame_data) / processing_time)
                if processing_time > 0
                else None
            ),
            "realtime_factor_at_source_fps": (
                float((len(frame_data) / processing_time) / args.fps)
                if processing_time > 0
                else None
            ),
        },
        "behavior_counts": (
            stats.get("behavior_analysis", {})
            .get("individual_summary", {})
            .get("behavior_counts", {})
        ),
        "limitations": [
            "No frame-level ground-truth labels were evaluated in this smoke test.",
            "Unique track IDs are not equivalent to unique bees because ID switches can occur.",
            "Entry/exit events are not validated on a two-second clip.",
        ],
    }
    with (output_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(
        frame_data["time_s"],
        frame_data["detections"],
        label="YOLOv8 detections",
        linewidth=1.5,
    )
    axis.plot(
        frame_data["time_s"],
        frame_data["confirmed_tracks"],
        label="confirmed tracks",
        linewidth=1.5,
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Count per frame")
    axis.set_title("Outside-hive detection and tracking counts")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "counts_over_time.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
