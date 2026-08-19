import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    track_counts = pd.Series(stats["track_history"], name="active_tracks")
    frame_stats = pd.DataFrame(
        {
            "frame": range(len(track_counts)),
            "time_seconds": pd.Series(range(len(track_counts))) / 10.0,
            "active_tracks": track_counts,
        }
    )
    frame_stats.to_csv(output_dir / "frame_track_counts.csv", index=False)

    behavior_counts = (
        stats.get("behavior_analysis", {})
        .get("individual_summary", {})
        .get("behavior_counts", {})
    )
    behavior_stats = pd.DataFrame(
        [{"behavior": key, "count": value} for key, value in behavior_counts.items()]
    )
    if not behavior_stats.empty:
        behavior_stats["percentage"] = (
            behavior_stats["count"] / behavior_stats["count"].sum() * 100
        )
    behavior_stats.to_csv(output_dir / "behavior_distribution.csv", index=False)

    processing_time = float(stats["processing_time"])
    total_frames = int(stats["total_frames"])
    throughput = total_frames / processing_time
    source_fps = 10.0

    summary = {
        "total_frames": total_frames,
        "processing_time_seconds": processing_time,
        "mean_seconds_per_frame": processing_time / total_frames,
        "throughput_fps": throughput,
        "source_fps": source_fps,
        "realtime_slowdown_factor": source_fps / throughput,
        "track_count_mean": float(track_counts.mean()),
        "track_count_median": float(track_counts.median()),
        "track_count_std": float(track_counts.std(ddof=0)),
        "track_count_min": int(track_counts.min()),
        "track_count_max": int(track_counts.max()),
        "track_count_final": int(track_counts.iloc[-1]),
        "reported_total_tracks": int(stats.get("total_tracks", 0)),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plt.figure(figsize=(9, 4.8))
    plt.plot(frame_stats["time_seconds"], frame_stats["active_tracks"], linewidth=2)
    plt.xlabel("Video time (s)")
    plt.ylabel("Reported active tracks")
    plt.title("Inside-hive demo: active track count by frame")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "track_count_over_time.png", dpi=180)
    plt.close()

    if not behavior_stats.empty:
        plt.figure(figsize=(7.5, 4.8))
        bars = plt.bar(
            behavior_stats["behavior"],
            behavior_stats["percentage"],
            color=["#d9912b", "#4589c7", "#6aa84f", "#8e63b6"][
                : len(behavior_stats)
            ],
        )
        plt.ylabel("Share of per-frame behavior records (%)")
        plt.title("Rule-based behavior output distribution")
        plt.bar_label(bars, fmt="%.2f%%", padding=3)
        plt.tight_layout()
        plt.savefig(output_dir / "behavior_distribution.png", dpi=180)
        plt.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
