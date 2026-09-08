"""Create optional human-readable artifacts during each competition run."""

from __future__ import annotations

import csv
import html
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    temporary = path.with_name(path.name + ".tmp")
    encoded.tofile(temporary)
    os.replace(temporary, path)


def _rows_by_frame(payload: Dict[str, Any]) -> Dict[int, List[List[Any]]]:
    key = "detections" if payload["task"] == "detection" else "tracks"
    grouped: Dict[int, List[List[Any]]] = defaultdict(list)
    for row in payload[key]:
        grouped[int(row[0])].append(row)
    return grouped


def _bbox(row: Sequence[Any], task: str) -> Tuple[float, float, float, float]:
    offset = 3 if task == "detection" else 2
    return tuple(float(value) for value in row[offset:offset + 4])


def _count_chart(counts: List[int], title: str) -> np.ndarray:
    width, height = 1280, 720
    left, top, right, bottom = 90, 70, 40, 90
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (left, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, height - bottom), (40, 40, 40), 2)
    cv2.line(canvas, (left, height - bottom),
             (width - right, height - bottom), (40, 40, 40), 2)
    maximum = max(max(counts, default=0), 1)
    plot_w, plot_h = width - left - right, height - top - bottom
    points = []
    denominator = max(len(counts) - 1, 1)
    for index, count in enumerate(counts):
        x = left + round(index * plot_w / denominator)
        y = height - bottom - round(count * plot_h / maximum)
        points.append((x, y))
    if len(points) > 1:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)],
                      False, (35, 105, 220), 2, cv2.LINE_AA)
    elif points:
        cv2.circle(canvas, points[0], 4, (35, 105, 220), -1)
    cv2.putText(canvas, f"max={maximum}", (12, top + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    average = sum(counts) / max(len(counts), 1)
    cv2.putText(canvas, f"frames={len(counts)}  average={average:.2f}",
                (left, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def _heatmap(grouped: Dict[int, List[List[Any]]], task: str,
             frame_width: int, frame_height: int) -> np.ndarray:
    grid_h, grid_w = 68, 120
    values = np.zeros((grid_h, grid_w), dtype=np.float32)
    for rows in grouped.values():
        for row in rows:
            left, top, width, height = _bbox(row, task)
            x = min(grid_w - 1, max(0, int((left + width / 2) * grid_w /
                                             max(frame_width, 1))))
            y = min(grid_h - 1, max(0, int((top + height / 2) * grid_h /
                                             max(frame_height, 1))))
            values[y, x] += 1
    values = cv2.GaussianBlur(values, (0, 0), sigmaX=2.2)
    normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, (frame_width, frame_height),
                      interpolation=cv2.INTER_CUBIC)


def _annotated_video(images: Sequence[Tuple[int, Path]], grouped,
                     payload: Dict[str, Any], destination: Path,
                     frame_size: Tuple[int, int], fps: float) -> None:
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    if temporary.exists():
        temporary.unlink()
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create annotated video: {destination}")
    task = payload["task"]
    trails: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    try:
        for position, (frame_id, image_path) in enumerate(images, start=1):
            raw = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"cannot decode visualization image: {image_path}")
            for row in grouped.get(frame_id, []):
                left, top, width, height = _bbox(row, task)
                p1 = (round(left), round(top))
                p2 = (round(left + width), round(top + height))
                color = (40, 210, 40) if task == "detection" else (30, 180, 255)
                cv2.rectangle(raw, p1, p2, color, 2)
                if task == "detection":
                    label = f"bee {float(row[2]):.2f}"
                else:
                    track_id = int(row[1])
                    label = f"ID {track_id}"
                    center = (round(left + width / 2), round(top + height / 2))
                    trails[track_id].append(center)
                    trails[track_id] = trails[track_id][-60:]
                    if len(trails[track_id]) > 1:
                        cv2.polylines(raw, [np.asarray(trails[track_id], np.int32)],
                                      False, color, 2, cv2.LINE_AA)
                cv2.putText(raw, label, (p1[0], max(18, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(raw, f"frame {frame_id}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2, cv2.LINE_AA)
            writer.write(raw)
            if position % 100 == 0 or position == len(images):
                sys.stderr.write(
                    f"[visualization] {payload['sequence']}: "
                    f"{position}/{len(images)} frames\n")
                sys.stderr.flush()
    finally:
        writer.release()
    os.replace(temporary, destination)


def generate_visual_outputs(payload: Dict[str, Any],
                            images: Sequence[Tuple[int, Path]],
                            executable: str | Path,
                            fps: float = 24.0) -> Path:
    """Generate figures, video, CSV and HTML beside the four EXEs."""
    root = Path(executable).resolve().parent / "output"
    folders = {name: root / name for name in
               ("figures", "videos", "data", "reports")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    sequence, team_id = payload["sequence"], payload["team_id"]
    basename = f"{sequence}-{team_id}"
    grouped = _rows_by_frame(payload)
    counts = [len(grouped.get(frame_id, []))
              for frame_id, _ in images]

    first = cv2.imdecode(np.fromfile(str(images[0][1]), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot decode visualization image: {images[0][1]}")
    frame_height, frame_width = first.shape[:2]
    _write_image(folders["figures"] / f"{basename}-counts.png",
                 _count_chart(counts, f"{sequence} object count by frame"))
    _write_image(folders["figures"] / f"{basename}-heatmap.png",
                 _heatmap(grouped, payload["task"], frame_width, frame_height))

    csv_path = folders["data"] / f"{basename}.csv"
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "object_count"])
        writer.writerows((frame_id, count)
                         for (frame_id, _), count in zip(images, counts))
    os.replace(temporary_csv, csv_path)

    video_path = folders["videos"] / f"{basename}.mp4"
    _annotated_video(images, grouped, payload, video_path,
                     (frame_width, frame_height), fps)

    report_path = folders["reports"] / f"{basename}.html"
    average = sum(counts) / max(len(counts), 1)
    report = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">
<title>{html.escape(basename)} 分析报告</title><body>
<h1>{html.escape(sequence)} 分析报告</h1>
<ul><li>队伍 ID：{html.escape(team_id)}</li><li>总帧数：{len(images)}</li>
<li>记录数：{payload['num_records']}</li><li>平均每帧目标数：{average:.2f}</li>
<li>单帧最大目标数：{max(counts, default=0)}</li></ul>
<h2>数量变化</h2><img src=\"../figures/{basename}-counts.png\" style=\"max-width:100%\">
<h2>空间热力图</h2><img src=\"../figures/{basename}-heatmap.png\" style=\"max-width:100%\">
<p><a href=\"../videos/{basename}.mp4\">查看标注视频</a>　
<a href=\"../data/{basename}.csv\">下载逐帧统计 CSV</a></p></body></html>"""
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, report_path)
    return root
