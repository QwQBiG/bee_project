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


def _count_trend(counts: Sequence[int]) -> Tuple[float, str]:
    """Compare the first and last third of a sequence."""
    window = max(1, len(counts) // 3)
    early = sum(counts[:window]) / window
    late = sum(counts[-window:]) / window
    change = (late - early) / max(early, 1.0)
    if change <= -0.35:
        label = "明显下降"
    elif change >= 0.35:
        label = "明显上升"
    else:
        label = "总体平稳"
    return change, label


def _pollen_score(image: np.ndarray, bbox: Sequence[float],
                  lower: np.ndarray, upper: np.ndarray) -> float:
    left, top, width, height = [int(round(value)) for value in bbox]
    top += max(height // 2, 0)
    height = max(height // 2, 1)
    crop = image[max(0, top):min(image.shape[0], top + height),
                 max(0, left):min(image.shape[1], left + width)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask) / mask.size)


def _outside_prediction(images: Sequence[Tuple[int, Path]], grouped,
                        payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    options = config.get("pollen_analysis", {})
    lower = np.asarray(options.get("hsv_lower", [15, 70, 70]), dtype=np.uint8)
    upper = np.asarray(options.get("hsv_upper", [45, 255, 255]), dtype=np.uint8)
    min_color = float(options.get("min_color_ratio", 0.12))
    min_ratio = float(options.get("min_pollen_ratio", 0.15))
    min_records = int(options.get("min_prediction_records", 20))
    confidence_threshold = float(
        config.get("detector", {}).get("outside", {}).get("conf", 0.25))
    task = payload["task"]
    track_samples: Dict[int, List[bool]] = defaultdict(list)
    observations: List[bool] = []
    for frame_id, image_path in images:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            continue
        for row in grouped.get(frame_id, []):
            if task == "detection" and float(row[2]) < confidence_threshold:
                continue
            positive = _pollen_score(
                image, _bbox(row, task), lower, upper) >= min_color
            if task == "tracking":
                track_samples[int(row[1])].append(positive)
            else:
                observations.append(positive)
    if task == "tracking":
        min_samples = int(options.get("min_samples", 3))
        positive_share = float(options.get("positive_sample_ratio", 0.60))
        usable = [samples for samples in track_samples.values()
                  if len(samples) >= min_samples]
        observations = [sum(samples) / len(samples) >= positive_share
                        for samples in usable]
        unit = "可分析轨迹"
    else:
        unit = "有效检测目标"
    total = len(observations)
    positives = sum(observations)
    ratio = positives / total if total else None
    if total < min_records:
        status, confidence = "unknown", "不足"
        message = "有效样本不足，当前仅展示统计结果，暂不形成稳定的营养趋势判断。"
    elif ratio is not None and ratio < min_ratio:
        status, confidence = "warning", "较低"
        message = "携粉候选比例低于设定阈值，存在花粉采集活跃度偏低的可能，建议结合天气、花源和连续时段复核。"
    else:
        status, confidence = "normal", "较低"
        message = "本时段携粉候选比例未低于设定阈值，暂未发现明显的花粉采集不足趋势。"
    return {
        "status": status,
        "label": {"normal": "暂未发现明显风险", "warning": "建议重点复核",
                  "unknown": "数据不足"}[status],
        "confidence": confidence,
        "message": message,
        "metrics": [(unit, total), ("携粉候选数", positives),
                    ("携粉候选比例", "暂无" if ratio is None else f"{ratio:.1%}")],
        "advice": "若连续多个可比时间窗均偏低，请检查外界花源与天气；确认花源不足后，再由养蜂人员决定是否补饲花粉饼。",
        "limitation": "该预测使用后足区域 HSV 颜色候选，不是经专项标注验证的花粉团模型，不能替代现场检查。",
    }


def _inside_prediction(payload: Dict[str, Any], grouped,
                       counts: Sequence[int]) -> Dict[str, Any]:
    change, trend = _count_trend(counts)
    records = int(payload.get("num_records", 0))
    frames = int(payload.get("num_frames", 0))
    metrics: List[Tuple[str, Any]] = [
        ("目标数量趋势", trend),
        ("首尾时段变化", f"{change:+.1%}"),
    ]
    warning_reasons: List[str] = []
    if change <= -0.35:
        warning_reasons.append("目标数量出现明显下降")
    if payload["task"] == "tracking":
        tracks: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        for frame_id, rows in grouped.items():
            for row in rows:
                left, top, width, height = _bbox(row, "tracking")
                tracks[int(row[1])].append(
                    (frame_id, left + width / 2, top + height / 2))
        speeds = []
        for points in tracks.values():
            points.sort()
            for previous, current in zip(points, points[1:]):
                delta = max(current[0] - previous[0], 1)
                speeds.append((((current[1] - previous[1]) ** 2 +
                                (current[2] - previous[2]) ** 2) ** 0.5) / delta)
        mean_speed = sum(speeds) / len(speeds) if speeds else None
        metrics.extend([
            ("独立轨迹数", len(tracks)),
            ("平均位移速度", "暂无" if mean_speed is None else f"{mean_speed:.2f} 像素/帧"),
        ])
        if mean_speed is not None and len(speeds) >= 20 and mean_speed < 0.5:
            warning_reasons.append("群体平均位移速度偏低")
    if frames < 10 or records < 20:
        status, confidence = "unknown", "不足"
        message = "有效帧数或目标记录不足，暂不能形成稳定的蜂群行为预警。"
    elif warning_reasons:
        status, confidence = "warning", "中等"
        message = "；".join(warning_reasons) + "，建议结合温度、季节和现场开箱情况复核。"
    else:
        status, confidence = "normal", "中等"
        message = "目标数量与活动趋势未触发当前阈值，暂未发现明显异常信号。"
    return {
        "status": status,
        "label": {"normal": "暂未发现明显异常", "warning": "建议重点复核",
                  "unknown": "数据不足"}[status],
        "confidence": confidence,
        "message": message,
        "metrics": metrics,
        "advice": "出现连续低活跃、数量骤降或局部聚集时，请检查巢温、通风、饲料储备、蜂王状态和病虫害迹象。",
        "limitation": "该预警依据检测框数量和轨迹位移，不包含温湿度、蜂王状态或实验室病原检测，不能替代养蜂人员诊断。",
    }


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


def _analysis_html(prediction: Dict[str, Any]) -> str:
    metrics = "".join(
        "<li><span>{}</span><b>{}</b></li>".format(
            html.escape(str(name)), html.escape(str(value)))
        for name, value in prediction["metrics"]
    )
    status = html.escape(str(prediction["status"]))
    return f"""<section class="assessment {status}">
<h2>自动分析与预警：{html.escape(str(prediction['label']))}</h2>
<p><b>预测置信度：</b>{html.escape(str(prediction['confidence']))}</p>
<ul class="metrics">{metrics}</ul>
<p>{html.escape(str(prediction['message']))}</p>
<p><b>处理建议：</b>{html.escape(str(prediction['advice']))}</p>
<small>{html.escape(str(prediction['limitation']))}</small>
</section>"""


def generate_visual_outputs(payload: Dict[str, Any],
                            images: Sequence[Tuple[int, Path]],
                            executable: str | Path,
                            fps: float = 24.0,
                            config: Dict[str, Any] | None = None) -> Path:
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
    runtime_config = config or {}
    if sequence.startswith("Outside-"):
        prediction = _outside_prediction(images, grouped, payload, runtime_config)
    else:
        prediction = _inside_prediction(payload, grouped, counts)
    analysis = _analysis_html(prediction)
    report = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{html.escape(basename)} 分析与预警报告</title>
<style>
body{{font-family:\"Microsoft YaHei\",sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#173128;background:#f6f7f2}}
h1{{font-size:34px}} .summary,.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;padding:0}}
.summary li,.metrics li{{list-style:none;background:white;border:1px solid #dce5dd;border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:8px}}
.summary b,.metrics b{{font-size:24px;color:#205f46}} .assessment{{margin:28px 0;padding:24px;border-radius:14px;background:white;border-top:7px solid #7c8b82}}
.assessment.warning{{border-color:#d77b20}} .assessment.normal{{border-color:#27865c}} .assessment.unknown{{border-color:#87928c}}
img{{max-width:100%;border-radius:12px;background:white}} a{{color:#12624a}} small{{color:#66726d}}
</style></head><body>
<h1>{html.escape(sequence)} 分析与预警报告</h1>
<ul class=\"summary\"><li><span>队伍 ID</span><b>{html.escape(team_id)}</b></li>
<li><span>总帧数</span><b>{len(images)}</b></li><li><span>记录数</span><b>{payload['num_records']}</b></li>
<li><span>平均每帧目标数</span><b>{average:.2f}</b></li><li><span>单帧最大目标数</span><b>{max(counts, default=0)}</b></li></ul>
{analysis}
<h2>数量变化</h2><img src=\"../figures/{basename}-counts.png\" alt=\"数量变化图\">
<h2>空间热力图</h2><img src=\"../figures/{basename}-heatmap.png\" alt=\"空间热力图\">
<p><a href=\"../videos/{basename}.mp4\">查看标注视频</a>　
<a href=\"../data/{basename}.csv\">下载逐帧统计 CSV</a></p></body></html>"""
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, report_path)
    return root
