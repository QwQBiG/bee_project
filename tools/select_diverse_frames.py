"""从完整视频候选帧中选择质量可用且视觉差异较大的标注帧。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import exp, log1p, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class FrameFeature:
    frame_index: int
    vector: List[float]
    quality: float
    brightness: float
    contrast: float
    blur_score: float
    edge_density: float


def euclidean(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("特征维度不一致")
    return sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def select_diverse(features: List[FrameFeature], count: int,
                   min_frame_gap: int = 90, temporal_weight: float = 0.25) -> List[FrameFeature]:
    if count < 1 or min_frame_gap < 0 or temporal_weight < 0:
        raise ValueError("选择参数无效")
    if not features:
        return []
    ordered = sorted(features, key=lambda item: item.frame_index)
    frame_span = max(ordered[-1].frame_index - ordered[0].frame_index, 1)
    selected = [max(ordered, key=lambda item: (item.quality, -item.frame_index))]
    while len(selected) < min(count, len(ordered)):
        candidates = [item for item in ordered if item not in selected and
                      all(abs(item.frame_index - previous.frame_index) >= min_frame_gap
                          for previous in selected)]
        if not candidates:
            candidates = [item for item in ordered if item not in selected]
        if not candidates:
            break

        def score(item: FrameFeature) -> tuple:
            visual = min(euclidean(item.vector, previous.vector) for previous in selected)
            temporal = min(abs(item.frame_index - previous.frame_index) / frame_span
                           for previous in selected)
            combined = (visual + temporal_weight * temporal) * (0.4 + 0.6 * item.quality)
            return combined, item.quality, -item.frame_index

        selected.append(max(candidates, key=score))
    return sorted(selected, key=lambda item: item.frame_index)


def image_feature(frame_index: int, frame) -> FrameFeature:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    brightness = float(np.mean(small) / 255.0)
    contrast = float(min(np.std(small) / 96.0, 1.0))
    laplacian_variance = float(cv2.Laplacian(small, cv2.CV_64F).var())
    blur_score = float(min(log1p(laplacian_variance) / 8.0, 1.0))
    edges = cv2.Canny(small, 50, 150)
    edge_density = float(np.mean(edges > 0))
    histogram = cv2.calcHist([small], [0], None, [16], [0, 256]).flatten()
    histogram = histogram / max(float(histogram.sum()), 1.0)
    exposure = exp(-((brightness - 0.5) / 0.45) ** 4)
    quality = max(0.0, min(exposure * (0.35 + 0.35 * contrast + 0.30 * blur_score), 1.0))
    vector = histogram.tolist() + [brightness, contrast, blur_score, edge_density]
    return FrameFeature(frame_index, [float(value) for value in vector], quality,
                        brightness, contrast, blur_score, edge_density)


def analyze_video(video_path: Path, candidate_frames: Iterable[int]) -> tuple[List[FrameFeature], Dict[int, object]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    features, decoded = [], {}
    for frame_index in sorted(set(map(int, candidate_frames))):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        features.append(image_feature(frame_index, frame))
        decoded[frame_index] = frame
    capture.release()
    return features, decoded


def build_selection(manifest_path: Path, output_dir: Path, frames_per_video: int,
                    min_frame_gap: int, included_splits: set[str], extract: bool) -> Dict:
    if "test" in included_splits:
        raise ValueError("测试集默认封存；本工具不允许选择 test 帧")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = source.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("清单不包含 videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for record in videos:
        if record.get("split") not in included_splits:
            continue
        video_path = Path(record["source_path"])
        features, decoded = analyze_video(video_path, record.get("planned_frames", []))
        selected = select_diverse(features, frames_per_video, min_frame_gap)
        if len(selected) < frames_per_video:
            raise RuntimeError(f"{video_path} 只选出 {len(selected)}/{frames_per_video} 帧")
        selection = [{key: value for key, value in asdict(item).items() if key != "vector"}
                     for item in selected]
        selected_indices = [item.frame_index for item in selected]
        new_record = {**record, "planned_frames": selected_indices,
                      "selection_method": "visual_quality_farthest_point_v1",
                      "selection_features": selection}
        records.append(new_record)
        if extract:
            import cv2
            frame_dir = output_dir / "frames" / record["video_id"]
            frame_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in selected_indices:
                target = frame_dir / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(target), decoded[frame_index],
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"无法写出抽帧: {target}")
    manifest = {
        "manifest_version": "1.0",
        "source_manifest": str(manifest_path.resolve()),
        "excluded_splits": sorted({"train", "val", "test"} - included_splits),
        "frames_per_video": frames_per_video,
        "min_frame_gap": min_frame_gap,
        "videos": records,
    }
    (output_dir / "selected_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="选择质量可用且视觉多样的非测试标注帧")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--min-frame-gap", type=int, default=90)
    parser.add_argument("--include-split", action="append", choices=["train", "val"],
                        help="可重复指定；默认包含 train 和 val")
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    included = set(args.include_split or ["train", "val"])
    try:
        manifest = build_selection(args.manifest, args.output, args.frames_per_video,
                                   args.min_frame_gap, included, args.extract)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps({"videos": len(manifest["videos"]),
                      "frames": sum(len(item["planned_frames"]) for item in manifest["videos"]),
                      "excluded_splits": manifest["excluded_splits"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
