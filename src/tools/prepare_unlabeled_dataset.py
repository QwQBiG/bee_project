"""盘点未标注视频并生成可复现的抽帧计划。

默认只生成清单；传入 ``--extract`` 才会写出抽帧图片和空白标注文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from annotation.schema import VideoAnnotation


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立未标注蜜蜂视频的数据清单和抽帧计划")
    parser.add_argument("input", help="包含原始视频的目录")
    parser.add_argument("output", help="清单、抽帧和空白标注的输出目录")
    parser.add_argument("--scene", choices=["auto", "unknown", "inside_ir", "inside_visible",
                                            "outside_entrance"],
                        default="auto", help="场景；auto 根据路径名称推断")
    parser.add_argument("--frames-per-video", type=int, default=120,
                        help="每个视频均匀抽取的候选帧数")
    parser.add_argument("--split-policy", choices=["unassigned", "hash"], default="unassigned",
                        help="默认不自动划分，避免同源视频片段泄漏到不同集合")
    parser.add_argument("--extract", action="store_true", help="实际写出候选帧 JPG")
    return parser.parse_args()


def discover_videos(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {root}")
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def infer_scene(path: Path) -> str:
    text = str(path).lower()
    if any(token in text for token in ("outside", "entrance", "vnbee", "巢外", "门口")):
        return "outside_entrance"
    if any(token in text for token in ("inside", "ir", "infrared", "巢内", "红外")):
        return "inside_ir"
    return "unknown"


def file_digest(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def assign_split(video_id: str) -> str:
    value = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 70 else "val" if value < 85 else "test"


def frame_plan(frame_count: int, requested: int) -> List[int]:
    if frame_count <= 0 or requested <= 0:
        return []
    count = min(frame_count, requested)
    if count == 1:
        return [frame_count // 2]
    return sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})


def probe_video_with_ffprobe(path: Path) -> Dict:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("读取视频需要 opencv-python 或 ffprobe")
    command = [
        executable, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe 无法读取视频: {path}: {completed.stderr.strip()}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        fps = float(Fraction(stream["avg_frame_rate"]))
        raw_count = str(stream.get("nb_frames", ""))
        frame_count = int(raw_count) if raw_count.isdigit() else round(float(stream["duration"]) * fps)
        metadata = {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "frame_count": frame_count,
        }
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"ffprobe 返回无效元数据: {path}") from error
    if min(metadata.values()) <= 0:
        raise RuntimeError(f"视频元数据无效: {path}")
    return metadata


def probe_video(path: Path) -> Dict:
    try:
        import cv2
    except ImportError:
        return probe_video_with_ffprobe(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if min(metadata.values()) <= 0:
        raise RuntimeError(f"视频元数据无效: {path}")
    return metadata


def extract_frames(video: Path, indices: Iterable[int], output_dir: Path) -> int:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("抽帧需要安装 opencv-python") from error
    capture = cv2.VideoCapture(str(video))
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if ok and cv2.imwrite(str(output_dir / f"frame_{frame_index:08d}.jpg"), frame):
            written += 1
    capture.release()
    return written


def prepare(args: argparse.Namespace) -> Dict:
    input_root, output_root = Path(args.input).resolve(), Path(args.output).resolve()
    videos = discover_videos(input_root)
    if not videos:
        raise RuntimeError(f"没有找到支持的视频文件: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for video in videos:
        metadata = probe_video(video)
        digest = file_digest(video)
        video_id = f"{video.stem}-{digest[:12]}"
        scene = infer_scene(video) if args.scene == "auto" else args.scene
        indices = frame_plan(metadata["frame_count"], args.frames_per_video)
        split = assign_split(video_id) if args.split_policy == "hash" else "unassigned"
        record = {"video_id": video_id, "source_path": str(video), "sha256": digest,
                  "scene": scene, "split": split, **metadata,
                  "planned_frames": indices}
        if args.extract:
            record["extracted_frames"] = extract_frames(video, indices, output_root / "frames" / video_id)
            VideoAnnotation(video_id=video_id, source_path=str(video), scene=scene,
                            **metadata, metadata={"split": record["split"], "sha256": digest}).save(
                                output_root / "annotations" / f"{video_id}.json")
        records.append(record)
    manifest = {"manifest_version": "1.0", "input_root": str(input_root), "videos": records}
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    try:
        manifest = prepare(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"已建立清单：{len(manifest['videos'])} 个视频")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
