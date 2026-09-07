"""Extract every video frame using the competition's sequential JPG names."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video: Path, output: Path, prefix: str, quality: int) -> int:
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*.jpg"))
    if existing:
        raise RuntimeError(
            f"output directory already contains {len(existing)} JPG files: {output}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            written += 1
            destination = output / f"{prefix}_{written:04d}.jpg"
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not encoded_ok:
                raise RuntimeError(f"cannot encode frame {written}")
            encoded.tofile(destination)
    finally:
        capture.release()
    if written == 0:
        raise RuntimeError("video contains no readable frames")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", choices=("im", "frame"), default="im")
    parser.add_argument("--quality", type=int, choices=range(1, 101), default=95)
    args = parser.parse_args()
    count = extract(args.video.resolve(), args.output.resolve(),
                    args.prefix, args.quality)
    print(f"extracted {count} frames to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
