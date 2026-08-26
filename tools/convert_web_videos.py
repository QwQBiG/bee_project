"""为已有结果视频生成浏览器可直接播放的 VP8 WebM 预览。"""

import argparse
from pathlib import Path

from app.media import make_browser_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("runtime_results"))
    args = parser.parse_args()

    videos = sorted(args.root.rglob("*_result.mp4"))
    for video in videos:
        print(f"Converting {video}")
        make_browser_preview(video)
    print(f"Converted {len(videos)} video(s).")


if __name__ == "__main__":
    main()
