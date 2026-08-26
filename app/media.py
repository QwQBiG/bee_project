"""Web 结果视频处理工具。"""

from pathlib import Path


def make_browser_preview(video_path: Path) -> Path:
    """从 OpenCV 的 mp4v 结果生成浏览器可播放的 VP8 WebM。"""
    import cv2

    preview_path = video_path.with_name(f"{video_path.stem}_preview.webm")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("无法读取结果视频")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(preview_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        preview_path.unlink(missing_ok=True)
        raise RuntimeError("当前 OpenCV 不支持生成浏览器预览视频")
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
    capture.release()
    writer.release()
    if frames == 0 or not preview_path.is_file() or preview_path.stat().st_size == 0:
        preview_path.unlink(missing_ok=True)
        raise RuntimeError("浏览器预览视频生成失败")
    return preview_path
