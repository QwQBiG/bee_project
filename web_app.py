"""面向普通用户的蜂群视频分析 Web 入口。"""

from __future__ import annotations

import json
import os
import shutil
import signal
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from main import apply_device_override, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
RUNTIME_ROOT = Path(
    os.environ.get("BEE_RUNTIME_DIR", PROJECT_ROOT / "runtime_results")
).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("BEE_MAX_UPLOAD_MB", "4096")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="蜂群视频智能分析", version="1.0.0")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# 模型推理通常会独占显卡或大量内存，默认一次只处理一个视频。
_executor = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("BEE_WORKERS", "1"))),
    thread_name_prefix="bee-analysis",
)


def _update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "job_dir"}


def _build_summary(mode: str, stats: dict[str, Any]) -> list[dict[str, str]]:
    seconds = float(stats.get("processing_time", 0) or 0)
    cards = [
        {"label": "分析场景", "value": "巢内" if mode == "inside" else "巢外"},
        {"label": "处理帧数", "value": f"{int(stats.get('total_frames', 0) or 0):,}"},
        {"label": "识别个体", "value": f"{int(stats.get('total_tracks', 0) or 0):,}"},
        {"label": "用时", "value": f"{seconds:.1f} 秒"},
    ]
    if mode == "outside":
        cards.extend([
            {"label": "进入事件", "value": str(stats.get("entry_events", 0))},
            {"label": "离开事件", "value": str(stats.get("exit_events", 0))},
        ])
    return cards


def _make_archive(output_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path != archive_path:
                archive.write(path, arcname=path.name)


def _make_browser_preview(video_path: Path) -> Path:
    """从 mp4v 结果生成 Chrome/Edge/Firefox 可播放的 VP8 WebM 预览。"""
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


def _run_analysis(job_id: str, video_path: Path, mode: str) -> None:
    job_dir = RUNTIME_ROOT / job_id
    output_dir = job_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_video = output_dir / f"{mode}_result.mp4"
    stats_path = output_dir / f"{mode}_stats.json"

    def on_progress(processed: int, total: int) -> None:
        progress = int(processed * 100 / total) if total > 0 else 0
        _update_job(
            job_id,
            status="processing",
            progress=max(1, min(progress, 99)),
            message=f"正在分析视频… {processed:,}/{total:,} 帧" if total else "正在分析视频…",
        )

    try:
        _update_job(job_id, status="processing", progress=1, message="正在加载识别模型…")
        config = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
        device = os.environ.get("BEE_DEVICE")
        config = apply_device_override(config, device)

        if mode == "outside":
            from inference.processor import OutsideHiveProcessor
            from visualization.outside_pollen_report import create_outside_pollen_report

            processor = OutsideHiveProcessor(config)
            stats = processor.process_video(
                str(video_path), str(result_video), False, on_progress
            )
            report_path = Path(create_outside_pollen_report(
                stats.get("pollen_analysis", {}),
                output_dir / "outside_pollen_report.html",
            ))
        else:
            from inference.processor import InsideHiveProcessor
            from visualization.inside_report import create_inside_report

            processor = InsideHiveProcessor(config)
            stats = processor.process_video(
                str(video_path), str(result_video), False, on_progress
            )
            report_path = Path(create_inside_report(
                stats.get("inside_metrics", {}),
                output_dir / "inside_analysis_report.html",
            ))

        _update_job(job_id, progress=99, message="正在生成浏览器预览视频…")
        preview_video = _make_browser_preview(result_video)

        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        archive_path = output_dir / "bee_analysis_results.zip"
        _make_archive(output_dir, archive_path)
        files = [
            {"label": "查看完整分析报告", "name": report_path.name, "kind": "report"},
            {"label": "下载统计数据（JSON）", "name": stats_path.name, "kind": "data"},
            {"label": "下载全部结果（ZIP）", "name": archive_path.name, "kind": "archive"},
        ]
        if result_video.exists() and result_video.stat().st_size > 0:
            files.insert(0, {"label": "下载标注视频", "name": result_video.name, "kind": "video"})

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="分析完成",
            summary=_build_summary(mode, stats),
            files=files,
            video=preview_video.name if preview_video.exists() else None,
        )
    except Exception as error:  # 后台任务必须把错误反馈给页面
        _update_job(
            job_id,
            status="failed",
            progress=0,
            message=f"分析失败：{error}",
        )
    finally:
        # 上传文件通常远大于结果数据；无论成功失败都及时释放磁盘空间。
        video_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((WEB_ROOT / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "preview": "vp8-webm"}


def _stop_process_after_response() -> None:
    """稍后发送中断信号，确保关闭提示能够先返回浏览器。"""
    time.sleep(0.6)
    os.kill(os.getpid(), signal.SIGINT)


@app.post("/api/shutdown")
def shutdown(request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else ""
    allow_remote = os.environ.get("BEE_ALLOW_REMOTE_SHUTDOWN", "").lower() in {
        "1", "true", "yes",
    }
    if client_host not in {"127.0.0.1", "::1", "testclient"} and not allow_remote:
        raise HTTPException(status_code=403, detail="只能在运行服务的电脑上关闭程序")
    threading.Thread(
        target=_stop_process_after_response,
        name="bee-shutdown",
        daemon=True,
    ).start()
    return {"status": "shutting_down", "message": "程序正在关闭"}


@app.post("/api/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(...),
    mode: str = Form(...),
) -> dict[str, Any]:
    if mode not in {"inside", "outside"}:
        raise HTTPException(status_code=400, detail="请选择巢内或巢外场景")
    extension = Path(video.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持此视频格式，请上传 MP4、AVI、MOV、MKV 或 WebM")

    job_id = uuid.uuid4().hex
    job_dir = RUNTIME_ROOT / job_id
    job_dir.mkdir(parents=True)
    upload_path = job_dir / f"input{extension}"
    size = 0
    try:
        with upload_path.open("wb") as target:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"视频过大，当前上限为 {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
                    )
                target.write(chunk)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await video.close()

    now = datetime.now(timezone.utc).isoformat()
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "视频已上传，等待开始分析…",
        "mode": mode,
        "filename": Path(video.filename or "video").name,
        "created_at": now,
        "summary": [],
        "files": [],
        "video": None,
        "job_dir": str(job_dir),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _executor.submit(_run_analysis, job_id, upload_path, mode)
    return _public_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或服务已重启")
        return _public_job(dict(job))


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_result_file(job_id: str, filename: str) -> FileResponse:
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="任务不存在")
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="文件名无效")
    path = RUNTIME_ROOT / job_id / "results" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="结果文件不存在")
    suffix = path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".zip": "application/zip",
    }.get(suffix)
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline" if suffix in {".mp4", ".webm", ".html"} else "attachment",
    )
