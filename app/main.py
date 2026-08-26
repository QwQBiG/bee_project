"""智能蜂场可视化分析平台 - FastAPI 入口。

运行方式：
    双击 start_web.bat
    或 python -m uvicorn app.main:app --port 8000
"""

import json
import logging
import os
import signal
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import init_engine
from .tasks import task_manager

logger = logging.getLogger("app.main")

DONE_STATUS = "done"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
RUNTIME_STATUS_PATH = BASE_DIR.parent / "runtime_environment.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="智能蜂场可视化分析平台",
    description="上传巢外/巢内蜂群视频，获取行为量化指标、异常识别与健康评估报告。",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """静态资源禁用缓存，避免浏览器加载旧版 JS/CSS 导致功能不一致。"""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.on_event("startup")
def startup_load_engine() -> None:
    """服务启动时加载推理模型（常驻内存）。

    模型加载失败不阻断服务启动（界面仍可用），
    具体任务执行时会以失败状态返回错误信息。
    """
    try:
        init_engine()
    except Exception as exc:  # noqa: BLE001
        logger.error("推理引擎启动失败，上传任务将无法执行: %s", exc)


# ---------- 页面路由 ----------

@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/result", include_in_schema=False)
@app.get("/result/{task_id}", include_in_schema=False)
def result_page(task_id: str = "") -> FileResponse:
    return FileResponse(STATIC_DIR / "result.html")


def _runtime_status() -> dict:
    try:
        return json.loads(RUNTIME_STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"device": os.environ.get("BEE_DEVICE", "auto")}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "preview": "vp8-webm", "runtime": _runtime_status()}


def _stop_process_after_response() -> None:
    time.sleep(0.6)
    os.kill(os.getpid(), signal.SIGINT)


@app.post("/api/shutdown")
def shutdown(request: Request) -> dict:
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


# ---------- 任务 API ----------

@app.post("/api/tasks")
async def create_task(
    mode: str = Form(...),
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    """上传视频并创建分析任务。

    Args:
        mode: outside | inside | multi
        files: 一个或多个视频文件（multi 模式需按顺序：巢外、巢内）
    """
    if mode not in ("outside", "inside", "multi"):
        return JSONResponse({"error": f"未知模式: {mode}"}, status_code=400)
    if not files:
        return JSONResponse({"error": "未上传任何视频文件"}, status_code=400)

    saved = []
    for upload in files:
        if not upload.filename:
            continue
        # 防路径穿越，仅保留文件名；加随机前缀避免同名覆盖
        safe_name = Path(upload.filename).name
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        with dest.open("wb") as f:
            f.write(await upload.read())
        saved.append({
            "name": safe_name,
            "path": str(dest),
            "size": dest.stat().st_size,
        })

    if not saved:
        return JSONResponse({"error": "文件保存失败"}, status_code=400)

    task_id = task_manager.create(mode, saved)
    return JSONResponse({"task_id": task_id, "mode": mode})


@app.get("/api/tasks")
def list_tasks() -> JSONResponse:
    """任务列表（倒序）。"""
    return JSONResponse({"tasks": task_manager.list()})


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> JSONResponse:
    """任务详情（含进度、状态、结果）。"""
    task = task_manager.get(task_id)
    if task is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return JSONResponse(task)


@app.get("/api/tasks/{task_id}/result")
def get_result(task_id: str) -> JSONResponse:
    """获取分析结果（仅完成时可用）。"""
    task = task_manager.get(task_id)
    if task is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    if task["status"] != DONE_STATUS:
        return JSONResponse({"error": "任务尚未完成"}, status_code=409)
    return JSONResponse(task["result"])


@app.get("/api/tasks/{task_id}/download/stats", response_model=None)
def download_stats(task_id: str):
    """下载统计 JSON。"""
    task = task_manager.get(task_id)
    if task is None or task["status"] != DONE_STATUS or not task["result"]:
        return JSONResponse({"error": "结果不存在"}, status_code=404)
    json_path = UPLOAD_DIR / f"{task_id}_stats.json"
    json_path.write_text(
        json.dumps(task["result"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return FileResponse(
        json_path, media_type="application/json", filename=f"{task_id}_stats.json"
    )
