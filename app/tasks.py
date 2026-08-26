"""后台任务管理（持久化 + 真实推理）。

- 任务记录持久化到 app/task_store.json，服务重启后自动恢复；
  恢复时发现 pending/running 的任务会标记为 failed（"服务重启中断"）。
- 处理过程调用 inference/processor.py 的真实推理：
    启动时由 app/engine.py 加载模型单例，
    每个任务开始前 reset_processor 重置内部累积状态，
    逐帧循环通过 progress_callback 上报进度。
- 轨迹质量模块未开发，结果中的 track_quality 为占位数据（见 adapters.py）。
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from visualization.inside_report import create_inside_report
from visualization.outside_pollen_report import create_outside_pollen_report

from .adapters import adapt_inside, adapt_multi, adapt_outside
from .engine import get_processors, reset_processor
from .media import make_browser_preview

logger = logging.getLogger("app.tasks")

# 任务状态
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STORE_PATH = BASE_DIR / "task_store.json"
# 进度写盘节流间隔（秒），避免逐帧落盘造成 IO 压力
PERSIST_INTERVAL = 5.0


class TaskManager:
    """持久化任务管理器：创建任务、后台推理处理、查询进度与结果。"""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        # 两个任务不得同时重置并使用同一组模型/GPU。
        self._inference_lock = threading.Lock()
        self._last_write: dict[str, float] = {}
        self.load()

    # ---------- 对外接口 ----------

    def create(self, mode: str, files: list[dict]) -> str:
        """创建任务并立即在后台启动处理线程。"""
        task_id = uuid.uuid4().hex[:12]
        task = {
            "id": task_id,
            "mode": mode,
            "files": files,          # [{name, path, size}]
            "status": PENDING,
            "progress": 0,
            "message": "排队中",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._tasks[task_id] = task
        self._persist()

        threading.Thread(target=self._run, args=(task_id,), daemon=True).start()
        return task_id

    def get(self, task_id: str) -> Optional[dict]:
        """获取任务（含进度/结果），不存在返回 None。"""
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None

    def list(self) -> list[dict]:
        """按创建时间倒序返回任务列表（不含完整结果，控制体积）。"""
        with self._lock:
            items = [
                {k: v for k, v in t.items() if k != "result"}
                for t in self._tasks.values()
            ]
        items.sort(key=lambda t: t["created_at"], reverse=True)
        return items

    # ---------- 持久化 ----------

    def load(self) -> None:
        """启动时从磁盘恢复任务记录。

        未完成任务（排队中/运行中）因服务重启已中断，统一标记为 failed，
        并保留记录供用户查看与重新上传。
        """
        if not STORE_PATH.exists():
            return
        try:
            with self._io_lock:
                data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            restored = 0
            with self._lock:
                for t in data:
                    if not isinstance(t, dict) or not t.get("id"):
                        continue
                    if t.get("status") in (PENDING, RUNNING):
                        t["status"] = FAILED
                        t["message"] = "已中断"
                        t["error"] = "服务重启，任务中断，请重新上传视频"
                        t["finished_at"] = now
                    self._tasks[t["id"]] = t
                    restored += 1
            logger.info("已从存储恢复 %d 条任务记录（中断任务已标记为失败）", restored)
        except Exception as exc:  # noqa: BLE001
            logger.warning("任务存储加载失败（忽略，从空列表开始）: %s", exc)

    def _persist(self) -> None:
        """原子写盘：tmp 文件 + rename，避免写一半损坏存储。"""
        try:
            with self._io_lock:
                with self._lock:
                    snapshot = json.dumps(
                        list(self._tasks.values()), ensure_ascii=False, indent=1)
                tmp = STORE_PATH.with_suffix(".tmp")
                tmp.write_text(snapshot, encoding="utf-8")
                tmp.replace(STORE_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("任务存储写入失败: %s", exc)

    # ---------- 内部实现 ----------

    def _update(self, task_id: str, **fields) -> None:
        """更新任务字段；状态变更立即落盘，进度更新按间隔节流落盘。"""
        need_persist = False
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            old_status = task.get("status")
            task.update(fields)
            if fields.get("status") != old_status:
                need_persist = True
            else:
                now = time.time()
                if now - self._last_write.get(task_id, 0) >= PERSIST_INTERVAL:
                    need_persist = True
                    self._last_write[task_id] = now
        if need_persist:
            self._persist()

    def _run(self, task_id: str) -> None:
        """后台真实推理：按模式调用处理器，推进进度并写入结果。"""
        try:
            with self._lock:
                mode = self._tasks[task_id]["mode"]
                files = self._tasks[task_id]["files"]
            if not files:
                raise ValueError("未收到有效视频文件")

            with self._inference_lock:
                processors = get_processors()
                self._update(task_id, status=RUNNING, progress=1, message="引擎初始化…")

                if mode == "multi":
                    if len(files) < 2:
                        raise ValueError("双路同步分析需要巢外与巢内两个视频")
                    outside_result = self._run_one(task_id, "outside", files[0], processors, 1, 50)
                    inside_result = self._run_one(task_id, "inside", files[1], processors, 50, 99)
                    result = adapt_multi(outside_result, inside_result)
                else:
                    result = self._run_one(task_id, mode, files[0], processors, 1, 99)

            self._update(
                task_id,
                status=DONE,
                progress=100,
                message="分析完成",
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("任务 %s 处理失败", task_id)
            self._update(
                task_id,
                status=FAILED,
                message="分析失败",
                error=str(exc),
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

    def _run_one(self, task_id: str, mode: str, file_info: dict,
                 processors: dict, progress_lo: int, progress_hi: int) -> dict:
        """处理单个视频：重置状态 → 真实推理（带进度回调）→ 生成报告。"""
        processor = processors[mode]
        reset_processor(processor)

        video_path = Path(file_info["path"])
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        annotated_path = UPLOAD_DIR / f"{task_id}_{mode}.mp4"
        report_path = UPLOAD_DIR / f"{task_id}_{mode}_report.html"

        lo, hi = progress_lo, progress_hi

        def progress_callback(frame: int, total: int) -> None:
            percent = lo + int(frame / max(total, 1) * (hi - lo))
            self._update(task_id, status=RUNNING, progress=percent,
                         message=f"推理中（{mode}，{percent}%）…")

        raw = processor.process_video(
            str(video_path), str(annotated_path), progress_callback=progress_callback)

        self._update(task_id, status=RUNNING, progress=progress_hi,
                     message="生成浏览器预览…")
        preview_path = make_browser_preview(annotated_path)

        # 生成详细 HTML 分析报告
        if mode == "outside":
            create_outside_pollen_report(raw["pollen_analysis"], str(report_path))
        else:
            create_inside_report(raw["inside_metrics"], str(report_path))

        annotated_url = f"/uploads/{preview_path.name}"
        report_url = f"/uploads/{report_path.name}"
        if mode == "outside":
            return adapt_outside(raw, video_path, annotated_url, report_url)
        return adapt_inside(raw, video_path, annotated_url, report_url)


# 全局单例
task_manager = TaskManager()
