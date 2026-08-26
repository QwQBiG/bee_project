"""通过 Web API 跑一个视频，用于部署前端到端冒烟验证。"""

import argparse
import time
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as web_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--mode", choices=("inside", "outside"), default="outside")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    client = TestClient(web_main.app)
    with args.video.open("rb") as source:
        response = client.post(
            "/api/tasks",
            data={"mode": args.mode},
            files={"files": (args.video.name, source, "video/mp4")},
        )
    response.raise_for_status()
    job = response.json()
    task_id = job["task_id"]
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/tasks/{task_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.5)

    print(f"status={job['status']} progress={job['progress']} message={job['message']}")
    print("files=" + ", ".join(item["name"] for item in job.get("files", [])))
    if job["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
