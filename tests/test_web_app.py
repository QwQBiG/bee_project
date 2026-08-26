from io import BytesIO

from fastapi.testclient import TestClient

import app.main as web_main


client = TestClient(web_main.app)


def test_home_and_health_are_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "智能蜂场可视化分析平台" in response.text
    assert "关闭程序" in response.text

    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["preview"] == "vp8-webm"
    assert "runtime" in health


def test_create_task_validates_mode_and_files():
    bad_mode = client.post(
        "/api/tasks",
        data={"mode": "unknown"},
        files={"files": ("sample.mp4", BytesIO(b"video"), "video/mp4")},
    )
    assert bad_mode.status_code == 400

    no_files = client.post("/api/tasks", data={"mode": "inside"})
    assert no_files.status_code == 422


def test_shutdown_endpoint_schedules_process_stop(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            started.append(kwargs)

        def start(self):
            started.append("started")

    monkeypatch.setattr(web_main.threading, "Thread", FakeThread)
    response = client.post("/api/shutdown")
    assert response.status_code == 200
    assert response.json()["status"] == "shutting_down"
    assert started[-1] == "started"
