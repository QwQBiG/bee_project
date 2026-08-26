from io import BytesIO

from fastapi.testclient import TestClient

import web_app


client = TestClient(web_app.app)


def test_home_and_health_are_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "蜂群视频智能分析" in response.text
    assert client.get("/api/health").json() == {
        "status": "ok",
        "preview": "vp8-webm",
    }


def test_summary_contains_common_and_outside_metrics():
    summary = web_app._build_summary("outside", {
        "total_frames": 125,
        "total_tracks": 8,
        "processing_time": 2.5,
        "entry_events": 3,
        "exit_events": 2,
    })
    values = {item["label"]: item["value"] for item in summary}
    assert values["分析场景"] == "巢外"
    assert values["处理帧数"] == "125"
    assert values["进入事件"] == "3"


def test_create_job_validates_mode_and_extension():
    bad_mode = client.post(
        "/api/jobs",
        data={"mode": "unknown"},
        files={"video": ("sample.mp4", BytesIO(b"video"), "video/mp4")},
    )
    assert bad_mode.status_code == 400

    bad_extension = client.post(
        "/api/jobs",
        data={"mode": "inside"},
        files={"video": ("sample.txt", BytesIO(b"text"), "text/plain")},
    )
    assert bad_extension.status_code == 400


def test_shutdown_endpoint_schedules_process_stop(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            started.append(kwargs)

        def start(self):
            started.append("started")

    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    response = client.post("/api/shutdown")
    assert response.status_code == 200
    assert response.json()["status"] == "shutting_down"
    assert started[-1] == "started"
