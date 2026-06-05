"""Integration tests for the web API via FastAPI's TestClient.

These run the app with isolated XDG/store directories so they never touch real projects
or the global voiceprint/lexicon/config stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.web.server import create_app
from app.web.settings import WebSettings


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    settings = WebSettings(
        host="127.0.0.1",
        port=0,
        projects_dir=projects_dir,
        store_dir=tmp_path / "store",
        open_browser=False,
        token=None,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "auth_required": False}


def test_projects_empty(client: TestClient) -> None:
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json()["projects"] == []


def test_unknown_project_is_404(client: TestClient) -> None:
    resp = client.get("/api/projects/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"


def test_ping_job_streams_progress_and_completes(client: TestClient) -> None:
    job_id = client.post("/api/jobs/ping").json()["job_id"]

    events: list[dict] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))

    kinds = [e.get("type") for e in events]
    assert "progress" in kinds
    assert {"type": "status", "status": "done"} in events
    assert events[-1] == {"type": "end"}

    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["status"] == "done"
    assert snapshot["result"] == {"pings": 5}


def test_config_masks_secrets_in_isolated_store(client: TestClient) -> None:
    body = client.get("/api/config").json()
    by_name = {k["name"]: k for k in body["keys"]}
    api_key = by_name["dashscope.api_key"]
    assert api_key["secret"] is True
    assert api_key["is_set"] is False  # isolated config has nothing set


def test_doctor_returns_checks(client: TestClient) -> None:
    body = client.get("/api/doctor").json()
    assert "checks" in body
    names = {c["name"] for c in body["checks"]}
    assert "python" in names
    assert "ffmpeg" in names


def test_speaker_save_marshals_decision(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The save endpoint must translate the JSON body into the exact service primitives."""
    import app.web.routers.speakers as speakers

    captured: dict = {}

    class FakeResult:
        mapping_path = tmp_path / "speaker_map.json"
        transcript_path = tmp_path / "t.txt"
        srt_path = tmp_path / "t.srt"
        reassignment = None

    def fake_save(project_dir, **kwargs):
        captured["project_dir"] = project_dir
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(speakers, "resolve_project_ref", lambda ref, _dir: tmp_path)
    monkeypatch.setattr(speakers, "save_speaker_review", fake_save)

    resp = client.post(
        "/api/speakers/p-x/save",
        json={
            "mapping": {"0": "Alice", "1": "Bob"},
            "person_mapping": {"0": 7},
            "person_public_mapping": {"0": "vpp-abc"},
            "ignored_speaker_ids": [2],
            "reassignments": [
                {
                    "sentence_id": 5,
                    "begin_time_ms": 100,
                    "end_time_ms": 200,
                    "original_speaker_id": 1,
                    "new_speaker_id": 0,
                }
            ],
        },
    )

    assert resp.status_code == 200
    # Speaker-id keys are parsed to ints; reassignment becomes a spec.
    assert captured["mapping"] == {0: "Alice", 1: "Bob"}
    assert captured["person_mapping"] == {0: 7}
    assert captured["person_public_mapping"] == {0: "vpp-abc"}
    assert list(captured["ignored_speaker_ids"]) == [2]
    spec = captured["reassignments"][0]
    assert spec.sentence_id == 5
    assert spec.new_speaker_id == 0
    assert spec.original_speaker_id == 1
    assert resp.json()["reassigned_count"] == 1
