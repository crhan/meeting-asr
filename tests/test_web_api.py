"""Integration tests for the web API via FastAPI's TestClient.

These run the app with isolated XDG/store directories so they never touch real projects
or the global voiceprint/lexicon/config stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# The web extra (fastapi/uvicorn/sse-starlette) is optional; skip this whole module
# cleanly when it is not installed instead of aborting collection for the entire suite.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.web.server import create_app  # noqa: E402
from app.web.settings import WebSettings  # noqa: E402


def _settings(tmp_path: Path, *, token: str | None) -> WebSettings:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(exist_ok=True)
    return WebSettings(
        host="127.0.0.1",
        port=0,
        projects_dir=projects_dir,
        store_dir=tmp_path / "store",
        open_browser=False,
        token=token,
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with TestClient(create_app(_settings(tmp_path, token=None))) as test_client:
        yield test_client


@pytest.fixture
def token_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A client for a token-protected (non-loopback-style) bind."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with TestClient(create_app(_settings(tmp_path, token="s3cret"))) as test_client:
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


def test_lexicon_writes_honor_store_dir(client: TestClient, tmp_path: Path) -> None:
    """Lexicon writes must land under --store-dir, never the real XDG dictionary."""
    resp = client.post(
        "/api/lexicon/terms",
        json={"canonical": "iSee", "category": "product", "aliases": ["IC"]},
    )
    assert resp.status_code == 200

    # The configured store fixture is tmp_path/"store"; the lexicon db must live under it.
    store_db = tmp_path / "store" / "lexicon" / "lexicon.sqlite"
    assert store_db.is_file()
    # And the term is readable back from that same isolated store.
    terms = client.get("/api/lexicon/terms").json()["terms"]
    assert any(t["canonical"] == "iSee" for t in terms)


def test_correction_accept_records_into_store_dir_lexicon(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accepting a proposal must learn into the --store-dir lexicon, not the real XDG one."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections
    from app.lexicon_store import get_lexicon_db_path

    captured: dict = {}

    class FakeSummary:
        accepted = True
        change_count = 1
        learned_count = 1
        corrected_named_transcript_path = None

    def fake_accept(**kwargs):
        captured.update(kwargs)
        return FakeSummary()

    monkeypatch.setattr(corrections, "resolve_project_ref", lambda ref, _dir: tmp_path)
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )
    monkeypatch.setattr(corrections, "load_manifest", lambda root: object())
    monkeypatch.setattr(
        corrections, "load_speaker_mapping_for_correction", lambda root: {}
    )
    monkeypatch.setattr(corrections, "accept_correction_for_review", fake_accept)

    resp = client.post("/api/corrections/p-x/accept", json={"selected_indices": [0]})
    assert resp.status_code == 200
    # The lexicon db handed to the accept path must be the store-dir one, not None/XDG.
    assert captured["lexicon_db"] == get_lexicon_db_path(tmp_path / "store")


def _drain_job(client: TestClient, job_id: str) -> dict:
    """Consume a job's SSE stream to completion, then return its snapshot."""
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data:") and '"end"' in line:
                break
    return client.get(f"/api/jobs/{job_id}").json()


def test_pipeline_run_serializes_by_project_and_uses_store_lexicon(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pipeline job must be keyed by its project dir and learn into the store lexicon."""
    from types import SimpleNamespace

    import app.web.routers.pipeline as pipeline
    from app.lexicon_store import get_lexicon_db_path

    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake-media")
    project_dir = tmp_path / "projects" / "p-abc"
    captured: dict = {}

    monkeypatch.setattr(
        pipeline,
        "create_or_reuse_project",
        lambda *a, **k: SimpleNamespace(
            project_dir=project_dir, manifest=object(), created=True
        ),
    )

    def fake_workflow(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        return SimpleNamespace(
            project=SimpleNamespace(
                manifest=SimpleNamespace(project_id="p-abc"), project_dir=project_dir
            ),
            transcription=SimpleNamespace(detected_speaker_count=1, sentence_count=3),
            applied_mapping={},
            meeting_summary=None,
            correction_summary=None,
        )

    monkeypatch.setattr(pipeline, "_run_project_workflow", fake_workflow)

    resp = client.post("/api/pipeline/run", json={"input_path": str(media)})
    assert resp.status_code == 200
    snapshot = _drain_job(client, resp.json()["job_id"])

    # Serialized by the resolved project dir (same key inline routes use), not unkeyed.
    assert snapshot["project_id"] == str(project_dir)
    assert snapshot["status"] == "done"
    # The workflow reuses that project and learns corrections into the store lexicon.
    assert captured["project_dir"] == project_dir
    assert captured["lexicon_db"] == get_lexicon_db_path(tmp_path / "store")


def test_health_is_open_and_reports_auth_required(token_client: TestClient) -> None:
    resp = token_client.get("/api/health")
    assert resp.status_code == 200  # health is never gated
    assert resp.json()["auth_required"] is True


def test_token_required_routes_reject_without_credential(
    token_client: TestClient,
) -> None:
    assert token_client.get("/api/projects").status_code == 401
    assert token_client.get("/api/auth/check").status_code == 401


def test_token_accepted_via_bearer_header(token_client: TestClient) -> None:
    headers = {"Authorization": "Bearer s3cret"}
    assert token_client.get("/api/auth/check", headers=headers).status_code == 200
    assert token_client.get("/api/projects", headers=headers).status_code == 200


def test_token_accepted_via_query_param(token_client: TestClient) -> None:
    """SSE/audio transports can't set headers; ?token= must authenticate them."""
    assert token_client.get("/api/auth/check?token=s3cret").status_code == 200
    assert token_client.get("/api/projects?token=s3cret").status_code == 200
    # Wrong token is still rejected on the query path.
    assert token_client.get("/api/projects?token=nope").status_code == 401


def test_capture_pending_blocks_store_writes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a capture transaction is pending, store mutations return 409 (not silent)."""
    import app.web.routers.voiceprints as voiceprints
    from app.core.voiceprint_review_service import CaptureConflictError

    # Sanity: with nothing pending, creating a person succeeds against the isolated store.
    assert (
        client.post("/api/voiceprints/people", json={"name": "Alice"}).status_code
        == 200
    )

    # CRUD writes go through REGISTRY.run_store_write; a pending capture makes it raise.
    def boom(_fn):
        raise CaptureConflictError("pending capture")

    monkeypatch.setattr(voiceprints.REGISTRY, "run_store_write", boom)
    resp = client.post("/api/voiceprints/people", json={"name": "Bob"})
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


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


def _save_body(*, with_reassignment: bool) -> dict:
    body: dict = {
        "mapping": {"0": "Alice"},
        "person_mapping": {},
        "person_public_mapping": {},
        "ignored_speaker_ids": [],
        "reassignments": [],
    }
    if with_reassignment:
        body["reassignments"] = [
            {
                "sentence_id": 5,
                "begin_time_ms": 100,
                "end_time_ms": 200,
                "original_speaker_id": 1,
                "new_speaker_id": 0,
            }
        ]
    return body


def test_speaker_reassignment_blocked_when_capture_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reassignment write touches the global store, so it must 409 while a capture pends."""
    import app.web.routers.speakers as speakers
    from app.core.voiceprint_review_service import CaptureConflictError

    monkeypatch.setattr(speakers, "resolve_project_ref", lambda ref, _dir: tmp_path)

    def boom(_fn):
        raise CaptureConflictError("pending capture")

    monkeypatch.setattr(speakers.REGISTRY, "run_store_write", boom)

    resp = client.post(
        "/api/speakers/p-x/save", json=_save_body(with_reassignment=True)
    )
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


def test_speaker_naming_only_save_bypasses_store_section(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Naming-only saves never touch the global store, so they must not enter run_store_write."""
    import app.web.routers.speakers as speakers

    class FakeResult:
        mapping_path = tmp_path / "speaker_map.json"
        transcript_path = tmp_path / "t.txt"
        srt_path = tmp_path / "t.srt"
        reassignment = None

    monkeypatch.setattr(speakers, "resolve_project_ref", lambda ref, _dir: tmp_path)
    monkeypatch.setattr(speakers, "save_speaker_review", lambda *_a, **_k: FakeResult())

    def fail(_fn):  # must not be reached on the naming-only path
        raise AssertionError(
            "naming-only save must not enter the store critical section"
        )

    monkeypatch.setattr(speakers.REGISTRY, "run_store_write", fail)

    resp = client.post(
        "/api/speakers/p-x/save", json=_save_body(with_reassignment=False)
    )
    assert resp.status_code == 200
