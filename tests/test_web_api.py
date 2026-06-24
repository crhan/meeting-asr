"""Integration tests for the web API via FastAPI's TestClient.

These run the app with isolated XDG/store directories so they never touch real projects
or the global voiceprint/lexicon/config stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# FastAPI is a default dependency; keep importorskip so a broken local environment fails
# cleanly at collection instead of aborting the entire suite.
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
    # base_url gives a loopback Host header: a tokenless bind now requires one (DNS-rebinding
    # guard), and TestClient's default "testserver" Host would otherwise be rejected.
    with TestClient(
        create_app(_settings(tmp_path, token=None)), base_url="http://127.0.0.1:8765"
    ) as test_client:
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
    # 127.0.0.1 bind: loopback, no token -> is_local True, auth_required False.
    assert resp.json() == {
        "status": "ok",
        "auth_required": False,
        "is_local": True,
    }


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
    # Neutral synthetic term -- real product/term mappings are user data (AGENTS.md), never
    # source fixtures.
    resp = client.post(
        "/api/lexicon/terms",
        json={"canonical": "Zigzag", "category": "product", "aliases": ["Zigzaq"]},
    )
    assert resp.status_code == 200

    # The configured store fixture is tmp_path/"store"; the lexicon db must live under it.
    store_db = tmp_path / "store" / "lexicon" / "lexicon.sqlite"
    assert store_db.is_file()
    # And the term is readable back from that same isolated store.
    terms = client.get("/api/lexicon/terms").json()["terms"]
    assert any(t["canonical"] == "Zigzag" for t in terms)


def test_correction_accept_records_into_store_dir_lexicon(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accepting a proposal must learn into the --store-dir lexicon, not the real XDG one."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections
    from app.lexicon_store import get_lexicon_db_path

    captured: dict = {}
    change = SimpleNamespace(
        sentence_id=1,
        original_text="a",
        corrected_text="b",
        change_type="x",
        reason="r",
    )
    proposal = SimpleNamespace(model="m", proposed_changes=[change])

    class FakeSummary:
        accepted = True
        change_count = 1
        learned_count = 1
        corrected_named_transcript_path = None

    def fake_accept(**kwargs):
        captured.update(kwargs)
        return FakeSummary()

    monkeypatch.setattr(
        corrections, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )
    monkeypatch.setattr(corrections, "load_manifest", lambda root: object())
    monkeypatch.setattr(
        corrections, "load_speaker_mapping_for_correction", lambda root: {}
    )
    monkeypatch.setattr(
        corrections, "load_correction_proposal", lambda paths, p: proposal
    )
    monkeypatch.setattr(corrections, "accept_correction_for_review", fake_accept)

    resp = client.post(
        "/api/corrections/p-x/accept",
        json={
            "selected_indices": [0],
            "proposal_id": corrections._proposal_id(proposal),
        },
    )
    assert resp.status_code == 200
    # The lexicon db handed to the accept path must be the store-dir one, not None/XDG.
    assert captured["lexicon_db"] == get_lexicon_db_path(tmp_path / "store")


def test_correction_accept_requires_proposal_id(client: TestClient) -> None:
    """Accept must be bound to the proposal the user reviewed."""
    resp = client.post("/api/corrections/p-x/accept", json={"selected_indices": [0]})
    assert resp.status_code == 422


def test_correction_accept_refuses_stale_proposal_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accept must refuse (409) when the reviewed proposal_id no longer matches the on-disk
    proposal -- a regenerate (another tab/CLI) would otherwise apply the reviewed indices to a
    different proposal and write the wrong subset."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections

    change = SimpleNamespace(
        sentence_id=1,
        original_text="a",
        corrected_text="b",
        change_type="x",
        reason="r",
    )
    proposal = SimpleNamespace(model="m", proposed_changes=[change])
    expected_id = corrections._proposal_id(proposal)

    accepted: dict = {}

    class FakeSummary:
        accepted = True
        change_count = 1
        learned_count = 1
        corrected_named_transcript_path = None

    monkeypatch.setattr(
        corrections, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )
    monkeypatch.setattr(corrections, "load_manifest", lambda root: object())
    monkeypatch.setattr(
        corrections, "load_speaker_mapping_for_correction", lambda root: {}
    )
    monkeypatch.setattr(
        corrections, "load_correction_proposal", lambda paths, p: proposal
    )

    def fake_accept(**kwargs):
        accepted["called"] = True
        return FakeSummary()

    monkeypatch.setattr(corrections, "accept_correction_for_review", fake_accept)

    # Stale id (proposal regenerated since review) -> 409, and accept never runs.
    stale = client.post(
        "/api/corrections/p-x/accept",
        json={"selected_indices": [0], "proposal_id": "0000staleid0000"},
    )
    assert stale.status_code == 409
    assert "called" not in accepted

    # Matching id -> accepted.
    ok = client.post(
        "/api/corrections/p-x/accept",
        json={"selected_indices": [0], "proposal_id": expected_id},
    )
    assert ok.status_code == 200
    assert accepted.get("called") is True


def test_correction_accept_holds_lexicon_store_lock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accept records learned contexts into the shared lexicon, so it must hold the lexicon
    store lock (not just the per-project lock), or it races concurrent lexicon writes."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections

    class FakeSummary:
        accepted = True
        change_count = 1
        learned_count = 1
        corrected_named_transcript_path = None

    change = SimpleNamespace(
        sentence_id=1,
        original_text="a",
        corrected_text="b",
        change_type="x",
        reason="r",
    )
    proposal = SimpleNamespace(model="m", proposed_changes=[change])

    monkeypatch.setattr(
        corrections, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )
    monkeypatch.setattr(corrections, "load_manifest", lambda root: object())
    monkeypatch.setattr(
        corrections, "load_speaker_mapping_for_correction", lambda root: {}
    )
    monkeypatch.setattr(
        corrections, "load_correction_proposal", lambda paths, p: proposal
    )
    monkeypatch.setattr(
        corrections, "accept_correction_for_review", lambda **k: FakeSummary()
    )

    # Spy on the very LockRegistry the route resolves via Depends(get_locks).
    locks = client.app.state.locks  # type: ignore[attr-defined]
    seen: list[str] = []
    original_acquire = locks.acquire
    monkeypatch.setattr(
        locks,
        "acquire",
        lambda *keys: (seen.extend(keys), original_acquire(*keys))[1],
    )

    resp = client.post(
        "/api/corrections/p-x/accept",
        json={
            "selected_indices": [0],
            "proposal_id": corrections._proposal_id(proposal),
        },
    )
    assert resp.status_code == 200
    assert "store:lexicon" in seen
    assert any(k.startswith("project:") for k in seen)


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
    # The run's destination is resolved READ-ONLY: for a project made with --project-dir it
    # is a non-canonical path, NOT the content-addressed identity. The job must be keyed by
    # this real path (the key inline routes use) and the create/manifest write deferred into
    # the job under that lock -- never done out here under a different (identity) lock.
    actual_dir = tmp_path / "custom" / "my-project"
    captured: dict = {}

    monkeypatch.setattr(
        pipeline, "resolve_project_dir_for_run", lambda *a, **k: actual_dir
    )

    def fake_workflow(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        return SimpleNamespace(
            project=SimpleNamespace(
                manifest=SimpleNamespace(project_id="p-abc"), project_dir=actual_dir
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

    # Keyed by the ACTUAL reused dir (not the content-addressed default), matching the key
    # inline speaker/correction saves take, so a run and an inline edit serialize.
    assert snapshot["project_id"] == str(actual_dir)
    assert snapshot["status"] == "done"
    # The workflow runs against that reused project and learns corrections into the
    # store-dir lexicon, not the real XDG one.
    assert captured["project_dir"] == actual_dir
    assert captured["lexicon_db"] == get_lexicon_db_path(tmp_path / "store")


def _stub_pipeline_run(
    pipeline, monkeypatch: pytest.MonkeyPatch, project_dir: Path
) -> None:
    """Stub the heavy run internals so /api/pipeline/run completes instantly."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        pipeline, "resolve_project_dir_for_run", lambda *a, **k: project_dir
    )
    monkeypatch.setattr(
        pipeline,
        "_run_project_workflow",
        lambda *a, **k: SimpleNamespace(
            project=SimpleNamespace(
                manifest=SimpleNamespace(project_id="p-abc"), project_dir=project_dir
            ),
            transcription=SimpleNamespace(detected_speaker_count=1, sentence_count=1),
            applied_mapping={},
            meeting_summary=None,
            correction_summary=None,
        ),
    )


def test_pipeline_run_holds_lexicon_and_voiceprint_store_locks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run learns into the lexicon (auto-accepted polish) and deletes global voiceprint
    samples (stabilization), so the job must hold BOTH store locks, not just the project
    lock -- otherwise it races inline lexicon/voiceprint writes."""
    import app.web.routers.pipeline as pipeline

    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake-media")
    _stub_pipeline_run(pipeline, monkeypatch, tmp_path / "projects" / "p-abc")

    locks = client.app.state.locks  # type: ignore[attr-defined]
    seen: list[str] = []
    original_acquire = locks.acquire
    monkeypatch.setattr(
        locks,
        "acquire",
        lambda *keys: (seen.extend(keys), original_acquire(*keys))[1],
    )

    resp = client.post("/api/pipeline/run", json={"input_path": str(media)})
    assert resp.status_code == 200
    snapshot = _drain_job(client, resp.json()["job_id"])
    assert snapshot["status"] == "done"
    assert "store:lexicon" in seen
    assert "store:voiceprints" in seen


def test_pipeline_run_refused_when_capture_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pending capture transaction must block a run before any heavy work: its rollback
    snapshot predates the run's voiceprint deletions and would silently undo them."""
    import app.web.routers.pipeline as pipeline

    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake-media")
    _stub_pipeline_run(pipeline, monkeypatch, tmp_path / "projects" / "p-abc")
    monkeypatch.setattr(pipeline.REGISTRY, "has_pending", lambda: True)

    resp = client.post("/api/pipeline/run", json={"input_path": str(media)})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "conflict"
    assert "capture" in body["detail"].lower()


def test_pipeline_summarize_refused_when_capture_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pending capture must block summarize: summarize_project rewrites project.json, and a
    later capture rollback restores the pre-capture snapshot -- silently dropping it."""
    import app.web.routers.pipeline as pipeline

    monkeypatch.setattr(pipeline, "resolve_web_project_ref", lambda ref, _s: tmp_path)

    def fail_summarize(*_a, **_k):  # must never run while a capture is pending
        raise AssertionError("summarize_project must not run with a capture pending")

    monkeypatch.setattr(pipeline, "summarize_project", fail_summarize)
    monkeypatch.setattr(pipeline.REGISTRY, "has_pending", lambda: True)

    resp = client.post("/api/pipeline/summarize/p-x", json={})
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "conflict"
    assert "capture" in body["detail"].lower()


def test_authenticated_url_brackets_ipv6_hosts() -> None:
    """IPv6 literal hosts must be bracketed in the printed/opened handoff URL."""
    from app.web.server import authenticated_url, base_url

    v6 = WebSettings(
        host="::1",
        port=8765,
        projects_dir=None,
        store_dir=None,
        open_browser=False,
        token="tok",
    )
    assert base_url(v6) == "http://[::1]:8765/"
    assert authenticated_url(v6) == "http://[::1]:8765/?token=tok"

    v4 = WebSettings(
        host="127.0.0.1",
        port=8765,
        projects_dir=None,
        store_dir=None,
        open_browser=False,
        token=None,
    )
    assert base_url(v4) == "http://127.0.0.1:8765/"


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


def test_require_auth_handles_non_ascii_tokens(tmp_path: Path) -> None:
    """secrets.compare_digest raises TypeError on non-ASCII str, which would surface as a
    500. require_auth must compare on bytes: a non-ASCII *presented* token is a clean 401
    (a header byte sequence decodes latin-1 to a non-ASCII str on the wire), and a non-ASCII
    *configured* token still authenticates when presented correctly. Called directly to
    bypass the HTTP client's own header encoding limits."""
    import pytest as _pytest
    from fastapi import HTTPException

    from app.web.deps import require_auth

    secret = "naïve-tök-密码"
    settings = _settings(tmp_path, token=secret)

    # Wrong, non-ASCII presented token -> 401, never a TypeError/500.
    with _pytest.raises(HTTPException) as exc:
        require_auth(settings=settings, authorization=None, token="пароль")
    assert exc.value.status_code == 401
    with _pytest.raises(HTTPException) as exc:
        require_auth(settings=settings, authorization="Bearer Ã©Ã¨", token=None)
    assert exc.value.status_code == 401

    # Correct non-ASCII token authenticates via both query and bearer paths (returns None).
    assert require_auth(settings=settings, authorization=None, token=secret) is None
    assert (
        require_auth(settings=settings, authorization=f"Bearer {secret}", token=None)
        is None
    )


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

    person = client.post("/api/voiceprints/people", json={"name": "Alice"}).json()
    captured: dict = {}

    class FakeResult:
        mapping_path = tmp_path / "speaker_map.json"
        transcript_path = tmp_path / "t.txt"
        srt_path = tmp_path / "t.srt"
        reassignment = None
        deletion = None
        created_person_count = 1

    def fake_save(project_dir, **kwargs):
        captured["project_dir"] = project_dir
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(speakers, "save_speaker_review", fake_save)

    resp = client.post(
        "/api/speakers/p-x/save",
        json={
            "review_revision": speakers._review_revision(tmp_path),
            "mapping": {"0": "Alice", "1": "Bob"},
            "person_mapping": {"0": 7},
            "person_public_mapping": {"0": person["public_id"]},
            "new_person_names": {"4": "Charlie"},
            "ignored_speaker_ids": [2],
            "deleted_speaker_ids": [3],
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
    assert captured["person_public_mapping"] == {0: person["public_id"]}
    assert captured["new_person_names"] == {4: "Charlie"}
    assert list(captured["ignored_speaker_ids"]) == [2]
    assert list(captured["deleted_speaker_ids"]) == [3]
    spec = captured["reassignments"][0]
    assert spec.sentence_id == 5
    assert spec.new_speaker_id == 0
    assert spec.original_speaker_id == 1
    assert resp.json()["reassigned_count"] == 1
    assert resp.json()["created_person_count"] == 1
    assert resp.json()["deleted_speaker_count"] == 1


def test_speaker_save_refuses_stale_review_revision(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A save based on an old review snapshot must not overwrite newer project files."""
    import app.web.routers.speakers as speakers

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )

    def _should_not_run(*_a, **_k):
        raise AssertionError("save ran despite stale review_revision")

    monkeypatch.setattr(speakers, "save_speaker_review", _should_not_run)

    resp = client.post(
        "/api/speakers/p-x/save",
        json={
            "review_revision": "not-current",
            "mapping": {"0": "Alice"},
            "person_mapping": {},
            "person_public_mapping": {},
            "ignored_speaker_ids": [],
            "reassignments": [],
        },
    )

    assert resp.status_code == 409
    assert "speaker review changed" in resp.json()["detail"]


def test_speaker_save_validates_public_person_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Public voiceprint bindings must be real current-store person ids."""
    import app.web.routers.speakers as speakers

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )

    def _should_not_run(*_a, **_k):
        raise AssertionError("save ran despite invalid person_public_mapping")

    monkeypatch.setattr(speakers, "save_speaker_review", _should_not_run)

    body = {
        "review_revision": speakers._review_revision(tmp_path),
        "mapping": {"0": "Alice"},
        "person_mapping": {},
        "person_public_mapping": {"0": "vpp-abc"},
        "ignored_speaker_ids": [],
        "reassignments": [],
    }
    bad_shape = client.post("/api/speakers/p-x/save", json=body)
    assert bad_shape.status_code == 400
    assert bad_shape.json()["error"] == "bad_request"

    body["person_public_mapping"] = {"0": "vpp-0000000000000001"}
    missing = client.post("/api/speakers/p-x/save", json=body)
    assert missing.status_code == 400
    assert "does not exist" in missing.json()["detail"]


def _save_body(*, with_reassignment: bool) -> dict:
    body: dict = {
        "review_revision": "test-revision",
        "mapping": {"0": "Alice"},
        "person_mapping": {},
        "person_public_mapping": {},
        "new_person_names": {},
        "ignored_speaker_ids": [],
        "reassignments": [],
        "deleted_speaker_ids": [],
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

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(speakers, "_require_current_revision", lambda *_a, **_k: None)

    def boom(_fn):
        raise CaptureConflictError("pending capture")

    monkeypatch.setattr(speakers.REGISTRY, "run_store_write", boom)

    resp = client.post(
        "/api/speakers/p-x/save", json=_save_body(with_reassignment=True)
    )
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


def test_speaker_new_person_save_enters_store_section(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Creating a person during speaker save mutates the voiceprint store, so it must lock."""
    import app.web.routers.speakers as speakers
    from app.core.voiceprint_review_service import CaptureConflictError

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(speakers, "_require_current_revision", lambda *_a, **_k: None)

    def boom(_fn):
        raise CaptureConflictError("pending capture")

    monkeypatch.setattr(speakers.REGISTRY, "run_store_write", boom)
    body = _save_body(with_reassignment=False)
    body["new_person_names"] = {"0": "Charlie"}

    resp = client.post("/api/speakers/p-x/save", json=body)

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
        deletion = None
        created_person_count = 0

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
    monkeypatch.setattr(speakers, "_require_current_revision", lambda *_a, **_k: None)
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


def _nonloopback_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, token: str
) -> TestClient:
    """Build a client for a non-loopback (token-protected, is_local=False) bind."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(exist_ok=True)
    settings = WebSettings(
        host="0.0.0.0",
        port=0,
        projects_dir=projects_dir,
        store_dir=tmp_path / "store",
        open_browser=False,
        token=token,
    )
    return TestClient(create_app(settings))


def test_reveal_secrets_refused_on_non_loopback_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token alone must not exfiltrate plaintext secrets over a networked bind."""
    with _nonloopback_client(tmp_path, monkeypatch, token="s3cret") as client:
        client.get("/api/health").raise_for_status()  # is_local must be False here
        assert client.get("/api/health").json()["is_local"] is False
        # Masked listing is fine.
        assert client.get("/api/config?token=s3cret").status_code == 200
        # Reveal is refused with 403, not silently honored. (HTTPException uses the
        # Starlette default body shape {"detail": ...}, like the 401 auth rejection.)
        denied = client.get("/api/config?reveal=true&token=s3cret")
        assert denied.status_code == 403
        assert "loopback" in denied.json()["detail"]


def test_reveal_secrets_allowed_on_loopback(client: TestClient) -> None:
    """Loopback may reveal: the gate permits it and the plaintext value comes back."""
    client.patch(
        "/api/config", json={"key": "dashscope.api_key", "value": "sk-test-123"}
    ).raise_for_status()
    body = client.get("/api/config?reveal=true")
    assert body.status_code == 200
    by_name = {k["name"]: k for k in body.json()["keys"]}
    assert by_name["dashscope.api_key"]["value"] == "sk-test-123"


def test_resolve_project_ref_restrict_blocks_escape(tmp_path: Path) -> None:
    """restrict_to_projects_dir denies refs resolving outside the projects parent."""
    from app.core.project_refs import resolve_project_ref

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    inside = projects_dir / "p-inside"
    inside.mkdir()
    (inside / "project.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "project.json").write_text("{}", encoding="utf-8")

    # CLI default (unrestricted): explicit on-disk paths still resolve -- behavior intact.
    assert resolve_project_ref(outside, projects_dir) == outside.resolve()
    # Web restriction: in-tree resolves, out-of-tree escape raises instead of resolving.
    assert (
        resolve_project_ref(inside, projects_dir, restrict_to_projects_dir=True)
        == inside.resolve()
    )
    with pytest.raises(ValueError):
        resolve_project_ref(outside, projects_dir, restrict_to_projects_dir=True)


def test_projects_list_skips_out_of_tree_symlinks(
    client: TestClient, tmp_path: Path
) -> None:
    """The web project list must not reveal projects symlinked from outside projects_dir."""
    from app.project_manager import create_project

    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")
    outside = tmp_path / "outside-project"
    create_project(
        source,
        title="Outside",
        projects_dir=tmp_path / "unused",
        project_dir=outside,
        meeting_time=None,
        hash_source=False,
    )
    (tmp_path / "projects" / "linked-outside").symlink_to(outside)

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json()["projects"] == []


def test_merge_preview_rejects_out_of_tree_ref(
    client: TestClient, tmp_path: Path
) -> None:
    """Body-supplied project refs cannot traverse out of the configured projects dir."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "project.json").write_text("{}", encoding="utf-8")
    resp = client.post(
        "/api/pipeline/merge-preview", json={"project_refs": [str(outside)]}
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"


def test_get_proposal_returns_404_when_none_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No pending polish proposal must surface as 404 (the correction page's empty state),
    not the 500 a bare RuntimeError from _resolve_json would produce."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections

    monkeypatch.setattr(
        corrections, "resolve_web_project_ref", lambda ref, _s: tmp_path
    )
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )

    def _no_proposal(paths, proposal_path):
        raise RuntimeError(f"No correction proposal found in {paths.root}")

    monkeypatch.setattr(corrections, "load_correction_proposal", _no_proposal)

    resp = client.get("/api/corrections/p-x/proposal")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"


def test_get_proposal_includes_audio_window(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Correction review needs the sentence time window so the web page can play the source."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections

    changes = [
        SimpleNamespace(
            sentence_id=7,
            speaker_name="Speaker A",
            begin_time_ms=4210,
            end_time_ms=5800,
            original_text="before",
            corrected_text="after",
            change_type="polish",
            reason="reason",
        ),
        SimpleNamespace(
            sentence_id=8,
            speaker_name="Speaker B",
            begin_time_ms=0,
            end_time_ms=0,
            original_text="legacy",
            corrected_text="legacy fixed",
            change_type="",
            reason="",
        ),
    ]
    proposal = SimpleNamespace(model="m", proposed_changes=changes)

    monkeypatch.setattr(
        corrections, "resolve_web_project_ref", lambda ref, _s: tmp_path
    )
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )
    monkeypatch.setattr(
        corrections, "load_correction_proposal", lambda paths, proposal_path: proposal
    )

    resp = client.get("/api/corrections/p-x/proposal")

    assert resp.status_code == 200
    rows = resp.json()["changes"]
    assert rows[0]["begin_time_ms"] == 4210
    assert rows[0]["end_time_ms"] == 5800
    assert rows[1]["begin_time_ms"] is None
    assert rows[1]["end_time_ms"] is None


def test_merge_apply_refuses_nonempty_dir_without_force(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Merge must not silently overwrite a non-empty output dir: force defaults to False
    (so write_merge_outputs' guard fires) and the conflict surfaces as 409, not 500."""
    import app.web.routers.pipeline as pipeline

    proj = tmp_path / "projects" / "p-a"
    proj.mkdir(parents=True)
    out_dir = tmp_path / "projects" / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("x", encoding="utf-8")  # non-empty

    # Guard fires before the (fake) result is used, so a dummy merge result is fine.
    monkeypatch.setattr(pipeline, "resolve_web_project_ref", lambda ref, _s: proj)
    monkeypatch.setattr(pipeline, "merge_projects", lambda *a, **k: object())

    resp = client.post(
        "/api/pipeline/merge",
        json={"project_refs": [str(proj)], "out_dir": str(out_dir)},
    )
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


def test_merge_apply_rejects_out_of_tree_output_dir(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Web merge output must stay under the configured projects dir."""
    import app.web.routers.pipeline as pipeline

    proj = tmp_path / "projects" / "p-a"
    proj.mkdir(parents=True)
    outside = tmp_path / "outside-merge"

    monkeypatch.setattr(pipeline, "resolve_web_project_ref", lambda ref, _s: proj)

    resp = client.post(
        "/api/pipeline/merge",
        json={"project_refs": [str(proj)], "out_dir": str(outside)},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"


def test_merge_apply_falls_back_to_default_projects_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no configured projects_dir, merge-apply must fall back to the default XDG
    projects dir (mirroring the read paths) instead of 400ing -- otherwise the feature is
    dead in the default `meeting-asr web` invocation while merge-preview still works."""
    from types import SimpleNamespace

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import app.web.routers.pipeline as pipeline
    from app.config import get_default_projects_dir

    default_root = get_default_projects_dir()
    proj = default_root / "p-a"
    proj.mkdir(parents=True)

    captured: dict = {}
    dummy_result = SimpleNamespace(
        merged_corrected=SimpleNamespace(sentences=[]),
        merged_raw=None,
        order_source="x",
        use_corrected=True,
        identities=[],
        mapping={},
        warnings=[],
    )

    def fake_write(result, out_dir, *, force):
        captured["out_dir"] = out_dir
        return SimpleNamespace(
            out_dir=out_dir,
            transcript=None,
            transcript_corrected=None,
            subtitle=None,
            subtitle_corrected=None,
            manifest=None,
        )

    monkeypatch.setattr(pipeline, "resolve_web_project_ref", lambda ref, _s: proj)
    monkeypatch.setattr(pipeline, "merge_projects", lambda *a, **k: dummy_result)
    monkeypatch.setattr(pipeline, "write_merge_outputs", fake_write)

    settings = WebSettings(
        host="127.0.0.1",
        port=0,
        projects_dir=None,  # the default `meeting-asr web` invocation
        store_dir=tmp_path / "store",
        open_browser=False,
        token=None,
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8765") as client:
        resp = client.post(
            "/api/pipeline/merge",
            json={"project_refs": [str(proj)], "out_dir": "merged-out"},
        )
    assert resp.status_code == 200
    # A relative out_dir rebases onto the default projects dir, and the bundle is written
    # there -- no longer refused for lack of an explicit --projects-dir.
    assert captured["out_dir"] == (default_root / "merged-out").resolve()
    assert resp.json()["out_dir"] == str((default_root / "merged-out").resolve())


def test_voiceprint_store_dir_rebases_onto_subdir(tmp_path: Path) -> None:
    """store_dir is the data root; the voiceprint store lives under <root>/voiceprints,
    mirroring the lexicon's <root>/lexicon rebasing. A bare store_dir would resolve the flat
    <root>/voiceprints.sqlite and miss the real DB."""
    s = WebSettings(
        host="127.0.0.1",
        port=1,
        projects_dir=None,
        store_dir=tmp_path,
        open_browser=False,
        token=None,
    )
    assert s.voiceprint_store_dir == tmp_path / "voiceprints"
    s_none = WebSettings(
        host="127.0.0.1",
        port=1,
        projects_dir=None,
        store_dir=None,
        open_browser=False,
        token=None,
    )
    assert s_none.voiceprint_store_dir is None


def test_voiceprint_library_resolves_db_under_voiceprints_subdir(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The library route must hand get_voiceprint_db_path the rebased voiceprints/ store,
    not the bare data-root store_dir, or a --store-dir copy reads an empty library."""
    import app.web.routers.voiceprints as vp

    captured: dict = {}

    def fake_db(store_dir):
        captured["store_dir"] = store_dir
        return tmp_path / "vp.sqlite"

    monkeypatch.setattr(vp, "get_voiceprint_db_path", fake_db)
    monkeypatch.setattr(vp, "list_voiceprint_speakers", lambda db: [])

    resp = client.get("/api/voiceprints/library")
    assert resp.status_code == 200
    assert captured["store_dir"] == tmp_path / "store" / "voiceprints"


def test_unknown_api_path_is_404_not_spa_html(client: TestClient) -> None:
    """An unknown /api/... path must 404 as an API miss, never fall back to index.html --
    otherwise a misspelled fetch gets 200 + HTML and fails later parsing it as JSON."""
    resp = client.get("/api/this-route-does-not-exist")
    assert resp.status_code == 404
    assert "text/html" not in resp.headers.get("content-type", "")


def test_authenticated_url_encodes_reserved_token_chars() -> None:
    """An explicit --token may contain URL-reserved chars; the handoff URL must
    percent-encode them so the SPA's URLSearchParams parse recovers the original."""
    from urllib.parse import parse_qs, urlparse

    from app.web.server import authenticated_url

    s = WebSettings(
        host="1.2.3.4",
        port=8765,
        projects_dir=None,
        store_dir=None,
        open_browser=False,
        token="a&b#c+d",
    )
    url = authenticated_url(s)
    assert "a&b#c+d" not in url  # raw reserved chars must not appear unencoded
    assert parse_qs(urlparse(url).query)["token"] == ["a&b#c+d"]


def test_voiceprint_lookup_miss_maps_to_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Voiceprint CRUD raises LookupError for a stale/mistyped ref; the web must return 404,
    not the generic 500 the fallback handler would give."""
    import app.web.routers.voiceprints as vp

    monkeypatch.setattr(vp, "get_voiceprint_db_path", lambda s: tmp_path / "vp.sqlite")

    def _boom(*_a, **_k):
        raise LookupError("No voiceprint person found for id: nope")

    monkeypatch.setattr(vp, "rename_voiceprint_person", _boom)

    resp = client.patch("/api/voiceprints/people/nope", json={"name": "X"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"


def test_naming_save_refused_when_capture_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A naming-only speaker save must be refused while a capture is pending: it skips
    run_store_write, but a later rollback restores the pre-capture project snapshot and
    would silently clobber the saved names."""
    import app.web.routers.speakers as speakers

    monkeypatch.setattr(speakers, "resolve_web_project_ref", lambda ref, _s: tmp_path)
    monkeypatch.setattr(speakers, "_require_current_revision", lambda *_a, **_k: None)
    monkeypatch.setattr(speakers.REGISTRY, "has_pending", lambda: True)

    def _should_not_run(*_a, **_k):
        raise AssertionError("save ran despite a pending capture")

    monkeypatch.setattr(speakers, "save_speaker_review", _should_not_run)

    resp = client.post(
        "/api/speakers/p-x/save",
        json={
            "review_revision": "test-revision",
            "mapping": {"0": "Alice"},
            "person_mapping": {},
            "person_public_mapping": {},
            "ignored_speaker_ids": [],
            "reassignments": [],
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"] == "conflict"


def test_capture_rollback_takes_project_lock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rollback restores the project snapshot, so it must hold that project's lock to
    serialize against a concurrent project-local write (speaker save / correction accept)."""
    import app.web.routers.voiceprints as vp

    proj = tmp_path / "projects" / "p-z"
    monkeypatch.setattr(vp.REGISTRY, "project_dir_for", lambda txn: proj)
    monkeypatch.setattr(vp.REGISTRY, "rollback", lambda txn: None)

    locks = client.app.state.locks  # type: ignore[attr-defined]
    seen: list[str] = []
    original_acquire = locks.acquire
    monkeypatch.setattr(
        locks,
        "acquire",
        lambda *keys: (seen.extend(keys), original_acquire(*keys))[1],
    )

    resp = client.post("/api/voiceprints/capture/transactions/txn-1/rollback")
    assert resp.status_code == 200
    assert f"project:{proj}" in seen


def test_merge_people_reads_survivor_under_store_lock(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The survivor must be read INSIDE the store-write critical section. Reading it after
    the lock released let a concurrent store mutation delete the survivor in between, raising
    a spurious 404 even though the merge committed -- so the read must be atomic with it."""
    import app.web.routers.voiceprints as vp
    from app.voiceprint_models import VoiceprintSpeakerRow

    survivor = VoiceprintSpeakerRow(
        speaker_id=1,
        public_id="vp-into",
        name="B",
        sample_count=2,
        project_count=1,
        embedded_sample_count=2,
        embedding_model_count=1,
        updated_at=None,
    )
    state = {"survivor_exists": True}
    events: list[str] = []

    monkeypatch.setattr(vp, "get_voiceprint_db_path", lambda _s: tmp_path / "vp.sqlite")
    monkeypatch.setattr(
        vp, "merge_voiceprint_people", lambda *_a, **_k: events.append("merge")
    )

    def _get(_ref, _db):
        events.append("read")
        return survivor if state["survivor_exists"] else None

    monkeypatch.setattr(vp, "get_voiceprint_person", _get)

    # Simulate a concurrent store mutation that deletes the survivor the instant the store
    # lock is released. A route that reads the survivor outside the lock would now miss it.
    real_run = vp._run

    async def racing_run(locks, fn):
        result = await real_run(locks, fn)
        state["survivor_exists"] = False
        return result

    monkeypatch.setattr(vp, "_run", racing_run)

    resp = client.post(
        "/api/voiceprints/people/merge",
        json={"from_ref": "vp-from", "into_ref": "vp-into"},
    )
    assert resp.status_code == 200
    assert resp.json()["public_id"] == "vp-into"
    # Both the merge and the survivor read happened before the simulated concurrent delete.
    assert events == ["merge", "read"]


def test_get_clip_extracts_via_atomic_temp_then_rename(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The clip cache must be written atomically (temp file + os.replace): is_file() turns
    True the instant ffmpeg creates the file, so a direct write could serve a half-written
    WAV (or two concurrent extractions could corrupt it)."""
    import app.web.routers.audio as audio

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    source = tmp_path / "audio.wav"
    source.write_bytes(b"src")
    monkeypatch.setattr(audio, "resolve_web_project_ref", lambda ref, _s: project_dir)
    monkeypatch.setattr(audio, "load_manifest", lambda _d: object())
    monkeypatch.setattr(audio, "resolve_project_audio_path", lambda _d, _m: source)

    seen: dict[str, str] = {}

    def fake_extract(_source, output, *, start_seconds, duration_seconds):
        # ffmpeg always targets a temp sibling, never the live cache path.
        seen["output"] = str(output)
        Path(output).write_bytes(b"CLIPDATA")
        return Path(output)

    monkeypatch.setattr(audio, "extract_audio_clip", fake_extract)

    resp = client.get("/api/projects/p-x/clip?begin_ms=0&end_ms=1000")
    assert resp.status_code == 200
    assert resp.content == b"CLIPDATA"
    # Staged to a temp, not the live cache path, AND the temp must end in .wav so ffmpeg can
    # infer the output format (a ".tmp" tail fails with "Unable to choose an output format").
    assert seen["output"].endswith(".wav")
    assert not seen["output"].endswith("0_1000.wav")

    clips_dir = project_dir / "tmp" / "web_clips"
    assert (clips_dir / "0_1000.wav").is_file()  # appeared only via the atomic rename
    assert list(clips_dir.glob("*.tmp.wav")) == []  # temp cleaned up


def test_capture_pending_endpoint_reports_and_clears(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The recovery banner's endpoint surfaces a pending capture (id + project) so a txn whose
    originating page is gone can still be accepted/rolled back, and returns null when none."""
    from types import SimpleNamespace

    import app.web.routers.voiceprints as vp

    monkeypatch.setattr(
        vp, "load_manifest", lambda _d: SimpleNamespace(project_id="p-xyz")
    )
    monkeypatch.setattr(
        vp.REGISTRY, "pending_transaction", lambda: ("txn-9", tmp_path / "proj")
    )
    resp = client.get("/api/voiceprints/capture/pending")
    assert resp.status_code == 200
    assert resp.json() == {"transaction_id": "txn-9", "project_id": "p-xyz"}

    monkeypatch.setattr(vp.REGISTRY, "pending_transaction", lambda: None)
    resp = client.get("/api/voiceprints/capture/pending")
    assert resp.status_code == 200
    assert resp.json() is None


def test_get_proposal_malformed_is_not_hidden_as_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed proposal raises RuntimeError too, but it is corruption, not absence: it
    must NOT be translated to the 404 'no pending proposal' empty state."""
    from types import SimpleNamespace

    import app.web.routers.corrections as corrections

    monkeypatch.setattr(
        corrections, "resolve_web_project_ref", lambda ref, _s: tmp_path
    )
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )

    def _malformed(paths, proposal_path):
        raise RuntimeError(f"Correction proposal must be a JSON object: {paths.root}")

    monkeypatch.setattr(corrections, "load_correction_proposal", _malformed)

    # The malformed RuntimeError must NOT be swallowed into the 404 empty state. It is not
    # caught by a specific handler, so it surfaces as a server error (TestClient re-raises it)
    # -- the point being it is never silently translated to 404 like a missing proposal.
    with pytest.raises(RuntimeError, match="must be a JSON object"):
        client.get("/api/corrections/p-x/proposal")


def test_get_sample_clip_rebases_to_configured_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Library clip playback must serve from the CONFIGURED store, not the row's absolute
    clip_path. Under a copied --store-dir that absolute path points at the ORIGINAL store, so
    serving it would read outside the configured copy (an isolation escape)."""
    from types import SimpleNamespace

    import app.web.routers.voiceprints as vp

    store = tmp_path / "store" / "voiceprints"
    store.mkdir(parents=True)
    monkeypatch.setattr(
        vp, "get_voiceprint_db_path", lambda _s: store / "voiceprints.sqlite"
    )

    # Copied store: the row's absolute clip_path is the ORIGINAL store's file (outside the
    # configured copy); clip_rel_path is store-relative.
    outside = tmp_path / "orig" / "clips" / "x.wav"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"ORIGINAL")
    row = SimpleNamespace(
        public_id="s1", clip_path=outside, clip_rel_path="clips/x.wav"
    )
    monkeypatch.setattr(vp, "list_voiceprint_samples", lambda ref, db: [row])

    # The original (outside) clip must NOT be served: the rebased in-store path is absent -> 404.
    assert client.get("/api/voiceprints/people/p1/clips/s1").status_code == 404

    # When the configured store actually holds the clip, it is served from there.
    in_store = store / "clips" / "x.wav"
    in_store.parent.mkdir(parents=True)
    in_store.write_bytes(b"COPY")
    resp = client.get("/api/voiceprints/people/p1/clips/s1")
    assert resp.status_code == 200
    assert resp.content == b"COPY"


def test_delete_person_with_zero_samples(client: TestClient) -> None:
    """A person created via the API has zero samples; the Delete-person endpoint must still
    remove it (it used to 404 because deletion resolved the speaker through its sample list)."""
    created = client.post("/api/voiceprints/people", json={"name": "Empty Person"})
    assert created.status_code == 200
    ref = created.json()["public_id"]

    resp = client.delete(f"/api/voiceprints/people/{ref}")
    assert resp.status_code == 200
    assert resp.json()["deleted_sample_count"] == 0

    # Really gone, and a second delete of the now-absent ref is a clean 404 (not a 500).
    assert client.delete(f"/api/voiceprints/people/{ref}").status_code == 404


def test_validate_capture_selection_detects_plan_drift() -> None:
    """Capture must refuse a selection whose stable (begin,end) no longer matches the freshly
    recomputed plan -- index-based rel_paths can otherwise embed the wrong audio if the project
    was edited between planning and capture."""
    from types import SimpleNamespace

    from app.core.voiceprint_review_service import CaptureConflictError
    from app.web.routers.voiceprints import _validate_capture_selection
    from app.web.schemas import SelectedCaptureClipIn

    planned = SimpleNamespace(
        speakers=[
            SimpleNamespace(
                name="Reviewer",
                person_public_id="vp-abc",
                clips=[
                    SimpleNamespace(
                        rel_path="speaker_0/clip_001.wav",
                        source_begin_time_ms=1000,
                        source_end_time_ms=2000,
                    )
                ],
            )
        ]
    )

    def sel(**over):
        base = dict(
            rel_path="speaker_0/clip_001.wav",
            begin_time_ms=1000,
            end_time_ms=2000,
            name="Reviewer",
            person_public_id="vp-abc",
        )
        base.update(over)
        return SelectedCaptureClipIn(**base)

    # Matching selection -> returns the validated rel_path set.
    assert _validate_capture_selection(planned, [sel()]) == frozenset(
        {"speaker_0/clip_001.wav"}
    )

    # Same rel_path now maps to a different time window (audio drift) -> refuse.
    with pytest.raises(CaptureConflictError):
        _validate_capture_selection(
            planned, [sel(begin_time_ms=5000, end_time_ms=6000)]
        )

    # Same rel_path + times but the speaker was renamed / rebound (identity drift) -> refuse.
    with pytest.raises(CaptureConflictError):
        _validate_capture_selection(planned, [sel(name="Someone Else")])
    with pytest.raises(CaptureConflictError):
        _validate_capture_selection(planned, [sel(person_public_id="vp-other")])

    # Selected rel_path no longer exists in the recomputed plan -> refuse.
    with pytest.raises(CaptureConflictError):
        _validate_capture_selection(planned, [sel(rel_path="speaker_9/clip_999.wav")])


def test_capture_run_rejects_unbounded_parameters(client: TestClient) -> None:
    """Capture run must bound expensive extraction parameters at the HTTP boundary."""
    resp = client.post(
        "/api/voiceprints/capture/p-x/run",
        json={
            "selected_clips": [],
            "sample_count": 999,
            "max_seconds": 999,
            "padding_seconds": 99,
        },
    )
    assert resp.status_code == 422


def test_tokenless_loopback_rejects_foreign_host(client: TestClient) -> None:
    """A tokenless loopback bind must reject requests whose Host is not a loopback name --
    that closes DNS rebinding, where a remote page rebinds to 127.0.0.1 and reaches the
    unauthenticated secret-reveal / mutating routes as same-origin (CORS does not help)."""
    # A rebinding attacker's Host (its own domain) -> 403, even though auth is otherwise off.
    rebind = client.get("/api/auth/check", headers={"Host": "evil.example.com"})
    assert rebind.status_code == 403

    # A genuine loopback Host is allowed through (auth still skipped on the loopback bind).
    ok = client.get("/api/auth/check", headers={"Host": "127.0.0.1:8765"})
    assert ok.status_code == 200
    ok_localhost = client.get("/api/auth/check", headers={"Host": "localhost:8765"})
    assert ok_localhost.status_code == 200


def test_cors_allows_only_vite_dev_origin(client: TestClient) -> None:
    """CORS must grant only the Vite dev origin, not any localhost port -- otherwise any local
    page could read loopback secret-reveal responses (GET /api/config?reveal=true) cross-origin
    and exfiltrate DashScope/OSS keys."""
    allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # A different localhost origin (another local dev server / XSS'd app) is NOT granted access.
    other = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert other.headers.get("access-control-allow-origin") is None


def test_delete_sample_resolves_stable_public_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a sample resolves its stable public_id to the current position, so a stale
    library pane cannot delete the wrong row via an index that shifted under it."""
    from types import SimpleNamespace

    import app.web.routers.voiceprints as vp

    rows = [
        SimpleNamespace(public_id="s1"),
        SimpleNamespace(public_id="s2"),
        SimpleNamespace(public_id="s3"),
    ]
    monkeypatch.setattr(vp, "list_voiceprint_samples", lambda ref, db: rows)

    captured: dict = {}

    def fake_delete(ref, index, *, db_path):
        captured["ref"] = ref
        captured["index"] = index
        return SimpleNamespace(public_id="s2")

    monkeypatch.setattr(vp, "delete_voiceprint_sample", fake_delete)

    resp = client.delete("/api/voiceprints/people/p1/samples/s2")
    assert resp.status_code == 200
    # "s2" is the 2nd row -> resolved to 1-based index 2, regardless of any client-side position.
    assert captured["index"] == 2
    assert resp.json()["deleted_sample_public_id"] == "s2"

    # A public_id absent from the current list is a clean 404, never a wrong-row delete.
    assert client.delete("/api/voiceprints/people/p1/samples/gone").status_code == 404


def test_set_and_clear_alias_disambiguation(client: TestClient) -> None:
    """The web lexicon must expose the disambiguate mutation, not just a read endpoint -- a
    web-only user has to be able to mark a context-ambiguous alias (so it is routed to LLM
    guidance instead of blanket replacement) and clear it again, mirroring the CLI."""
    created = client.post(
        "/api/lexicon/terms",
        json={"canonical": "Canonical Term", "category": "unknown", "aliases": ["amb"]},
    )
    assert created.status_code == 200

    marked = client.post(
        "/api/lexicon/disambiguations",
        json={
            "term": "Canonical Term",
            "alias": "amb",
            "guidance": "resolve by context",
        },
    )
    assert marked.status_code == 200
    body = marked.json()
    assert body is not None
    assert body["alias"] == "amb"
    assert body["guidance"] == "resolve by context"
    assert any(
        d["alias"] == "amb" for d in client.get("/api/lexicon/disambiguations").json()
    )

    # Empty guidance clears it (null response) and drops it from the list.
    cleared = client.post(
        "/api/lexicon/disambiguations",
        json={"term": "Canonical Term", "alias": "amb", "guidance": ""},
    )
    assert cleared.status_code == 200
    assert cleared.json() is None
    assert not any(
        d["alias"] == "amb" for d in client.get("/api/lexicon/disambiguations").json()
    )


def _vp_sample_row(public_id: str, speaker: str, status: str = "active"):
    """A minimal stand-in for VoiceprintSampleRow carrying every field _sample_out reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        sample_id=hash(public_id) & 0xFFFF,
        public_id=public_id,
        speaker_public_id=speaker,
        speaker_name="Someone",
        project_id="p-test",
        source_begin_time_ms=0,
        source_end_time_ms=1000,
        transcript_text="hi",
        sample_status=status,
        clip_rel_path=f"clips/{public_id}.wav",
    )


def test_set_sample_status_returns_valid_delete_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PATCH /samples/{id}/status must return the sample's real 1-based position within its
    person's list (the documented delete key), not a hardcoded 0 the delete endpoint rejects."""
    import app.web.routers.voiceprints as vp

    monkeypatch.setattr(vp, "get_voiceprint_db_path", lambda _s: tmp_path / "vp.sqlite")
    monkeypatch.setattr(
        vp,
        "update_voiceprint_sample_status",
        lambda pid, status, db: _vp_sample_row(pid, "p1", status),
    )
    # The target ("s2") sits second in its person's ordered sample list -> index must be 2.
    monkeypatch.setattr(
        vp,
        "list_voiceprint_samples",
        lambda ref, db: [
            _vp_sample_row("s1", "p1"),
            _vp_sample_row("s2", "p1"),
            _vp_sample_row("s3", "p1"),
        ],
    )

    resp = client.patch(
        "/api/voiceprints/samples/s2/status", json={"status": "quarantined"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["public_id"] == "s2"
    assert body["status"] == "quarantined"
    assert body["identity_confirmed"] is False
    assert body["matching_enabled"] is False
    # The bug returned 0 here; a 1-based delete key is never 0.
    assert body["index"] == 2


def test_exclude_quality_samples_preserves_confirmed_identity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bulk quality exclusion keeps identity confirmation while disabling matching."""
    from types import SimpleNamespace

    import app.web.routers.voiceprints as vp

    monkeypatch.setattr(vp, "get_voiceprint_db_path", lambda _s: tmp_path / "vp.sqlite")
    report = SimpleNamespace(
        people=[
            SimpleNamespace(
                samples=[
                    SimpleNamespace(
                        sample_public_id="ok",
                        status="active",
                        label="ok",
                    ),
                    SimpleNamespace(
                        sample_public_id="bad-active",
                        status="active",
                        label="critical",
                    ),
                    SimpleNamespace(
                        sample_public_id="bad-verified",
                        status="verified-active",
                        label="warning",
                    ),
                ]
            )
        ]
    )
    monkeypatch.setattr(vp, "analyze_voiceprint_quality", lambda **_kwargs: report)
    updates: dict[str, str] = {}

    def fake_update(sample_public_id: str, status: str, _db):
        updates[sample_public_id] = status
        return _vp_sample_row(sample_public_id, "p1", status)

    monkeypatch.setattr(vp, "update_voiceprint_sample_status", fake_update)

    resp = client.post("/api/voiceprints/people/p1/quality/exclude", json={})

    assert resp.status_code == 200
    assert resp.json()["sample_public_ids"] == ["bad-active", "bad-verified"]
    assert updates == {
        "bad-active": "quarantined",
        "bad-verified": "verified-quarantined",
    }
