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
    monkeypatch.setattr(corrections, "accept_correction_for_review", fake_accept)

    resp = client.post("/api/corrections/p-x/accept", json={"selected_indices": [0]})
    assert resp.status_code == 200
    # The lexicon db handed to the accept path must be the store-dir one, not None/XDG.
    assert captured["lexicon_db"] == get_lexicon_db_path(tmp_path / "store")


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

    resp = client.post("/api/corrections/p-x/accept", json={"selected_indices": [0]})
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

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
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

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )

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

    monkeypatch.setattr(
        speakers, "resolve_web_project_ref", lambda ref, _settings: tmp_path
    )
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

    monkeypatch.setattr(corrections, "resolve_web_project_ref", lambda ref, _s: tmp_path)
    monkeypatch.setattr(
        corrections, "project_paths", lambda root: SimpleNamespace(root=root)
    )

    def _no_proposal(paths, proposal_path):
        raise RuntimeError(f"No correction proposal found in {paths.root}")

    monkeypatch.setattr(corrections, "load_correction_proposal", _no_proposal)

    resp = client.get("/api/corrections/p-x/proposal")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"


def test_merge_apply_refuses_nonempty_dir_without_force(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Merge must not silently overwrite a non-empty output dir: force defaults to False
    (so write_merge_outputs' guard fires) and the conflict surfaces as 409, not 500."""
    import app.web.routers.pipeline as pipeline

    proj = tmp_path / "projects" / "p-a"
    proj.mkdir(parents=True)
    out_dir = tmp_path / "out"
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
    monkeypatch.setattr(speakers.REGISTRY, "has_pending", lambda: True)

    def _should_not_run(*_a, **_k):
        raise AssertionError("save ran despite a pending capture")

    monkeypatch.setattr(speakers, "save_speaker_review", _should_not_run)

    resp = client.post(
        "/api/speakers/p-x/save",
        json={
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
    assert seen["output"].endswith(".wav.tmp")  # staged, not written in place

    clips_dir = project_dir / "tmp" / "web_clips"
    assert (clips_dir / "0_1000.wav").is_file()  # appeared only via the atomic rename
    assert list(clips_dir.glob("*.tmp")) == []  # temp cleaned up
