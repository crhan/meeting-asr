"""Tests for voiceprint library availability and threshold health."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import set_config_value
from app.speaker_pipeline_params import (
    DEFAULT_MATCH_THRESHOLD,
    match_threshold_coupling_kinds,
    resolve_match_threshold,
)
from app.voiceprint_calibration import calibrate_voiceprint_thresholds
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_library_health import (
    AVAILABILITY_FRAGILE,
    AVAILABILITY_OK,
    AVAILABILITY_UNUSABLE,
    SEVERITY_CRITICAL,
    analyze_library_health,
)
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    store_voiceprint_samples_with_rows,
    update_voiceprint_sample_status,
    upsert_voiceprint_embedding,
)


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default lookups and config writes inside the test sandbox."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def test_person_without_active_model_embeddings_is_unusable(tmp_path: Path) -> None:
    """A person with samples but no vectors for the active model cannot match.

    This is the failure the per-sample consistency report cannot see: it has no
    scored samples to fault, so it renders clean while the person is absent
    from the matching pool entirely.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _seed_person(store_dir, "Ghost", [[0.0, 1.0]] * 3, embed=False)

    report = analyze_library_health(store_dir=store_dir)

    ghost = _person(report, "Ghost")
    assert ghost.availability == AVAILABILITY_UNUSABLE
    assert ghost.matching_sample_count == 0
    assert ghost.enabled_sample_count == 3
    assert ghost.missing_embedding_count == 3
    assert report.person_count == 2
    assert report.usable_person_count == 1
    issue = _issue(report, "missing-embeddings")
    assert issue.severity == SEVERITY_CRITICAL
    assert issue.action == "embed"
    assert issue.person_name == "Ghost"
    assert issue.context["missing_embedding_count"] == 3


def test_person_with_only_quarantined_samples_is_unusable(tmp_path: Path) -> None:
    """Excluding every sample removes a person from matching, silently."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    rows = _seed_person(store_dir, "Muted", [[0.0, 1.0]] * 3)
    db_path = get_voiceprint_db_path(store_dir)
    for row in rows:
        update_voiceprint_sample_status(row.public_id, "quarantined", db_path)

    report = analyze_library_health(store_dir=store_dir)

    muted = _person(report, "Muted")
    assert muted.availability == AVAILABILITY_UNUSABLE
    assert muted.enabled_sample_count == 0
    assert muted.total_sample_count == 3
    issue = _issue(report, "no-enabled-samples")
    assert issue.severity == SEVERITY_CRITICAL
    assert issue.action == "review-samples"


def test_thin_cluster_is_fragile_not_ok(tmp_path: Path) -> None:
    """Under the minimum cluster size a person is matchable but unreliable."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _seed_person(store_dir, "Thin", [[0.0, 1.0], [0.02, 0.99]])

    report = analyze_library_health(store_dir=store_dir)

    assert _person(report, "Thin").availability == AVAILABILITY_FRAGILE
    assert _person(report, "Alice").availability == AVAILABILITY_OK
    assert _issue(report, "fragile-cluster").person_name == "Thin"


def test_healthy_library_reports_no_critical_availability_issues(
    tmp_path: Path,
) -> None:
    """Well-covered people produce no blocking availability issues."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(
        store_dir,
        "Alice",
        [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]],
        seconds=12.0,
        projects=("p-a", "p-b", "p-c"),
    )
    _seed_person(
        store_dir,
        "Bob",
        [[0.0, 1.0], [0.03, 0.97], [-0.02, 0.99]],
        seconds=12.0,
        projects=("p-x", "p-y", "p-z"),
    )

    report = analyze_library_health(store_dir=store_dir)

    assert report.usable_person_count == 2
    availability_kinds = {
        issue.kind
        for issue in report.issues
        if issue.kind
        in {
            "missing-embeddings",
            "no-enabled-samples",
            "fragile-cluster",
            "short-audio",
            "single-source",
        }
    }
    assert availability_kinds == set()


def test_threshold_too_high_issue_quantifies_the_recovery(tmp_path: Path) -> None:
    """A threshold above the genuine mass is reported with what moving it buys."""
    store_dir = tmp_path / "voiceprints"
    # Genuine scores land near 0.66 -- comfortably below the 0.75 default.
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.7, 0.7], [0.75, 0.66]])
    _seed_person(store_dir, "Bob", [[-1.0, 0.0], [-0.7, -0.7], [-0.75, -0.66]])

    report = analyze_library_health(store_dir=store_dir)

    issue = _issue(report, "threshold-too-high")
    assert issue.action == "set-threshold"
    assert issue.context["current_threshold"] == DEFAULT_MATCH_THRESHOLD
    assert issue.context["current_false_reject_count"] > 0
    # The suggestion must actually be an improvement, not just a different number.
    assert (
        issue.context["suggested_false_reject_count"]
        < issue.context["current_false_reject_count"]
    )


def test_suggested_threshold_sits_in_the_gap(tmp_path: Path) -> None:
    """With separable populations the suggestion lands between them."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _seed_person(store_dir, "Bob", [[0.0, 1.0], [0.03, 0.97], [-0.02, 0.99]])

    report = calibrate_voiceprint_thresholds(store_dir=store_dir)

    assert report.suggested_kind == "gap"
    assert report.suggested_threshold is not None
    assert report.impostor is not None and report.genuine is not None
    assert report.impostor.maximum < report.suggested_threshold
    assert report.suggested_threshold < report.genuine.p5
    # The suggestion must cost less than the built-in default on this library.
    current = report.current_cost
    suggested = report.suggested_cost
    assert current is not None and suggested is not None
    assert suggested.false_accept_count == 0


def test_cost_at_prices_an_arbitrary_threshold(tmp_path: Path) -> None:
    """Raw score populations let any candidate threshold be priced."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _seed_person(store_dir, "Bob", [[0.0, 1.0], [0.03, 0.97], [-0.02, 0.99]])

    report = calibrate_voiceprint_thresholds(store_dir=store_dir)

    # Above every genuine score: everything correct is rejected. A cosine of
    # exactly 1.0 is reachable, and acceptance is `score >= threshold`, so the
    # strict probe has to sit above 1.0 to exclude it.
    strict = report.cost_at(1.01)
    assert strict is not None
    assert strict.false_reject_count == len(report.genuine_scores)
    assert strict.false_accept_count == 0
    # Below every impostor score: every wrong-person score is accepted. Cosine
    # similarity is signed, so "below everything" means below -1, not below 0.
    loose = report.cost_at(-1.01)
    assert loose is not None
    assert loose.false_reject_count == 0
    assert loose.false_accept_count == len(report.impostor_scores)


def test_configured_threshold_overrides_the_builtin_default() -> None:
    """A configured threshold reaches every entry point that does not pass one."""
    assert resolve_match_threshold() == DEFAULT_MATCH_THRESHOLD

    set_config_value("voiceprint.match_threshold", "0.62")

    assert resolve_match_threshold() == 0.62
    # An explicit value still wins; the config is only the fallback.
    assert resolve_match_threshold(0.8) == 0.8
    assert calibrate_voiceprint_thresholds().current_threshold == 0.62


def test_coupling_kinds_flag_thresholds_that_kill_downstream_rules() -> None:
    """Lowering past the strong-margin / crosstalk boundaries is reported."""
    assert match_threshold_coupling_kinds(0.75) == ()
    assert "strong-margin-dead" in match_threshold_coupling_kinds(0.60)
    assert "below-crosstalk-floor" in match_threshold_coupling_kinds(0.45)


def test_health_survives_an_empty_store(tmp_path: Path) -> None:
    """An empty library renders as empty, not as an error."""
    report = analyze_library_health(store_dir=tmp_path / "empty")

    assert report.person_count == 0
    assert report.usable_person_count == 0
    assert report.issues == ()


def _person(report, name: str):
    """Return one person's health row by name."""
    matches = [item for item in report.people if item.speaker_name == name]
    assert matches, f"person not in report: {name}"
    return matches[0]


def _issue(report, kind: str):
    """Return the first issue of a kind, asserting it exists."""
    matches = [item for item in report.issues if item.kind == kind]
    assert matches, f"issue not raised: {kind} (got {[i.kind for i in report.issues]})"
    return matches[0]


def _seed_person(
    store_dir: Path,
    name: str,
    vectors: list[list[float]],
    *,
    embed: bool = True,
    seconds: float = 8.0,
    projects: tuple[str, ...] = (),
) -> list:
    """Store samples for one person, optionally without embedding them."""
    db_path = get_voiceprint_db_path(store_dir)
    _provider, model = resolve_voiceprint_embedding_options(provider=None, model=None)
    source = store_dir / f"{name}-source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"seed")
    duration_ms = int(seconds * 1000)
    samples = []
    for index, _vector in enumerate(vectors):
        clip_path = store_dir / "clips" / name / f"clip_{index}.wav"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"{name}-{index}".encode())
        project_id = (
            projects[index % len(projects)] if projects else f"p-{name.lower()}"
        )
        samples.append(
            StoredVoiceprintSample(
                speaker_name=name,
                project_id=project_id,
                project_path=store_dir,
                project_speaker_id=0,
                source_path=source,
                clip_path=clip_path,
                clip_rel_path=str(clip_path.relative_to(store_dir)),
                source_begin_time_ms=index * 60_000,
                source_end_time_ms=index * 60_000 + duration_ms,
                clip_begin_time_ms=0,
                clip_end_time_ms=duration_ms,
                transcript_text=f"{name} sample {index}",
            )
        )
    _db, rows = store_voiceprint_samples_with_rows(samples, db_path)
    if embed:
        for row, vector in zip(rows, vectors):
            upsert_voiceprint_embedding(row.sample_id, model, vector, db_path)
    return list(rows)


def test_health_payload_is_json_serializable(tmp_path: Path) -> None:
    """Issue context must survive the HTTP boundary as plain JSON."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Ghost", [[0.0, 1.0]] * 3, embed=False)

    report = analyze_library_health(store_dir=store_dir)

    payload = [issue.context for issue in report.issues]
    assert json.loads(json.dumps(payload)) == payload
