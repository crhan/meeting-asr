"""Tests for voiceprint library availability and threshold health."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.config import set_config_value
from app.speaker_pipeline_params import (
    DEFAULT_MATCH_THRESHOLD,
    match_threshold_coupling_kinds,
    resolve_match_threshold,
)
from app.voiceprint_calibration import ConfusablePair, calibrate_voiceprint_thresholds
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_library_health import (
    AVAILABILITY_FRAGILE,
    AVAILABILITY_OK,
    AVAILABILITY_UNUSABLE,
    CONFUSABLE_WARNING_MARGIN,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    _confusable_issue,
    analyze_library_health,
)
from app.voiceprint_quality import analyze_voiceprint_quality
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
    clip_seconds: float | None = None,
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
                clip_end_time_ms=(
                    duration_ms if clip_seconds is None else int(clip_seconds * 1000)
                ),
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


def test_health_cli_reports_unusable_people_and_next_command(tmp_path: Path) -> None:
    """`voiceprint health` names the blocked person and the command that fixes it."""
    from typer.testing import CliRunner

    from app.cli import app

    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _seed_person(store_dir, "Ghost", [[0.0, 1.0]] * 3, embed=False)

    result = CliRunner().invoke(
        app, ["voiceprint", "health", "--store-dir", str(store_dir)]
    )
    as_json = CliRunner().invoke(
        app, ["voiceprint", "health", "--store-dir", str(store_dir), "--json"]
    )

    assert result.exit_code == 0
    assert "People matchable: 1/2" in result.output
    assert "Ghost" in result.output
    assert "meeting-asr voiceprint embed" in result.output
    assert as_json.exit_code == 0
    payload = json.loads(as_json.output)
    assert payload["usable_person_count"] == 1
    assert any(i["kind"] == "missing-embeddings" for i in payload["issues"])


def test_missing_clip_is_not_offered_a_backfill(tmp_path: Path) -> None:
    """A sample whose audio is gone must not be reported as "backfill embeddings".

    The backfill physically cannot read the clip, so it would run, report
    success and change nothing -- the button reads as broken. Report it as an
    unrecoverable sample that needs deleting or re-capturing instead.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    rows = _seed_person(store_dir, "Lost", [[0.0, 1.0]] * 3, embed=False)
    for row in rows:
        row.clip_path.unlink()

    report = analyze_library_health(store_dir=store_dir)

    lost = _person(report, "Lost")
    assert lost.availability == AVAILABILITY_UNUSABLE
    assert lost.missing_embedding_count == 3
    assert lost.missing_clip_count == 3
    assert lost.embeddable_count == 0
    issue = _issue(report, "missing-clips")
    assert issue.action == "review-samples"
    assert issue.severity == SEVERITY_CRITICAL
    # The embed action must NOT be offered for these.
    assert not [
        item
        for item in report.issues
        if item.person_name == "Lost" and item.action == "embed"
    ]


def test_embedding_rebases_a_clip_path_from_a_moved_store(tmp_path: Path) -> None:
    """A stale absolute clip_path must not stop embedding when the file is there.

    `clip_path` records the store the clip was first written to; after the
    library moves (different XDG_DATA_HOME, relocated volume) every row points
    somewhere that no longer exists while the clips sit intact under the active
    store. Playback and deletion already rebase via clip_rel_path; embedding
    must too, or those people stay permanently unmatchable.
    """
    from app.voiceprint_audio import resolve_voiceprint_sample_source
    from app.voiceprint_store import list_voiceprint_samples

    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Moved", [[1.0, 0.0], [0.0, 1.0]], embed=False)
    db_path = get_voiceprint_db_path(store_dir)
    # Rewrite clip_path to a store that no longer exists, keeping clip_rel_path.
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE voiceprint_samples SET clip_path = '/gone/' || clip_rel_path"
        )

    rows = list_voiceprint_samples("Moved", db_path)
    for row in rows:
        assert not row.clip_path.exists()
        resolved = resolve_voiceprint_sample_source(row, store_dir=store_dir)
        assert resolved.exists()
        assert resolved.is_relative_to(store_dir.resolve())

    # Health must therefore treat them as embeddable, not as lost clips.
    report = analyze_library_health(store_dir=store_dir)
    moved = _person(report, "Moved")
    assert moved.missing_clip_count == 0
    assert moved.embeddable_count == 2


def test_samples_recorded_over_another_speaker_are_reported(tmp_path: Path) -> None:
    """A reference recorded during crosstalk is a mixture, and nothing said so.

    Consistency and availability both pass such a sample: it has a vector, it
    is enabled, and it agrees with the other samples of the same contaminated
    set. The only place the problem is visible is the source transcript, so the
    check joins back to it.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    _seed_person(store_dir, "Crowded", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _write_project(
        projects_dir,
        "p-crowded",
        # Speaker 1 takes the floor the instant speaker 0 stops, for every
        # sample range the fixture seeded (index * 60s, 8s long).
        sentences=[
            item
            for index in range(3)
            for item in (
                {
                    "begin_time_ms": index * 60_000,
                    "end_time_ms": index * 60_000 + 8_000,
                    "text": "这是被采样的那个人说的一段完整发言内容。",
                    "speaker_id": 0,
                },
                {
                    "begin_time_ms": index * 60_000 + 8_000,
                    "end_time_ms": index * 60_000 + 14_000,
                    "text": "这是另一个人紧接着说的话，中间没有停顿。",
                    "speaker_id": 1,
                },
            )
        ],
    )

    report = analyze_library_health(store_dir=store_dir, projects_dir=projects_dir)

    issue = _issue(report, "overlapped-samples")
    assert issue.person_name == "Crowded"
    assert issue.context["overlapped_count"] == 3
    assert issue.action == "review-samples"


def test_overlap_check_is_absent_without_a_projects_directory(
    tmp_path: Path,
) -> None:
    """Not checked must never render as checked and clean."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Unknown", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])

    report = analyze_library_health(store_dir=store_dir)

    assert not [item for item in report.issues if item.kind == "overlapped-samples"]


def _write_project(root: Path, project_id: str, *, sentences: list[dict]) -> Path:
    """Write the minimal project files the overlap check reads."""
    project_dir = root / project_id
    (project_dir / "asr").mkdir(parents=True)
    created_at = "2026-05-11T12:00:00+08:00"
    (project_dir / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_id": project_id,
                "title": "会议",
                "created_at": created_at,
                "updated_at": created_at,
                "status": "corrected",
                "source": {
                    "path": f"source/{project_id}.mp3",
                    "filename": f"{project_id}.mp3",
                    "size_bytes": 1,
                    "mtime": created_at,
                    "meeting_time": created_at,
                },
                "audio": {"duration_seconds": 3600.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(
            {
                "full_text": "".join(item["text"] for item in sentences),
                "sentences": sentences,
                "detected_speakers": sorted({s["speaker_id"] for s in sentences}),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return project_dir


def test_a_person_with_no_samples_at_all_is_reported(tmp_path: Path) -> None:
    """Zero samples is the most unmatchable state there is, not an absence.

    Deriving the people list from stored samples drops anyone created but
    never captured, which lets the report say every person is usable while a
    name sits in the library matching nothing.
    """
    from app.voiceprint_people import create_voiceprint_person

    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Captured", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    create_voiceprint_person("Empty", get_voiceprint_db_path(store_dir))

    report = analyze_library_health(store_dir=store_dir)

    empty = _person(report, "Empty")
    assert empty.total_sample_count == 0
    assert empty.availability == AVAILABILITY_UNUSABLE
    assert report.person_count == 2
    assert report.usable_person_count == 1
    issue = _issue(report, "no-samples")
    assert issue.person_name == "Empty"
    assert issue.severity == SEVERITY_CRITICAL
    assert issue.action == "capture"


def test_threshold_issue_counts_describe_the_full_population() -> None:
    """Scaled costs must be paired with the population they were scaled to.

    The cost counts are scaled back to the whole library, so quoting the capped
    export length beside them yields sentences like "3000 of 2000 same-person
    scores were rejected" -- a report that visibly contradicts itself.
    """
    from app.voiceprint_calibration import (
        ScoreDistribution,
        VoiceprintCalibrationReport,
    )
    from app.voiceprint_library_health import _threshold_issues

    calibration = VoiceprintCalibrationReport(
        model="m",
        person_count=2,
        scored_person_count=2,
        sample_count=10,
        genuine=ScoreDistribution(
            count=6_000, minimum=0.5, p5=0.55, median=0.8, p95=0.9, maximum=0.95
        ),
        impostor=ScoreDistribution(
            count=10_000, minimum=0.1, p5=0.2, median=0.4, p95=0.7, maximum=0.8
        ),
        eer_threshold=0.65,
        eer_rate=0.1,
        low_impostor_threshold=0.82,
        current_threshold=0.6,
        warnings=(),
        # Far fewer exported scores than the populations above: this is exactly
        # the downsampled state the counts have to survive.
        genuine_scores=tuple([0.55] * 50 + [0.85] * 50),
        impostor_scores=tuple([0.3] * 50 + [0.75] * 50),
        suggested_threshold=0.82,
        suggested_reason="gap midpoint",
        suggested_kind="gap",
    )

    issues = _threshold_issues(calibration)

    assert issues
    context = issues[0].context
    assert context["genuine_count"] == 6_000
    assert context["impostor_count"] == 10_000
    # And the cost it is quoted beside is on the same scale.
    assert context["current_false_accept_count"] == 5_000


def test_matching_seconds_counts_the_clip_not_the_sentence(tmp_path: Path) -> None:
    """Capture truncates a long sentence; health must not credit the whole one.

    A 40s utterance stored as a 12s clip contributed 12s of reference audio.
    Counting 40 would silence a "too little audio" warning that is still true,
    and would mis-rank sourcing, which reads the same number.
    """
    store_dir = tmp_path / "voiceprints"
    # Three 40s sentences, each stored as a 12s clip.
    _seed_person(
        store_dir,
        "Truncated",
        [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]],
        seconds=40.0,
        clip_seconds=12.0,
    )

    report = analyze_library_health(store_dir=store_dir)

    person = _person(report, "Truncated")
    assert person.matching_seconds == 36.0
    # 36s still clears the healthy bar; 120s would have been claimed before.
    assert person.matching_seconds < 3 * 40.0


def _clean_sentences(count: int) -> list[dict]:
    """Transcript where the sampled speaker is alone at every sampled range."""
    return [
        {
            "begin_time_ms": index * 60_000,
            "end_time_ms": index * 60_000 + 8_000,
            "text": "这是被采样的那个人说的一段完整发言内容。",
            "speaker_id": 0,
        }
        for index in range(count)
    ]


def test_samples_from_a_deleted_project_are_reported(tmp_path: Path) -> None:
    """A project can be deleted long after its clips entered the library.

    Nothing about matching breaks -- the clips live in the store -- so both the
    availability and the consistency checks pass such a sample in silence. What
    is gone is the ability to check it: the overlap join has no transcript to
    read, so the sample is counted clean without ever being examined.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    _seed_person(
        store_dir,
        "Split",
        [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]],
        projects=("p-live", "p-gone"),
    )
    # Only one of the two projects still exists on disk.
    _write_project(projects_dir, "p-live", sentences=_clean_sentences(3))

    report = analyze_library_health(store_dir=store_dir, projects_dir=projects_dir)

    issue = _issue(report, "orphaned-samples")
    assert issue.context["orphaned_count"] == 1
    assert issue.context["matching_sample_count"] == 3
    # Some samples are still verifiable, so this is context, not a warning.
    assert issue.severity == SEVERITY_INFO
    assert issue.action == "capture"


def test_a_wholly_orphaned_person_is_a_warning(tmp_path: Path) -> None:
    """When every matching sample is orphaned, the clean record is untested.

    The person still matches and still scores well against their own centroid
    -- which is exactly the self-fulfilling agreement of a set nobody can
    inspect. Recapturing from a project that exists is the only way out, so the
    severity has to rise above the informational case.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    _seed_person(store_dir, "Orphan", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])

    report = analyze_library_health(store_dir=store_dir, projects_dir=projects_dir)

    issue = _issue(report, "orphaned-samples")
    assert issue.context["orphaned_count"] == 3
    assert issue.context["matching_sample_count"] == 3
    assert issue.severity == SEVERITY_WARNING
    assert "nothing about this voiceprint has actually been verified" in issue.detail


def test_orphan_check_is_absent_without_a_projects_directory(tmp_path: Path) -> None:
    """No projects dir means "not checked", not "every project is deleted"."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Unknown", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])

    report = analyze_library_health(store_dir=store_dir)

    assert not [item for item in report.issues if item.kind == "orphaned-samples"]


def _ray(score: float) -> list[float]:
    """Return a unit vector whose cosine against ``[1.0, 0.0]`` is ``score``."""
    return [score, math.sqrt(max(0.0, 1.0 - score * score))]


def _arc(degrees: float, norm: float = 1.0) -> list[float]:
    """Return the vector at ``degrees`` from the x axis, length ``norm``.

    Angles make the competition legible: the cosine between two samples is
    just the cosine of the angle between them, so a cluster's spread and its
    distance to a neighbour can be read straight off the numbers. ``norm`` is
    for the cases that turn on how much a sample pulls its centroid, which
    stored embeddings do differ in.
    """
    radians = math.radians(degrees)
    return [norm * math.cos(radians), norm * math.sin(radians)]


def _person_issue(report, kind: str, name: str):
    """Return one person's issue of a kind, asserting it exists."""
    matches = [
        item for item in report.issues if item.kind == kind and item.person_name == name
    ]
    assert matches, (
        f"issue not raised: {kind} for {name} "
        f"(got {[(i.kind, i.person_name) for i in report.issues]})"
    )
    return matches[0]


def test_a_pair_the_pipeline_actually_swaps_is_critical(tmp_path: Path) -> None:
    """A wrong name is claimed only when the other person wins the ranking.

    Alice's cluster is wide (two samples at -30 degrees, one at +30) and Bob
    sits tightly in the middle of it. For that outlying +30 sample, Bob's
    centroid beats Alice's own leave-one-out centroid -- the stand-in for an
    unseen probe -- and clears the bar, so matching would attach Bob's name.

    Neither existing view sees it: availability is perfect for both, and Bob's
    cluster is flawlessly self-consistent.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [_arc(-30), _arc(-30), _arc(30)])
    _seed_person(store_dir, "Bob", [_arc(0)] * 3)

    report = analyze_library_health(store_dir=store_dir)

    issue = _person_issue(report, "confusable-people", "Alice")
    assert issue.severity == SEVERITY_CRITICAL
    assert issue.action == "capture"
    assert issue.context["other_name"] == "Bob"
    # Only the outlier: the two samples at -30 tie against their own
    # leave-one-out centroid, and a tie is not a win.
    assert issue.context["crossing_count"] == 1
    assert issue.context["sample_count"] == 3
    assert "1 of 3" in issue.title
    # The premise: the per-sample consistency report sees nothing wrong.
    quality = analyze_voiceprint_quality(store_dir=store_dir)
    assert quality.people and quality.suspicious_count == 0
    # Bob still wins his own samples, but by 0.018 -- warned, not blamed.
    bob = _person_issue(report, "confusable-people", "Bob")
    assert bob.severity == SEVERITY_WARNING
    assert bob.context["crossing_count"] == 0


def test_similar_but_separable_people_are_not_reported(tmp_path: Path) -> None:
    """A high score against another centroid is not by itself a wrong match.

    Both clusters are internally identical, so every sample scores 1.0 against
    its own leave-one-out centroid and 0.82 against the other person's. The
    absolute 0.82 is far above the 0.75 bar, but matching ranks candidates and
    accepts the winner, so these probes always land on themselves. Reporting
    them would send the operator off to re-capture perfectly good audio.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0]] * 3)
    _seed_person(store_dir, "Alicia", [_ray(0.82)] * 3)

    report = analyze_library_health(store_dir=store_dir)

    assert not [item for item in report.issues if item.kind == "confusable-people"]


def test_only_the_person_holding_the_stray_sample_is_flagged(tmp_path: Path) -> None:
    """A lost lead blames its owner, not the person they drifted onto.

    The evidence is directional on purpose: Alice keeps one sample that Bob
    wins, while every sample of Bob's is still comfortably his own. A
    symmetric centroid-distance check would have blamed Bob too and sent the
    operator to re-capture audio that is not the problem.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [_arc(-40), _arc(-40), _arc(40)])
    _seed_person(store_dir, "Bob", [_arc(90)] * 3)

    report = analyze_library_health(store_dir=store_dir)

    issue = _person_issue(report, "confusable-people", "Alice")
    assert issue.context["other_name"] == "Bob"
    assert issue.context["min_lead"] < 0
    assert not [
        item
        for item in report.issues
        if item.kind == "confusable-people" and item.person_name == "Bob"
    ]


def test_a_thin_win_warns_before_the_order_reverses(tmp_path: Path) -> None:
    """Winning by a hair is a warning, so the fix can precede the mistake."""
    store_dir = tmp_path / "voiceprints"
    # Alice's outlier beats Bob on its own leave-one-out centroid by 0.030.
    _seed_person(store_dir, "Alice", [_arc(-22), _arc(-22), _arc(22)])
    _seed_person(store_dir, "Bob", [_arc(68.4)] * 3)

    report = analyze_library_health(store_dir=store_dir)

    issue = _person_issue(report, "confusable-people", "Alice")
    assert issue.severity == SEVERITY_WARNING
    assert issue.context["crossing_count"] == 0
    assert 0 < issue.context["min_lead"] < CONFUSABLE_WARNING_MARGIN
    assert "by only" in issue.title
    # Bob wins his own samples by 0.759 and stays out of the queue.
    assert not [
        item
        for item in report.issues
        if item.kind == "confusable-people" and item.person_name == "Bob"
    ]
    assert json.loads(json.dumps(issue.context)) == issue.context


def test_well_separated_people_raise_nothing(tmp_path: Path) -> None:
    """The check must stay quiet on a library that is actually fine."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0]] * 3)
    _seed_person(store_dir, "Bob", [[0.0, 1.0]] * 3)

    report = analyze_library_health(store_dir=store_dir)

    assert not [item for item in report.issues if item.kind == "confusable-people"]


def test_confusable_check_follows_the_configured_threshold(tmp_path: Path) -> None:
    """Whether a lost lead is a wrong *name* depends on the library's own bar.

    Bob already outranks Alice on her outlying sample, but at 0.64 he falls
    short of the 0.75 default, so matching leaves it for manual review --
    a warning, not a wrong name. Configure 0.60 and the very same evidence
    becomes an automatic wrong name. A check frozen to the module-level
    constant would keep reporting the library as clean.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [_arc(-50), _arc(-50), _arc(50)])
    _seed_person(store_dir, "Bob", [_arc(0)] * 3)

    warned = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "Alice"
    )
    assert warned.severity == SEVERITY_WARNING
    assert warned.context["crossing_count"] == 0
    assert warned.context["min_lead"] < 0
    assert warned.context["threshold"] == DEFAULT_MATCH_THRESHOLD

    set_config_value("voiceprint.match_threshold", "0.60")

    escalated = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "Alice"
    )
    assert escalated.severity == SEVERITY_CRITICAL
    assert escalated.context["threshold"] == 0.60
    assert escalated.context["crossing_count"] == 1


def test_a_single_sample_person_claims_no_swap(tmp_path: Path) -> None:
    """Without a leave-one-out centroid the competition cannot be judged.

    One sample is its own centroid, so "does the other person win" has no
    honest answer. Claiming a wrong name from that would be a guess; the
    person is still visible through `fragile-cluster`, so nothing goes
    silently clean.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [_arc(0)])
    _seed_person(store_dir, "Bob", [_arc(2)] * 3)

    report = analyze_library_health(store_dir=store_dir)

    assert not [
        item
        for item in report.issues
        if item.kind == "confusable-people" and item.person_name == "Alice"
    ]
    assert _person_issue(report, "fragile-cluster", "Alice")


def test_confusable_pair_survives_the_cli_json_payload(tmp_path: Path) -> None:
    """`voiceprint health --json` must carry the pair, not just the count."""
    from typer.testing import CliRunner

    from app.cli import app

    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [_arc(-30), _arc(-30), _arc(30)])
    _seed_person(store_dir, "Bob", [_arc(0)] * 3)

    result = CliRunner().invoke(
        app, ["voiceprint", "health", "--store-dir", str(store_dir), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    pairs = [item for item in payload["issues"] if item["kind"] == "confusable-people"]
    assert pairs, payload["issues"]
    assert pairs[0]["context"]["other_name"] in {"Alice", "Bob"}


def test_strong_margin_acceptance_counts_as_a_wrong_name(tmp_path: Path) -> None:
    """A sub-threshold winner that runs away with it is still a wrong name.

    甲's outlier scores 0.707 as 乙 -- below the 0.75 bar -- but 0.0 as itself,
    so 乙 leads the runner-up by more than STRONG_MARGIN_ACCEPT_MARGIN and
    `_acceptance_decision` attaches the name anyway. Testing `score >=
    threshold` alone would report this as "lands in manual review", which is
    the opposite of what happens.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "甲", [_arc(-45), _arc(-45), _arc(45)])
    _seed_person(store_dir, "乙", [_arc(0)] * 3)

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "甲"
    )

    assert issue.severity == SEVERITY_CRITICAL
    assert issue.context["crossing_count"] == 1
    # The winning score is *under* the acceptance threshold: only the
    # strong-margin rule makes this a wrong name.
    assert issue.context["best_score"] < DEFAULT_MATCH_THRESHOLD
    assert issue.context["best_score"] == pytest.approx(0.707, abs=1e-3)


def test_a_project_centroid_can_be_what_takes_the_name(tmp_path: Path) -> None:
    """Candidates are scored the way production scores them, per project.

    乙 recorded in two very different sessions, so their whole-person centroid
    sits between the two and scores 甲's outlier only 0.34. Production does
    not use that number: `_score_known_vector` takes the best of the
    whole-person centroid and each stable per-project centroid, and 乙's first
    session sits almost exactly on the outlier. Scoring whole-person centroids
    only would report a warning here and miss the wrong name entirely.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "甲", [_arc(-55), _arc(-55), _arc(55)])
    _seed_person(
        store_dir,
        "乙",
        [_arc(50), _arc(50), _arc(200), _arc(200)],
        projects=("p1", "p1", "p2", "p2"),
    )

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "甲"
    )

    assert issue.severity == SEVERITY_CRITICAL
    assert issue.context["other_name"] == "乙"
    assert issue.context["crossing_count"] == 1
    # ~0.996 from the p1 centroid, not the ~0.34 the whole-person centroid
    # would have produced.
    assert issue.context["best_score"] > 0.9


def test_only_the_winning_candidate_is_blamed(tmp_path: Path) -> None:
    """Beating the owner is not enough -- the name attached is the top one.

    甲's outlier is outscored by both 乙 and 丙, and both clear the threshold,
    but production names it 丙. Recording every candidate that beats the owner
    would let the report claim 乙's name is being attached when it never is.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "甲", [_arc(-45), _arc(-45), _arc(45)])
    _seed_person(store_dir, "乙", [_arc(20)] * 3)
    _seed_person(store_dir, "丙", [_arc(35)] * 3)

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "甲"
    )

    assert issue.severity == SEVERITY_CRITICAL
    assert issue.context["crossing_count"] == 1
    assert issue.context["other_name"] == "丙"


def test_a_tie_keeps_the_name_when_its_owner_sorts_first(tmp_path: Path) -> None:
    """An exact tie is not a coin flip -- it is decided, and decided the same way.

    Two of AAA's samples score their own leave-one-out centroid and ZZZ's
    centroid identically. `_ranked_matches` sorts stably over the order
    `list_voiceprint_embeddings` returns, which is `ORDER BY speakers.name`,
    so AAA comes first and keeps the name. Only the genuine outlier crosses.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "AAA", [_arc(-30), _arc(-30), _arc(30)])
    _seed_person(store_dir, "ZZZ", [_arc(0)] * 3)

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "AAA"
    )

    # Only the outlier, whose 0.866 genuinely beats its own 0.500.
    assert issue.context["crossing_count"] == 1


def test_a_tie_is_a_wrong_name_when_the_rival_sorts_first(tmp_path: Path) -> None:
    """The same tie, the other way round, is a real mislabel and must be said.

    Identical geometry to the test above with the names swapped, so AAA now
    sorts ahead of the owner and takes every tied sample. Production would
    attach AAA's name to all three; reporting only the outlier -- or treating
    a tie as the owner's win on principle -- would hide two real mislabels.
    This is the case duplicate library entries produce.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "ZZZ", [_arc(-30), _arc(-30), _arc(30)])
    _seed_person(store_dir, "AAA", [_arc(0)] * 3)

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "ZZZ"
    )

    assert issue.severity == SEVERITY_CRITICAL
    assert issue.context["crossing_count"] == 3
    assert issue.context["other_name"] == "AAA"


def test_centroids_weight_samples_by_their_stored_norm(tmp_path: Path) -> None:
    """Averaging must happen on raw vectors, the way matching does it.

    Embeddings are stored exactly as the model produced them and their norms
    genuinely differ -- 1.48x between the smallest and largest on the
    reference library -- so a longer vector pulls the centroid further.
    `_known_speaker_vectors` averages raw and normalizes the result; scaling
    each sample to unit length first re-weights the average and can aim the
    centroid somewhere production never puts it.

    BBB is built to make that gap visible: one sample of norm 10 at 0 degrees
    and one of norm 1 at 80 degrees. Averaged raw, the centroid lands at 5.5
    degrees; averaged after normalizing, at 40 degrees. AAA's first sample
    (own score 0.900) is taken by BBB at 0.995 under the real weighting and
    keeps its own name at 0.766 under the wrong one.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "AAA", [_arc(0), _arc(25.8)])
    _seed_person(store_dir, "BBB", [_arc(0, norm=10.0), _arc(80)])

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "AAA"
    )

    assert issue.severity == SEVERITY_CRITICAL
    assert issue.context["other_name"] == "BBB"
    # Both samples, not just the one BBB wins either way.
    assert issue.context["crossing_count"] == 2
    # 0.995 comes from the raw-weighted centroid; pre-normalizing would cap
    # the best score at 0.969.
    assert issue.context["best_score"] == pytest.approx(0.995, abs=1e-3)


def test_a_tie_below_the_bar_still_says_the_rival_ranks_first() -> None:
    """A tie leaves the lead at 0.000 while the rival is nonetheless ahead.

    Duplicate library entries produce exactly equal scores, and the stable
    production ordering then puts whichever name sorts first in front. When
    that tied score is under the acceptance cutoff nothing is attached, so
    there is no crossing -- but the owner is not winning either. Reading the
    winner off ``min_lead`` would see 0.000 and announce that the right name
    still wins, which is the one thing that is not true here.

    Built directly rather than through a store: an exact float tie needs
    bit-identical centroids, and seeding two people to collide that precisely
    would test the arithmetic of the fixture rather than this branch.
    """
    pair = ConfusablePair(
        person_public_id="vpp-owner",
        person_name="ZZZ",
        other_public_id="vpp-rival",
        other_name="AAA",
        best_score=0.42,
        min_lead=0.0,
        crossing_sample_public_ids=(),
        sample_count=3,
        outranked_count=2,
    )

    issue = _confusable_issue(pair, DEFAULT_MATCH_THRESHOLD)

    assert issue.severity == SEVERITY_WARNING
    assert issue.context["crossing_count"] == 0
    assert issue.context["outranked_count"] == 2
    assert "ranks behind" in issue.title
    assert "the right name is not winning" in issue.detail
    # The fallback wording must not be the one that fires here.
    assert "still wins" not in issue.detail


def test_a_tie_that_is_accepted_is_not_called_a_higher_score() -> None:
    """Winning a draw on name order is a different fault from scoring higher.

    Two entries that draw on every sample are almost always one person
    entered twice; telling the operator to go capture more audio for "both of
    them" would be advice for a problem they do not have.
    """
    pair = ConfusablePair(
        person_public_id="vpp-owner",
        person_name="ZZZ",
        other_public_id="vpp-rival",
        other_name="AAA",
        best_score=0.93,
        min_lead=0.0,
        crossing_sample_public_ids=("vps-1", "vps-2"),
        sample_count=2,
        outranked_count=2,
        tied_win_count=2,
        accept_reasons=("threshold",),
    )

    issue = _confusable_issue(pair, DEFAULT_MATCH_THRESHOLD)

    assert issue.severity == SEVERITY_CRITICAL
    assert "scores higher" not in issue.detail
    assert "ranks AAA ahead" in issue.detail
    assert "exact draw" in issue.detail
    assert "entered into the library twice" in issue.detail


def test_a_score_win_is_not_described_as_a_draw() -> None:
    """The draw wording must stay off a crossing that was won on score."""
    pair = ConfusablePair(
        person_public_id="vpp-owner",
        person_name="ZZZ",
        other_public_id="vpp-rival",
        other_name="AAA",
        best_score=0.93,
        min_lead=-0.2,
        crossing_sample_public_ids=("vps-1",),
        sample_count=3,
        outranked_count=1,
        tied_win_count=0,
        accept_reasons=("threshold",),
    )

    issue = _confusable_issue(pair, DEFAULT_MATCH_THRESHOLD)

    assert "draw" not in issue.detail
    assert "ranks AAA ahead" in issue.detail


def test_a_genuine_narrow_win_still_reads_as_a_win() -> None:
    """The fallback stays reachable: nobody outranked, so the owner does win."""
    pair = ConfusablePair(
        person_public_id="vpp-owner",
        person_name="ZZZ",
        other_public_id="vpp-rival",
        other_name="AAA",
        best_score=0.42,
        min_lead=0.02,
        crossing_sample_public_ids=(),
        sample_count=3,
        outranked_count=0,
    )

    issue = _confusable_issue(pair, DEFAULT_MATCH_THRESHOLD)

    assert issue.severity == SEVERITY_WARNING
    assert "beats" in issue.title
    assert "The right name still wins" in issue.detail


def test_strong_margin_acceptance_is_not_described_as_clearing_the_bar(
    tmp_path: Path,
) -> None:
    """Saying a 0.707 winner "clears 0.75" would send the operator the wrong way.

    They would raise the threshold and watch the wrong name survive it,
    because the strong-margin rule -- not the bar -- is what attached it.
    """
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "甲", [_arc(-45), _arc(-45), _arc(45)])
    _seed_person(store_dir, "乙", [_arc(0)] * 3)

    issue = _person_issue(
        analyze_library_health(store_dir=store_dir), "confusable-people", "甲"
    )

    assert issue.context["accept_reason"] == "strong-margin"
    assert "strong-margin" in issue.detail
    assert f"clears the {DEFAULT_MATCH_THRESHOLD:.2f} bar" not in issue.detail
