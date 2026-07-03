"""CLI observability for explicitly ignored speakers."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project, load_manifest

runner = CliRunner()


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for ignored-speaker tests."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _write_three_speaker_sentences(path: Path) -> None:
    """Write a normalized sentences.json with three distinct speakers."""
    sentences = []
    for index in range(6):
        sentences.append(
            {
                "begin_time_ms": index * 1000,
                "end_time_ms": index * 1000 + 800,
                "text": f"Speaker0 sample {index}.",
                "speaker_id": 0,
                "sentence_id": index * 3 + 1,
            }
        )
        sentences.append(
            {
                "begin_time_ms": index * 1000 + 100,
                "end_time_ms": index * 1000 + 900,
                "text": f"Speaker1 sample {index}.",
                "speaker_id": 1,
                "sentence_id": index * 3 + 2,
            }
        )
        sentences.append(
            {
                "begin_time_ms": index * 1000 + 200,
                "end_time_ms": index * 1000 + 950,
                "text": f"Speaker2 sample {index}.",
                "speaker_id": 2,
                "sentence_id": index * 3 + 3,
            }
        )
    payload = {
        "full_text": "".join(sentence["text"] for sentence in sentences),
        "detected_speakers": [0, 1, 2],
        "sentences": sentences,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_speaker_ignore(project_dir: Path, ignored: list[int]) -> Path:
    """Write speakers/speaker_ignore.json without going through apply."""
    path = project_dir / "speakers" / "speaker_ignore.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ignored_speakers": ignored}, ensure_ascii=False), encoding="utf-8"
    )
    return path


def _write_matches(project_dir: Path, *, threshold: float = 0.75) -> None:
    """Write a speaker_matches.json fixture covering all three speakers."""
    payload = {
        "provider": "local-speechbrain",
        "model": "test",
        "threshold": threshold,
        "matches": [
            {
                "speaker_id": 0,
                "label": "Speaker A",
                "name": "欧丁",
                "score": 0.91,
                "accepted": True,
                "accepted_name": "欧丁",
                "best_name": "欧丁",
                "best_score": 0.91,
                "threshold": threshold,
            },
            {
                "speaker_id": 1,
                "label": "Speaker B",
                "name": None,
                "score": 0.6,
                "accepted": False,
                "best_name": "敬悦",
                "best_score": 0.6,
                "accepted_name": None,
                "threshold": threshold,
            },
            {
                "speaker_id": 2,
                "label": "Speaker C",
                "name": None,
                "score": 0.4,
                "accepted": False,
                "best_name": "墨泪",
                "best_score": 0.4,
                "accepted_name": None,
                "threshold": threshold,
            },
        ],
    }
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_project_show_json_exposes_ignored_speakers_and_status(tmp_path: Path) -> None:
    """``project show --json`` should expose ignored_speakers and per-speaker status."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    _write_matches(project_dir)
    _write_speaker_ignore(project_dir, [2])
    manifest = load_manifest(project_dir)
    manifest.speakers.update({"detected_ids": [0, 1, 2], "mapped": {"0": "欧丁"}})
    from app.project_manager import save_manifest

    save_manifest(project_dir, manifest)

    result = runner.invoke(app, ["project", "show", str(project_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["detected_speakers"] == [0, 1, 2]
    assert payload["ignored_speakers"] == [2]
    speakers = {row["speaker_id"]: row for row in payload["speakers"]}
    assert speakers[0]["status"] == "matched"
    assert speakers[0]["name"] == "欧丁"
    assert speakers[0]["match"]["candidate"] == "欧丁"
    assert speakers[1]["status"] == "below-threshold"
    assert speakers[1]["match"]["candidate"] == "敬悦"
    assert speakers[2]["status"] == "ignored"
    assert speakers[2]["ignored"] is True
    assert speakers[2]["sample_count"] > 0


def test_project_show_json_preserves_no_candidate_status(tmp_path: Path) -> None:
    """Speakers with a no-candidate match row must keep their voiceprint state."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    payload = {
        "threshold": 0.75,
        "matches": [
            {
                "speaker_id": 0,
                "label": "Speaker A",
                "name": None,
                "score": 0.0,
                "accepted": False,
                "best_name": None,
                "best_score": None,
                "threshold": 0.75,
            },
            {
                "speaker_id": 1,
                "label": "Speaker B",
                "name": None,
                "score": 0.0,
                "accepted": False,
                "best_name": None,
                "best_score": None,
                "threshold": 0.75,
            },
        ],
    }
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    manifest = load_manifest(project_dir)
    # Manually-named speaker should still expose the underlying voiceprint state.
    manifest.speakers.update({"detected_ids": [0, 1, 2], "mapped": {"1": "敬悦"}})
    from app.project_manager import save_manifest

    save_manifest(project_dir, manifest)

    result = runner.invoke(app, ["project", "show", str(project_dir), "--json"])

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    speakers = {row["speaker_id"]: row for row in body["speakers"]}
    # speaker_id=0 has no name and no candidate: still surfaces as no-candidate.
    assert speakers[0]["status"] == "no-candidate"
    # speaker_id=1 was manually named but voiceprint had no candidate; downstream
    # agents need to see both signals (status + name), not a synthesized "matched".
    assert speakers[1]["status"] == "no-candidate"
    assert speakers[1]["name"] == "敬悦"
    # speaker_id=2 has no match row at all: falls back to "unnamed".
    assert speakers[2]["status"] == "unnamed"


def test_project_show_json_marks_unnamed_speakers_when_no_match(tmp_path: Path) -> None:
    """Speakers without matches and without names should report ``status=unnamed``."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")

    result = runner.invoke(app, ["project", "show", str(project_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    statuses = {row["speaker_id"]: row["status"] for row in payload["speakers"]}
    assert statuses == {0: "unnamed", 1: "unnamed", 2: "unnamed"}
    assert payload["ignored_speakers"] == []


def test_project_speakers_inspect_treats_ignored_as_resolved(tmp_path: Path) -> None:
    """Speaker inspect should label ignored speakers and stop recommending review."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    _write_speaker_ignore(project_dir, [2])

    payload = {
        "provider": "local-speechbrain",
        "model": "test",
        "threshold": 0.75,
        "matches": [
            {
                "speaker_id": 0,
                "label": "Speaker A",
                "name": "欧丁",
                "score": 0.91,
                "accepted": True,
                "accepted_name": "欧丁",
                "threshold": 0.75,
            },
            {
                "speaker_id": 1,
                "label": "Speaker B",
                "name": "敬悦",
                "score": 0.92,
                "accepted": True,
                "accepted_name": "敬悦",
                "threshold": 0.75,
            },
            {
                "speaker_id": 2,
                "label": "Speaker C",
                "name": None,
                "score": 0.4,
                "accepted": False,
                "best_name": "墨泪",
                "best_score": 0.4,
                "threshold": 0.75,
            },
        ],
    }
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    result = runner.invoke(
        app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"]
    )

    assert result.exit_code == 0, result.output
    assert "Speaker C (speaker_id=2)" in result.output
    assert "Status: ignored" in result.output
    # Ignored speaker must not advertise a below-threshold match candidate.
    assert "best=墨泪" not in result.output
    # The only remaining unresolved speakers are matched, so no review prompt.
    assert "Recommended next step:" not in result.output


def test_project_speakers_inspect_still_prompts_when_non_ignored_remains(
    tmp_path: Path,
) -> None:
    """When a non-ignored speaker still needs review, inspect must still prompt."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    _write_speaker_ignore(project_dir, [2])
    _write_matches(project_dir)
    manifest = load_manifest(project_dir)

    result = runner.invoke(
        app, ["project", "speakers", "inspect", str(project_dir), "--sample-count", "1"]
    )

    assert result.exit_code == 0, result.output
    assert "Status: ignored" in result.output
    assert "best=墨泪" not in result.output  # ignored speaker shouldn't expose match
    assert (
        f"Recommended next step: meeting-asr project speakers review {manifest.project_id}"
        in result.output
    )


def test_project_show_renders_ignored_status_in_voiceprint_table(
    tmp_path: Path,
) -> None:
    """Project show should render ignored speakers as ``ignored`` in the match table."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    _write_matches(project_dir)
    _write_speaker_ignore(project_dir, [2])

    result = runner.invoke(app, ["project", "show", str(project_dir)])

    assert result.exit_code == 0, result.output
    assert "ignored" in result.output


def test_project_review_summary_skips_ignored_speaker(tmp_path: Path) -> None:
    """Speaker review summary should not advertise ignored speakers as unresolved."""
    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    _write_speaker_ignore(project_dir, [2])
    payload = {
        "threshold": 0.75,
        "matches": [
            {
                "speaker_id": 0,
                "label": "Speaker A",
                "name": "欧丁",
                "score": 0.91,
                "accepted": True,
                "accepted_name": "欧丁",
                "threshold": 0.75,
            },
            {
                "speaker_id": 1,
                "label": "Speaker B",
                "name": "敬悦",
                "score": 0.93,
                "accepted": True,
                "accepted_name": "敬悦",
                "threshold": 0.75,
            },
            {
                "speaker_id": 2,
                "label": "Speaker C",
                "name": None,
                "score": 0.4,
                "accepted": False,
                "best_name": "墨泪",
                "best_score": 0.4,
                "threshold": 0.75,
            },
        ],
    }
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "review",
            str(project_dir),
            "--summary",
            "--store-dir",
            str(tmp_path / "voiceprints"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Speaker C: below-threshold" not in result.output
    assert "Recommended next step" not in result.output
    assert "speaker_id=2 status=ignored" in result.output
    assert "match=ignored" in result.output


def test_project_has_unresolved_match_ignores_ignored_speakers(tmp_path: Path) -> None:
    """project_has_unresolved_match should treat ignored speakers as resolved."""
    from app.speaker_match_status import project_has_unresolved_match

    project_dir = _sample_project(tmp_path)
    _write_three_speaker_sentences(project_dir / "asr" / "sentences.json")
    payload = {
        "threshold": 0.75,
        "matches": [
            {
                "speaker_id": 2,
                "label": "Speaker C",
                "name": None,
                "score": 0.4,
                "accepted": False,
                "best_name": "墨泪",
                "best_score": 0.4,
                "threshold": 0.75,
            },
        ],
    }
    (project_dir / "speakers" / "speaker_matches.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    assert project_has_unresolved_match(project_dir) is True
    assert project_has_unresolved_match(project_dir, ignored_speaker_ids={2}) is False
