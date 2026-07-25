"""Tests for finding where more voiceprint samples can be harvested."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_sample_sourcing import (
    DEFICIT_SHORT_AUDIO,
    DEFICIT_SINGLE_SOURCE,
    EVIDENCE_NAME,
    EVIDENCE_PERSON_MAP,
    REASON_ALREADY_HARVESTED,
    REASON_NAME_ONLY,
    REASON_NEW_PROJECT,
    REASON_RETRY_QUARANTINED,
    REASON_THIN_SUPPLY,
    find_sample_sources,
    sample_source_payload,
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
    """Keep default store and projects lookups inside the test sandbox."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


def test_person_map_link_finds_an_unharvested_project(tmp_path: Path) -> None:
    """A linked speaker in a never-sampled project is the top recommendation.

    This is the case the old ``capture`` action could not express: the person
    is short on audio, a meeting full of their speech exists, and nothing on
    screen said which meeting.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Alice", sample_count=3, project_id="p-old")
    person_id = rows[0].speaker_public_id
    _make_project(
        projects_dir,
        "p-old",
        sentences=_speech(0, count=4, speaker_id=0),
        speaker_map={0: "Alice"},
        person_map={0: person_id},
    )
    _make_project(
        projects_dir,
        "p-new",
        sentences=_speech(0, count=12, speaker_id=2),
        speaker_map={2: "Alice"},
        person_map={2: person_id},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert report.person_name == "Alice"
    assert DEFICIT_SINGLE_SOURCE in report.deficits
    assert report.scanned_project_count == 2
    top = report.sources[0]
    assert top.project_id == "p-new"
    assert top.speaker_id == 2
    assert top.evidence == EVIDENCE_PERSON_MAP
    assert REASON_NEW_PROJECT in top.reasons
    assert top.candidate_count == 12
    assert report.new_project_count == 1


def test_display_name_match_finds_an_unlinked_speaker(tmp_path: Path) -> None:
    """A speaker can be named in a project without ever being linked.

    Attribution that only reads ``speaker_person_map.json`` misses these
    entirely, which is the common real shape: the operator renamed the speaker
    during review and never captured them into the library from there.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Bao", sample_count=3, project_id="p-old")
    person_id = rows[0].speaker_public_id
    _make_project(
        projects_dir,
        "p-named",
        sentences=_speech(0, count=8, speaker_id=1),
        speaker_map={1: "Bao"},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert [source.project_id for source in report.sources] == ["p-named"]
    source = report.sources[0]
    assert source.evidence == EVIDENCE_NAME
    assert REASON_NAME_ONLY in source.reasons
    assert source.speaker_name == "Bao"


def test_speaker_linked_to_another_person_is_never_claimed_by_name(
    tmp_path: Path,
) -> None:
    """A person link outranks a matching display name, and blocks it."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    mine = _seed_person(store_dir, "Chen", sample_count=3, project_id="p-old")
    other = _seed_person(store_dir, "Other", sample_count=3, project_id="p-old")
    _make_project(
        projects_dir,
        "p-ambiguous",
        sentences=_speech(0, count=8, speaker_id=1),
        speaker_map={1: "Chen"},
        person_map={1: other[0].speaker_public_id},
    )

    report = find_sample_sources(
        mine[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert report.sources == ()


def test_placeholder_names_never_attribute(tmp_path: Path) -> None:
    """Placeholder speaker names are not identities and must not match."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "待确认发言人2", sample_count=3, project_id="p-old")
    _make_project(
        projects_dir,
        "p-placeholder",
        sentences=_speech(0, count=8, speaker_id=1),
        speaker_map={1: "待确认发言人2"},
    )

    report = find_sample_sources(
        rows[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert report.sources == ()


def test_already_captured_utterances_are_excluded_from_supply(
    tmp_path: Path,
) -> None:
    """Re-offering stored clips would advertise audio that adds nothing."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    sentences = _speech(0, count=10, speaker_id=0)
    rows = _seed_person(
        store_dir,
        "Dai",
        sample_count=3,
        project_id="p-one",
        # Overlap the first three transcript segments exactly.
        ranges=[(item["begin_time_ms"], item["end_time_ms"]) for item in sentences[:3]],
    )
    _make_project(
        projects_dir,
        "p-one",
        sentences=sentences,
        speaker_map={0: "Dai"},
        person_map={0: rows[0].speaker_public_id},
    )

    report = find_sample_sources(
        rows[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )

    source = report.sources[0]
    assert source.candidate_count == 7
    assert REASON_ALREADY_HARVESTED in source.reasons


def test_project_that_only_yielded_quarantined_clips_is_still_unharvested(
    tmp_path: Path,
) -> None:
    """Quarantined samples contribute nothing to matching, so the room is fresh.

    Ranking it as "already harvested" would hide the one project able to fix a
    single-source person; the retry reason exists so the caller can explain
    that the previous attempt was rejected rather than never made.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    good = _seed_person(store_dir, "Er", sample_count=3, project_id="p-good")
    person_id = good[0].speaker_public_id
    bad = _seed_person(
        store_dir,
        "Er",
        sample_count=3,
        project_id="p-rejected",
        ranges=[(500_000, 505_000), (510_000, 515_000), (520_000, 525_000)],
    )
    db_path = get_voiceprint_db_path(store_dir)
    for row in bad:
        update_voiceprint_sample_status(row.public_id, "quarantined", db_path)
    _make_project(
        projects_dir,
        "p-good",
        sentences=_speech(0, count=4, speaker_id=0),
        speaker_map={0: "Er"},
        person_map={0: person_id},
    )
    _make_project(
        projects_dir,
        "p-rejected",
        sentences=_speech(0, count=14, speaker_id=0),
        speaker_map={0: "Er"},
        person_map={0: person_id},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    top = report.sources[0]
    assert top.project_id == "p-rejected"
    assert REASON_NEW_PROJECT in top.reasons
    assert REASON_RETRY_QUARANTINED in top.reasons
    assert top.quarantined_sample_count == 3
    assert top.matching_sample_count == 0


def test_supply_outranks_a_thin_source_for_a_short_audio_person(
    tmp_path: Path,
) -> None:
    """Someone short on seconds should be sent where the seconds are."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(
        store_dir, "Feng", sample_count=3, project_id="p-a", seconds=2.0
    )
    person_id = rows[0].speaker_public_id
    _make_project(
        projects_dir,
        "p-thin",
        sentences=_speech(0, count=1, speaker_id=0),
        speaker_map={0: "Feng"},
        person_map={0: person_id},
    )
    _make_project(
        projects_dir,
        "p-rich",
        sentences=_speech(0, count=20, speaker_id=0),
        speaker_map={0: "Feng"},
        person_map={0: person_id},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert DEFICIT_SHORT_AUDIO in report.deficits
    assert [source.project_id for source in report.sources] == ["p-rich", "p-thin"]
    thin = report.sources[1]
    assert REASON_THIN_SUPPLY in thin.reasons


def test_low_information_segments_are_not_counted_as_supply(
    tmp_path: Path,
) -> None:
    """Backchannel fragments cannot characterize a voice, so they are not stock."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Guo", sample_count=3, project_id="p-old")
    filler = [
        {
            "begin_time_ms": index * 20_000,
            "end_time_ms": index * 20_000 + 700,
            "text": "对对对",
            "speaker_id": 0,
        }
        for index in range(12)
    ]
    _make_project(
        projects_dir,
        "p-filler",
        sentences=filler,
        speaker_map={0: "Guo"},
        person_map={0: rows[0].speaker_public_id},
    )

    report = find_sample_sources(
        rows[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert report.sources == ()


def test_project_without_a_transcript_is_reported_not_swallowed(
    tmp_path: Path,
) -> None:
    """A broken project must not silently shrink the search."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Hu", sample_count=3, project_id="p-old")
    project_dir = _make_project(
        projects_dir,
        "p-broken",
        sentences=_speech(0, count=6, speaker_id=0),
        speaker_map={0: "Hu"},
        person_map={0: rows[0].speaker_public_id},
    )
    (project_dir / "asr" / "sentences.json").unlink()

    report = find_sample_sources(
        rows[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )

    assert report.sources == ()
    assert [item.reason for item in report.skipped] == ["no-transcript"]
    assert report.scanned_project_count == 1


def test_sources_offer_capture_plan_clips_from_the_given_store(
    tmp_path: Path,
) -> None:
    """Clips must be the capture plan's own, resolved against the given store.

    Two things ride on this. The clips carry ``rel_path``, which is the only
    identity a capture run accepts, so a caller can capture them without
    re-picking. And the planner resolves each speaker's library person from the
    store — defaulting that lookup would make an isolated ``--store-dir`` run
    identify people from the real library instead.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Lin", sample_count=3, project_id="p-old")
    person_id = rows[0].speaker_public_id
    _make_project(
        projects_dir,
        "p-new",
        sentences=_speech(0, count=9, speaker_id=0),
        speaker_map={0: "Lin"},
        person_map={0: person_id},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    source = report.sources[0]
    assert source.person_public_id == person_id
    assert source.clips
    assert all(clip.rel_path.endswith(".wav") for clip in source.clips)
    # Pre-selected picks stay spread across the timeline rather than bunching
    # at the top of the score order.
    picked = [clip for clip in source.clips if clip.recommended]
    assert 0 < len(picked) <= 3
    assert len({clip.begin_time_ms for clip in picked}) == len(picked)


def test_sources_survive_a_project_that_cannot_be_planned(tmp_path: Path) -> None:
    """A source with no capture plan still ranks; it just cannot one-click."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Ma", sample_count=3, project_id="p-old")
    project_dir = _make_project(
        projects_dir,
        "p-new",
        sentences=_speech(0, count=8, speaker_id=0),
        speaker_map={0: "Ma"},
        person_map={0: rows[0].speaker_public_id},
    )
    # A capture plan needs named speakers; remove the names but keep the link
    # so attribution still succeeds while planning cannot.
    (project_dir / "speakers" / "speaker_map.json").unlink()

    report = find_sample_sources(
        rows[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )

    source = report.sources[0]
    assert source.candidate_count == 8
    assert source.clips == ()


def test_payload_is_json_serializable(tmp_path: Path) -> None:
    """Automation output must survive a JSON round-trip unchanged."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Jin", sample_count=3, project_id="p-old")
    _make_project(
        projects_dir,
        "p-new",
        sentences=_speech(0, count=6, speaker_id=0),
        speaker_map={0: "Jin"},
        person_map={0: rows[0].speaker_public_id},
    )

    report = find_sample_sources(
        rows[0].speaker_public_id, projects_dir=projects_dir, store_dir=store_dir
    )
    payload = json.loads(json.dumps(sample_source_payload(report)))

    assert payload["person_name"] == "Jin"
    assert payload["sources"][0]["project_id"] == "p-new"
    assert payload["sources"][0]["clips"]
    assert payload["scanned_project_count"] == 1


def test_unknown_person_yields_an_empty_report(tmp_path: Path) -> None:
    """A missing person is a normal empty answer, not a crash."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    _seed_person(store_dir, "Kang", sample_count=3, project_id="p-old")

    report = find_sample_sources(
        "vpp-does-not-exist", projects_dir=projects_dir, store_dir=store_dir
    )

    assert report.sources == ()
    assert report.health is None
    assert report.person_name == ""


# ---------------------------------------------------------------------------
# fixtures


def _speech(
    start_ms: int, *, count: int, speaker_id: int, gap_ms: int = 20_000
) -> list[dict]:
    """Build well-formed, well-separated sentences worth capturing."""
    return [
        {
            "begin_time_ms": start_ms + index * gap_ms,
            "end_time_ms": start_ms + index * gap_ms + 8_000,
            "text": f"这是第{index}句用于声纹采样的完整发言内容，长度足够并且信息量充分。",
            "speaker_id": speaker_id,
        }
        for index in range(count)
    ]


def _make_project(
    root: Path,
    project_id: str,
    *,
    sentences: list[dict],
    speaker_map: dict[int, str] | None = None,
    person_map: dict[int, str] | None = None,
    title: str = "会议",
) -> Path:
    """Write a minimal on-disk project the sourcing scan can read."""
    project_dir = root / project_id
    (project_dir / "asr").mkdir(parents=True)
    (project_dir / "speakers").mkdir(parents=True)
    created_at = "2026-05-11T12:00:00+08:00"
    manifest = {
        "schema_version": 1,
        "project_id": project_id,
        "title": title,
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
    }
    (project_dir / "project.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(
            {
                "full_text": "".join(item["text"] for item in sentences),
                "sentences": sentences,
                "detected_speakers": sorted(
                    {
                        item["speaker_id"]
                        for item in sentences
                        if item["speaker_id"] is not None
                    }
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if speaker_map:
        (project_dir / "speakers" / "speaker_map.json").write_text(
            json.dumps({str(k): v for k, v in speaker_map.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    if person_map:
        (project_dir / "speakers" / "speaker_person_map.json").write_text(
            json.dumps({str(k): v for k, v in person_map.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    return project_dir


def _seed_person(
    store_dir: Path,
    name: str,
    *,
    sample_count: int,
    project_id: str,
    seconds: float = 8.0,
    ranges: list[tuple[int, int]] | None = None,
) -> list:
    """Store embedded samples for one person, sourced from one project."""
    db_path = get_voiceprint_db_path(store_dir)
    _provider, model = resolve_voiceprint_embedding_options(provider=None, model=None)
    source = store_dir / f"{name}-source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"seed")
    duration_ms = int(seconds * 1000)
    samples = []
    for index in range(sample_count):
        clip_path = store_dir / "clips" / name / project_id / f"clip_{index}.wav"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"{name}-{project_id}-{index}".encode())
        if ranges is not None:
            begin, end = ranges[index]
        else:
            begin, end = index * 90_000, index * 90_000 + duration_ms
        samples.append(
            StoredVoiceprintSample(
                speaker_name=name,
                project_id=project_id,
                project_path=store_dir,
                project_speaker_id=0,
                source_path=source,
                clip_path=clip_path,
                clip_rel_path=str(clip_path.relative_to(store_dir)),
                source_begin_time_ms=begin,
                source_end_time_ms=end,
                clip_begin_time_ms=0,
                clip_end_time_ms=end - begin,
                transcript_text=f"{name} sample {index}",
            )
        )
    _db, rows = store_voiceprint_samples_with_rows(samples, db_path)
    for offset, row in enumerate(rows):
        upsert_voiceprint_embedding(row.sample_id, model, [1.0, offset * 0.01], db_path)
    return list(rows)


def test_overlap_risk_clips_are_never_pre_selected(tmp_path: Path) -> None:
    """A pre-ticked checkbox is a recommendation; never recommend a mixture.

    Measured on a real two-party call, clips taken where the other person is
    within half a second stored 10-13% of that person's voice and pulled the
    speaker's centroid toward them. Such clips stay listed and selectable --
    a thin source may offer nothing else -- but must not arrive checked.
    """
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Ni", sample_count=3, project_id="p-old")
    person_id = rows[0].speaker_public_id
    # Speaker 0's turns each butt straight against speaker 1's.
    sentences: list[dict] = []
    for index in range(8):
        base = index * 30_000
        sentences.append(
            {
                "begin_time_ms": base,
                "end_time_ms": base + 8_000,
                "text": f"这是第{index}句用于声纹采样的完整发言内容，长度足够信息量充分。",
                "speaker_id": 0,
            }
        )
        sentences.append(
            {
                "begin_time_ms": base + 8_000,
                "end_time_ms": base + 14_000,
                "text": f"这是另一个人的第{index}句回应内容，同样足够长且信息充分。",
                "speaker_id": 1,
            }
        )
    _make_project(
        projects_dir,
        "p-crosstalk",
        sentences=sentences,
        speaker_map={0: "Ni", 1: "Other"},
        person_map={0: person_id},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    source = report.sources[0]
    assert source.clips, "clips should still be offered"
    assert all(clip.overlap_risk for clip in source.clips)
    assert not any(clip.recommended for clip in source.clips)


def test_clean_clips_win_the_default_picks_over_overlapping_ones(
    tmp_path: Path,
) -> None:
    """With a clean alternative available, the crowded segment loses."""
    store_dir = tmp_path / "voiceprints"
    projects_dir = tmp_path / "projects"
    rows = _seed_person(store_dir, "Ou", sample_count=3, project_id="p-old")
    person_id = rows[0].speaker_public_id
    sentences: list[dict] = []
    # Four clean turns with wide silence around them.
    for index in range(4):
        base = index * 60_000
        sentences.append(
            {
                "begin_time_ms": base,
                "end_time_ms": base + 8_000,
                "text": f"这是第{index}句独自连续发言的内容，长度足够信息量充分。",
                "speaker_id": 0,
            }
        )
    # One crowded turn, immediately followed by the other speaker.
    sentences.append(
        {
            "begin_time_ms": 300_000,
            "end_time_ms": 312_000,
            "text": "这是一段很长很完整信息量也很充分的发言但紧接着别人就开口了。",
            "speaker_id": 0,
        }
    )
    sentences.append(
        {
            "begin_time_ms": 312_000,
            "end_time_ms": 318_000,
            "text": "这是另一个人紧接着说的话，中间没有任何停顿。",
            "speaker_id": 1,
        }
    )
    _make_project(
        projects_dir,
        "p-mixed",
        sentences=sentences,
        speaker_map={0: "Ou", 1: "Other"},
        person_map={0: person_id},
    )

    report = find_sample_sources(
        person_id, projects_dir=projects_dir, store_dir=store_dir
    )

    clips = {clip.begin_time_ms: clip for clip in report.sources[0].clips}
    crowded = clips[300_000]
    assert crowded.overlap_risk
    assert not crowded.recommended
    picked = [clip for clip in clips.values() if clip.recommended]
    assert picked and all(not clip.overlap_risk for clip in picked)
