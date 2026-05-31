"""Tests for cross-segment transcript merge."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app import transcript_merge as tm


# ---------------------------------------------------------------------------
# Synthetic project builders
# ---------------------------------------------------------------------------


def _sentence(
    speaker_id: int | None, text: str, begin_ms: int, end_ms: int, sid: int
) -> dict:
    return {
        "begin_time_ms": begin_ms,
        "end_time_ms": end_ms,
        "text": text,
        "speaker_id": speaker_id,
        "sentence_id": sid,
    }


def _meaningful(speaker_id: int, base_ms: int, *, who: str, sid: int) -> list[dict]:
    """Two substantial sentences so the low-information filter keeps the track."""
    return [
        _sentence(
            speaker_id,
            f"{who}这边先同步一下本周稳定性专项的整体进展和接下来的安排。",
            base_ms,
            base_ms + 4000,
            sid,
        ),
        _sentence(
            speaker_id,
            f"{who}觉得这个方案的关键风险还是在资源协调和上线节奏上面。",
            base_ms + 4000,
            base_ms + 8000,
            sid + 1,
        ),
    ]


def _make_project(
    root: Path,
    project_id: str,
    *,
    title: str = "周会",
    meeting_time: str | None = "2026-05-11T11:00:00+08:00",
    created_at: str = "2026-05-11T12:00:00+08:00",
    duration_seconds: float | None = 100.0,
    sentences: list[dict],
    speaker_map: dict[int, str] | None = None,
    person_map: dict[int, str] | None = None,
    ignored: set[int] | None = None,
    corrected_sentences: list[dict] | None = None,
) -> Path:
    project_dir = root / project_id
    (project_dir / "asr").mkdir(parents=True)
    (project_dir / "speakers").mkdir(parents=True)
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
            "meeting_time": meeting_time,
        },
        "audio": (
            {"duration_seconds": duration_seconds}
            if duration_seconds is not None
            else {}
        ),
    }
    (project_dir / "project.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    def _write_sentences(name: str, items: list[dict]) -> None:
        payload = {
            "full_text": "".join(item["text"] for item in items),
            "sentences": items,
            "detected_speakers": sorted(
                {item["speaker_id"] for item in items if item["speaker_id"] is not None}
            ),
        }
        (project_dir / "asr" / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    _write_sentences("sentences.json", sentences)
    if corrected_sentences is not None:
        _write_sentences("sentences_corrected.json", corrected_sentences)
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
    if ignored:
        (project_dir / "speakers" / "speaker_ignore.json").write_text(
            json.dumps({"ignored_speakers": sorted(ignored)}, ensure_ascii=False),
            encoding="utf-8",
        )
    return project_dir


def _resolver(names: dict[str, str]) -> Callable[[str], str | None]:
    return lambda vpp: names.get(vpp)


def _no_store() -> Callable[[str], str | None]:
    return lambda vpp: None


def _identity_for(result: tm.MergeResult, display_name: str) -> dict:
    matches = [i for i in result.identities if i["display_name"] == display_name]
    assert len(matches) == 1, f"expected one {display_name!r}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# Name normalization unit tests
# ---------------------------------------------------------------------------


def test_name_fold_strips_ime_spaces_and_normalizes() -> None:
    assert tm.name_fold("墨泪 ") == tm.name_fold("墨泪")
    assert tm.name_fold("墨泪 ") == tm.name_fold("墨泪")
    # Fullwidth parentheses normalize to halfwidth via NFKC, kept whole.
    assert tm.name_fold("张辉洲（尺木）") == tm.name_fold("张辉洲(尺木)")


def test_is_placeholder_name() -> None:
    assert tm.is_placeholder_name("待确认发言人2")
    assert tm.is_placeholder_name("说话人")
    assert tm.is_placeholder_name("")
    assert tm.is_placeholder_name(None)
    assert not tm.is_placeholder_name("墨泪")
    assert not tm.is_placeholder_name("张辉洲(尺木)")


# ---------------------------------------------------------------------------
# Cross-segment speaker unification
# ---------------------------------------------------------------------------


def test_cross_segment_vpp_unifies_named_and_unnamed(tmp_path: Path) -> None:
    """Same vpp across segments is one person, even when one segment is unnamed."""
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        sentences=_meaningful(0, 0, who="墨泪", sid=1),
        speaker_map={0: "墨泪"},
        person_map={0: "vpp-1111"},
    )
    # Segment B: same person (vpp-1111) but this segment never named them.
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(2, 0, who="墨泪", sid=1),
        person_map={2: "vpp-1111"},
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    墨泪 = _identity_for(result, "墨泪")
    assert 墨泪["identity_kind"] == "vpp"
    assert 墨泪["vpp"] == "vpp-1111"
    assert {m["order"] for m in 墨泪["members"]} == {0, 1}
    # Only one global id; both segments map their local id onto it.
    assert len([i for i in result.identities if i["identity_kind"] == "vpp"]) == 1


def test_store_name_overrides_segment_local_name(tmp_path: Path) -> None:
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        sentences=_meaningful(0, 0, who="老名字", sid=1),
        speaker_map={0: "旧花名"},
        person_map={0: "vpp-1111"},
    )
    result = tm.merge_projects([seg], vpp_name_resolver=_resolver({"vpp-1111": "墨泪"}))
    assert any(i["display_name"] == "墨泪" for i in result.identities)
    assert not any(i["display_name"] == "旧花名" for i in result.identities)


def test_name_to_vpp_promotion_default_on(tmp_path: Path) -> None:
    """A name-only speaker folds onto a matching voiceprint identity."""
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        sentences=_meaningful(0, 0, who="墨泪", sid=1),
        speaker_map={0: "墨泪"},  # name only, no vpp
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(1, 0, who="墨泪", sid=1),
        speaker_map={1: "墨泪"},
        person_map={1: "vpp-1111"},
    )
    merged = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    墨泪 = _identity_for(merged, "墨泪")
    assert 墨泪["identity_kind"] == "vpp"
    assert 墨泪.get("promoted_from_name") is True

    split = tm.merge_projects(
        [seg_a, seg_b], name_to_vpp=False, vpp_name_resolver=_no_store()
    )
    assert len([i for i in split.identities if i["display_name"] == "墨泪"]) == 2


def test_intra_project_vpp_collapse(tmp_path: Path) -> None:
    """Two local ids with the same vpp inside one segment collapse to one."""
    sentences = _meaningful(0, 0, who="宵恩", sid=1) + _meaningful(
        3, 9000, who="宵恩", sid=10
    )
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        sentences=sentences,
        speaker_map={0: "宵恩"},
        person_map={0: "vpp-9999", 3: "vpp-9999"},
    )
    result = tm.merge_projects([seg], vpp_name_resolver=_no_store())
    宵恩 = _identity_for(result, "宵恩")
    assert {(m["local_speaker_id"]) for m in 宵恩["members"]} == {0, 3}


def test_intra_project_name_collapse(tmp_path: Path) -> None:
    sentences = _meaningful(0, 0, who="墨泪", sid=1) + _meaningful(
        2, 9000, who="墨泪", sid=10
    )
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        sentences=sentences,
        speaker_map={0: "墨泪", 2: "墨泪"},
    )
    result = tm.merge_projects([seg], vpp_name_resolver=_no_store())
    墨泪 = _identity_for(result, "墨泪")
    assert {m["local_speaker_id"] for m in 墨泪["members"]} == {0, 2}


# ---------------------------------------------------------------------------
# Anonymous / placeholder handling
# ---------------------------------------------------------------------------


def test_placeholder_name_is_anonymous_and_never_merged(tmp_path: Path) -> None:
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        sentences=_meaningful(0, 0, who="某甲", sid=1),
        speaker_map={0: "待确认发言人2"},
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(0, 0, who="某乙", sid=1),
        speaker_map={0: "待确认发言人2"},
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    anon = [i for i in result.identities if i["identity_kind"] == "anon"]
    assert len(anon) == 2
    assert all(i["display_name"].startswith("Speaker ") for i in anon)


def test_anonymous_speakers_get_distinct_labels(tmp_path: Path) -> None:
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        sentences=_meaningful(0, 0, who="某甲", sid=1),
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(0, 0, who="某乙", sid=1),
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    labels = {i["display_name"] for i in result.identities}
    assert labels == {"Speaker A", "Speaker B"}


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_continuous_timeline_offsets_and_monotonic(tmp_path: Path) -> None:
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        duration_seconds=100.0,
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        duration_seconds=50.0,
        sentences=_meaningful(0, 0, who="乙", sid=1),
        speaker_map={0: "乙"},
        person_map={0: "vpp-b"},
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    assert [m.clock_offset_ms for m in result.metas_raw] == [0, 100_000]
    times = [(s.begin_time_ms, s.end_time_ms) for s in result.merged_raw.sentences]
    # Strictly non-decreasing begins, every cue ends after it begins.
    assert times == sorted(times)
    assert all(end > begin for begin, end in times)
    # Second segment's first sentence starts at the offset.
    assert result.merged_raw.sentences[2].begin_time_ms == 100_000


def test_naive_and_aware_meeting_time_sort_without_error(tmp_path: Path) -> None:
    aware = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(0, 0, who="晚", sid=1),
        speaker_map={0: "晚"},
        person_map={0: "vpp-late"},
    )
    naive = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11 11:00:00",  # no timezone
        sentences=_meaningful(0, 0, who="早", sid=1),
        speaker_map={0: "早"},
        person_map={0: "vpp-early"},
    )
    result = tm.merge_projects([aware, naive], vpp_name_resolver=_no_store())
    assert result.order_source == "meeting_time"
    # Naive 11:00 (assumed +08:00) sorts before aware 13:00.
    assert [m.project_id for m in result.metas_raw] == [
        "p-bbbbbbbbbbbbbbbb",
        "p-aaaaaaaaaaaaaaaa",
    ]


# ---------------------------------------------------------------------------
# Corrected variant, ignored, dedupe, single segment
# ---------------------------------------------------------------------------


def test_corrected_fallback_to_raw_when_segment_missing(tmp_path: Path) -> None:
    raw_a = _meaningful(0, 0, who="甲", sid=1)
    corrected_a = _meaningful(0, 0, who="甲修", sid=1)
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        sentences=raw_a,
        corrected_sentences=corrected_a,
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(0, 0, who="乙", sid=1),  # no corrected
        speaker_map={0: "乙"},
        person_map={0: "vpp-b"},
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    assert result.use_corrected is True
    assert result.metas_corrected is not None
    assert [m.corrected for m in result.metas_corrected] == [True, False]
    assert any("无 polish" in w for w in result.warnings)


def test_ignored_speaker_kept_anonymous(tmp_path: Path) -> None:
    sentences = _meaningful(0, 0, who="主讲", sid=1) + _meaningful(
        2, 9000, who="杂音", sid=10
    )
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        sentences=sentences,
        speaker_map={0: "主讲"},
        person_map={0: "vpp-a", 2: "vpp-x"},  # speaker 2 has a vpp but is ignored
        ignored={2},
    )
    result = tm.merge_projects(
        [seg], vpp_name_resolver=_resolver({"vpp-x": "不该出现"})
    )
    # Ignored speaker is kept but anonymous, never attributed to its voiceprint.
    assert not any(i["display_name"] == "不该出现" for i in result.identities)
    anon = [i for i in result.identities if i["identity_kind"] == "anon"]
    assert len(anon) == 1
    assert anon[0]["sentence_count"] == 2
    assert result.metas_raw[0].ignored_speaker_count == 1


def test_duplicate_project_ref_deduped(tmp_path: Path) -> None:
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        duration_seconds=100.0,
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    result = tm.merge_projects([seg, seg], vpp_name_resolver=_no_store())
    assert len(result.metas_raw) == 1
    assert any("重复" in w for w in result.warnings)


def test_single_segment_degenerates_without_headers(tmp_path: Path) -> None:
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    result = tm.merge_projects([seg], vpp_name_resolver=_no_store())
    text = tm.render_merged_text(
        result.merged_raw, result.metas_raw, result.mapping, corrected=False
    )
    assert "# ━━" not in text  # no segment header
    assert "甲:" in text
    assert len(result.metas_raw) == 1
    # The corrected variant also degenerates without headers.
    corrected_source = result.merged_corrected or result.merged_raw
    corrected_metas = result.metas_corrected or result.metas_raw
    corrected_text = tm.render_merged_text(
        corrected_source, corrected_metas, result.mapping, corrected=True
    )
    assert "# ━━" not in corrected_text


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def test_write_merge_outputs_creates_package(tmp_path: Path) -> None:
    seg_a = _make_project(
        tmp_path / "store",
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        duration_seconds=100.0,
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
        corrected_sentences=_meaningful(0, 0, who="甲", sid=1),
    )
    seg_b = _make_project(
        tmp_path / "store",
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        duration_seconds=50.0,
        sentences=_meaningful(0, 0, who="乙", sid=1),
        speaker_map={0: "乙"},
        person_map={0: "vpp-b"},
        corrected_sentences=_meaningful(0, 0, who="乙", sid=1),
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    out_dir = tmp_path / "out"
    outputs = tm.write_merge_outputs(result, out_dir)

    assert outputs.transcript.exists()
    assert outputs.transcript_corrected is not None
    assert outputs.subtitle.exists()
    assert outputs.manifest.exists()

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["timeline_mode"] == "concatenated"
    assert manifest["order_source"] == "meeting_time"
    assert [s["part"] for s in manifest["segments"]] == ["段1", "段2"]
    # Both segments carry polished text, so each is flagged corrected.
    assert [s["corrected"] for s in manifest["segments"]] == [True, True]
    assert manifest["meeting"]["participants"] == ["甲", "乙"]

    transcript = outputs.transcript.read_text(encoding="utf-8")
    assert "# ━━ 段1/2" in transcript and "# ━━ 段2/2" in transcript

    srt = outputs.subtitle.read_text(encoding="utf-8")
    # SRT indices are continuous across segments.
    assert srt.splitlines()[0] == "1"

    # Corrected subtitle is a valid, continuously-numbered SRT too.
    assert outputs.subtitle_corrected is not None
    assert outputs.subtitle_corrected.exists()
    assert "merged_corrected" in outputs.subtitle_corrected.name
    corrected_srt = outputs.subtitle_corrected.read_text(encoding="utf-8")
    assert corrected_srt.splitlines()[0] == "1"


def test_write_refuses_nonempty_without_force(tmp_path: Path) -> None:
    seg = _make_project(
        tmp_path / "store",
        "p-aaaaaaaaaaaaaaaa",
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    result = tm.merge_projects([seg], vpp_name_resolver=_no_store())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        tm.write_merge_outputs(result, out_dir)
    # --force allows overwriting.
    outputs = tm.write_merge_outputs(result, out_dir, force=True)
    assert outputs.manifest.exists()


def test_no_projects_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tm.merge_projects([], vpp_name_resolver=_no_store())


def test_low_information_filter_propagates(tmp_path: Path) -> None:
    """include_low_information must thread through to load_transcript_result."""
    filler = [
        _sentence(1, "嗯。", 9000, 9500, 10),
        _sentence(1, "对。", 9500, 10000, 11),
        _sentence(1, "啊。", 10000, 10500, 12),
    ]
    seg = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        sentences=_meaningful(0, 0, who="主讲", sid=1) + filler,
        speaker_map={0: "主讲"},
        person_map={0: "vpp-a"},  # speaker 1 is an anonymous filler-only track
    )
    filtered = tm.merge_projects([seg], vpp_name_resolver=_no_store())
    kept = tm.merge_projects(
        [seg], include_low_information=True, vpp_name_resolver=_no_store()
    )
    assert len(filtered.merged_raw.sentences) == 2
    assert len(kept.merged_raw.sentences) == 5
    assert not any(i["identity_kind"] == "anon" for i in filtered.identities)
    assert any(i["identity_kind"] == "anon" for i in kept.identities)


def test_sort_falls_back_to_cli_order_when_timestamps_unparseable(
    tmp_path: Path,
) -> None:
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time=None,
        created_at="未知",  # unparseable
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time=None,
        created_at="未知",
        sentences=_meaningful(0, 0, who="乙", sid=1),
        speaker_map={0: "乙"},
        person_map={0: "vpp-b"},
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    assert result.order_source == "cli_fallback"
    assert [m.project_id for m in result.metas_raw] == [
        "p-aaaaaaaaaaaaaaaa",
        "p-bbbbbbbbbbbbbbbb",
    ]
    assert any("回退命令行顺序" in w for w in result.warnings)


def test_name_conflicts_recorded_for_same_vpp(tmp_path: Path) -> None:
    """Distinct local names for one voiceprint person are recorded, not lost."""
    seg_a = _make_project(
        tmp_path,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "墨泪"},
        person_map={0: "vpp-1111"},
    )
    seg_b = _make_project(
        tmp_path,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        sentences=_meaningful(1, 0, who="乙", sid=1),
        speaker_map={1: "墨泪哥"},  # same vpp, different name
        person_map={1: "vpp-1111"},
    )
    result = tm.merge_projects([seg_a, seg_b], vpp_name_resolver=_no_store())
    vpp_ids = [i for i in result.identities if i.get("vpp") == "vpp-1111"]
    assert len(vpp_ids) == 1
    assert vpp_ids[0]["name_conflicts"] == ["墨泪", "墨泪哥"]


def test_slice_by_counts() -> None:
    assert tm._slice_by_counts([1, 2, 3, 4, 5], [2, 3]) == [[1, 2], [3, 4, 5]]
    assert tm._slice_by_counts([1, 2], [2, 0]) == [[1, 2], []]
    assert tm._slice_by_counts([], [0]) == [[]]


def test_project_merge_cli_end_to_end(tmp_path: Path) -> None:
    """The CLI command resolves refs, writes a package, and emits JSON."""
    from typer.testing import CliRunner

    from app.cli import app

    store = tmp_path / "store"
    seg_a = _make_project(
        store,
        "p-aaaaaaaaaaaaaaaa",
        meeting_time="2026-05-11T11:00:00+08:00",
        duration_seconds=100.0,
        sentences=_meaningful(0, 0, who="甲", sid=1),
        speaker_map={0: "甲"},
        person_map={0: "vpp-a"},
    )
    seg_b = _make_project(
        store,
        "p-bbbbbbbbbbbbbbbb",
        meeting_time="2026-05-11T13:00:00+08:00",
        duration_seconds=50.0,
        sentences=_meaningful(0, 0, who="乙", sid=1),
        speaker_map={0: "乙"},
        person_map={0: "vpp-b"},
    )
    out_dir = tmp_path / "out"
    empty_store = tmp_path / "empty-vp"  # no voiceprint db -> resolver returns None
    empty_store.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "project",
            "merge",
            str(seg_a),
            str(seg_b),
            "--out",
            str(out_dir),
            "--store-dir",
            str(empty_store),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["timeline_mode"] == "concatenated"
    assert [s["part"] for s in payload["segments"]] == ["段1", "段2"]
    outputs = payload["outputs"]
    assert set(outputs) == {
        "out_dir",
        "transcript",
        "transcript_corrected",
        "subtitle",
        "subtitle_corrected",
        "manifest",
    }
    assert Path(outputs["transcript"]).exists()
    assert Path(outputs["manifest"]).exists()
