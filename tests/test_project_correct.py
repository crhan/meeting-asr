"""Tests for editor-driven project vocabulary correction."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

from typer.testing import CliRunner

from app.commands import project as project_commands
from app.commands import project_correct as project_correct_commands
from app.cli import app
from app.config import Settings
from app.core.progress import CliProgressEvent
from app.correction_llm import (
    LlmCorrectionResult,
    LlmPolishItem,
    LlmReplacementRule,
    LlmStrictPolishResult,
)
from app.correction_types import CorrectionEditOptions, CorrectionEditSummary
from app.project_manager import create_project, load_manifest, project_paths
from app.speaker_tui import SentenceCorrectionEdit, SpeakerReviewDecision

runner = CliRunner()


def test_project_correct_edit_writes_corrected_outputs_and_learns_context(tmp_path: Path) -> None:
    """Editing the review file should write corrected artifacts and lexicon context."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")
    lexicon_db = tmp_path / "lexicon.sqlite"

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
            "--yes",
            "--lexicon-db",
            str(lexicon_db),
            "--category",
            "system",
        ],
    )

    assert result.exit_code == 0
    assert "Vocabulary correction accepted." in result.output
    assert "Changed sentences: 1" in result.output
    assert "Learned contexts: 1" in result.output
    assert "艾赛" in (project_dir / "asr" / "sentences.json").read_text(encoding="utf-8")
    assert "iSee" in (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert "敬悦: 我们看一下iSee系统。" in (
        project_dir / "exports" / "transcript_named_corrected.txt"
    ).read_text(encoding="utf-8")
    assert (project_dir / "exports" / "subtitle_named_corrected.srt").exists()
    assert (project_dir / "corrections" / "asr_hotwords.json").exists()
    hotwords = json.loads((project_dir / "corrections" / "asr_hotwords.json").read_text(encoding="utf-8"))
    assert hotwords["dashscope_vocabulary"] == [{"text": "iSee", "weight": 4}]
    assert _fetch_one(lexicon_db, "SELECT canonical FROM terms") == "iSee"
    assert _fetch_one(lexicon_db, "SELECT alias FROM aliases") == "艾赛"
    assert _fetch_one(lexicon_db, "SELECT category FROM terms") == "system"


def test_project_correct_edit_no_open_only_creates_review_file(tmp_path: Path) -> None:
    """No-open mode should let users inspect the generated review file without applying changes."""
    project_dir = _sample_project(tmp_path)

    result = runner.invoke(app, ["project", "correct", "edit", str(project_dir), "--no-open"])
    review_files = list((project_dir / "tmp" / "corrections").glob("review_*.md"))

    assert result.exit_code == 0
    assert "Changed sentences: 0" in result.output
    assert review_files
    assert "meeting-asr: sentence_id=1" in review_files[0].read_text(encoding="utf-8")
    assert not (project_dir / "asr" / "sentences_corrected.json").exists()


def test_project_correct_edit_can_leave_proposal_pending(tmp_path: Path) -> None:
    """Without acceptance, edit should produce proposal files but not final artifacts."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Vocabulary correction proposal ready." in result.output
    assert "Correction proposal left pending." in result.output
    assert list((project_dir / "tmp" / "corrections").glob("proposal_*.json"))
    assert not (project_dir / "asr" / "sentences_corrected.json").exists()


def test_project_correct_accept_applies_latest_proposal(tmp_path: Path) -> None:
    """Accept command should apply a pending proposal and learn contexts."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")
    lexicon_db = tmp_path / "lexicon.sqlite"
    runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    result = runner.invoke(
        app,
        ["project", "correct", "accept", str(project_dir), "--lexicon-db", str(lexicon_db)],
    )

    assert result.exit_code == 0
    assert "Vocabulary correction accepted." in result.output
    assert "iSee" in (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert _fetch_one(lexicon_db, "SELECT canonical FROM terms") == "iSee"


def test_project_correct_edit_can_use_existing_review_file(tmp_path: Path) -> None:
    """Existing edited review files should be reusable for proposal generation."""
    project_dir = _sample_project(tmp_path)
    runner.invoke(app, ["project", "correct", "edit", str(project_dir), "--no-open"])
    review_file = next((project_dir / "tmp" / "corrections").glob("review_*.md"))
    review_file.write_text(review_file.read_text(encoding="utf-8").replace("艾赛", "iSee"), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--review-file",
            str(review_file),
            "--from-original",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Sample changes: 1" in result.output


def test_project_correct_understands_whole_ascii_term_replacement(tmp_path: Path) -> None:
    """Partial character diffs inside ASCII terms should learn the whole term."""
    project_dir = _sample_project_with_text(tmp_path, "我们看一下 ic 系统。")
    editor_script = _editor_script(tmp_path, "ic", "isee")

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "ic -> isee" in result.output
    assert "C -> see" not in result.output
    assert "c -> see" not in result.output


def test_project_correct_uses_ai_understanding_for_chinese_terms(tmp_path: Path, monkeypatch) -> None:
    """Chinese sample corrections should use model-inferred word boundaries."""
    project_dir = _sample_project_with_text(tmp_path, "我们要建设云原声平台。")
    editor_script = _editor_script(tmp_path, "云原声", "云原生")

    monkeypatch.setattr(
        "app.correction_understanding.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )
    monkeypatch.setattr(
        "app.correction_understanding.infer_vocabulary_replacements",
        lambda **_: [LlmReplacementRule("云原声", "云原生", "建设", "平台")],
    )
    monkeypatch.setattr(
        "app.transcript_corrections.propose_vocabulary_corrections",
        lambda **kwargs: LlmCorrectionResult("云原声应为云原生", {}, kwargs["model"]),
    )

    result = runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-proposal-open",
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "云原声 -> 云原生" in result.output
    assert "声 -> 生" not in result.output
    proposal = _latest_proposal(project_dir)
    assert proposal["sample_changes"][0]["replacements"][0]["wrong_text"] == "云原声"
    assert proposal["proposed_changes"][0]["replacements"][0]["wrong_text"] == "云原声"


def test_project_correct_polish_creates_non_lexicon_proposal(tmp_path: Path, monkeypatch) -> None:
    """Transcript polish should propose sentence rewrites without learning vocabulary replacements."""
    bad_text = "这个入参的时候输出什么，然后出参的时候输出什么，就类有点类似于出参跟入参记录起来"
    good_text = "入参的时候输出什么，出参的时候输出什么，有点类似于把入参和出参记录起来。"
    project_dir = _sample_project_with_text(tmp_path, bad_text)

    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )
    monkeypatch.setattr(
        "app.transcript_corrections.propose_transcript_polish",
        lambda **kwargs: LlmCorrectionResult("修复口语语序", {"c0": good_text}, kwargs["model"]),
    )

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir), "--legacy-polish"], input="n\n")

    assert result.exit_code == 0
    assert "Transcript polish proposal ready." in result.output
    assert "Correction proposal left pending." in result.output
    proposal = _latest_proposal(project_dir)
    assert proposal["category"] == "polish"
    assert proposal["sample_changes"] == []
    assert proposal["proposed_changes"][0]["corrected_text"] == good_text
    assert proposal["proposed_changes"][0]["replacements"] == []
    assert not (project_dir / "asr" / "sentences_corrected.json").exists()
    diff_result = runner.invoke(app, ["project", "correct", "diff", str(project_dir)])
    assert diff_result.exit_code == 0
    assert good_text in diff_result.output


def test_project_correct_polish_runs_batches_in_parallel(tmp_path: Path, monkeypatch) -> None:
    """Transcript polish should run independent LLM batches concurrently."""
    texts = [f"第 {index} 句话需要轻量修复。" for index in range(1, 91)]
    project_dir = _sample_project_with_sentences(tmp_path, texts)
    lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(
            dashscope_api_key="key",
            dashscope_base_url=None,
            dashscope_correction_model="qwen-test",
            dashscope_correction_concurrency=3,
        ),
    )

    def fake_propose_transcript_polish(**kwargs):
        nonlocal active_calls, max_active_calls
        candidates = kwargs["candidates"]
        with lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.05)
        with lock:
            active_calls -= 1
        first = candidates[0]
        return LlmCorrectionResult("并发修复", {first.candidate_id: first.text + " 已修复。"}, kwargs["model"])

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish", fake_propose_transcript_polish)

    result = runner.invoke(
        app,
        ["project", "correct", "polish", str(project_dir), "--concurrency", "3", "--legacy-polish"],
        input="n\n",
    )

    assert result.exit_code == 0
    assert max_active_calls == 3
    proposal = _latest_proposal(project_dir)
    assert proposal["category"] == "polish"
    assert len(proposal["proposed_changes"]) == 3


def test_project_correct_polish_reports_batch_progress(tmp_path: Path, monkeypatch) -> None:
    """Transcript polish progress should expose batches, parallelism, and completion."""
    texts = [f"第 {index} 句话需要轻量修复。" for index in range(1, 91)]
    project_dir = _sample_project_with_sentences(tmp_path, texts)
    events = []

    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(
            dashscope_api_key="key",
            dashscope_base_url=None,
            dashscope_correction_model="qwen-test",
            dashscope_correction_concurrency=3,
        ),
    )

    def fake_propose_transcript_polish(**kwargs):
        first = kwargs["candidates"][0]
        return LlmCorrectionResult("并发修复", {first.candidate_id: first.text + " 已修复。"}, kwargs["model"])

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish", fake_propose_transcript_polish)

    summary = project_correct_commands.prepare_transcript_polish_for_review(
        paths=project_paths(project_dir),
        manifest=load_manifest(project_dir),
        speaker_mapping={0: "敬悦"},
        options=CorrectionEditOptions(
            open_editor=False,
            open_proposal=False,
            category="polish",
            model="qwen-test",
            polish_concurrency=3,
            polish_legacy=True,
        ),
        progress=events.append,
    )

    polish_events = [
        event
        for event in events
        if event.description and "Generating transcript polish proposal" in event.description
    ]
    assert summary.proposed_change_count == 3
    assert polish_events
    assert polish_events[0].total == 3
    assert polish_events[0].completed == 0
    assert "batches 0/3" in polish_events[0].description
    assert "parallel 3" in polish_events[0].description
    assert "active 3" in polish_events[0].description
    assert polish_events[-1].total == 3
    assert polish_events[-1].completed == 3
    assert "batches 3/3" in polish_events[-1].description
    assert "active 0" in polish_events[-1].description


def test_project_correct_polish_command_wires_progress_reporter(tmp_path: Path, monkeypatch) -> None:
    """The standalone polish command should not drop batch progress events."""
    project_dir = _sample_project_with_text(tmp_path, "这个句子需要润色。")
    events = []

    def fake_run_with_progress(operation, **kwargs):
        assert kwargs["description"] == "Generating transcript polish proposal"
        assert kwargs["enabled"] is True
        assert kwargs["structured_log"] is False
        return operation(events.append)

    def fake_prepare_transcript_polish(**kwargs):
        progress = kwargs["progress"]
        progress(CliProgressEvent("Generating transcript polish proposal | batches 0/1"))
        return CorrectionEditSummary(
            review_path=project_dir / "tmp" / "corrections" / "review_polish_test.md",
            proposal_path=None,
            proposal_diff_path=None,
            proposal_json_path=None,
            change_count=0,
            sample_change_count=0,
            proposed_change_count=0,
            learned_count=0,
            accepted=False,
            model="qwen-test",
            model_error=None,
            understanding=[],
            corrected_sentences_path=None,
            corrected_transcript_path=None,
            corrected_named_transcript_path=None,
            corrected_srt_path=None,
            hotwords_path=None,
            applied_path=None,
            lexicon_db=None,
        )

    monkeypatch.setattr(project_correct_commands, "run_with_progress", fake_run_with_progress)
    monkeypatch.setattr(project_correct_commands, "prepare_transcript_polish", fake_prepare_transcript_polish)

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    assert result.exit_code == 0
    assert events
    assert "batches 0/1" in events[0].description


def test_project_correct_polish_strict_reports_agent_log_batches(tmp_path: Path, monkeypatch) -> None:
    """Strict polish should expose batch state to structured agent logs."""
    texts = [f"第 {index} 句话需要清理。" for index in range(1, 25)]
    project_dir = _sample_project_with_sentences(tmp_path, texts)
    events = []

    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(
            dashscope_api_key="key",
            dashscope_base_url=None,
            dashscope_correction_model="qwen-test",
            dashscope_correction_concurrency=2,
        ),
    )

    def fake_strict(**kwargs):
        first = kwargs["candidates"][0]
        return LlmStrictPolishResult(
            "清噪",
            [LlmPolishItem(first.candidate_id, first.text + " 已清理。", "filler", "测试")],
            kwargs["model"],
        )

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", fake_strict)

    project_correct_commands.prepare_transcript_polish_for_review(
        paths=project_paths(project_dir),
        manifest=load_manifest(project_dir),
        speaker_mapping={0: "敬悦"},
        options=CorrectionEditOptions(
            open_editor=False,
            open_proposal=False,
            category="polish",
            model="qwen-test",
            polish_concurrency=2,
        ),
        progress=events.append,
    )

    heartbeat_events = [event for event in events if event.log_kind == "heartbeat" and event.stage == "polish"]
    assert heartbeat_events
    assert dict(heartbeat_events[0].log_fields)["batch"] == "0/2"
    assert dict(heartbeat_events[-1].log_fields)["batch"] == "2/2"


def test_project_correct_polish_default_uses_strict_path_and_records_change_type(
    tmp_path: Path, monkeypatch
) -> None:
    """Polish without --legacy-polish must call the strict LLM path and persist change_type/reason."""
    sentences = [
        "然后用用这个CLI去拿。",          # dup: 用用 -> 用
        "我觉得这个方案可能不太需要做。",  # protected: 我觉得 + 可能 must survive
        "用codekex这个嘛，又不用钱。",     # term: codekex -> Codex
    ]
    project_dir = _sample_project_with_sentences(tmp_path, sentences)
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )

    def fake_strict(**kwargs):
        items = []
        for cand in kwargs["candidates"]:
            if "用用" in cand.text:
                items.append(LlmPolishItem(cand.candidate_id, "然后用这个CLI去拿。", "dup", "ASR 同字重复"))
            elif "codekex" in cand.text:
                items.append(LlmPolishItem(cand.candidate_id, "用Codex这个嘛，又不用钱。", "term", "ASR 误识别"))
        return LlmStrictPolishResult("修字+清噪", items, kwargs["model"])

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", fake_strict)

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    assert result.exit_code == 0, result.output
    proposal = _latest_proposal(project_dir)
    changes = proposal["proposed_changes"]
    assert {c["change_type"] for c in changes} == {"dup", "term"}
    types_to_text = {c["change_type"]: (c["original_text"], c["corrected_text"]) for c in changes}
    assert types_to_text["dup"][1] == "然后用这个CLI去拿。"
    assert types_to_text["term"][1] == "用Codex这个嘛，又不用钱。"
    # Sentence containing protected words ("我觉得"/"可能") was not proposed by LLM,
    # so it must remain absent from the proposal entirely.
    assert all("我觉得" not in c["original_text"] for c in changes)


def test_project_correct_polish_strict_surfaces_total_failure(tmp_path: Path, monkeypatch) -> None:
    """When every strict batch fails, polish must surface the legacy-style recovery context, not a silent no-op."""
    project_dir = _sample_project_with_text(tmp_path, "用codekex这个嘛。")
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )

    def always_fail(**_kwargs):
        raise TimeoutError("read timeout")

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", always_fail)

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    assert result.exit_code == 0
    manifest = load_manifest(project_dir)
    assert "Transcript polish" in result.output
    assert "Model fallback:" in result.output
    assert "stage=polish" in result.output
    assert f"project_id={manifest.project_id}" in result.output
    assert f"meeting-asr project correct polish {manifest.project_id}" in result.output
    proposal_files = sorted((project_dir / "tmp" / "corrections").glob("proposal_*.json"))
    assert proposal_files == []


def test_project_correct_polish_strict_reports_partial_failure(tmp_path: Path, monkeypatch) -> None:
    """When some strict batches fail but others succeed, polish must surface partial failure via model_error."""
    sentences = [f"第 {i} 句话需要修复。" for i in range(1, 25)]  # 2 batches of 12
    project_dir = _sample_project_with_sentences(tmp_path, sentences)
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )
    call_state = {"count": 0}

    def half_fail(**kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise TimeoutError("read timeout")
        first = kwargs["candidates"][0]
        return LlmStrictPolishResult(
            "ok",
            [LlmPolishItem(first.candidate_id, first.text + " 修复后。", "punct", "")],
            kwargs["model"],
        )

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", half_fail)

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    assert result.exit_code == 0
    assert "Model fallback:" in result.output
    assert "Strict polish completed with partial failures" in result.output
    proposal = _latest_proposal(project_dir)
    assert proposal["model_error"] and "partial failures" in proposal["model_error"]
    assert len(proposal["proposed_changes"]) >= 1


def test_project_correct_polish_strict_guard_rejects_protected_word_deletion(
    tmp_path: Path, monkeypatch
) -> None:
    """Strict guard must reject any LLM proposal that strips a protected attitude word."""
    project_dir = _sample_project_with_text(tmp_path, "这个我觉得可能不需要做。")
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )

    def fake_strict(**kwargs):
        cand = kwargs["candidates"][0]
        return LlmStrictPolishResult(
            "wrongly stripped attitude",
            [LlmPolishItem(cand.candidate_id, "这个不需要做。", "filler", "压成事实")],
            kwargs["model"],
        )

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", fake_strict)

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    assert result.exit_code == 0
    # Guard nuked the only proposed change → no proposal file; sidecar still records the rejection.
    proposal_files = sorted((project_dir / "tmp" / "corrections").glob("proposal_*.json"))
    assert proposal_files == []
    sidecars = sorted((project_dir / "tmp" / "corrections").glob("polish_strict_meta_*.json"))
    sidecar = json.loads(sidecars[-1].read_text(encoding="utf-8"))
    decisions = [it["decision"] for it in sidecar["items"] if it["decision"].startswith("reject")]
    assert any("protected_word_deleted" in d for d in decisions)


def test_project_correct_accept_select_indices(tmp_path: Path, monkeypatch) -> None:
    """`accept --select` should apply only the listed proposed changes."""
    project_dir = _sample_project_with_sentences(
        tmp_path, ["然后用用这个CLI。", "用codekex这个嘛。"]
    )
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )

    def fake_strict(**kwargs):
        items = []
        for cand in kwargs["candidates"]:
            if "用用" in cand.text:
                items.append(LlmPolishItem(cand.candidate_id, "然后用这个CLI。", "dup", ""))
            else:
                items.append(LlmPolishItem(cand.candidate_id, "用Codex这个嘛。", "term", ""))
        return LlmStrictPolishResult("两类", items, kwargs["model"])

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", fake_strict)

    runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")
    proposal_before = _latest_proposal(project_dir)
    assert len(proposal_before["proposed_changes"]) == 2

    accept_result = runner.invoke(
        app, ["project", "correct", "accept", str(project_dir), "--select", "1"]
    )
    assert accept_result.exit_code == 0, accept_result.output
    corrected = (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert "Codex" in corrected
    assert "用用" in corrected  # change index 0 (dup) NOT applied


def test_project_correct_accept_types_filter(tmp_path: Path, monkeypatch) -> None:
    """`accept --types term` should apply only changes whose primary change_type is term."""
    project_dir = _sample_project_with_sentences(
        tmp_path, ["然后用用这个CLI。", "用codekex这个嘛。"]
    )
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )

    def fake_strict(**kwargs):
        items = []
        for cand in kwargs["candidates"]:
            if "用用" in cand.text:
                items.append(LlmPolishItem(cand.candidate_id, "然后用这个CLI。", "dup|filler", ""))
            else:
                items.append(LlmPolishItem(cand.candidate_id, "用Codex这个嘛。", "term", ""))
        return LlmStrictPolishResult("两类", items, kwargs["model"])

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", fake_strict)

    runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    accept_result = runner.invoke(
        app, ["project", "correct", "accept", str(project_dir), "--types", "term"]
    )
    assert accept_result.exit_code == 0, accept_result.output
    corrected = (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert "Codex" in corrected
    assert "用用" in corrected  # dup-tagged change NOT applied


def test_project_correct_accept_rejects_unknown_type_token(tmp_path: Path, monkeypatch) -> None:
    """Invalid --types tokens must produce an error rather than silently ignore them."""
    project_dir = _sample_project_with_text(tmp_path, "用codekex这个嘛。")
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )
    monkeypatch.setattr(
        "app.transcript_corrections.propose_transcript_polish_strict",
        lambda **kwargs: LlmStrictPolishResult(
            "x",
            [LlmPolishItem(kwargs["candidates"][0].candidate_id, "用Codex这个嘛。", "term", "")],
            kwargs["model"],
        ),
    )
    runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    accept_result = runner.invoke(
        app, ["project", "correct", "accept", str(project_dir), "--types", "bogus"]
    )
    assert accept_result.exit_code != 0
    assert "bogus" in accept_result.output


def test_project_correct_accept_rejects_stale_proposal(tmp_path: Path, monkeypatch) -> None:
    """Accept should fail if the source transcript changed after proposal generation."""
    project_dir = _sample_project_with_text(tmp_path, "用codekex这个嘛。")
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )
    monkeypatch.setattr(
        "app.transcript_corrections.propose_transcript_polish_strict",
        lambda **kwargs: LlmStrictPolishResult(
            "x",
            [LlmPolishItem(kwargs["candidates"][0].candidate_id, "用Codex这个嘛。", "term", "")],
            kwargs["model"],
        ),
    )
    runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")
    source_path = project_dir / "asr" / "sentences.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["sentences"][0]["text"] = "这句话已经被别人改过。"
    source_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    accept_result = runner.invoke(app, ["project", "correct", "accept", str(project_dir)])

    assert accept_result.exit_code != 0
    assert "Correction proposal is stale" in accept_result.output
    assert not (project_dir / "asr" / "sentences_corrected.json").exists()


def test_project_correct_polish_proposal_markdown_groups_by_change_type(
    tmp_path: Path, monkeypatch
) -> None:
    """When change_type is present, the proposal markdown should render grouped sections."""
    project_dir = _sample_project_with_sentences(
        tmp_path, ["然后用用这个CLI。", "用codekex这个嘛。"]
    )
    monkeypatch.setattr(
        "app.transcript_corrections.load_settings",
        lambda **_: Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_correction_model="qwen-test"),
    )

    def fake_strict(**kwargs):
        items = []
        for cand in kwargs["candidates"]:
            if "用用" in cand.text:
                items.append(LlmPolishItem(cand.candidate_id, "然后用这个CLI。", "dup", "同字重复"))
            else:
                items.append(LlmPolishItem(cand.candidate_id, "用Codex这个嘛。", "term", "ASR 误识别"))
        return LlmStrictPolishResult("两类", items, kwargs["model"])

    monkeypatch.setattr("app.transcript_corrections.propose_transcript_polish_strict", fake_strict)
    runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

    proposal_md = sorted((project_dir / "tmp" / "corrections").glob("proposal_*.md"))[-1].read_text(encoding="utf-8")
    assert "Proposed Changes (grouped by change_type)" in proposal_md
    assert "### dup" in proposal_md
    assert "### term" in proposal_md
    assert "Counts:" in proposal_md
    assert "dup=1" in proposal_md
    assert "term=1" in proposal_md


def test_project_transcript_show_can_select_corrected_output(tmp_path: Path) -> None:
    """Corrected transcript artifacts should be viewable through project transcript show."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")

    runner.invoke(
        app,
        [
            "project",
            "correct",
            "edit",
            str(project_dir),
            "--editor",
            f"{sys.executable} {editor_script}",
            "--no-ai",
            "--no-proposal-open",
            "--yes",
        ],
    )
    result = runner.invoke(app, ["project", "transcript", "show", str(project_dir), "--kind", "corrected"])

    assert result.exit_code == 0
    assert "iSee" in result.output


def test_project_review_correction_action_uses_editor_correction_flow(tmp_path: Path) -> None:
    """Project review's correction action should reuse the CLI editor-diff workflow."""
    project_dir = _sample_project(tmp_path)
    editor_script = _editor_script(tmp_path, "艾赛", "iSee")
    lexicon_db = tmp_path / "lexicon.sqlite"
    options = project_commands.ProjectReviewCorrectionOptions(
        edit_options=CorrectionEditOptions(
            editor=f"{sys.executable} {editor_script}",
            open_editor=True,
            open_proposal=False,
            category="system",
            lexicon_db=lexicon_db,
            use_ai=False,
        ),
        yes=True,
    )

    project_commands._handle_speaker_review_decision(
        project_dir,
        SpeakerReviewDecision(saved=True, mapping={0: "敬悦"}, action="correct"),
        options,
    )

    assert "iSee" in (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert "敬悦: 我们看一下iSee系统。" in (
        project_dir / "exports" / "transcript_named_corrected.txt"
    ).read_text(encoding="utf-8")
    assert _fetch_one(lexicon_db, "SELECT category FROM terms") == "system"


def test_project_review_inline_correction_does_not_open_editor(tmp_path: Path) -> None:
    """TUI sentence edits should reuse correction processing without launching Code/editor."""
    project_dir = _sample_project(tmp_path)
    lexicon_db = tmp_path / "lexicon.sqlite"
    options = project_commands.ProjectReviewCorrectionOptions(
        edit_options=CorrectionEditOptions(
            editor="definitely-missing-editor",
            open_editor=True,
            open_proposal=True,
            category="system",
            lexicon_db=lexicon_db,
            use_ai=False,
        ),
        yes=True,
    )

    project_commands._handle_speaker_review_decision(
        project_dir,
        SpeakerReviewDecision(
            saved=True,
            mapping={0: "敬悦"},
            action="correct-inline",
            correction_edit=SentenceCorrectionEdit(
                sentence_id=1,
                speaker_id=0,
                begin_time_ms=1000,
                end_time_ms=1500,
                original_text="我们看一下艾赛系统。",
                corrected_text="我们看一下iSee系统。",
            ),
        ),
        options,
    )

    assert "iSee" in (project_dir / "asr" / "sentences_corrected.json").read_text(encoding="utf-8")
    assert "敬悦: 我们看一下iSee系统。" in (
        project_dir / "exports" / "transcript_named_corrected.txt"
    ).read_text(encoding="utf-8")
    assert _fetch_one(lexicon_db, "SELECT canonical FROM terms") == "iSee"


def test_project_review_inline_correction_keeps_multiple_tui_edits(tmp_path: Path) -> None:
    """Project review should process every staged TUI text edit in one proposal."""
    project_dir = _sample_project_with_sentences(
        tmp_path,
        ["我们看一下艾赛系统。", "AS服务需要修正。"],
    )
    lexicon_db = tmp_path / "lexicon.sqlite"
    options = project_commands.ProjectReviewCorrectionOptions(
        edit_options=CorrectionEditOptions(
            editor="definitely-missing-editor",
            open_editor=True,
            open_proposal=True,
            category="system",
            lexicon_db=lexicon_db,
            use_ai=False,
        ),
        yes=True,
    )

    project_commands._handle_speaker_review_decision(
        project_dir,
        SpeakerReviewDecision(
            saved=True,
            mapping={0: "敬悦"},
            action="correct-inline",
            correction_edits=(
                SentenceCorrectionEdit(1, 0, 1000, 1500, "我们看一下艾赛系统。", "我们看一下iSee系统。"),
                SentenceCorrectionEdit(2, 0, 2000, 2500, "AS服务需要修正。", "IaaS服务需要修正。"),
            ),
        ),
        options,
    )

    corrected = (project_dir / "exports" / "transcript_named_corrected.txt").read_text(encoding="utf-8")
    assert "敬悦: 我们看一下iSee系统。" in corrected
    assert "敬悦: IaaS服务需要修正。" in corrected
    proposal = _latest_proposal(project_dir)
    assert len(proposal["sample_changes"]) == 2


def test_project_review_can_accept_selected_correction_changes(tmp_path: Path) -> None:
    """Accepting a proposal with selected indices should exclude rejected changes."""
    project_dir = _sample_project_with_sentences(
        tmp_path,
        ["我们看一下艾赛系统。", "AS服务需要修正。"],
    )
    paths = project_paths(project_dir)
    manifest = load_manifest(project_dir)
    options = CorrectionEditOptions(use_ai=False, lexicon_db=tmp_path / "lexicon.sqlite")
    summary = project_correct_commands.prepare_inline_corrections_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping={0: "敬悦"},
        correction_edits=[
            SentenceCorrectionEdit(1, 0, 1000, 1500, "我们看一下艾赛系统。", "我们看一下iSee系统。"),
            SentenceCorrectionEdit(2, 0, 2000, 2500, "AS服务需要修正。", "IaaS服务需要修正。"),
        ],
        options=options,
    )

    project_correct_commands.accept_correction_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping={0: "敬悦"},
        proposal_path=summary.proposal_json_path,
        lexicon_db=options.lexicon_db,
        selected_change_indices=(0,),
    )

    corrected = (project_dir / "exports" / "transcript_named_corrected.txt").read_text(encoding="utf-8")
    assert "敬悦: 我们看一下iSee系统。" in corrected
    assert "敬悦: AS服务需要修正。" in corrected
    assert "IaaS服务需要修正" not in corrected


def _sample_project(tmp_path: Path) -> Path:
    """Create a project fixture with one mapped speaker and one ASR error."""
    return _sample_project_with_text(tmp_path, "我们看一下艾赛系统。")


def _sample_project_with_text(tmp_path: Path, text: str) -> Path:
    """Create a project fixture with one mapped speaker and custom transcript text."""
    return _sample_project_with_sentences(tmp_path, [text])


def _sample_project_with_sentences(tmp_path: Path, texts: list[str]) -> Path:
    """Create a project fixture with one mapped speaker and custom transcript sentences."""
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "projects" / "demo"
    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path / "projects",
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    sentence_payloads = [_sentence_payload(index, text) for index, text in enumerate(texts, start=1)]
    sentences = {
        "full_text": "\n".join(texts),
        "detected_speakers": [0],
        "sentences": sentence_payloads,
    }
    (project_dir / "asr" / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    (project_dir / "speakers" / "speaker_map.json").write_text('{"0": "敬悦"}\n', encoding="utf-8")
    return project_dir


def _sentence_payload(index: int, text: str) -> dict:
    """Build one transcript sentence payload."""
    begin = index * 1000
    return {
        "begin_time_ms": begin,
        "end_time_ms": begin + 500,
        "text": text,
        "speaker_id": 0,
        "sentence_id": index,
    }


def _editor_script(tmp_path: Path, old: str, new: str) -> Path:
    """Write an editor script that replaces text in the review file."""
    script = tmp_path / f"editor_{old}_{new}.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "path = Path(sys.argv[1])",
                f"path.write_text(path.read_text(encoding='utf-8').replace({old!r}, {new!r}), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return script


def _latest_proposal(project_dir: Path) -> dict:
    """Load the latest correction proposal JSON."""
    proposal_path = sorted((project_dir / "tmp" / "corrections").glob("proposal_*.json"))[-1]
    return json.loads(proposal_path.read_text(encoding="utf-8"))


def _fetch_one(db_path: Path, query: str) -> str:
    """Fetch a single SQLite string value."""
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(query).fetchone()
    assert row is not None
    return str(row[0])
