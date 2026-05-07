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
from app.correction_llm import LlmCorrectionResult, LlmReplacementRule
from app.correction_types import CorrectionEditOptions
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

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir)], input="n\n")

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

    result = runner.invoke(app, ["project", "correct", "polish", str(project_dir), "--concurrency", "3"], input="n\n")

    assert result.exit_code == 0
    assert max_active_calls == 3
    proposal = _latest_proposal(project_dir)
    assert proposal["category"] == "polish"
    assert len(proposal["proposed_changes"]) == 3


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
