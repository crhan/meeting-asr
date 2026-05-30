"""Tests for transcript polish evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.correction_llm import LlmPolishItem
from app.polish_evaluation import evaluate_polish_cases, load_polish_eval_cases

runner = CliRunner()


def test_default_polish_eval_cases_pass_offline() -> None:
    """The checked-in polish eval set should pass under its offline proposals."""
    cases = load_polish_eval_cases(Path("evals/polish_cases.jsonl"))

    summary = evaluate_polish_cases(cases)

    assert summary.success
    assert summary.total >= 8


def test_polish_eval_catches_model_overreach(tmp_path: Path) -> None:
    """A model proposal that rewrites a no-change case should fail."""
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "uncertain",
                "original_text": "你有以假换体钉钉投。",
                "expected_decision": "no_change",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_polish_eval_cases(cases_path)

    summary = evaluate_polish_cases(
        cases,
        {
            "uncertain": LlmPolishItem(
                "uncertain", "你有一个钉钉投屏。", "term", "猜测修复"
            )
        },
    )

    assert not summary.success
    assert summary.results[0].reason == "changed a no-change case"


def test_project_correct_eval_polish_command_reports_summary() -> None:
    """The CLI should expose the default polish eval set, all cases passing."""
    total = len(load_polish_eval_cases(Path("evals/polish_cases.jsonl")))

    result = runner.invoke(app, ["project", "correct", "eval-polish"])

    assert result.exit_code == 0, result.output
    assert f"Polish eval: {total}/{total} passed" in result.output


def test_project_correct_eval_polish_command_fails_on_bad_case(tmp_path: Path) -> None:
    """The CLI should fail non-zero when cases do not pass."""
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "bad",
                "original_text": "然后用用这个CLI。",
                "expected_decision": "change",
                "expected_text": "然后用这个CLI。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["project", "correct", "eval-polish", "--cases", str(cases_path)]
    )

    assert result.exit_code != 0
    assert "FAIL bad" in result.output
