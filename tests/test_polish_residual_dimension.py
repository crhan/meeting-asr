"""Lock the residual-noise dimension wired into the polish eval (CI gate).

Residual noise (filler / uncollapsed repeat left IN the output) is ORTHOGONAL to
the keep/reject guard axis: a polish can correctly survive the guard — and even
match its expected text — yet still be dirty. These tests prove the eval gate
catches that, and that the bundled clean cases stay clean. Synthetic, no
lexicon/network (CI-safe).
"""

from __future__ import annotations

from app.polish_evaluation import (
    PolishEvalCase,
    default_polish_eval_path,
    evaluate_polish_cases,
    load_polish_eval_cases,
)


def _case(**kw) -> PolishEvalCase:
    base = dict(
        case_id="t",
        original_text="",
        expected_decision="change",
        expected_text=None,
        proposed_text=None,
        change_type="dup",
    )
    base.update(kw)
    return PolishEvalCase(**base)


def test_residual_dimension_catches_dirty_accepted_output() -> None:
    """A kept output that matches its expectation but is still dirty fails the gate."""
    dirty = "A就是就是B。"
    case = _case(
        case_id="residual_demo",
        original_text="A就是就是B啊。",
        expected_decision="change",
        expected_text=dirty,  # in-code dirty expectation to isolate the dimension
        proposed_text=dirty,
        change_type="filler",
    )
    summary = evaluate_polish_cases([case])
    # the case itself passes its own criteria (kept + text matches expected)...
    assert summary.results[0].passed is True
    # ...but the residual dimension catches the dirty accepted output
    assert len(summary.residual_dirty) == 1
    assert summary.residual_dirty[0].case_id == "residual_demo"
    assert any("chunk_dup" in r for r in summary.residual_dirty[0].reasons)
    # so the overall gate fails
    assert summary.success is False


def test_clean_accepted_output_passes_dimension() -> None:
    """A kept, clean output produces no residual hit and the gate can succeed."""
    case = _case(
        case_id="clean",
        original_text="这个方案就就是可以。",
        expected_decision="change",
        expected_text="这个方案就是可以。",
        proposed_text="这个方案就是可以。",
        change_type="dup",
    )
    summary = evaluate_polish_cases([case])
    assert summary.residual_dirty == ()
    assert summary.results[0].passed is True
    assert summary.success is True


def test_rejected_dirty_proposal_is_not_a_residual_hit() -> None:
    """Residual only judges ACCEPTED outputs; a rejected proposal isn't counted.

    The proposal is dirty (就是就是) AND trips the guard — it introduces an ASCII
    token (XYZ) absent from the original, so the ascii-hallucination rule rejects
    it. actual_text then falls back to the clean original, so there is no residual
    hit even though the proposed text was dirty.
    """
    case = _case(
        case_id="rejected",
        original_text="这个就是这个意思。",
        expected_decision="reject",
        proposed_text="这个就是就是 XYZ 意思。",
        change_type="dup",
    )
    summary = evaluate_polish_cases([case])
    assert summary.results[0].actual_decision.startswith("reject")
    assert summary.residual_dirty == ()


def test_bundled_eval_cases_have_no_residual_dirt() -> None:
    """Regression gate: no bundled case's accepted output carries residual noise."""
    cases = load_polish_eval_cases(default_polish_eval_path())
    summary = evaluate_polish_cases(cases)
    assert summary.residual_dirty == (), summary.residual_dirty
