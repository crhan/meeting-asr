"""Evaluation helpers for transcript polish quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.correction_llm import LlmCorrectionCandidate, LlmPolishItem
from app.models import SentenceSegment
from app.residual_noise import residual_noise
from app.transcript_corrections import _is_change_type_allowed, _polish_guard

PolishExpectedDecision = Literal["change", "no_change", "reject"]


@dataclass(frozen=True, slots=True)
class PolishEvalCase:
    """One transcript polish evaluation case.

    Args:
        case_id: Stable case identifier.
        original_text: Source ASR sentence.
        expected_decision: Expected final behavior.
        expected_text: Required output for ``change`` cases.
        proposed_text: Candidate proposal used by offline/adversarial eval.
        change_type: Candidate polish change type.
        previous_text: Previous transcript sentence for cross-sentence borrow checks.
        next_text: Next transcript sentence for cross-sentence borrow checks.
        must_keep: Terms that must remain in the accepted output.
        category: Case category, such as dup/filler/term/protected/borrow.
        source: Human-readable source note.
        rationale: Why this case exists.
    """

    case_id: str
    original_text: str
    expected_decision: PolishExpectedDecision
    expected_text: str | None = None
    proposed_text: str | None = None
    change_type: str = ""
    previous_text: str = ""
    next_text: str = ""
    must_keep: tuple[str, ...] = ()
    category: str = ""
    source: str = ""
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class PolishEvalCaseResult:
    """Evaluation result for one polish case."""

    case_id: str
    passed: bool
    expected_decision: PolishExpectedDecision
    actual_decision: str
    expected_text: str
    actual_text: str
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResidualHit:
    """One accepted polish output that still carried residual noise."""

    case_id: str
    text: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolishEvalSummary:
    """Aggregate polish evaluation result."""

    total: int
    passed: int
    failed: int
    results: tuple[PolishEvalCaseResult, ...]
    residual_dirty: tuple[ResidualHit, ...] = ()

    @property
    def success(self) -> bool:
        """Return whether all cases passed and no accepted output is dirty.

        Residual noise in an accepted output is a dimension ORTHOGONAL to the
        keep/reject guard axis: a polish can correctly survive the guard (and even
        match its expected text) yet still leave a filler / uncollapsed repeat in.
        The eval gate fails on either a case failure or any residual-dirty output.
        """
        return self.failed == 0 and not self.residual_dirty


def default_polish_eval_path() -> Path:
    """Return the repository default polish eval case path."""
    return Path(__file__).resolve().parents[2] / "evals" / "polish_cases.jsonl"


def load_polish_eval_cases(path: Path) -> list[PolishEvalCase]:
    """Load JSONL polish evaluation cases.

    Args:
        path: JSONL file path.

    Returns:
        Parsed evaluation cases.
    """
    cases: list[PolishEvalCase] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        cases.append(_case_from_payload(payload, line_number=line_number, path=path))
    return cases


def evaluate_polish_cases(
    cases: list[PolishEvalCase],
    proposed_items: dict[str, LlmPolishItem] | None = None,
) -> PolishEvalSummary:
    """Evaluate polish cases with either supplied items or per-case proposals.

    Args:
        cases: Evaluation cases.
        proposed_items: Optional model output keyed by case id.

    Returns:
        Aggregate summary.
    """
    results = tuple(_evaluate_case(case, proposed_items or {}) for case in cases)
    passed = sum(1 for result in results if result.passed)
    residual_dirty = tuple(
        ResidualHit(result.case_id, result.actual_text, tuple(reasons))
        for result in results
        if result.actual_decision == "kept"
        and (reasons := residual_noise(result.actual_text))
    )
    return PolishEvalSummary(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
        residual_dirty=residual_dirty,
    )


def cases_to_llm_candidates(
    cases: list[PolishEvalCase],
) -> list[LlmCorrectionCandidate]:
    """Convert change/no-change cases into LLM candidates for live model eval."""
    return [
        LlmCorrectionCandidate(
            candidate_id=case.case_id,
            sentence_id=index,
            speaker_name="Speaker",
            text=case.original_text,
        )
        for index, case in enumerate(cases)
        if case.expected_decision != "reject"
    ]


def _case_from_payload(
    payload: dict, *, line_number: int, path: Path
) -> PolishEvalCase:
    """Build one case from a decoded JSON object."""
    required = ("id", "original_text", "expected_decision")
    for key in required:
        if not payload.get(key):
            raise ValueError(f"{path}:{line_number}: missing required field {key!r}")
    expected_decision = payload["expected_decision"]
    if expected_decision not in {"change", "no_change", "reject"}:
        raise ValueError(
            f"{path}:{line_number}: invalid expected_decision {expected_decision!r}"
        )
    must_keep = payload.get("must_keep") or []
    if isinstance(must_keep, str):
        must_keep = [must_keep]
    return PolishEvalCase(
        case_id=str(payload["id"]),
        original_text=str(payload["original_text"]),
        expected_decision=expected_decision,
        expected_text=_optional_text(payload.get("expected_text")),
        proposed_text=_optional_text(payload.get("proposed_text")),
        change_type=str(payload.get("change_type") or ""),
        previous_text=str(payload.get("previous_text") or ""),
        next_text=str(payload.get("next_text") or ""),
        must_keep=tuple(str(item) for item in must_keep),
        category=str(payload.get("category") or ""),
        source=str(payload.get("source") or ""),
        rationale=str(payload.get("rationale") or ""),
    )


def _optional_text(value: object) -> str | None:
    """Return a non-empty string or None."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _evaluate_case(
    case: PolishEvalCase, proposed_items: dict[str, LlmPolishItem]
) -> PolishEvalCaseResult:
    """Evaluate one case."""
    item = proposed_items.get(case.case_id)
    proposed_text = item.corrected_text if item is not None else case.proposed_text
    change_type = item.change_type if item is not None else case.change_type
    if proposed_text is None:
        return _evaluate_absent_proposal(case)
    actual_decision = _proposal_decision(case, proposed_text, change_type)
    actual_text = proposed_text if actual_decision == "kept" else case.original_text
    return _judge_case(case, actual_decision, actual_text)


def _evaluate_absent_proposal(case: PolishEvalCase) -> PolishEvalCaseResult:
    """Evaluate a case where the model produced no proposal."""
    if case.expected_decision == "no_change":
        return _result(
            case, True, "no_change", case.original_text, "no proposal expected"
        )
    return _result(case, False, "no_change", case.original_text, "expected a proposal")


def _proposal_decision(
    case: PolishEvalCase, proposed_text: str, change_type: str
) -> str:
    """Return kept/reject/no_change for one proposed edit."""
    if proposed_text == case.original_text:
        return "no_change"
    if not _is_change_type_allowed(change_type):
        return f"reject_unknown_type:{change_type}"
    verdict = _polish_guard(
        1, _sentences_for_case(case), case.original_text, proposed_text
    )
    if verdict is not None:
        return f"reject:{verdict}"
    return "kept"


def _sentences_for_case(case: PolishEvalCase) -> list[SentenceSegment]:
    """Build minimal sentence context for guard checks."""
    return [
        SentenceSegment(0, 1000, case.previous_text, None, 0),
        SentenceSegment(1000, 2000, case.original_text, None, 1),
        SentenceSegment(2000, 3000, case.next_text, None, 2),
    ]


def _judge_case(
    case: PolishEvalCase, actual_decision: str, actual_text: str
) -> PolishEvalCaseResult:
    """Judge the final behavior against expected behavior."""
    if case.expected_decision == "reject":
        passed = actual_decision.startswith("reject")
        reason = (
            "rejected unsafe proposal" if passed else "unsafe proposal was not rejected"
        )
        return _result(case, passed, actual_decision, actual_text, reason)
    if case.expected_decision == "no_change":
        passed = actual_text == case.original_text and actual_decision != "kept"
        reason = "left unchanged" if passed else "changed a no-change case"
        return _result(case, passed, actual_decision, actual_text, reason)
    expected_text = case.expected_text or ""
    keeps_required = all(term in actual_text for term in case.must_keep)
    passed = (
        actual_decision == "kept" and actual_text == expected_text and keeps_required
    )
    reason = (
        "accepted expected correction"
        if passed
        else _change_failure_reason(case, actual_decision, actual_text)
    )
    return _result(case, passed, actual_decision, actual_text, reason)


def _change_failure_reason(
    case: PolishEvalCase, actual_decision: str, actual_text: str
) -> str:
    """Return a compact failure reason for expected-change cases."""
    if actual_decision != "kept":
        return f"expected kept change, got {actual_decision}"
    missing = [term for term in case.must_keep if term not in actual_text]
    if missing:
        return f"missing protected term(s): {', '.join(missing)}"
    return "corrected text mismatch"


def _result(
    case: PolishEvalCase,
    passed: bool,
    actual_decision: str,
    actual_text: str,
    reason: str,
) -> PolishEvalCaseResult:
    """Build one result object."""
    return PolishEvalCaseResult(
        case_id=case.case_id,
        passed=passed,
        expected_decision=case.expected_decision,
        actual_decision=actual_decision,
        expected_text=case.expected_text or case.original_text,
        actual_text=actual_text,
        category=case.category,
        reason=reason,
    )
