"""Term-level understanding for transcript correction samples."""

from __future__ import annotations

import re
from dataclasses import asdict, replace

from app.config import load_settings
from app.correction_llm import (
    LlmCorrectionSample,
    LlmReplacementRule,
    infer_vocabulary_replacements,
)
from app.correction_types import (
    CorrectionChange,
    CorrectionEditOptions,
    CorrectionReplacement,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def refine_sample_replacements(
    sample_changes: list[CorrectionChange],
    options: CorrectionEditOptions,
) -> tuple[list[CorrectionChange], str | None]:
    """Use AI to refine Chinese sample replacements when local diff is ambiguous."""
    if (
        not sample_changes
        or not options.use_ai
        or not _needs_contextual_replacement_inference(sample_changes)
    ):
        return sample_changes, None
    try:
        settings = load_settings(require_oss=False, require_dashscope=True)
        model = options.model or settings.dashscope_correction_model
        rules = infer_vocabulary_replacements(
            samples=_llm_samples(sample_changes),
            settings=settings,
            model=model,
        )
        return _apply_llm_replacement_rules(sample_changes, rules), None
    except Exception as exc:
        return sample_changes, f"replacement understanding failed: {exc}"


def join_model_errors(first_error: str | None, second_error: str | None) -> str | None:
    """Join optional model diagnostic messages."""
    errors = [item for item in (first_error, second_error) if item]
    return "; ".join(errors) if errors else None


def matching_correction_replacements(
    change: CorrectionChange,
    rules: list[CorrectionReplacement],
) -> list[CorrectionReplacement]:
    """Return replacement rules grounded in one changed sentence."""
    replacements = []
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        if (
            rule.wrong_text not in change.original_text
            or rule.corrected_text not in change.corrected_text
        ):
            continue
        key = (rule.wrong_text, rule.corrected_text)
        if key in seen:
            continue
        seen.add(key)
        left_context, right_context = replacement_context(
            change.original_text, rule.wrong_text
        )
        replacements.append(
            CorrectionReplacement(
                wrong_text=rule.wrong_text,
                corrected_text=rule.corrected_text,
                left_context=left_context or rule.left_context,
                right_context=right_context or rule.right_context,
            )
        )
    return replacements


def replacement_context(text: str, wrong_text: str) -> tuple[str, str]:
    """Return local context around one replacement term."""
    index = text.find(wrong_text)
    if index < 0:
        return "", ""
    end = index + len(wrong_text)
    return text[max(0, index - 24) : index].strip(), text[end : end + 24].strip()


def _needs_contextual_replacement_inference(
    sample_changes: list[CorrectionChange],
) -> bool:
    """Return whether sample replacements contain Chinese terms needing model context."""
    return any(
        _ambiguous_chinese_span(replacement.wrong_text)
        or _ambiguous_chinese_span(replacement.corrected_text)
        for change in sample_changes
        for replacement in change.replacements
    )


def _apply_llm_replacement_rules(
    sample_changes: list[CorrectionChange],
    rules: list[LlmReplacementRule],
) -> list[CorrectionChange]:
    """Replace local diff spans with model-inferred term-level rules when valid."""
    if not rules:
        return sample_changes
    refined = []
    for change in sample_changes:
        replacements = _matching_llm_replacements(change, rules)
        refined.append(
            replace(change, replacements=replacements or change.replacements)
        )
    return refined


def _matching_llm_replacements(
    change: CorrectionChange,
    rules: list[LlmReplacementRule],
) -> list[CorrectionReplacement]:
    """Return model rules that are grounded in one before/after sentence pair."""
    replacements = []
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        if (
            rule.wrong_text not in change.original_text
            or rule.corrected_text not in change.corrected_text
        ):
            continue
        key = (rule.wrong_text, rule.corrected_text)
        if key in seen:
            continue
        seen.add(key)
        replacements.append(_replacement_from_llm_rule(change, rule))
    return replacements


def _replacement_from_llm_rule(
    change: CorrectionChange, rule: LlmReplacementRule
) -> CorrectionReplacement:
    """Build one grounded replacement from a model-inferred rule."""
    left_context, right_context = replacement_context(
        change.original_text, rule.wrong_text
    )
    return CorrectionReplacement(
        wrong_text=rule.wrong_text,
        corrected_text=rule.corrected_text,
        left_context=rule.left_context or left_context,
        right_context=rule.right_context or right_context,
    )


def _llm_samples(changes: list[CorrectionChange]) -> list[LlmCorrectionSample]:
    """Build model samples from user-edited sentence changes."""
    return [
        LlmCorrectionSample(
            original_text=change.original_text,
            corrected_text=change.corrected_text,
            replacements=[asdict(replacement) for replacement in change.replacements],
        )
        for change in changes
    ]


def _has_cjk(text: str) -> bool:
    """Return whether text contains Chinese characters."""
    return bool(CJK_RE.search(text))


def _ambiguous_chinese_span(text: str) -> bool:
    """Return whether a diff span is too short to trust as a Chinese term."""
    return _has_cjk(text) and len(text.strip()) <= 1
