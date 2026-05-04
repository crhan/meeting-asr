"""DashScope-assisted vocabulary correction proposal generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import dashscope
from dashscope import Generation

from app.config import Settings
from app.utils import retry


@dataclass(frozen=True, slots=True)
class LlmCorrectionCandidate:
    """One sentence candidate sent to the correction model."""

    candidate_id: str
    sentence_id: int | None
    speaker_name: str
    text: str


@dataclass(frozen=True, slots=True)
class LlmCorrectionSample:
    """One user-edited sample used as correction evidence."""

    original_text: str
    corrected_text: str
    replacements: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class LlmCorrectionResult:
    """Structured correction proposal returned by a text model."""

    understanding: str
    corrected_text_by_id: dict[str, str]
    model: str


@dataclass(frozen=True, slots=True)
class LlmReplacementRule:
    """One term-level replacement inferred from edited samples."""

    wrong_text: str
    corrected_text: str
    left_context: str = ""
    right_context: str = ""


def propose_vocabulary_corrections(
    *,
    samples: list[LlmCorrectionSample],
    candidates: list[LlmCorrectionCandidate],
    settings: Settings,
    model: str,
) -> LlmCorrectionResult:
    """
    Ask DashScope to propose full-document vocabulary corrections.

    Args:
        samples: User-edited examples that define the correction intent.
        candidates: Candidate transcript sentences to inspect.
        settings: Runtime DashScope settings.
        model: DashScope text generation model id.

    Returns:
        Structured correction proposal.
    """
    _configure_dashscope(settings)
    prompt = _build_prompt(samples, candidates)

    def _call() -> Any:
        response = Generation.call(
            model=model,
            api_key=settings.dashscope_api_key,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            temperature=0.1,
        )
        _raise_for_generation_error(response)
        return response

    content = _extract_generation_text(retry(_call, attempts=3, delay_seconds=1.0))
    return _parse_result(content, model=model, candidate_ids={item.candidate_id for item in candidates})


def infer_vocabulary_replacements(
    *,
    samples: list[LlmCorrectionSample],
    settings: Settings,
    model: str,
) -> list[LlmReplacementRule]:
    """
    Ask DashScope to infer term-level replacements from edited samples.

    Args:
        samples: User-edited examples with before/after text and local diff spans.
        settings: Runtime DashScope settings.
        model: DashScope text generation model id.

    Returns:
        Validated term-level replacement rules.
    """
    _configure_dashscope(settings)
    prompt = _build_replacement_prompt(samples)

    def _call() -> Any:
        response = Generation.call(
            model=model,
            api_key=settings.dashscope_api_key,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            temperature=0.1,
        )
        _raise_for_generation_error(response)
        return response

    content = _extract_generation_text(retry(_call, attempts=3, delay_seconds=1.0))
    return _parse_replacement_rules(content)


def _configure_dashscope(settings: Settings) -> None:
    """
    Configure DashScope SDK globals.

    Args:
        settings: Runtime settings.

    Returns:
        None.
    """
    dashscope.api_key = settings.dashscope_api_key
    if settings.dashscope_base_url:
        for attr in ("base_http_api_url", "base_url"):
            if hasattr(dashscope, attr):
                setattr(dashscope, attr, settings.dashscope_base_url)


def _system_prompt() -> str:
    """
    Return the fixed correction system prompt.

    Returns:
        System prompt text.
    """
    return "你是会议转写词汇纠错助手。只输出 JSON，不要输出 Markdown，不要解释。"


def _build_prompt(samples: list[LlmCorrectionSample], candidates: list[LlmCorrectionCandidate]) -> str:
    """
    Build a bounded correction prompt.

    Args:
        samples: User-edited correction examples.
        candidates: Candidate transcript sentences.

    Returns:
        Prompt text.
    """
    payload = {
        "samples": [_sample_payload(item) for item in samples],
        "candidates": [_candidate_payload(item) for item in candidates],
    }
    return (
        "根据用户已经人工修改过的样例，推断会议转写里的专有词、人名、系统名纠错规则，"
        "然后只对 candidates 里的句子提出必要修改。\n"
        "要求：\n"
        "1. 样例修改是最高优先级证据。\n"
        "2. 只修复词汇识别错误，不要润色、总结、扩写或改变说话风格。\n"
        "3. 如果某个 candidate 不需要修改，不要返回它。\n"
        "4. corrected_text 必须保留原句结构，只替换必要词汇。\n"
        "5. 返回 JSON 对象，字段为 understanding 和 corrections。\n"
        "6. corrections 是数组，每项包含 id, corrected_text, reason。\n\n"
        f"输入：\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _build_replacement_prompt(samples: list[LlmCorrectionSample]) -> str:
    """
    Build a prompt for extracting term-level correction rules.

    Args:
        samples: User-edited correction examples.

    Returns:
        Prompt text.
    """
    payload = {"samples": [_sample_payload(item) for item in samples]}
    return (
        "根据用户人工修改过的会议转写样例，抽取真正的词汇级纠错规则。\n"
        "要求：\n"
        "1. original_text 和 corrected_text 是最高优先级证据。\n"
        "2. replacements 只是程序本地 diff 的初步结果，可能把中文词或英文术语切碎，不能盲信。\n"
        "3. 中文没有空格边界时，必须结合整句语境判断专有词、人名、系统名的完整边界。\n"
        "4. wrong_text 必须原样出现在 original_text 中，corrected_text 必须原样出现在 corrected_text 中。\n"
        "5. 只抽取词汇识别错误，不要抽取语气、标点、润色或句式调整。\n"
        "6. 选择最短的自洽术语边界，不要把整句或无关上下文放进 wrong_text。\n"
        "7. 返回 JSON 对象，字段为 replacements；replacements 是数组，每项包含 "
        "wrong_text, corrected_text, left_context, right_context, reason。\n\n"
        f"输入：\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _sample_payload(sample: LlmCorrectionSample) -> dict[str, object]:
    """Return one JSON-ready sample payload."""
    return {
        "original_text": sample.original_text,
        "corrected_text": sample.corrected_text,
        "replacements": sample.replacements,
    }


def _candidate_payload(candidate: LlmCorrectionCandidate) -> dict[str, object]:
    """Return one JSON-ready candidate payload."""
    return {
        "id": candidate.candidate_id,
        "sentence_id": candidate.sentence_id,
        "speaker_name": candidate.speaker_name,
        "text": candidate.text,
    }


def _raise_for_generation_error(response: Any) -> None:
    """
    Raise when DashScope returns an error response.

    Args:
        response: DashScope response object.

    Returns:
        None.
    """
    status_code = getattr(response, "status_code", None)
    if status_code and int(status_code) >= 400:
        message = getattr(response, "message", None) or getattr(response, "code", None) or response
        raise RuntimeError(f"DashScope correction failed: HTTP {status_code} {message}")


def _extract_generation_text(response: Any) -> str:
    """
    Extract text from common DashScope generation response shapes.

    Args:
        response: DashScope generation response.

    Returns:
        Generated text.
    """
    output = getattr(response, "output", None)
    text = _field(output, "text")
    if text:
        return str(text)
    choices = _field(output, "choices") or []
    if choices:
        message = _field(choices[0], "message")
        content = _field(message, "content")
        if content:
            return str(content)
    raise RuntimeError("DashScope correction response did not contain generated text.")


def _parse_result(text: str, *, model: str, candidate_ids: set[str]) -> LlmCorrectionResult:
    """
    Parse model JSON into a validated correction result.

    Args:
        text: Raw model output.
        model: Model used for generation.
        candidate_ids: Allowed candidate ids.

    Returns:
        Validated correction result.
    """
    payload = _load_json_object(text)
    understanding = str(payload.get("understanding") or "").strip()
    corrections = _parse_corrections(payload.get("corrections"), candidate_ids)
    return LlmCorrectionResult(understanding, corrections, model)


def _parse_replacement_rules(text: str) -> list[LlmReplacementRule]:
    """
    Parse model JSON into replacement rules.

    Args:
        text: Raw model output.

    Returns:
        Validated replacement rules.
    """
    payload = _load_json_object(text)
    value = payload.get("replacements")
    if not isinstance(value, list):
        return []
    rules = []
    for item in value:
        if not isinstance(item, dict):
            continue
        wrong_text = str(item.get("wrong_text") or "").strip()
        corrected_text = str(item.get("corrected_text") or "").strip()
        if wrong_text and corrected_text and wrong_text != corrected_text:
            rules.append(
                LlmReplacementRule(
                    wrong_text=wrong_text,
                    corrected_text=corrected_text,
                    left_context=str(item.get("left_context") or "").strip(),
                    right_context=str(item.get("right_context") or "").strip(),
                )
            )
    return rules


def _load_json_object(text: str) -> dict[str, Any]:
    """
    Load a JSON object from raw model text.

    Args:
        text: Raw model output.

    Returns:
        Parsed JSON object.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match is None:
            raise RuntimeError(f"DashScope correction response was not JSON: {text}") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise RuntimeError("DashScope correction response JSON must be an object.")
    return payload


def _parse_corrections(value: object, candidate_ids: set[str]) -> dict[str, str]:
    """
    Parse and validate correction rows.

    Args:
        value: Raw JSON value.
        candidate_ids: Allowed candidate ids.

    Returns:
        Mapping from candidate id to corrected text.
    """
    if not isinstance(value, list):
        return {}
    parsed: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("id") or "")
        corrected_text = str(item.get("corrected_text") or "").strip()
        if candidate_id in candidate_ids and corrected_text:
            parsed[candidate_id] = corrected_text
    return parsed


def _field(value: Any, name: str) -> Any:
    """Return an attribute or mapping field from SDK response objects."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
