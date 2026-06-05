"""DashScope-assisted vocabulary correction proposal generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.dashscope_chat import generate_chat_text
from app.utils import retry

DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS = 120


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


@dataclass(frozen=True, slots=True)
class LlmPolishItem:
    """One strict polish proposal item with self-tagged change type."""

    candidate_id: str
    corrected_text: str
    change_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class LlmStrictPolishResult:
    """Strict polish result with per-item metadata."""

    understanding: str
    items: list[LlmPolishItem]
    model: str


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
    prompt = _build_prompt(samples, candidates)

    def _call() -> str:
        return generate_chat_text(
            settings=settings,
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            request_timeout=DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
            temperature=0.1,
            enable_thinking=False,
        )

    content = retry(_call, attempts=3, delay_seconds=1.0)
    return _parse_result(
        content, model=model, candidate_ids={item.candidate_id for item in candidates}
    )


def propose_transcript_polish(
    *,
    candidates: list[LlmCorrectionCandidate],
    settings: Settings,
    model: str,
) -> LlmCorrectionResult:
    """
    Ask DashScope to propose safe full-transcript wording polish.

    Args:
        candidates: Transcript sentences to inspect.
        settings: Runtime DashScope settings.
        model: DashScope text generation model id.

    Returns:
        Structured sentence-level polish proposal.
    """
    prompt = _build_polish_prompt(candidates)

    def _call() -> str:
        return generate_chat_text(
            settings=settings,
            model=model,
            messages=[
                {"role": "system", "content": _polish_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            request_timeout=DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
            temperature=0.1,
            enable_thinking=False,
        )

    content = retry(_call, attempts=1, delay_seconds=1.0)
    return _parse_result(
        content, model=model, candidate_ids={item.candidate_id for item in candidates}
    )


def propose_transcript_polish_strict(
    *,
    candidates: list[LlmCorrectionCandidate],
    settings: Settings,
    model: str,
    request_timeout: int = DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
    attempts: int = 3,
    disambiguations: list[tuple[str, str]] | None = None,
) -> LlmStrictPolishResult:
    """
    Strict polish: only typo / term / case fixes, with self-tagged change_type.

    Args:
        candidates: Transcript sentences to inspect.
        settings: Runtime DashScope settings.
        model: DashScope text generation model id.
        request_timeout: Per-call HTTP timeout in seconds. Larger models like
            qwen3-235b-a22b sometimes need 240s+ for batched polish.
        attempts: Retry attempts for transient timeouts.
        disambiguations: Optional ``(surface, guidance)`` pairs from the lexicon
            for context-dependent ASCII terms.

    Returns:
        Per-item strict polish result.
    """
    prompt = _build_polish_strict_prompt(candidates, disambiguations)

    def _call() -> str:
        return generate_chat_text(
            settings=settings,
            model=model,
            messages=[
                {"role": "system", "content": _polish_strict_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            request_timeout=request_timeout,
            temperature=0.0,
            enable_thinking=False,
        )

    content = retry(_call, attempts=attempts, delay_seconds=2.0)
    return _parse_strict_polish(
        content, model=model, candidate_ids={item.candidate_id for item in candidates}
    )


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
    prompt = _build_replacement_prompt(samples)

    def _call() -> str:
        return generate_chat_text(
            settings=settings,
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            request_timeout=DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
            temperature=0.1,
            enable_thinking=False,
        )

    content = retry(_call, attempts=3, delay_seconds=1.0)
    return _parse_replacement_rules(content)


def _system_prompt() -> str:
    """
    Return the fixed correction system prompt.

    Returns:
        System prompt text.
    """
    return "你是会议转写词汇纠错助手。只输出 JSON，不要输出 Markdown，不要解释。"


def _polish_system_prompt() -> str:
    """
    Return the fixed transcript polish system prompt.

    Returns:
        System prompt text.
    """
    return "你是会议转写可读性修复助手。只输出 JSON，不要输出 Markdown，不要解释。"


def _polish_strict_system_prompt() -> str:
    """
    Strict polish: clean ASR noise for downstream summarization agents.

    Returns:
        System prompt text.
    """
    return (
        "你是会议转写的【降噪+纠错】助手，服务对象是下游的纪要/总结 agent，不是人类听众。"
        "你的目标：清除所有对最终总结无意义的 ASR 噪音（重复、填充、卡顿、失败启动、改口前半），"
        "纠正明显的字面错误，但保留所有承载事实/确定度/共识信号的内容。"
        "情感本身（卡顿、犹豫强度）不重要——但描述确定度的态度词（我觉得/可能/或许）和"
        "寻求共识的词（对吧/是不是）必须保留，因为它们决定下游纪要写"
        "『决议』还是『提议』、『共识』还是『单方陈述』。"
        "只输出 JSON，不要 Markdown，不要解释。"
    )


def _build_prompt(
    samples: list[LlmCorrectionSample], candidates: list[LlmCorrectionCandidate]
) -> str:
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


def _build_polish_prompt(candidates: list[LlmCorrectionCandidate]) -> str:
    """
    Build a prompt for safe transcript wording polish.

    Args:
        candidates: Candidate transcript sentences.

    Returns:
        Prompt text.
    """
    payload = {"candidates": [_candidate_payload(item) for item in candidates]}
    return (
        "逐句检查会议 ASR 转写，提出轻量的可读性修复。\n"
        "目标：把明显破碎的口语 ASR 文本修成可读会议记录，但不能改变说话人的事实含义。\n"
        "允许修改：\n"
        "1. 删除明显重复的口头填充、卡顿和无意义重复。\n"
        "2. 修复 ASR 导致的中文语序断裂、词语错位和不自然表达。\n"
        "3. 根据同一句上下文修正明显术语顺序，例如入参/出参、输入/输出这类成对概念。\n"
        "4. 修正明显的专有词、系统名、技术词误识别，但只在上下文足够明确时修改。\n"
        "禁止修改：\n"
        "1. 不要总结、扩写、补充新信息或改变技术结论。\n"
        "2. 不要改变人名、数字、时间、任务归属、否定/肯定语义。\n"
        "3. 没有把握的句子不要返回。\n"
        "4. 每个 corrected_text 仍然必须是一句转写文本，不要输出解释。\n"
        "返回 JSON 对象，字段为 understanding 和 corrections。\n"
        "understanding 用一句话概括本批次主要修复类型。\n"
        "corrections 是数组，每项包含 id, corrected_text, reason。\n\n"
        f"输入：\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _polish_disambiguation_block(
    disambiguations: list[tuple[str, str]] | None,
) -> str:
    """
    Render the lexicon-driven term-disambiguation section.

    Args:
        disambiguations: ``(surface, guidance)`` pairs; business knowledge stays
            in the lexicon, this layer only formats whatever it is handed.

    Returns:
        A prompt block ending in a blank line, or ``""`` when there is nothing
        to disambiguate (keeping the prompt byte-identical to before).
    """
    if not disambiguations:
        return ""
    lines = [
        "【术语消歧 — 下列 ASCII 词同音歧义，必须按本句语境判断，"
        "符合所述义项才改，否则保持原样，禁止无脑替换】",
    ]
    for surface, guidance in disambiguations:
        lines.append(f"  - 「{surface}」：{guidance}")
    return "\n".join(lines) + "\n\n"


def _build_polish_strict_prompt(
    candidates: list[LlmCorrectionCandidate],
    disambiguations: list[tuple[str, str]] | None = None,
) -> str:
    """
    Build the strict polish prompt: aggressive ASR-noise removal + typo fix,
    with hard guards against cross-sentence borrow and ASCII hallucination,
    and mandatory protection of confidence/confirmation modifiers.

    Args:
        candidates: Transcript sentences to inspect.
        disambiguations: Optional ``(surface, guidance)`` pairs from the lexicon
            for context-dependent ASCII terms (e.g. AC -> Acme only when it means
            the platform). Business terms come from config, never hardcoded here.

    Returns:
        Prompt text.
    """
    payload = {"candidates": [_candidate_payload(item) for item in candidates]}
    disambiguation_block = _polish_disambiguation_block(disambiguations)
    return (
        "下面是一批连续的会议转写句子，每条独立判断，**绝对禁止**跨条引用、合并或借词。\n"
        "目标：清除 ASR 无意义噪音 + 修字面错，让下游纪要 agent 拿到干净文本。\n"
        "\n"
        "【允许并鼓励的修改 — 每条必须自标 change_type】\n"
        "  - typo：中文错别字（同音/形近），上下文足够明确时\n"
        "  - term：专有名词、产品名、技术词的 ASR 误识别（codekex→Codex、踏包→拓扑包、A键→AI）\n"
        "  - case：英文术语大小写或拼写规范化（cosay→Cosay、crown→cron、CRI→CLI）\n"
        "  - punct：标点或空格修正\n"
        "  - dup：删任意长度的同字重复（用用→用、再再→再、我我我我我我→我、卡卡卡卡→卡）\n"
        "  - filler：删纯填充音节和节奏停顿（嗯/啊/呃；以及连续重复的『那个 那个』『这个 这个』『就是 就是』）\n"
        "  - restart：删失败启动和改口前半（『我...我就...我就说我们项目好了』→『我就说我们项目好了』；\n"
        "             『他是他比如说同学C』→『比如说同学C』）。只删被改口抛弃的前半，新内容必须保留\n"
        "  - emphasis：删强调性词级重复（『我的我的我的我的PPT』→『我的PPT』；『一个一个框架』→『一个框架』）\n"
        "\n"
        "【必须保留 — 这些是下游 agent 判断决议/共识的事实信号，禁止删除或弱化】\n"
        "  - 态度/确定度词：我觉得 / 我认为 / 我感觉 / 可能 / 也许 / 或许 / 大概 / 应该 / 似乎 / 估计\n"
        "    （它们决定下游写『决议』vs『提议』）\n"
        "  - 寻求共识词：对吧 / 对吗 / 是不是 / 是吧 / 你说呢 / 你看呢 / 你觉得呢 / 行不行 / 好不好\n"
        "    （它们决定下游写『共识』vs『单方陈述』）\n"
        "  - 决策动词：同意 / 反对 / 决定 / 确认 / 不同意 / 不行 / 可以 / 不可以\n"
        "  - 任何实词：人名 / 数字 / 时间 / 术语 / 动作 / 对象\n"
        "\n"
        "【绝对禁止 — 违反即污染下游】\n"
        "  1. 禁止从相邻句子借用任何内容（哪怕该句看起来缺主语/指代不明，也保持原样）\n"
        "  2. 禁止补全说话人未说出的内容\n"
        "  3. 禁止凭空发明 ASCII：英文/数字纠正必须是对原句已有 ASCII 词的拼写修正"
        "（CRI→CLI 允许；中文『猫提卡→Mattika』『迪玛→Dima』『武一→WuYi』禁止）\n"
        "  4. 禁止把整句重写为完全不同的措辞（修字+删噪音 OK，重构句式 NO）\n"
        "  5. 禁止删除上面【必须保留】列表里的任何词\n"
        "  6. 不确定就不要返回该条；宁可漏修不要错改\n"
        "\n"
        f"{disambiguation_block}"
        "【标尺样例 — 按这个尺度改，不要自由发挥】\n"
        "  - 『我们就好像好像没什么要说的，我想想』→『我们好像没什么要说的，我想想』\n"
        "    只删重复和失败启动，不删仍然承载主语/事实的『我们』。\n"
        "  - 『我可以让我A键读到就行』→『我可以让 AI 读到就行』\n"
        "    A键 是 AI 的 ASR 错误，中文和英文术语之间保留空格。\n"
        "  - 『这个就是action，好的，对吧』→『这就是 action，对吧？』\n"
        "    删除无意义口头确认『好的』，但必须保留『对吧』这种共识探询词。\n"
        "\n"
        "【输出格式】\n"
        "  {\n"
        '    "understanding": "本批次主修了哪些类型",\n'
        '    "corrections": [\n'
        '      {"id": "c123", "corrected_text": "...",'
        ' "change_type": "typo|term|case|punct|dup|filler|restart|emphasis",'
        ' "reason": "为什么"}\n'
        "    ]\n"
        "  }\n"
        "无需修改的句子不要返回。\n"
        "\n"
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


def _parse_result(
    text: str, *, model: str, candidate_ids: set[str]
) -> LlmCorrectionResult:
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


def _parse_strict_polish(
    text: str, *, model: str, candidate_ids: set[str]
) -> LlmStrictPolishResult:
    """
    Parse strict polish JSON into per-item results with change_type.

    Args:
        text: Raw model output.
        model: Model used for generation.
        candidate_ids: Allowed candidate ids.

    Returns:
        Strict polish result.
    """
    payload = _load_json_object(text)
    understanding = str(payload.get("understanding") or "").strip()
    raw = payload.get("corrections")
    items: list[LlmPolishItem] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            candidate_id = str(entry.get("id") or "")
            corrected_text = str(entry.get("corrected_text") or "").strip()
            change_type = (
                str(entry.get("change_type") or "").strip().lower() or "unknown"
            )
            reason = str(entry.get("reason") or "").strip()
            if candidate_id in candidate_ids and corrected_text:
                items.append(
                    LlmPolishItem(
                        candidate_id=candidate_id,
                        corrected_text=corrected_text,
                        change_type=change_type,
                        reason=reason,
                    )
                )
    return LlmStrictPolishResult(understanding=understanding, items=items, model=model)


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
    Load a JSON object from raw model text, with best-effort repair when the
    response is mildly malformed.

    Failure modes seen on real DashScope traffic (qwen-plus correction model):

    * ``strict`` JSON rejecting unescaped control characters (literal ``\\n``
      inside a ``corrected_text`` value).
    * Output truncated mid-array because generation hit ``max_tokens`` before
      the model could close ``]}``.
    * One correction item has an unescaped quote or missing comma; the rest
      of the batch is structurally fine.

    Strategy:

    1. ``json.loads(strict=False)`` — handles unescaped control chars without
       any other relaxation.
    2. If that fails, walk the text with ``JSONDecoder.raw_decode`` looking for
       any individually-valid ``{...}`` object. A salvaged ``{"understanding":
       "", "corrections": [...]}`` payload lets the caller keep all complete
       items rather than dropping the whole batch.

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
        payload = json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        payload = _salvage_json_object(cleaned, original=text)
    if not isinstance(payload, dict):
        raise RuntimeError("DashScope correction response JSON must be an object.")
    return payload


def _salvage_json_object(text: str, *, original: str) -> dict[str, Any]:
    """
    Recover a usable JSON object from a partially-malformed model response.

    Walks ``text`` with ``JSONDecoder.raw_decode``: if any ``{...}`` is itself
    valid JSON we keep it. If we find a dict that already looks like the
    expected root (carries ``corrections`` or ``replacements``) we return it
    directly. Otherwise we collect every standalone item that carries an
    ``id`` (correction items always do) and return them under a synthetic
    ``corrections`` key so the caller's normal filtering still applies.

    Args:
        text: Cleaned model output (code fences stripped).
        original: Raw model output retained only for the no-JSON error message.

    Returns:
        Parsed dict, possibly synthesized from individually-valid items.
    """
    decoder = json.JSONDecoder(strict=False)
    items: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        next_brace = text.find("{", index)
        if next_brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, next_brace)
        except json.JSONDecodeError:
            index = next_brace + 1
            continue
        if isinstance(obj, dict) and ("corrections" in obj or "replacements" in obj):
            return obj
        if isinstance(obj, dict) and "id" in obj:
            items.append(obj)
        index = end
    if not items:
        raise RuntimeError(f"DashScope correction response was not JSON: {original}")
    return {"understanding": "", "corrections": items}


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
