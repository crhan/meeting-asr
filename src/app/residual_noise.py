"""Detect residual ASR noise a transcript polish failed to remove.

Polish should strip spoken noise — fillers (嗯/呃), uncollapsed phrase repeats
(就是就是), multi-char runs (对对对). This flags such noise left IN the polish
OUTPUT, the opposite of the guard's job: the guard catches polish that deleted
too much, and nothing else catches polish that left noise in.

Detectors are PRECISION-FIRST: a flag should almost always be real noise, so a
residual-noise rate is trustworthy. A 7-judge verification (precision per
detector + recall on not-flagged + an adversarial re-check of every claimed false
positive, over a stratified sample of real polish outputs) measured:

    filler 96% -> 100% after dropping 额    triple 100%    chunk_dup 100%
    char_dup (single-char doubling) 24.6%  -> DROPPED

so only the reliable detectors ship:

  - single-char doubling (在在 / 我我 / 的的) is NOT detected. Chinese word
    boundaries routinely place the same char adjacent legitimately (现在|在 ->
    在在, 在这|这里 -> 这这), which a regex cannot separate from a real stutter
    without a segmenter — 46/46 sampled flags were legit. Genuine single-char
    stutters are out of this deterministic detector's scope (they need audio).
  - 额 is not a filler: it is the first/second char of common words (额度, 金额,
    余额, 额外) and was the only filler false positive. 嗯/呃 never are.
"""

from __future__ import annotations

import re

# 嗯/呃 are never part of legitimate words. 啊/哦/吧/呢/嘛 are real sentence
# particles and are deliberately excluded.
_FILLER_RE = re.compile(r"[嗯呃]")
# Three or more identical CJK chars in a row (对对对, 是是是).
_TRIPLE_RE = re.compile(r"([一-鿿])\1{2,}")
# An adjacent repeat of a 2-4 char CJK chunk (就是就是, 干嘛干嘛, 什么什么).
_CHUNK_DUP_RE = re.compile(r"([一-鿿]{2,4})\1")


def residual_noise(text: str) -> list[str]:
    """Return reason codes for residual noise in one polish output.

    Args:
        text: A polish output sentence.

    Returns:
        Reason codes like ``["filler:呃", "chunk_dup:就是"]``; empty when clean.
    """
    reasons: list[str] = []
    fillers = _FILLER_RE.findall(text)
    if fillers:
        reasons.append(f"filler:{''.join(sorted(set(fillers)))}")
    triples = _TRIPLE_RE.findall(text)
    if triples:
        reasons.append(f"triple:{','.join(sorted(set(triples)))}")
    chunks = {m.group(1) for m in _CHUNK_DUP_RE.finditer(text)}
    if chunks:
        reasons.append(f"chunk_dup:{','.join(sorted(chunks))}")
    return reasons


def is_residual_clean(text: str) -> bool:
    """Return whether the text carries no detectable residual noise.

    Args:
        text: A polish output sentence.

    Returns:
        True when no residual-noise detector fires.
    """
    return not residual_noise(text)
