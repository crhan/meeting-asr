"""Lock the residual-noise detector (precision over recall).

The detector flags noise a polish FAILED to remove. A 7-judge verification
(precision per detector + recall on not-flagged + an adversarial re-check of every
claimed false positive) set the trustworthy detector list:

    filler 100% (after dropping 额)   triple 100%   chunk_dup 100%
    char_dup 24.6% -> DROPPED

So single-char doubling (在在, 我我, 的的) is intentionally NOT flagged: Chinese
word boundaries place the same char adjacent legitimately (现在|在 -> 在在), which
a regex can't separate from a real stutter. These synthetic cases pin the kept
detectors AND the deliberate non-detection; no eval data or lexicon (CI-safe).
"""

from __future__ import annotations

from app.residual_noise import is_residual_clean, residual_noise


def _kinds(text: str) -> set[str]:
    return {r.split(":")[0] for r in residual_noise(text)}


def test_clean_polish_has_no_residual() -> None:
    """A fully cleaned polish output flags nothing."""
    assert residual_noise("这个方案我觉得可以，下周上线。") == []
    assert is_residual_clean("这个方案我觉得可以，下周上线。")


def test_flags_residual_filler() -> None:
    """A leftover 嗯/呃 filler is flagged; legit 啊/吧 particles are not."""
    assert "filler" in _kinds("呃这个方案可以。")
    assert "filler" in _kinds("这个嗯方案可以。")
    assert _kinds("这个方案可以啊，对吧？") == set()  # 啊/吧 are legit particles


def test_filler_e_dropped_no_false_positive_on_legit_words() -> None:
    """额 is not a filler: 额度/金额/额外 must not be flagged (the verified FP)."""
    for legit in ("买一个 100 的额度。", "这笔金额不小。", "还有额外的成本。"):
        assert _kinds(legit) == set(), legit


def test_single_char_doubling_not_flagged() -> None:
    """char_dup is dropped: single-char doubles are NOT flagged (24.6% precision).

    Real stutters like 在在 are lost, but so are the far more numerous legit
    word-boundary doubles (现在|在, 在这|这里) the regex couldn't separate.
    """
    for text in ("我们这个在在落地的时候。", "这个的的问题。", "我我觉得不行。", "现在在一个状态。"):
        assert "char_dup" not in _kinds(text), text


def test_legit_reduplication_not_flagged() -> None:
    """Genuine Chinese reduplication is never flagged (was the char_dup risk)."""
    for legit in ("我去看看。", "他常常迟到。", "刚刚说完。", "我们天天加班。", "试试这个。"):
        assert _kinds(legit) == set(), legit


def test_flags_chunk_and_triple() -> None:
    """Uncollapsed phrase repeats and 3+ char runs are flagged."""
    assert "chunk_dup" in _kinds("就是就是这个意思。")
    assert "chunk_dup" in _kinds("然后干嘛干嘛的。")
    assert "triple" in _kinds("对对对，没问题。")


def test_legit_chunk_not_overflagged() -> None:
    """A non-repeated phrase is clean (chunk detector needs an adjacent repeat)."""
    assert "chunk_dup" not in _kinds("这个那个都行。")
