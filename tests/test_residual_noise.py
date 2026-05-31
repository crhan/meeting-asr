"""Lock the residual-noise scanner's detectors (precision over recall).

This scans polish OUTPUT for noise the polish failed to remove. The single-char
detector is the precision-critical one: it must flag stutter-prone function words
(在在, 的的, 我我) but NOT legitimate Chinese reduplication (看看, 常常, 刚刚,
天天) — a stoplist of legit doubles over-fires (~91% of outputs), so the scanner
uses a closed allowlist of stutter-prone chars instead. These synthetic cases
pin that boundary; no eval data or lexicon needed (CI-safe).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.scan_residual_noise import residual_noise  # noqa: E402


def _kinds(text: str) -> set[str]:
    return {r.split(":")[0] for r in residual_noise(text)}


def test_clean_polish_has_no_residual() -> None:
    """A fully cleaned polish output flags nothing."""
    assert residual_noise("这个方案我觉得可以，下周上线。") == []


def test_flags_residual_filler() -> None:
    """A leftover 嗯/呃/额 filler is flagged; legit 啊/吧 particles are not."""
    assert "filler" in _kinds("呃这个方案可以。")
    assert "filler" in _kinds("这个嗯方案可以。")
    assert _kinds("这个方案可以啊，对吧？") == set()  # 啊/吧 are legit particles


def test_flags_stutter_prone_char_double() -> None:
    """Doubled function words / pronouns are uncollapsed stutter."""
    assert "char_dup" in _kinds("我们这个在在落地的时候。")  # the user's example
    assert "char_dup" in _kinds("这个的的问题。")
    assert "char_dup" in _kinds("我我觉得不行。")


def test_legit_reduplication_not_flagged() -> None:
    """Genuine Chinese reduplication must NOT be flagged as stutter."""
    for legit in ("我去看看。", "他常常迟到。", "刚刚说完。", "我们天天加班。", "试试这个。"):
        assert "char_dup" not in _kinds(legit), legit


def test_flags_chunk_and_triple() -> None:
    """Uncollapsed phrase repeats and 3+ char runs are flagged."""
    assert "chunk_dup" in _kinds("就是就是这个意思。")
    assert "triple" in _kinds("对对对，没问题。")


def test_legit_chunk_not_overflagged() -> None:
    """A non-repeated phrase is clean (chunk detector needs an adjacent repeat)."""
    assert "chunk_dup" not in _kinds("这个那个都行。")
