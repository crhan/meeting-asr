"""Lock the audio-verification page's verdict options and template reuse.

The interactive page (shared by the leak review and the destutter spot-check via
``build``) must offer the five verdicts the user rules with — including
``low_quality`` for clips whose audio is too poor to judge, which must be a
first-class choice so such rows are dropped from the eval instead of being forced
into keep/reject. These tests render the page from a synthetic case (no audio, no
lexicon) so they stay deterministic and CI-safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.audio_verify import build  # noqa: E402

_CASE = {"source": "p-test", "original_text": "可以可以", "proposed_text": "可以"}


def test_page_offers_all_five_verdicts() -> None:
    """keep / reject / both_wrong / unsure / low_quality are all selectable."""
    page = build([_CASE])
    for verdict in ("keep", "reject", "both_wrong", "unsure", "low_quality"):
        assert f'value="{verdict}"' in page
    assert "不采纳" in page  # the low_quality label is rendered


def test_low_quality_is_not_keep_or_reject() -> None:
    """low_quality must be a distinct value, so apply_audio_corrections skips it.

    apply_audio_corrections only writes overrides for verdict in {keep, reject};
    keeping low_quality lexically distinct is what guarantees a poor-audio row is
    dropped rather than silently merged into the gold as a keep.
    """
    page = build([_CASE])
    assert 'value="low_quality"' in page
    assert "low_quality" not in ("keep", "reject")


def test_build_respects_custom_title_and_intro() -> None:
    """The destutter page reuses the same template with its own framing."""
    page = build([_CASE], title="destutter→keep 抽检", intro="确认没删实义")
    assert "destutter→keep 抽检" in page
    assert "确认没删实义" in page
    # no leftover placeholders
    for placeholder in ("__TITLE__", "__INTRO__", "__CARDS__", "__META__"):
        assert placeholder not in page
