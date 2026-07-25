"""Authoritative speaker-pipeline thresholds and their coupling contracts.

Every number here interacts with the others across subsystem boundaries, so
they live in one leaf module instead of being scattered as per-module
literals. When tuning, read the coupling notes first; ``meeting-asr
voiceprint calibrate`` computes store-specific evidence for the aggregate
acceptance layer.

Layer overview (top = coarser decisions, bottom = finer):

1. **Aggregate acceptance** (``speaker_matching``): a project speaker's
   averaged probe vector against library people. ``DEFAULT_MATCH_THRESHOLD``
   accepts outright; below it, a clearly leading candidate is still accepted
   via the strong-margin rule (``STRONG_MARGIN_ACCEPT_SCORE`` +
   ``STRONG_MARGIN_ACCEPT_MARGIN``).
2. **Per-sentence identity** (``speaker_sample_matching``): one sentence
   against library people. Deliberately looser than layer 1
   (``DEFAULT_SAMPLE_IDENTITY_THRESHOLD``) because single sentences are
   noisy; conflict/ambiguity margins decide when stabilization may move a
   sentence, and ``DEFAULT_FOREIGN_REASSIGN_THRESHOLD`` guards reassignment
   out of unnamed clusters.
3. **In-project cluster geometry** (``speaker_cluster_quality``): sentences
   against their own track centroid; no library involved.

Cross-layer contracts to keep in mind:

- Re-split promotion (``ResplitParams.promote_score`` = 0.62 in
  ``speaker_resplit``) mints and seeds a speaker on weaker evidence than
  ``DEFAULT_MATCH_THRESHOLD`` accepts. The seeded name survives later
  rematches only through ``apply_project_speakers``'s merge semantics —
  raising the accept threshold does NOT retroactively unseat those seeds.
- ``sentence_reassignment`` reruns aggregate matching after every
  reassignment batch; it must use the SAME acceptance parameters as
  ``project run``'s match stage, otherwise one stabilization pass could flip
  decisions the user already saw. That is why the rematch defaults below are
  aliases, not copies.
- Crosstalk flagging (``CrosstalkParams`` in ``speaker_crosstalk``) only
  fires strictly below the acceptance layer (score floor 0.5 <
  ``DEFAULT_MATCH_THRESHOLD``); raising the floor above the strong-margin
  score would start tagging speakers the acceptance layer can still accept.
"""

from __future__ import annotations

# --- Layer 1: aggregate probe acceptance (speaker_matching) -----------------
# Minimum best-candidate score for automatic naming.
DEFAULT_MATCH_THRESHOLD = 0.75
# Below the threshold, accept when the best candidate still scores at least
# this AND leads the runner-up by the margin below.
STRONG_MARGIN_ACCEPT_SCORE = 0.65
STRONG_MARGIN_ACCEPT_MARGIN = 0.25

# Probe construction shared by `project run`, `project speakers match`, and
# the post-reassignment rematch in sentence_reassignment.
DEFAULT_MATCH_SAMPLE_COUNT = 2
DEFAULT_MATCH_MAX_SECONDS = 12.0
DEFAULT_MATCH_PADDING_SECONDS = 0.5

# --- Layer 2: per-sentence identity diagnostics (speaker_sample_matching) ---
# Minimum per-sentence score treated as identity evidence.
DEFAULT_SAMPLE_IDENTITY_THRESHOLD = 0.45
# Required lead of another person over the assigned one for a conflict.
DEFAULT_IDENTITY_CONFLICT_MARGIN = 0.08
# |margin| below this is ambiguous rather than a conflict.
DEFAULT_IDENTITY_AMBIGUOUS_MARGIN = 0.05
# Minimum top-match score to pull a sentence out of an unnamed cluster.
DEFAULT_FOREIGN_REASSIGN_THRESHOLD = 0.55

# --- Layer 3: in-project cluster geometry (speaker_cluster_quality) ---------
# Edge threshold for same-speaker connected components.
CLUSTER_SAME_SPEAKER_THRESHOLD = 0.60
# Two track centroids at least this close are merge candidates.
CLUSTER_MERGE_THRESHOLD = 0.62


def resolve_match_threshold(explicit: float | None = None) -> float:
    """
    Resolve the acceptance threshold from an explicit override or config.

    ``DEFAULT_MATCH_THRESHOLD`` was tuned for the original embedding model;
    a different model shifts the whole genuine/impostor score scale, so the
    acceptance layer is configurable per library via
    ``voiceprint.match_threshold``. Every entry point that does not take an
    explicit threshold from the user MUST resolve through here, otherwise a
    configured threshold would apply to `project run` but not to the rematch
    that stabilization triggers right after it.

    Args:
        explicit: Threshold supplied on the command line or API call.

    Returns:
        Explicit value, else the configured value, else the built-in default.
    """
    if explicit is not None:
        return explicit
    from app.config import get_configured_match_threshold

    configured = get_configured_match_threshold()
    if configured is not None:
        return configured
    return DEFAULT_MATCH_THRESHOLD


def match_threshold_coupling_kinds(threshold: float) -> tuple[str, ...]:
    """
    Return stable identifiers for the contracts a threshold would violate.

    The prose in :func:`match_threshold_coupling_warnings` is English; a
    localized UI needs the same findings as tokens it can word itself.

    Args:
        threshold: Candidate acceptance threshold.

    Returns:
        Kinds among ``"strong-margin-dead"`` and ``"below-crosstalk-floor"``.
    """
    from app.speaker_crosstalk import DEFAULT_CROSSTALK_SCORE_FLOOR

    kinds: list[str] = []
    if threshold <= STRONG_MARGIN_ACCEPT_SCORE:
        kinds.append("strong-margin-dead")
    if threshold <= DEFAULT_CROSSTALK_SCORE_FLOOR:
        kinds.append("below-crosstalk-floor")
    return tuple(kinds)


def match_threshold_coupling_warnings(threshold: float) -> tuple[str, ...]:
    """
    Return contract violations a candidate threshold would introduce.

    The acceptance layer is coupled to the strong-margin rule below it and to
    the crosstalk score floor below that (see module docstring). Moving the
    threshold past either boundary does not crash anything — it silently
    turns a rule into dead code — so callers surface these to the human
    before writing the value.

    Args:
        threshold: Candidate acceptance threshold.

    Returns:
        Human-readable warnings; empty when the threshold is contract-safe.
    """
    from app.speaker_crosstalk import DEFAULT_CROSSTALK_SCORE_FLOOR

    warnings: list[str] = []
    if threshold <= STRONG_MARGIN_ACCEPT_SCORE:
        warnings.append(
            f"threshold {threshold:.2f} is at or below the strong-margin accept "
            f"score {STRONG_MARGIN_ACCEPT_SCORE:.2f}; the strong-margin rescue "
            "rule becomes dead code because everything it would rescue is "
            "already accepted outright"
        )
    if threshold <= DEFAULT_CROSSTALK_SCORE_FLOOR:
        warnings.append(
            f"threshold {threshold:.2f} is at or below the crosstalk score "
            f"floor {DEFAULT_CROSSTALK_SCORE_FLOOR:.2f}; speakers the acceptance layer "
            "can now accept would also be eligible for crosstalk flagging"
        )
    return tuple(warnings)


__all__ = [
    "CLUSTER_MERGE_THRESHOLD",
    "CLUSTER_SAME_SPEAKER_THRESHOLD",
    "DEFAULT_IDENTITY_AMBIGUOUS_MARGIN",
    "DEFAULT_IDENTITY_CONFLICT_MARGIN",
    "DEFAULT_FOREIGN_REASSIGN_THRESHOLD",
    "DEFAULT_MATCH_MAX_SECONDS",
    "DEFAULT_MATCH_PADDING_SECONDS",
    "DEFAULT_MATCH_SAMPLE_COUNT",
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_SAMPLE_IDENTITY_THRESHOLD",
    "STRONG_MARGIN_ACCEPT_MARGIN",
    "STRONG_MARGIN_ACCEPT_SCORE",
    "match_threshold_coupling_kinds",
    "match_threshold_coupling_warnings",
    "resolve_match_threshold",
]
