"""Evaluate voiceprint embedding impact across projects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.project_refs import list_projects
from app.project_manager import load_manifest
from app.speaker_labeling import load_ignored_speakers
from app.speaker_match_status import best_candidate_name, best_candidate_score, match_threshold
from app.speaker_matching import SpeakerMatchSummary, match_project_speakers, preview_project_speaker_matches

DEFAULT_DECLINE_THRESHOLD = 0.05
DEFAULT_HISTORICAL_LIMIT = 20


@dataclass(frozen=True, slots=True)
class VoiceprintScoreChange:
    """One before/after speaker score comparison."""

    speaker_id: int
    label: str
    before_name: str | None
    before_score: float | None
    after_name: str | None
    after_score: float | None
    delta: float | None
    status: str
    threshold: float | None = None

    @property
    def is_critical(self) -> bool:
        """Return whether this change can indicate a wrong identity."""
        return self.status in {"changed-best", "lost-candidate"} or self.is_below_threshold

    @property
    def is_warning(self) -> bool:
        """Return whether this change is a non-critical score decline."""
        return self.status == "declined" and not self.is_critical

    @property
    def is_below_threshold(self) -> bool:
        """Return whether the new score fell below the acceptance threshold."""
        if self.status not in {"declined", "lost-candidate"}:
            return False
        if self.threshold is None:
            return False
        if self.after_score is None:
            return True
        return self.after_score < self.threshold


@dataclass(frozen=True, slots=True)
class VoiceprintProjectEvaluation:
    """Voiceprint score comparison for one project."""

    project_dir: Path
    project_id: str
    title: str
    current: bool
    changes: tuple[VoiceprintScoreChange, ...]

    @property
    def improved_count(self) -> int:
        """Return the number of speakers whose best score improved."""
        return sum(1 for item in self.changes if item.status == "improved")

    @property
    def declined_count(self) -> int:
        """Return the number of speakers whose best score declined."""
        return sum(1 for item in self.changes if item.status == "declined")

    @property
    def changed_best_count(self) -> int:
        """Return the number of speakers whose best candidate changed."""
        return sum(1 for item in self.changes if item.status == "changed-best")

    @property
    def warning_count(self) -> int:
        """Return the number of score declines that stayed above threshold."""
        return sum(1 for item in self.changes if item.is_warning)

    @property
    def critical_count(self) -> int:
        """Return the number of identity-changing or below-threshold changes."""
        return sum(1 for item in self.changes if item.is_critical)

    @property
    def risk_count(self) -> int:
        """Return the number of actionable warning or critical regressions."""
        return self.warning_count + self.critical_count


@dataclass(frozen=True, slots=True)
class VoiceprintEvaluationSummary:
    """Voiceprint embedding evaluation summary."""

    current: VoiceprintProjectEvaluation
    historical: tuple[VoiceprintProjectEvaluation, ...]

    @property
    def historical_project_count(self) -> int:
        """Return how many historical projects were checked."""
        return len(self.historical)

    @property
    def historical_risk_count(self) -> int:
        """Return total risky historical speaker changes."""
        return sum(project.risk_count for project in self.historical)

    @property
    def historical_warning_count(self) -> int:
        """Return total historical warning changes."""
        return sum(project.warning_count for project in self.historical)

    @property
    def historical_critical_count(self) -> int:
        """Return total historical critical changes."""
        return sum(project.critical_count for project in self.historical)


def evaluate_voiceprint_embedding(
    project_dir: Path,
    *,
    store_dir: Path | None,
    provider: str | None,
    endpoint: str | None,
    model: str | None,
    threshold: float = 0.75,
    sample_count: int = 2,
    max_seconds: float = 12.0,
    padding_seconds: float = 0.5,
    decline_threshold: float = DEFAULT_DECLINE_THRESHOLD,
    max_historical_projects: int = DEFAULT_HISTORICAL_LIMIT,
) -> VoiceprintEvaluationSummary:
    """
    Re-match the current project and dry-run historical projects.

    Args:
        project_dir: Current project root.
        store_dir: Optional global voiceprint store.
        provider: Provider override; endpoint and model are optional overrides.
    """
    project_root = project_dir.expanduser().resolve()
    current = _current_evaluation(
        project_root,
        store_dir=store_dir,
        provider=provider,
        endpoint=endpoint,
        model=model,
        threshold=threshold,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        decline_threshold=decline_threshold,
    )
    historical = _historical_evaluations(
        project_root,
        store_dir=store_dir,
        provider=provider,
        endpoint=endpoint,
        model=model,
        threshold=threshold,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        decline_threshold=decline_threshold,
        limit=max_historical_projects,
    )
    return VoiceprintEvaluationSummary(current, tuple(historical))


def _current_evaluation(
    project_root: Path,
    *,
    store_dir: Path | None,
    provider: str | None,
    endpoint: str | None,
    model: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    decline_threshold: float,
) -> VoiceprintProjectEvaluation:
    """Persist a fresh current-project match and compare it to the previous one."""
    before = _load_match_rows(project_root)
    after = match_project_speakers(
        project_root,
        store_dir=store_dir,
        provider=provider,
        endpoint=endpoint,
        model=model,
        threshold=threshold,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        progress=None,
    )
    return _project_evaluation(project_root, True, before, after, decline_threshold)


def _historical_evaluations(
    project_root: Path,
    *,
    store_dir: Path | None,
    provider: str | None,
    endpoint: str | None,
    model: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
    decline_threshold: float,
    limit: int,
) -> list[VoiceprintProjectEvaluation]:
    """Dry-run matching for historical projects with existing match files."""
    projects = [item.project_dir for item in list_projects(project_root.parent).projects]
    candidates = [item for item in projects if item.resolve() != project_root and _match_path(item).exists()]
    evaluations: list[VoiceprintProjectEvaluation] = []
    for candidate in candidates[:limit]:
        before = _load_match_rows(candidate)
        after = _preview_matches(candidate, store_dir, provider, endpoint, model, threshold, sample_count, max_seconds, padding_seconds)
        if after is not None:
            evaluations.append(_project_evaluation(candidate, False, before, after, decline_threshold))
    return evaluations


def _preview_matches(
    project_dir: Path,
    store_dir: Path | None,
    provider: str | None,
    endpoint: str | None,
    model: str | None,
    threshold: float,
    sample_count: int,
    max_seconds: float,
    padding_seconds: float,
) -> SpeakerMatchSummary | None:
    """Return dry-run matches, ignoring unusable historical projects."""
    try:
        return preview_project_speaker_matches(
            project_dir,
            store_dir=store_dir,
            provider=provider,
            endpoint=endpoint,
            model=model,
            threshold=threshold,
            sample_count=sample_count,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            progress=None,
        )
    except Exception:  # noqa: BLE001
        return None


def _project_evaluation(
    project_dir: Path,
    current: bool,
    before: dict[int, dict[str, Any]],
    after: SpeakerMatchSummary,
    decline_threshold: float,
) -> VoiceprintProjectEvaluation:
    """Build one project comparison."""
    manifest = load_manifest(project_dir)
    ignored_speaker_ids = _ignored_speaker_ids(project_dir, manifest)
    changes = tuple(
        _score_change(item, before.get(item.speaker_id), decline_threshold)
        for item in after.matches
        if item.speaker_id not in ignored_speaker_ids
    )
    return VoiceprintProjectEvaluation(project_dir, manifest.project_id, manifest.title, current, changes)


def _score_change(match: object, before: dict[str, Any] | None, decline_threshold: float) -> VoiceprintScoreChange:
    """Compare one speaker match row before and after embedding."""
    before_name = best_candidate_name(before or {})
    before_score = best_candidate_score(before or {})
    after_name = best_candidate_name(match)
    after_score = best_candidate_score(match)
    delta = _score_delta(before_score, after_score)
    status = _change_status(before_name, after_name, delta, decline_threshold)
    return VoiceprintScoreChange(
        int(getattr(match, "speaker_id")),
        str(getattr(match, "label")),
        before_name,
        before_score,
        after_name,
        after_score,
        delta,
        status,
        match_threshold(match),
    )


def _change_status(
    before_name: str | None,
    after_name: str | None,
    delta: float | None,
    decline_threshold: float,
) -> str:
    """Classify one score change for human review."""
    if before_name and after_name and before_name != after_name:
        return "changed-best"
    if before_name and not after_name:
        return "lost-candidate"
    if delta is None:
        return "new" if after_name and not before_name else "unchanged"
    if delta <= -decline_threshold:
        return "declined"
    if delta > 0:
        return "improved"
    return "unchanged"


def _score_delta(before: float | None, after: float | None) -> float | None:
    """Return score delta when both sides have a score."""
    if before is None or after is None:
        return None
    return after - before


def _load_match_rows(project_dir: Path) -> dict[int, dict[str, Any]]:
    """Load persisted speaker match rows by project speaker id."""
    path = _match_path(project_dir)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("matches", []) if isinstance(payload, dict) else []
    return {int(row["speaker_id"]): dict(row) for row in rows if isinstance(row, dict) and "speaker_id" in row}


def _ignored_speaker_ids(project_dir: Path, manifest: Any) -> set[int]:
    """Return speaker ids that should not participate in voiceprint risk checks."""
    ignored = load_ignored_speakers(project_dir / "speakers" / "speaker_ignore.json")
    for value in manifest.speakers.get("ignored", []):
        ignored.add(int(value))
    return ignored


def _match_path(project_dir: Path) -> Path:
    """Return a project's persisted speaker match path."""
    return project_dir / "speakers" / "speaker_matches.json"
