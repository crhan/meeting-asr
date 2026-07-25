"""Tests for voiceprint match-threshold calibration."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.voiceprint_calibration import calibrate_voiceprint_thresholds
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    store_voiceprint_samples_with_rows,
    upsert_voiceprint_embedding,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_xdg_data_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default voiceprint lookups inside the test sandbox."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def test_calibrate_separates_genuine_and_impostor_populations(
    tmp_path: Path,
) -> None:
    """Distinct people produce separable distributions and an EER between them."""
    store_dir = tmp_path / "voiceprints"
    # Alice clusters around [1, 0]; Bob clusters around [0, 1].
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.98, 0.05], [0.99, -0.03]])
    _seed_person(store_dir, "Bob", [[0.0, 1.0], [0.03, 0.97], [-0.02, 0.99]])

    report = calibrate_voiceprint_thresholds(store_dir=store_dir)

    assert report.person_count == 2
    assert report.scored_person_count == 2
    assert report.sample_count == 6
    assert report.genuine is not None and report.impostor is not None
    # Same-person scores are near 1; cross-person scores are near 0.
    assert report.genuine.minimum > 0.9
    assert report.impostor.maximum < 0.2
    assert report.eer_threshold is not None
    assert report.impostor.maximum < report.eer_threshold < report.genuine.minimum
    assert report.low_impostor_threshold is not None
    assert report.current_threshold == 0.75


def test_calibrate_reports_thin_library_warnings(tmp_path: Path) -> None:
    """Single-person or single-sample stores degrade with explicit warnings."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0]])

    report = calibrate_voiceprint_thresholds(store_dir=store_dir)

    assert report.person_count == 1
    assert report.scored_person_count == 0
    assert report.genuine is None
    assert report.impostor is None
    assert report.eer_threshold is None
    assert any("only 1 embedded" in warning for warning in report.warnings)
    assert any("fewer than 2 people" in warning for warning in report.warnings)


def test_calibrate_cli_renders_report_and_json(tmp_path: Path) -> None:
    """The CLI prints distributions and supports --json."""
    store_dir = tmp_path / "voiceprints"
    _seed_person(store_dir, "Alice", [[1.0, 0.0], [0.99, 0.02]])
    _seed_person(store_dir, "Bob", [[0.0, 1.0], [0.02, 0.99]])

    text = runner.invoke(
        app, ["voiceprint", "calibrate", "--store-dir", str(store_dir)]
    )
    as_json = runner.invoke(
        app, ["voiceprint", "calibrate", "--store-dir", str(store_dir), "--json"]
    )

    assert text.exit_code == 0
    assert "Equal-error threshold" in text.output
    assert "Current accept threshold: 0.75" in text.output
    assert as_json.exit_code == 0
    assert '"eer_threshold"' in as_json.output


def _seed_person(store_dir: Path, name: str, vectors: list[list[float]]) -> None:
    """Store embedded matching-pool samples for one person."""
    db_path = get_voiceprint_db_path(store_dir)
    _provider, model = resolve_voiceprint_embedding_options(provider=None, model=None)
    source = store_dir / f"{name}-source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"seed")
    samples = []
    for index, _vector in enumerate(vectors):
        clip_path = store_dir / "clips" / name / f"clip_{index}.wav"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(f"{name}-{index}".encode())
        samples.append(
            StoredVoiceprintSample(
                speaker_name=name,
                project_id=f"p-{name.lower()}",
                project_path=store_dir,
                project_speaker_id=0,
                source_path=source,
                clip_path=clip_path,
                clip_rel_path=str(clip_path.relative_to(store_dir)),
                source_begin_time_ms=index * 1000,
                source_end_time_ms=index * 1000 + 900,
                clip_begin_time_ms=0,
                clip_end_time_ms=900,
                transcript_text=f"{name} sample {index}",
            )
        )
    _db, rows = store_voiceprint_samples_with_rows(samples, db_path)
    for row, vector in zip(rows, vectors):
        upsert_voiceprint_embedding(row.sample_id, model, vector, db_path)


def test_cost_counts_scale_past_the_export_cap() -> None:
    """A downsampled export must not report the cap as the real cost.

    Above ``MAX_EXPORTED_SCORES`` the stored score arrays are an evenly spaced
    view of the library, so counting them directly would tell a large library
    it has exactly 2000 wrong matches -- the export limit wearing a cost's
    clothes. The rate is what the sample measures; the count is that rate
    applied to the true population.
    """
    from app.voiceprint_calibration import (
        ScoreDistribution,
        VoiceprintCalibrationReport,
    )

    # 100 exported impostor scores standing in for 10_000 real observations,
    # half of them at or above 0.60.
    exported = tuple([0.4] * 50 + [0.8] * 50)
    report = VoiceprintCalibrationReport(
        model="m",
        person_count=2,
        scored_person_count=2,
        sample_count=10,
        genuine=None,
        impostor=ScoreDistribution(
            count=10_000, minimum=0.4, p5=0.4, median=0.6, p95=0.8, maximum=0.8
        ),
        eer_threshold=None,
        eer_rate=None,
        low_impostor_threshold=None,
        current_threshold=0.6,
        warnings=(),
        impostor_scores=exported,
    )

    cost = report.cost_at(0.6)

    assert cost is not None
    assert cost.false_accept_rate == pytest.approx(0.5)
    assert cost.false_accept_count == 5000


def test_cost_counts_are_exact_below_the_export_cap() -> None:
    """Small libraries are not downsampled, so scaling must be a no-op there."""
    from app.voiceprint_calibration import (
        ScoreDistribution,
        VoiceprintCalibrationReport,
    )

    exported = (0.4, 0.4, 0.8, 0.8)
    report = VoiceprintCalibrationReport(
        model="m",
        person_count=2,
        scored_person_count=2,
        sample_count=4,
        genuine=None,
        impostor=ScoreDistribution(
            count=4, minimum=0.4, p5=0.4, median=0.6, p95=0.8, maximum=0.8
        ),
        eer_threshold=None,
        eer_rate=None,
        low_impostor_threshold=None,
        current_threshold=0.6,
        warnings=(),
        impostor_scores=exported,
    )

    cost = report.cost_at(0.6)

    assert cost is not None
    assert cost.false_accept_count == 2


def test_downsampling_keeps_the_extreme_scores() -> None:
    """The tail is the whole point; striding must not shave it off.

    A handful of impostors above the threshold is exactly what makes a
    threshold unsafe, and it lives at the top of the sorted array. Dropping the
    last element would let `cost_at` price such a threshold at zero false
    accepts -- the report would be most wrong precisely when it matters.
    """
    from app.voiceprint_calibration import MAX_EXPORTED_SCORES, _exported_scores

    population = [i / 10_000 for i in range(MAX_EXPORTED_SCORES * 3)]

    exported = _exported_scores(population)

    assert len(exported) == MAX_EXPORTED_SCORES
    assert exported[0] == round(min(population), 4)
    assert exported[-1] == round(max(population), 4)


def test_a_lone_high_impostor_survives_downsampling() -> None:
    """One score above the threshold must not vanish into the sampling grid."""
    from app.voiceprint_calibration import MAX_EXPORTED_SCORES, _exported_scores

    population = [0.10] * (MAX_EXPORTED_SCORES * 4) + [0.99]

    exported = _exported_scores(population)

    assert 0.99 in exported
