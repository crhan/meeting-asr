# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a unified voiceprint quality review workflow in the TUI, including global library quality inspection, per-sample playback, in-place refresh, and sample status updates.
- Added voiceprint sample lifecycle status `verified-active` for human-confirmed samples that should stay usable for matching without being treated as quality risks.
- Added localized Chinese and English explanations for voiceprint quality reasons in the TUI.
- Added voiceprint sample candidate pools: capture planning now exposes up to 12 candidates per speaker while marking only the requested top samples as `recommended`.
- Added visible capture diagnostics for each candidate sample, including `recommended` / `candidate`, selection score, and duration/text/boundary scoring details.
- Added embedding-centrality selection before final voiceprint sample persistence, so final stored samples prefer candidates close to the speaker candidate cluster.

### Changed

- Voiceprint capture sample selection no longer relies on the longest transcript segments only; it now scores candidate segments by duration, text information content, and speaker-boundary safety.
- Voiceprint capture now prefers time-diverse samples and avoids low-information filler fragments.
- Voiceprint embedding now uses normalized audio clips as the standard path before embedding.
- Voiceprint quality review now plays normalized clips when available.
- Project speaker matching now uses cached project probe embeddings and parallel matching to reduce repeated work.
- Voiceprint Review and Voiceprint Quality views now display clearer colors and refreshed quality state without leaving the TUI.

### Fixed

- Fixed duplicate voiceprint samples caused by the same audio clip being captured from duplicate historical projects; identical audio hashes for the same speaker are now deduplicated.
- Fixed Rich markup leaking into the voiceprint quality TUI as literal `[dim]` / `[cyan]` text.
- Fixed stale quality review state after changing sample status.
- Fixed misleading critical quality display for unchanged historical speaker-match scores.

## [0.1.0] - 2026-05-09

### Added

- Initial public release of the project-based Meeting-ASR CLI.
- Added project creation, transcription, transcript export, speaker review, voiceprint matching, correction review, and release automation foundations.

[Unreleased]: https://github.com/crhan/meeting-asr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/crhan/meeting-asr/releases/tag/v0.1.0
