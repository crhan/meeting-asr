"""Project directory layout: which artifacts are recomputable and which are not.

A project directory mixes two kinds of derived data:

1. **Recomputable intermediates.** Probe clips, cluster clips, sample clips, web
   preview clips, TUI playback clips, speaker-apply previews. Every one of them
   is cut from the project source with ffmpeg; deleting them costs CPU on the
   next run and nothing else.
2. **Durable artifacts.** Embedding vectors bought from an embedding model, LLM
   correction proposals, and the review files a human edited by hand. Deleting
   them either costs money to rebuild or destroys work that cannot be rebuilt at
   all.

``tmp/`` holds the first kind *only*. That is the whole point of this module:
``rm -rf <project>/tmp`` must always be a safe operation, and
``meeting-asr project clean`` offers exactly that.

Historically both kinds lived under ``tmp/`` -- the shared clip embedding cache
at ``tmp/voiceprint_clips/clip_embeddings.json``, the probe embedding cache at
``tmp/voiceprint_match/probe_embeddings.json``, and the whole correction review
workflow at ``tmp/corrections/``. A directory named ``tmp`` that you do not dare
delete is a directory that only grows, so those three moved out to
``embeddings/`` and ``corrections/``.

Projects created before the move keep their data at the old paths.
:func:`migrate_project_layout` relocates it in place, and is called from every
accessor that touches those artifacts, so an old project migrates itself the
first time anything reads or writes the cache. Paths already *recorded* inside
manifests and proposal JSON (``tmp/corrections/proposal_x.json``) stay readable
through :func:`resolve_recorded_project_path`.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.project_models import KeptPath, ProjectCleanSummary

LOGGER = logging.getLogger(__name__)

TMP_DIR_NAME = "tmp"
EMBEDDING_DIR_NAME = "embeddings"
CORRECTION_DIR_NAME = "corrections"

CLIP_EMBEDDING_CACHE_RELATIVE_PATH = Path(EMBEDDING_DIR_NAME) / "clip_embeddings.json"
PROBE_EMBEDDING_CACHE_RELATIVE_PATH = Path(EMBEDDING_DIR_NAME) / "probe_embeddings.json"
CORRECTION_REVIEW_RELATIVE_DIR = Path(CORRECTION_DIR_NAME)

LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH = (
    Path(TMP_DIR_NAME) / "voiceprint_clips" / "clip_embeddings.json"
)
LEGACY_PROBE_EMBEDDING_CACHE_RELATIVE_PATH = (
    Path(TMP_DIR_NAME) / "voiceprint_match" / "probe_embeddings.json"
)
LEGACY_CORRECTION_REVIEW_RELATIVE_DIR = Path(TMP_DIR_NAME) / CORRECTION_DIR_NAME

#: Legacy durable locations, mapped to where the same artifact lives now. The
#: correction directory is a directory-to-directory mapping; the two caches are
#: file-to-file.
_LEGACY_FILE_RELOCATIONS: tuple[tuple[Path, Path], ...] = (
    (
        LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH,
        CLIP_EMBEDDING_CACHE_RELATIVE_PATH,
    ),
    (
        LEGACY_PROBE_EMBEDDING_CACHE_RELATIVE_PATH,
        PROBE_EMBEDDING_CACHE_RELATIVE_PATH,
    ),
)
_LEGACY_DIR_RELOCATIONS: tuple[tuple[Path, Path], ...] = (
    (LEGACY_CORRECTION_REVIEW_RELATIVE_DIR, CORRECTION_REVIEW_RELATIVE_DIR),
)


@dataclass(frozen=True, slots=True)
class LayoutMigration:
    """What :func:`migrate_project_layout` did to one project directory.

    Attributes:
        moved: Destination paths that now hold relocated durable artifacts.
        merged: Destination cache files that absorbed entries from a legacy
            copy (both copies existed, so their vectors were unioned).
        blocked: Legacy paths left in place because the destination already held
            a different file with the same name. Nothing is ever overwritten,
            and ``project clean`` refuses to delete a blocked path.
    """

    moved: tuple[Path, ...] = ()
    merged: tuple[Path, ...] = ()
    blocked: tuple[Path, ...] = ()

    @property
    def changed(self) -> bool:
        """Return whether anything moved."""
        return bool(self.moved or self.merged)


def project_tmp_dir(project_root: Path) -> Path:
    """Return the recomputable-intermediates directory of one project."""
    return project_root / TMP_DIR_NAME


def clip_embedding_cache_path(project_root: Path) -> Path:
    """Return the shared clip embedding cache path (paid vectors, durable)."""
    return project_root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH


def probe_embedding_cache_path(project_root: Path) -> Path:
    """Return the speaker probe embedding cache path (paid vectors, durable)."""
    return project_root / PROBE_EMBEDDING_CACHE_RELATIVE_PATH


def correction_review_dir(project_root: Path) -> Path:
    """Return the directory holding correction review files and proposals."""
    return project_root / CORRECTION_REVIEW_RELATIVE_DIR


def migrate_project_layout(
    project_root: Path, *, dry_run: bool = False
) -> LayoutMigration:
    """
    Move durable artifacts out of ``tmp/`` into their permanent home.

    Idempotent and cheap: when no legacy path exists the function is a handful
    of ``exists()`` calls, so accessors can call it unconditionally. Nothing is
    ever overwritten -- a name collision leaves the legacy copy alone and is
    reported through :attr:`LayoutMigration.blocked`.

    Args:
        project_root: Project root directory.
        dry_run: Only report what would move. ``project clean`` uses this so a
            dry run really writes nothing, instead of quietly relocating a few
            hundred megabytes while claiming to be a preview.

    Returns:
        Record of what was (or, for a dry run, would be) relocated.
    """
    legacy_root = project_tmp_dir(project_root)
    if not legacy_root.is_dir():
        return LayoutMigration()
    moved: list[Path] = []
    merged: list[Path] = []
    blocked: list[Path] = []
    for relative_legacy, relative_target in _LEGACY_FILE_RELOCATIONS:
        _migrate_cache_file(
            project_root / relative_legacy,
            project_root / relative_target,
            moved=moved,
            merged=merged,
            blocked=blocked,
            dry_run=dry_run,
        )
    for relative_legacy, relative_target in _LEGACY_DIR_RELOCATIONS:
        _migrate_directory(
            project_root / relative_legacy,
            project_root / relative_target,
            moved=moved,
            blocked=blocked,
            dry_run=dry_run,
        )
    if (moved or merged) and not dry_run:
        LOGGER.info(
            "Relocated %d durable artifact(s) out of %s",
            len(moved) + len(merged),
            legacy_root,
        )
    return LayoutMigration(tuple(moved), tuple(merged), tuple(blocked))


def resolve_recorded_project_path(project_root: Path, recorded: object) -> Path:
    """
    Resolve a project-relative artifact path recorded before the layout change.

    Manifests and proposal JSON written by older versions point at
    ``tmp/corrections/...``. The file now lives at ``corrections/...``, so a
    literal join yields a path that does not exist. Prefer the recorded location
    when it is still there (a half-migrated project, or a path that was never
    relocated) and fall back to the relocated one.

    Args:
        project_root: Project root directory.
        recorded: Path string or ``Path`` as stored in JSON.

    Returns:
        Absolute path; the literal join when neither candidate exists, so error
        messages keep naming what was actually recorded.
    """
    path = Path(str(recorded or ""))
    if path.is_absolute():
        return path
    candidate = project_root / path
    if candidate.exists():
        return candidate
    relocated = relocate_legacy_relative_path(path)
    if relocated is not None and (project_root / relocated).exists():
        return project_root / relocated
    return candidate


def relocate_legacy_relative_path(relative: Path) -> Path | None:
    """
    Map a legacy project-relative path to its post-migration location.

    Args:
        relative: Project-relative path as recorded on disk.

    Returns:
        The relocated project-relative path, or None when the input is not a
        legacy durable path.
    """
    for legacy, target in _LEGACY_FILE_RELOCATIONS:
        if relative == legacy:
            return target
    for legacy, target in _LEGACY_DIR_RELOCATIONS:
        if relative.is_relative_to(legacy):
            return target / relative.relative_to(legacy)
    return None


def durable_paths_under_tmp(project_root: Path) -> tuple[Path, ...]:
    """
    Return legacy durable artifacts still sitting under ``tmp/``.

    After :func:`migrate_project_layout` this is empty unless a name collision
    blocked a move. ``project clean`` uses it to keep its promise that only
    recomputable data is removed.

    Args:
        project_root: Project root directory.

    Returns:
        Existing legacy durable paths, absolute.
    """
    found: list[Path] = []
    for relative_legacy, _target in _LEGACY_FILE_RELOCATIONS:
        path = project_root / relative_legacy
        if path.exists():
            found.append(path)
    for relative_legacy, _target in _LEGACY_DIR_RELOCATIONS:
        path = project_root / relative_legacy
        if path.is_dir() and any(path.iterdir()):
            found.append(path)
    return tuple(found)


def clean_project_tmp(project_root: Path, *, apply: bool) -> ProjectCleanSummary:
    """
    Clear one project's recomputable intermediates.

    Durable artifacts are relocated out of ``tmp/`` first, and anything that
    could not be relocated (a name collision) is kept rather than deleted -- the
    command must never be able to destroy data that costs money or hand work to
    rebuild.

    Args:
        project_root: Project root directory.
        apply: Delete for real; a false value only measures.

    Returns:
        What was, or would be, removed.
    """
    root = project_root.expanduser().resolve()
    migration = migrate_project_layout(root, dry_run=not apply)
    tmp_root = project_tmp_dir(root)
    if tmp_root.is_symlink():
        # Deleting a symlink removes the LINK. Walking it instead would delete the
        # target's children -- someone who redirected scratch to another disk
        # shares that directory with whatever else lives there, so this command
        # would erase data outside the project while leaving the symlink behind.
        # Migration above already pulled durable artifacts to safety; deleting is
        # where we stop and hand the decision back.
        return ProjectCleanSummary(
            project_dir=root,
            removed=(),
            freed_bytes=0,
            relocated=migration.moved + migration.merged,
            kept=(
                KeptPath(
                    tmp_root,
                    "tmp/ is a symlink; refusing to delete through it "
                    "(remove the link yourself if the target is disposable)",
                ),
            ),
            applied=apply,
        )
    if not tmp_root.is_dir():
        return ProjectCleanSummary(
            project_dir=root,
            removed=(),
            freed_bytes=0,
            relocated=migration.moved + migration.merged,
            kept=(),
            applied=apply,
        )
    # Durable data still physically under tmp/ falls in two buckets: what a
    # collision blocked from moving (stays for good, must never be deleted) and,
    # on a dry run, what --apply would relocate. Neither may be counted as freed
    # space, or the preview would advertise paid vectors as reclaimable disk.
    blocked = tuple(migration.blocked)
    still_durable = durable_paths_under_tmp(root)
    leaving = tuple(path for path in still_durable if path not in set(blocked))
    protected = blocked + leaving
    removable: list[Path] = []
    kept: list[KeptPath] = []
    for entry in sorted(tmp_root.iterdir()):
        if any(path == entry or path.is_relative_to(entry) for path in blocked):
            kept.append(KeptPath(entry, "still holds durable data"))
            continue
        if _is_excluded(entry, leaving) or any(
            path.is_relative_to(entry) for path in leaving
        ):
            if not _holds_content_outside(entry, protected):
                # Everything in here is on its way to a durable directory, so
                # the entry disappears with the migration rather than being
                # deleted. An entry that also holds clips stays removable.
                continue
        removable.append(entry)
    freed = sum(_tree_size(entry, exclude=protected) for entry in removable)
    if apply:
        survivors: list[Path] = []
        for entry in removable:
            reason = _remove_entry(entry)
            if reason is None:
                continue
            # The data is still on disk, so it is neither removed nor freed.
            # ignore_errors would have reported a read-only mount or an EACCES
            # as a successful cleanup and counted every byte as reclaimed.
            survivors.append(entry)
            kept.append(KeptPath(entry, reason))
            freed -= _tree_size(entry, exclude=protected)
        if survivors:
            removable = [entry for entry in removable if entry not in set(survivors)]
        if not kept:
            _rmdir_if_empty(tmp_root)
    return ProjectCleanSummary(
        project_dir=root,
        removed=tuple(removable),
        freed_bytes=freed,
        relocated=migration.moved + migration.merged,
        kept=tuple(kept),
        applied=apply,
    )


def _remove_entry(entry: Path) -> str | None:
    """Delete one tmp entry, returning why it survived, or None on success.

    Deletion is verified rather than assumed: ``shutil.rmtree`` can fail partway
    and leave a tree behind, so a clean return is not proof the path is gone.
    """
    try:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
    except OSError as error:
        return f"could not be removed: {error.strerror or error}"
    if entry.exists() or entry.is_symlink():
        return "could not be removed: still present after deletion"
    return None


def _holds_content_outside(path: Path, exclude: tuple[Path, ...]) -> bool:
    """Return whether a tree holds anything not covered by ``exclude``."""
    if path.is_symlink() or path.is_file():
        return not _is_excluded(path, exclude)
    return any(
        (child.is_file() or child.is_symlink()) and not _is_excluded(child, exclude)
        for child in path.rglob("*")
    )


def _is_excluded(path: Path, exclude: tuple[Path, ...]) -> bool:
    """Return whether a path sits at or below one of the excluded paths."""
    return any(path == item or path.is_relative_to(item) for item in exclude)


def _tree_size(path: Path, *, exclude: tuple[Path, ...] = ()) -> int:
    """Return the total byte size of a file or directory tree."""
    if path.is_symlink():
        return 0
    if path.is_file():
        if _is_excluded(path, exclude):
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink() or not child.is_file():
            continue
        if _is_excluded(child, exclude):
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _migrate_cache_file(
    legacy: Path,
    target: Path,
    *,
    moved: list[Path],
    merged: list[Path],
    blocked: list[Path],
    dry_run: bool = False,
) -> None:
    """Relocate one JSON vector cache, merging when both copies exist."""
    if not legacy.is_file():
        return
    if not target.exists():
        if dry_run:
            moved.append(target)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy.replace(target)
        except OSError:
            LOGGER.warning("Could not relocate %s to %s", legacy, target)
            blocked.append(legacy)
            return
        moved.append(target)
        _rmdir_if_empty(legacy.parent)
        return
    union = _merged_vector_cache(target, legacy)
    if union is None:
        blocked.append(legacy)
        return
    if dry_run:
        merged.append(target)
        return
    target.write_text(
        json.dumps(union, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    legacy.unlink(missing_ok=True)
    _rmdir_if_empty(legacy.parent)
    merged.append(target)


def _merged_vector_cache(target: Path, legacy: Path) -> dict[str, object] | None:
    """Return the union of two vector caches, or None when either is unreadable.

    Both files are ``{cache_key: vector}`` maps keyed by a hash of everything
    that shapes the embedding, so identical keys hold identical vectors and the
    union cannot lose or corrupt anything. Entries only present in the legacy
    copy are exactly the vectors this migration exists to save.
    """
    target_payload = _read_json_object(target)
    legacy_payload = _read_json_object(legacy)
    if target_payload is None or legacy_payload is None:
        return None
    union = dict(legacy_payload)
    union.update(target_payload)
    return union


def _rmdir_if_empty(path: Path) -> None:
    """Drop a legacy directory once its durable file has moved out.

    ``tmp/voiceprint_clips`` held nothing but the cache, so it would otherwise
    linger as an empty directory in every migrated project.
    """
    try:
        path.rmdir()
    except OSError:
        return


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Read a JSON object from disk, or None when absent or malformed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _migrate_directory(
    legacy: Path,
    target: Path,
    *,
    moved: list[Path],
    blocked: list[Path],
    dry_run: bool = False,
) -> None:
    """Relocate every entry of a legacy directory, never overwriting."""
    if not legacy.is_dir():
        return
    entries = sorted(legacy.iterdir())
    if not entries:
        if not dry_run:
            _rmdir_if_empty(legacy)
        return
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        destination = target / entry.name
        if destination.exists():
            LOGGER.warning(
                "Keeping %s where it is: %s already exists", entry, destination
            )
            blocked.append(entry)
            continue
        if dry_run:
            moved.append(destination)
            continue
        try:
            shutil.move(str(entry), str(destination))
        except OSError:
            LOGGER.warning("Could not relocate %s to %s", entry, destination)
            blocked.append(entry)
            continue
        moved.append(destination)
    if not dry_run and not any(legacy.iterdir()):
        legacy.rmdir()


__all__ = [
    "CLIP_EMBEDDING_CACHE_RELATIVE_PATH",
    "CORRECTION_DIR_NAME",
    "CORRECTION_REVIEW_RELATIVE_DIR",
    "EMBEDDING_DIR_NAME",
    "LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH",
    "LEGACY_CORRECTION_REVIEW_RELATIVE_DIR",
    "LEGACY_PROBE_EMBEDDING_CACHE_RELATIVE_PATH",
    "PROBE_EMBEDDING_CACHE_RELATIVE_PATH",
    "TMP_DIR_NAME",
    "LayoutMigration",
    "clean_project_tmp",
    "clip_embedding_cache_path",
    "correction_review_dir",
    "durable_paths_under_tmp",
    "migrate_project_layout",
    "probe_embedding_cache_path",
    "project_tmp_dir",
    "relocate_legacy_relative_path",
    "resolve_recorded_project_path",
]
