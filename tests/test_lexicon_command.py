"""Tests for lexicon hotword CLI commands."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.commands import lexicon as lexicon_commands
from app.config import Settings
from app.correction_hotwords import hotword_hash
from app.lexicon_store import (
    AsrVocabularyState,
    LexiconContext,
    get_asr_vocabulary_state,
    get_lexicon_term,
    import_lexicon_payload,
    list_asr_hotwords,
    list_lexicon_correction_rules,
    list_lexicon_disambiguations,
    record_lexicon_contexts,
    save_asr_vocabulary_state,
    set_alias_disambiguation,
    upsert_lexicon_term,
)

runner = CliRunner()


def _wrong_texts(db_path: Path) -> set[str]:
    """Return the surface forms covered by deterministic lexicon correction."""
    return {rule.wrong_text for rule in list_lexicon_correction_rules(db_path=db_path)}


def test_set_alias_disambiguation_excludes_alias_from_blanket_rules(
    tmp_path: Path,
) -> None:
    """A disambiguated alias must drop out of deterministic replacement."""
    db_path = tmp_path / "lexicon.sqlite"
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC", "阿克米"),
        status="active",
        db_path=db_path,
    )
    assert {"AC", "阿克米"} <= _wrong_texts(db_path)

    entry = set_alias_disambiguation(
        term="Acme",
        alias="AC",
        guidance="指 Acme 平台时改成 Acme；指个人贡献者角色时保持原样",
        db_path=db_path,
    )

    assert entry is not None and entry.canonical == "Acme"
    # AC is now LLM-resolved by context; 阿克米 stays a deterministic rule.
    assert "AC" not in _wrong_texts(db_path)
    assert "阿克米" in _wrong_texts(db_path)
    listed = list_lexicon_disambiguations(db_path=db_path)
    assert [(d.alias, d.guidance) for d in listed] == [(entry.alias, entry.guidance)]


def test_disambiguation_excludes_context_derived_rule(tmp_path: Path) -> None:
    """A disambiguated surface must drop out even when accepted contexts back it.

    Regression: blanket rules come from both aliases AND accepted context rows;
    excluding only the alias path left the context-derived AC->Acme rule alive.
    """
    db_path = tmp_path / "lexicon.sqlite"
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC",),
        status="active",
        db_path=db_path,
    )
    record_lexicon_contexts(
        [
            LexiconContext(
                canonical="Acme",
                wrong_text="AC",
                corrected_text="Acme",
                left_context="把",
                right_context="维修平台",
                category="system",
                speaker_name="敬悦",
                project_id="p-demo",
                sentence_id=1,
                source="test",
            )
        ],
        db_path=db_path,
    )
    assert "AC" in _wrong_texts(db_path)

    set_alias_disambiguation(
        term="Acme", alias="AC", guidance="按语境判断", db_path=db_path
    )

    assert "AC" not in _wrong_texts(db_path)


def test_set_alias_disambiguation_clear_restores_blanket_rule(tmp_path: Path) -> None:
    """Clearing the guidance returns the alias to deterministic correction."""
    db_path = tmp_path / "lexicon.sqlite"
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC",),
        status="active",
        db_path=db_path,
    )
    set_alias_disambiguation(
        term="Acme", alias="AC", guidance="按语境判断", db_path=db_path
    )
    assert "AC" not in _wrong_texts(db_path)

    cleared = set_alias_disambiguation(
        term="Acme", alias="AC", guidance="", db_path=db_path
    )

    assert cleared is None
    assert "AC" in _wrong_texts(db_path)
    assert list_lexicon_disambiguations(db_path=db_path) == []


def test_lexicon_disambiguate_command(tmp_path: Path) -> None:
    """The disambiguate CLI marks an alias ambiguous and persists the guidance."""
    db_path = tmp_path / "lexicon.sqlite"
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC",),
        status="active",
        db_path=db_path,
    )

    result = runner.invoke(
        app,
        [
            "lexicon",
            "disambiguate",
            "Acme",
            "AC",
            "指 Acme 平台时改成 Acme；指角色时保持原样",
            "--lexicon-db",
            str(db_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cleared"] is False
    assert payload["disambiguation"]["alias"] == "AC"
    assert "AC" not in _wrong_texts(db_path)


def _add_term_with_disambiguated_alias(db_path: Path, guidance: str) -> None:
    """Create a term with one disambiguated alias (AC) and one blanket alias.

    Uses neutral placeholder fixtures only; concrete business terms must never be
    hardcoded into this public repo (root AGENTS.md), not even in tests.
    """
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC", "阿克米"),
        status="active",
        db_path=db_path,
    )
    set_alias_disambiguation(
        term="Acme", alias="AC", guidance=guidance, db_path=db_path
    )


def test_lexicon_show_marks_disambiguated_alias(tmp_path: Path) -> None:
    """Show text must flag disambiguated aliases and print the full guidance."""
    db_path = tmp_path / "lexicon.sqlite"
    guidance = "指产品平台时改成 Acme，指其他义项时保持原样"
    _add_term_with_disambiguated_alias(db_path, guidance)

    result = runner.invoke(app, ["lexicon", "show", "Acme", "--lexicon-db", str(db_path)])

    assert result.exit_code == 0, result.output
    assert "AC (asr_error) [ambiguous]" in result.output
    assert f"Guidance: {guidance}" in result.output
    # The blanket alias stays unmarked.
    assert "阿克米 (asr_error)" in result.output
    assert "阿克米 (asr_error) [ambiguous]" not in result.output


def test_lexicon_show_renders_multiline_guidance(tmp_path: Path) -> None:
    """Multi-line guidance must keep continuation lines aligned under Guidance:."""
    db_path = tmp_path / "lexicon.sqlite"
    guidance = "产品甲=平台一\n产品乙=平台二\n拿不准保留原词"
    _add_term_with_disambiguated_alias(db_path, guidance)

    result = runner.invoke(app, ["lexicon", "show", "Acme", "--lexicon-db", str(db_path)])

    indent = " " * len("    Guidance: ")
    assert result.exit_code == 0, result.output
    assert "    Guidance: 产品甲=平台一" in result.output
    assert f"{indent}产品乙=平台二" in result.output
    assert f"{indent}拿不准保留原词" in result.output


def test_lexicon_show_json_includes_disambiguation(tmp_path: Path) -> None:
    """Show JSON must expose disambiguation so agents can review without SQL."""
    db_path = tmp_path / "lexicon.sqlite"
    guidance = "指产品平台时改成 Acme，指其他义项时保持原样"
    _add_term_with_disambiguated_alias(db_path, guidance)

    result = runner.invoke(
        app, ["lexicon", "show", "Acme", "--lexicon-db", str(db_path), "--json"]
    )
    payload = json.loads(result.output)
    by_alias = {alias["alias"]: alias for alias in payload["aliases"]}

    assert result.exit_code == 0
    assert by_alias["AC"]["disambiguation"] == guidance
    assert by_alias["阿克米"]["disambiguation"] is None


def test_lexicon_list_marks_ambiguous_count(tmp_path: Path) -> None:
    """List must surface how many aliases go through polish disambiguation."""
    db_path = tmp_path / "lexicon.sqlite"
    _add_term_with_disambiguated_alias(db_path, "按语境判断")

    json_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path), "--json"]
    )
    table_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path)]
    )
    term = json.loads(json_result.output)["terms"][0]

    assert json_result.exit_code == 0
    assert term["alias_count"] == 2
    assert term["ambiguous_alias_count"] == 1
    assert table_result.exit_code == 0
    assert "ambiguous" in table_result.output


def test_lexicon_export_import_preserves_disambiguation(tmp_path: Path) -> None:
    """Export/import must not silently drop disambiguation guidance."""
    db_path = tmp_path / "lexicon.sqlite"
    imported_db = tmp_path / "imported.sqlite"
    output = tmp_path / "lexicon.json"
    guidance = "指产品平台时改成 Acme，指其他义项时保持原样"
    _add_term_with_disambiguated_alias(db_path, guidance)

    export_result = runner.invoke(
        app, ["lexicon", "export", "--lexicon-db", str(db_path), "--output", str(output)]
    )
    import_result = runner.invoke(
        app, ["lexicon", "import", str(output), "--lexicon-db", str(imported_db)]
    )
    show_result = runner.invoke(
        app, ["lexicon", "show", "Acme", "--lexicon-db", str(imported_db), "--json"]
    )
    payload = json.loads(show_result.output)
    by_alias = {alias["alias"]: alias for alias in payload["aliases"]}

    assert export_result.exit_code == 0
    assert import_result.exit_code == 0
    assert by_alias["AC"]["disambiguation"] == guidance
    assert by_alias["阿克米"]["disambiguation"] is None
    # The disambiguated alias stays out of blanket replacement after import.
    assert "AC" not in _wrong_texts(imported_db)


def test_lexicon_import_clears_stale_disambiguation(tmp_path: Path) -> None:
    """Importing a blanket alias must clear stale local guidance (source wins).

    A restore/sync from an export where the alias was cleared must not leave the
    target's old guidance behind, or the alias stays wrongly excluded from
    blanket correction and is fed to polish with outdated instructions.
    """
    source_db = tmp_path / "source.sqlite"
    target_db = tmp_path / "target.sqlite"
    output = tmp_path / "lexicon.json"
    # Source: AC is a plain blanket alias (no disambiguation).
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC",),
        status="active",
        db_path=source_db,
    )
    runner.invoke(
        app, ["lexicon", "export", "--lexicon-db", str(source_db), "--output", str(output)]
    )
    # Target already marks AC ambiguous with stale guidance.
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC",),
        status="active",
        db_path=target_db,
    )
    set_alias_disambiguation(
        term="Acme", alias="AC", guidance="stale guidance", db_path=target_db
    )
    assert "AC" not in _wrong_texts(target_db)

    import_result = runner.invoke(
        app, ["lexicon", "import", str(output), "--lexicon-db", str(target_db)]
    )
    show_result = runner.invoke(
        app, ["lexicon", "show", "Acme", "--lexicon-db", str(target_db), "--json"]
    )
    by_alias = {
        alias["alias"]: alias for alias in json.loads(show_result.output)["aliases"]
    }

    assert import_result.exit_code == 0
    assert by_alias["AC"]["disambiguation"] is None
    # Cleared guidance returns AC to deterministic blanket replacement.
    assert "AC" in _wrong_texts(target_db)


def test_lexicon_import_legacy_payload_preserves_guidance(tmp_path: Path) -> None:
    """A legacy export that omits the disambiguation key must keep local guidance.

    Absent key = "no opinion" (predates the field); only an explicit null clears
    stale guidance. Importing an old backup must not silently wipe guidance the
    user configured after that backup was taken.
    """
    target_db = tmp_path / "target.sqlite"
    upsert_lexicon_term(
        canonical="Acme",
        category="system",
        description="",
        aliases=("AC",),
        status="active",
        db_path=target_db,
    )
    set_alias_disambiguation(
        term="Acme", alias="AC", guidance="keep me", db_path=target_db
    )
    # Legacy payload: the alias object has no disambiguation key at all.
    legacy_payload = {
        "schema_version": 2,
        "terms": [
            {
                "canonical": "Acme",
                "category": "system",
                "aliases": [{"alias": "AC", "alias_type": "asr_error"}],
            }
        ],
    }

    import_lexicon_payload(legacy_payload, db_path=target_db)
    detail = get_lexicon_term("Acme", db_path=target_db)
    ac_alias = next(alias for alias in detail.aliases if alias.alias == "AC")

    assert ac_alias.disambiguation == "keep me"
    assert "AC" not in _wrong_texts(target_db)


def test_lexicon_add_list_show_and_stats(tmp_path: Path) -> None:
    """Local lexicon commands should manage the vocabulary knowledge base."""
    db_path = tmp_path / "lexicon.sqlite"

    add_result = runner.invoke(
        app,
        [
            "lexicon",
            "add",
            "Acme",
            "--category",
            "system",
            "--description",
            "platform name",
            "--alias",
            "阿克米",
            "--lexicon-db",
            str(db_path),
        ],
    )
    list_result = runner.invoke(app, ["lexicon", "list", "--lexicon-db", str(db_path)])
    plain_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path), "--plain"]
    )
    show_result = runner.invoke(
        app, ["lexicon", "show", "阿克米", "--lexicon-db", str(db_path)]
    )
    stats_result = runner.invoke(
        app, ["lexicon", "stats", "--lexicon-db", str(db_path)]
    )

    assert add_result.exit_code == 0
    assert "Lexicon term saved." in add_result.output
    assert list_result.exit_code == 0
    assert re.search(r"lex-[0-9a-f]{16}", list_result.output)
    assert "Acme" in list_result.output
    assert "system" in list_result.output
    assert plain_result.exit_code == 0
    assert (
        plain_result.output.splitlines()[0]
        == "id\tterm\tcategory\tstatus\taliases\tcontexts\tupdated"
    )
    assert re.search(
        r"lex-[0-9a-f]{16}\tAcme\tsystem\tactive\t1\t0\t", plain_result.output
    )
    assert "╭" not in plain_result.output
    assert show_result.exit_code == 0
    assert "ID: lex-" in show_result.output
    assert "Term: Acme" in show_result.output
    assert "阿克米 (asr_error)" in show_result.output
    assert stats_result.exit_code == 0
    assert "Terms: 1 active / 0 inactive / 1 total" in stats_result.output
    assert "ASR hotwords: 1" in stats_result.output


def test_lexicon_show_json_prints_term_detail(tmp_path: Path) -> None:
    """Show should offer a stable JSON view for agents and scripts."""
    db_path = tmp_path / "lexicon.sqlite"
    runner.invoke(
        app,
        [
            "lexicon",
            "add",
            "Acme",
            "--category",
            "system",
            "--alias",
            "阿克米",
            "--lexicon-db",
            str(db_path),
        ],
    )

    result = runner.invoke(
        app, ["lexicon", "show", "Acme", "--lexicon-db", str(db_path), "--json"]
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["term"]["canonical"] == "Acme"
    assert re.fullmatch(r"lex-[0-9a-f]{16}", payload["term"]["public_id"])
    assert payload["aliases"][0]["alias"] == "阿克米"


def test_lexicon_commands_accept_prefixed_public_id(tmp_path: Path) -> None:
    """Lifecycle commands should target terms by stable prefixed public id."""
    db_path = tmp_path / "lexicon.sqlite"
    runner.invoke(
        app,
        [
            "lexicon",
            "add",
            "Acme",
            "--category",
            "system",
            "--alias",
            "阿克米",
            "--lexicon-db",
            str(db_path),
        ],
    )
    list_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path), "--json"]
    )
    public_id = json.loads(list_result.output)["terms"][0]["public_id"]

    show_result = runner.invoke(
        app, ["lexicon", "show", public_id, "--lexicon-db", str(db_path), "--json"]
    )
    delete_result = runner.invoke(
        app, ["lexicon", "delete", public_id, "--lexicon-db", str(db_path), "--yes"]
    )
    all_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path), "--status", "all"]
    )

    assert re.fullmatch(r"lex-[0-9a-f]{16}", public_id)
    assert show_result.exit_code == 0
    assert json.loads(show_result.output)["term"]["canonical"] == "Acme"
    assert delete_result.exit_code == 0
    assert f"ID: {public_id}" in delete_result.output
    assert public_id in all_result.output
    assert "inactive" in all_result.output


def test_lexicon_list_backfills_public_ids_for_existing_database(
    tmp_path: Path,
) -> None:
    """Opening an existing v1 lexicon database should create stable public ids."""
    db_path = tmp_path / "old.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata(key, value) VALUES ('schema_version', '1');
            CREATE TABLE terms (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              canonical TEXT NOT NULL UNIQUE,
              category TEXT NOT NULL,
              description TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE aliases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              term_id INTEGER NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
              alias TEXT NOT NULL,
              alias_type TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(term_id, alias, alias_type)
            );
            CREATE TABLE contexts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              term_id INTEGER NOT NULL REFERENCES terms(id) ON DELETE CASCADE,
              wrong_text TEXT NOT NULL,
              corrected_text TEXT NOT NULL,
              left_context TEXT NOT NULL,
              right_context TEXT NOT NULL,
              speaker_name TEXT,
              project_id TEXT NOT NULL,
              sentence_id INTEGER,
              source TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE asr_hotword_vocabularies (
              target_model TEXT NOT NULL,
              endpoint TEXT NOT NULL,
              vocabulary_hash TEXT NOT NULL,
              vocabulary_id TEXT NOT NULL,
              hotword_count INTEGER NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(target_model, endpoint)
            );
            INSERT INTO terms(canonical, category) VALUES ('Acme', 'system');
            """
        )

    result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path), "--json"]
    )
    payload = json.loads(result.output)
    public_id = payload["terms"][0]["public_id"]

    assert result.exit_code == 0
    assert re.fullmatch(r"lex-[0-9a-f]{16}", public_id)
    with sqlite3.connect(db_path) as connection:
        stored_public_id = connection.execute(
            "SELECT public_id FROM terms WHERE canonical = 'Acme'"
        ).fetchone()[0]
        schema_version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    assert stored_public_id == public_id
    assert schema_version == "2"


def test_lexicon_delete_deactivates_term_and_hotword(tmp_path: Path) -> None:
    """Deactivate should keep history while removing active hotwords."""
    db_path = tmp_path / "lexicon.sqlite"
    runner.invoke(
        app,
        [
            "lexicon",
            "add",
            "Acme",
            "--category",
            "system",
            "--alias",
            "阿克米",
            "--lexicon-db",
            str(db_path),
        ],
    )

    delete_result = runner.invoke(
        app, ["lexicon", "delete", "Acme", "--lexicon-db", str(db_path), "--yes"]
    )
    active_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path)]
    )
    all_result = runner.invoke(
        app, ["lexicon", "list", "--lexicon-db", str(db_path), "--status", "all"]
    )
    hotwords_result = runner.invoke(
        app, ["lexicon", "hotwords", "list", "--lexicon-db", str(db_path)]
    )

    assert delete_result.exit_code == 0
    assert "Lexicon term deactivated." in delete_result.output
    assert "No lexicon terms." in active_result.output
    assert "inactive" in all_result.output
    assert "Hotwords: 0" in hotwords_result.output


def test_lexicon_delete_prompt_shows_term_details(tmp_path: Path) -> None:
    """Interactive delete should show the resolved object before asking."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("阿克米", "Acme")], db_path=db_path)
    public_id = json.loads(
        runner.invoke(
            app, ["lexicon", "list", "--lexicon-db", str(db_path), "--json"]
        ).output
    )["terms"][0]["public_id"]

    result = runner.invoke(
        app,
        ["lexicon", "delete", public_id, "--lexicon-db", str(db_path)],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Lexicon Delete Review" in result.output
    assert public_id in result.output
    assert "Term" in result.output
    assert "Acme" in result.output
    assert "Aliases" in result.output
    assert "阿克米 (asr_error)" in result.output
    assert "Contexts" in result.output
    assert "p-demo#1" in result.output
    assert "Proceed to deactivate this lexicon term?" in result.output
    assert "Lexicon term deactivated." in result.output


def test_lexicon_export_import_round_trip(tmp_path: Path) -> None:
    """Exported local lexicon JSON should import into another database."""
    db_path = tmp_path / "lexicon.sqlite"
    imported_db = tmp_path / "imported.sqlite"
    output = tmp_path / "lexicon.json"
    record_lexicon_contexts([_context("阿克米", "Acme")], db_path=db_path)

    export_result = runner.invoke(
        app,
        ["lexicon", "export", "--lexicon-db", str(db_path), "--output", str(output)],
    )
    import_result = runner.invoke(
        app,
        ["lexicon", "import", str(output), "--lexicon-db", str(imported_db)],
    )
    show_result = runner.invoke(
        app, ["lexicon", "show", "Acme", "--lexicon-db", str(imported_db)]
    )

    assert export_result.exit_code == 0
    assert "Lexicon exported." in export_result.output
    assert import_result.exit_code == 0
    assert "Terms: 1" in import_result.output
    assert show_result.exit_code == 0
    assert "Term: Acme" in show_result.output
    assert "ID: lex-" in show_result.output
    assert "阿克米 (asr_error)" in show_result.output
    assert "p-demo#1" in show_result.output


def test_lexicon_hotwords_export_writes_dashscope_table(tmp_path: Path) -> None:
    """Hotword export should use accepted correction terms."""
    db_path = tmp_path / "lexicon.sqlite"
    output = tmp_path / "hotwords.json"
    record_lexicon_contexts([_context("阿克米", "Acme")], db_path=db_path)

    result = runner.invoke(
        app,
        [
            "lexicon",
            "hotwords",
            "export",
            "--lexicon-db",
            str(db_path),
            "--output",
            str(output),
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "ASR hotwords exported." in result.output
    assert payload["dashscope_vocabulary"] == [{"text": "Acme", "weight": 4}]


def test_lexicon_hotwords_list_prints_local_terms(tmp_path: Path) -> None:
    """List should show the local hotwords without writing an artifact."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("阿克米", "Acme")], db_path=db_path)

    result = runner.invoke(
        app, ["lexicon", "hotwords", "list", "--lexicon-db", str(db_path)]
    )

    assert result.exit_code == 0
    assert "Hotwords: 1" in result.output
    assert "1. Acme weight=4 category=system" in result.output


def test_lexicon_hotwords_sync_dry_run_does_not_require_api_key(tmp_path: Path) -> None:
    """Dry-run sync should be usable for inspection without DashScope credentials."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("莫", "墨总")], db_path=db_path)

    result = runner.invoke(
        app,
        ["lexicon", "hotwords", "sync", "--lexicon-db", str(db_path), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "ASR hotword sync dry run." in result.output
    assert "Hotwords: 1" in result.output


def test_lexicon_hotwords_status_reports_current_cache(tmp_path: Path) -> None:
    """Status should compare local hotword hash with the cached vocabulary id."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("阿克米", "Acme")], db_path=db_path)
    hotwords = list_asr_hotwords(db_path=db_path)
    save_asr_vocabulary_state(
        AsrVocabularyState(
            "fun-asr",
            "https://dashscope.aliyuncs.com/api/v1",
            hotword_hash(hotwords),
            "vocab-demo",
            len(hotwords),
        ),
        db_path=db_path,
    )

    result = runner.invoke(
        app, ["lexicon", "hotwords", "status", "--lexicon-db", str(db_path)]
    )

    assert result.exit_code == 0
    assert "Cache: current" in result.output
    assert "Cached vocabulary ID: vocab-demo" in result.output


def test_lexicon_hotwords_clear_cache_removes_matching_state(tmp_path: Path) -> None:
    """Clear-cache should remove stale local vocabulary ids without DashScope access."""
    db_path = tmp_path / "lexicon.sqlite"
    save_asr_vocabulary_state(
        AsrVocabularyState(
            "fun-asr", "https://dashscope.aliyuncs.com/api/v1", "hash", "vocab-demo", 1
        ),
        db_path=db_path,
    )

    result = runner.invoke(
        app,
        [
            "lexicon",
            "hotwords",
            "clear-cache",
            "--lexicon-db",
            str(db_path),
            "--vocabulary-id",
            "vocab-demo",
        ],
    )

    assert result.exit_code == 0
    assert "ASR hotword cache cleared." in result.output
    assert (
        get_asr_vocabulary_state(
            target_model="fun-asr",
            endpoint="https://dashscope.aliyuncs.com/api/v1",
            db_path=db_path,
        )
        is None
    )


def test_lexicon_hotwords_remote_list_uses_dashscope_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote-list should expose DashScope vocabulary rows."""
    monkeypatch.setattr(lexicon_commands, "load_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(
        lexicon_commands,
        "list_remote_asr_vocabularies",
        lambda **kwargs: [{"vocabulary_id": "vocab-demo", "status": "OK"}],
    )

    result = runner.invoke(app, ["lexicon", "hotwords", "remote-list", "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["vocabularies"] == [{"vocabulary_id": "vocab-demo", "status": "OK"}]


def test_lexicon_hotwords_remote_show_prints_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote-show should display one vocabulary payload."""
    monkeypatch.setattr(lexicon_commands, "load_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(
        lexicon_commands,
        "query_remote_asr_vocabulary",
        lambda **kwargs: {
            "status": "OK",
            "target_model": "fun-asr",
            "vocabulary": [{"text": "Acme", "weight": 4}],
        },
    )

    result = runner.invoke(app, ["lexicon", "hotwords", "remote-show", "vocab-demo"])

    assert result.exit_code == 0
    assert "Vocabulary ID: vocab-demo" in result.output
    assert "1. Acme weight=4" in result.output


def test_lexicon_hotwords_remote_delete_can_clear_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote-delete should require confirmation and optionally clear matching cache."""
    db_path = tmp_path / "lexicon.sqlite"
    deleted = []
    save_asr_vocabulary_state(
        AsrVocabularyState(
            "fun-asr", "https://dashscope.aliyuncs.com/api/v1", "hash", "vocab-demo", 1
        ),
        db_path=db_path,
    )
    monkeypatch.setattr(lexicon_commands, "load_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(
        lexicon_commands,
        "delete_remote_asr_vocabulary",
        lambda **kwargs: deleted.append(kwargs["vocabulary_id"]),
    )

    result = runner.invoke(
        app,
        [
            "lexicon",
            "hotwords",
            "remote-delete",
            "vocab-demo",
            "--yes",
            "--clear-cache",
            "--lexicon-db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert deleted == ["vocab-demo"]
    assert "Deleted remote ASR vocabulary: vocab-demo" in result.output
    assert (
        get_asr_vocabulary_state(
            target_model="fun-asr",
            endpoint="https://dashscope.aliyuncs.com/api/v1",
            db_path=db_path,
        )
        is None
    )


def _context(wrong: str, corrected: str) -> LexiconContext:
    """Build one accepted lexicon context."""
    return LexiconContext(
        canonical=corrected,
        wrong_text=wrong,
        corrected_text=corrected,
        left_context="",
        right_context="",
        category="system",
        speaker_name=None,
        project_id="p-demo",
        sentence_id=1,
        source="test",
    )


def _settings() -> Settings:
    """Build minimal settings for remote command tests."""
    return Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )
