"""Tests for the runtime agent self-discovery contract."""

from __future__ import annotations

import json
import sys

from typer.testing import CliRunner

from app.cli import app, main
from app.commands.agent import COMMANDS_META, SIDE_EFFECT_ENUM

runner = CliRunner()


def test_agent_guide_supports_sections_and_json() -> None:
    """Agents should be able to fetch only the guide section they need."""
    section_list = runner.invoke(app, ["agent-guide", "--list-sections"])
    section_result = runner.invoke(
        app, ["agent-guide", "--section", "workflow", "--json"]
    )
    voiceprint_result = runner.invoke(
        app, ["agent-guide", "--section", "voiceprint", "--json"]
    )

    assert section_list.exit_code == 0
    assert "workflow" in section_list.output
    assert "rerun-and-caching" in section_list.output
    assert "review-and-voiceprints" in section_list.output
    assert "reporting-back" in section_list.output
    assert section_result.exit_code == 0
    payload = json.loads(section_result.output)
    assert payload["schema_version"] == 1
    assert payload["cmd"] == "agent-guide"
    assert payload["ok"] is True
    assert payload["data"]["section"] == "workflow"
    assert "project run <video>" in payload["data"]["markdown"]
    assert voiceprint_result.exit_code == 0
    voiceprint_payload = json.loads(voiceprint_result.output)
    assert voiceprint_payload["data"]["section"] == "review-and-voiceprints"
    assert "quarantined" in voiceprint_payload["data"]["markdown"]
    assert "must not participate in matching" in voiceprint_payload["data"]["markdown"]
    rerun_result = runner.invoke(app, ["agent-guide", "--section", "rerun", "--json"])
    assert rerun_result.exit_code == 0
    rerun_payload = json.loads(rerun_result.output)
    assert rerun_payload["data"]["section"] == "rerun-and-caching"
    assert "project rerun <project-id>" in rerun_payload["data"]["markdown"]


def test_commands_json_exposes_side_effects() -> None:
    """The commands contract should tell agents which commands write or call networks."""
    result = runner.invoke(app, ["commands", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    commands = {row["name"]: row for row in payload["data"]["commands"]}
    assert payload["cmd"] == "commands"
    assert "project run" in commands
    assert "project rerun" in commands
    assert "network-io" in commands["project run"]["side_effects"]
    assert "fs-write" in commands["project run"]["side_effects"]
    assert "oss-upload" in commands["project rerun"]["side_effects"]
    assert "existing project audio" in commands["project rerun"]["side_effect_notes"][0]
    assert "--no-summarize" in commands["project run"]["side_effect_notes"][0]
    assert commands["project review"]["interactive"] is True
    assert commands["project delete"]["conditional_side_effects"]["--permanent"] == [
        "destructive"
    ]


def test_commands_schema_is_available() -> None:
    """Agents should not have to infer the metadata schema from examples."""
    result = runner.invoke(app, ["commands", "--schema"])
    json_result = runner.invoke(app, ["commands", "--schema", "--json"])

    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert schema["title"] == "meeting-asr commands --json data schema"
    assert "commands" in schema["required"]
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["data"]["title"] == "meeting-asr commands --json data schema"


def test_version_json_exposes_supported_features() -> None:
    """The version probe should be a cheap compatibility handshake."""
    result = runner.invoke(app, ["version", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    features = payload["data"]["supported_features"]
    assert payload["cmd"] == "version"
    assert features["agent_guide"] is True
    assert features["commands_json"] is True
    assert features["version_json"] is True
    assert features["reusable_project_audio"] is True
    assert features["project_rerun_command"] is True
    assert features["agent_guide_voiceprint_policy"] is True
    assert features["voiceprint_inactive_samples_excluded"] is True


def test_root_version_json_wrapper(monkeypatch, capsys) -> None:
    """The installed CLI should support the proxyctl-style --version --json probe."""
    monkeypatch.setattr(sys, "argv", ["meeting-asr", "--version", "--json"])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["cmd"] == "version"
    assert payload["data"]["supported_features"]["version_json"] is True


def test_command_side_effects_use_declared_enum() -> None:
    """Command metadata should stay parseable instead of inventing ad hoc labels."""
    known = set(SIDE_EFFECT_ENUM)
    for command in COMMANDS_META:
        assert isinstance(command["side_effects"], list)
        assert set(command["side_effects"]) <= known
        conditional = command.get("conditional_side_effects") or {}
        for effects in conditional.values():
            assert isinstance(effects, list)
            assert set(effects) <= known
