"""Tests for root command boundaries."""

from __future__ import annotations

from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


def test_root_version_option_prints_version() -> None:
    """Root CLI should expose a stable --version option."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("meeting-asr ")


def test_root_paths_command_prints_json(monkeypatch, tmp_path) -> None:
    """Global state paths should be discoverable for humans and scripts."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    result = runner.invoke(app, ["paths", "--json"])

    assert result.exit_code == 0
    assert '"key": "projects"' in result.output
    assert str(tmp_path / "data" / "meeting-asr" / "projects") in result.output


def test_root_help_command_prints_root_and_nested_help() -> None:
    """Git-like help command should expose root and subcommand help."""
    no_args_result = runner.invoke(app, [])
    native_help_result = runner.invoke(app, ["--help"])
    root_result = runner.invoke(app, ["help"])
    nested_result = runner.invoke(app, ["help", "project", "list"])

    assert no_args_result.exit_code == 0
    assert "Quick start:" in no_args_result.output
    assert native_help_result.exit_code == 0
    assert "Quick start:" in native_help_result.output
    assert root_result.exit_code == 0
    assert "Quick start:" in root_result.output
    assert "meeting-asr project run <video>" in root_result.output
    assert nested_result.exit_code == 0
    assert "Usage:" in nested_result.output
    assert "meeting-asr project list [OPTIONS]" in nested_result.output
    assert "--plain" in nested_result.output


def test_short_help_option_is_supported_for_root_and_nested_commands() -> None:
    """Common CLI convention -h should work like --help."""
    root_result = runner.invoke(app, ["-h"])
    group_result = runner.invoke(app, ["project", "-h"])
    command_result = runner.invoke(app, ["project", "list", "-h"])
    direct_command_result = runner.invoke(app, ["doctor", "-h"])

    assert root_result.exit_code == 0
    assert "Quick start:" in root_result.output
    assert "--help, -h" in root_result.output
    assert group_result.exit_code == 0
    assert "Usage:" in group_result.output
    assert "project [OPTIONS]" in group_result.output
    assert "--help" in group_result.output
    assert "-h" in group_result.output
    assert command_result.exit_code == 0
    assert "Usage:" in command_result.output
    assert "project list [OPTIONS]" in command_result.output
    assert "--help" in command_result.output
    assert "-h" in command_result.output
    assert direct_command_result.exit_code == 0
    assert "Usage:" in direct_command_result.output
    assert "doctor [OPTIONS]" in direct_command_result.output
    assert "--help" in direct_command_result.output
    assert "-h" in direct_command_result.output


def test_root_help_command_can_render_chinese(monkeypatch) -> None:
    """Help command should render Meeting-ASR-owned text in Chinese."""
    option_result = runner.invoke(app, ["--lang", "zh", "help", "project", "list"])
    locale_result = runner.invoke(app, [], env={"LC_ALL": "zh_CN.UTF-8"})
    native_help_result = runner.invoke(app, ["--help"], env={"LANG": "zh_CN.UTF-8"})
    monkeypatch.setenv("MEETING_ASR_LANG", "zh")
    env_result = runner.invoke(app, ["help"])

    assert option_result.exit_code == 0
    assert "用法:" in option_result.output
    assert "列出默认或指定项目库里的项目。" in option_result.output
    assert "--plain                      输出稳定的制表符分隔文本。" in option_result.output
    assert locale_result.exit_code == 0
    assert "用于 DashScope 会议转写的项目化 CLI。" in locale_result.output
    assert native_help_result.exit_code == 0
    assert "快速开始:" in native_help_result.output
    assert env_result.exit_code == 0
    assert "快速开始:" in env_result.output
    assert "命令" in env_result.output


def test_native_subcommand_help_uses_localized_renderer() -> None:
    """Native no-args and --help paths should not fall back to Typer English."""
    group_result = runner.invoke(app, ["--lang", "zh", "project"])
    leaf_result = runner.invoke(app, ["--lang", "zh", "project", "list", "--help"])

    assert group_result.exit_code == 0
    assert "用法:" in group_result.output
    assert "管理项目化转写流程。" in group_result.output
    assert "创建项目目录和 project.json 元数据。" in group_result.output
    assert "Create a project directory" not in group_result.output
    assert "--help, -h" in group_result.output
    assert leaf_result.exit_code == 0
    assert "用法:" in leaf_result.output
    assert "列出默认或指定项目库里的项目。" in leaf_result.output
    assert "--projects-dir" in leaf_result.output
    assert "指定项目库目录。" in leaf_result.output
    assert "--help, -h" in leaf_result.output


def test_localized_help_leads_with_examples_and_translated_options() -> None:
    """High-frequency Chinese help should include examples and translated option text."""
    run_result = runner.invoke(app, ["--lang", "zh", "project", "run", "--help"])
    delete_result = runner.invoke(app, ["--lang", "zh", "project", "delete", "--help"])
    hotwords_result = runner.invoke(app, ["--lang", "zh", "lexicon", "hotwords", "sync", "--help"])

    assert run_result.exit_code == 0
    assert "示例" in run_result.output
    assert "meeting-asr project run ~/Downloads/meeting.mp4" in run_result.output
    assert "指定会议总结使用的" in run_result.output
    assert "Generate title and summary after ASR." not in run_result.output
    assert delete_result.exit_code == 0
    assert "meeting-asr project delete p-292d10c1232b79a0" in delete_result.output
    assert "跳过确认提示。" in delete_result.output
    assert "Physically remove instead of moving to trash." not in delete_result.output
    assert hotwords_result.exit_code == 0
    assert "meeting-asr lexicon hotwords sync --dry-run" in hotwords_result.output
    assert "指定 DashScope ASR 目标模型。" in hotwords_result.output
    assert "Only render local hotword table." not in hotwords_result.output


def test_help_flag_overrides_parse_errors() -> None:
    """clig.dev expects adding -h to invalid input to show help."""
    root_result = runner.invoke(app, ["--bad", "-h"])
    direct_result = runner.invoke(app, ["doctor", "--bad", "-h"])
    group_result = runner.invoke(app, ["--lang", "zh", "project", "--bad", "-h"])
    leaf_result = runner.invoke(app, ["--lang", "zh", "project", "list", "--bad", "-h"])

    assert root_result.exit_code == 0
    assert "Quick start:" in root_result.output
    assert "No such option" not in root_result.output
    assert direct_result.exit_code == 0
    assert "Usage:" in direct_result.output
    assert "meeting-asr doctor [OPTIONS]" in direct_result.output
    assert "No such option" not in direct_result.output
    assert group_result.exit_code == 0
    assert "用法:" in group_result.output
    assert "meeting-asr project [OPTIONS]" in group_result.output
    assert "管理项目化转写流程。" in group_result.output
    assert "No such option" not in group_result.output
    assert leaf_result.exit_code == 0
    assert "用法:" in leaf_result.output
    assert "meeting-asr project list [OPTIONS]" in leaf_result.output
    assert "列出默认或指定项目库里的项目。" in leaf_result.output
    assert "No such option" not in leaf_result.output


def test_root_lang_rejects_invalid_value_without_traceback() -> None:
    """Invalid language should fail as a CLI parameter error."""
    result = runner.invoke(app, ["--lang", "bad", "help"])

    assert result.exit_code != 0
    assert "Invalid value for --lang" in result.output
    assert "Traceback" not in result.output


def test_chinese_parse_errors_are_actionable() -> None:
    """Parse errors should be localized and point to the relevant help command."""
    command_result = runner.invoke(app, ["--lang", "zh", "project", "nope"])
    option_result = runner.invoke(app, ["--lang", "zh", "project", "list", "--bad"])
    argument_result = runner.invoke(app, ["--lang", "zh", "project", "delete"])

    assert command_result.exit_code == 2
    assert "用法: meeting-asr project [OPTIONS] COMMAND [ARGS]..." in command_result.output
    assert "没有这个命令：nope" in command_result.output
    assert "meeting-asr project -h" in command_result.output
    assert "No such command" not in command_result.output
    assert option_result.exit_code == 2
    assert "没有这个选项：--bad" in option_result.output
    assert "meeting-asr project list -h" in option_result.output
    assert "No such option" not in option_result.output
    assert argument_result.exit_code == 2
    assert "缺少必填参数：PROJECT" in argument_result.output
    assert "meeting-asr project delete -h" in argument_result.output
    assert "Missing argument" not in argument_result.output


def test_top_level_audio_command_is_not_registered() -> None:
    """Audio preparation should stay under project workflows."""
    result = runner.invoke(app, ["audio", "extract"])

    assert result.exit_code != 0
    assert "No such command" in result.output
