"""Localized Rich help rendering for Meeting-ASR-owned CLI help."""

from __future__ import annotations

import click
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.presentation.cli.i18n import current_cli_language
from app.presentation.cli.output import cli_console

ROOT_INTRO = {
    "en": "Project-based CLI for DashScope meeting transcription workflows.",
    "zh": "用于 DashScope 会议转写的项目化 CLI。",
}
ROOT_SECTIONS = {
    "en": (
        ("Quick start", ("meeting-asr project run <video>", "meeting-asr project review <project-id-or-path>")),
        ("Inspect state", ("meeting-asr project list", "meeting-asr paths", "meeting-asr doctor")),
    ),
    "zh": (
        ("快速开始", ("meeting-asr project run <video>", "meeting-asr project review <project-id-or-path>")),
        ("常用查看", ("meeting-asr project list", "meeting-asr paths", "meeting-asr doctor")),
    ),
}
LABELS = {
    "en": {
        "usage": "Usage",
        "examples": "Examples",
        "arguments": "Arguments",
        "options": "Options",
        "commands": "Commands",
    },
    "zh": {"usage": "用法", "examples": "示例", "arguments": "参数", "options": "选项", "commands": "命令"},
}
COMMAND_ZH = {
    (): "用于 DashScope 会议转写的项目化 CLI。",
    ("doctor",): "检查运行依赖和全局配置。",
    ("help",): "显示 root 或嵌套命令帮助。",
    ("paths",): "显示 Meeting-ASR 配置、数据、缓存和存储路径。",
    ("config",): "管理全局 XDG 配置。",
    ("project",): "管理项目化转写流程。",
    ("project", "create"): "创建项目目录和 project.json 元数据。",
    ("project", "prepare"): "提取项目音频，不启动云端转写。",
    ("project", "transcribe"): "转写项目并写入结构化产物。",
    ("project", "summarize"): "基于已转写项目生成会议标题和回忆索引。",
    ("project", "run"): "创建项目、转写、生成回忆索引并自动匹配 speaker。",
    ("project", "list"): "列出 XDG 项目库里的项目。",
    ("project", "show"): "显示项目概览和产物查看方式。",
    ("project", "update"): "更新可编辑的项目元数据。",
    ("project", "delete"): "删除项目，默认移动到 Meeting-ASR 回收站。",
    ("project", "review"): "打开推荐的人类 review 流程，检查项目产物和未确认 speaker 姓名。",
    ("project", "status"): "打印项目状态摘要。",
    ("project", "git-init"): "为人工编辑的项目文件初始化可选 Git 跟踪。",
    ("project", "speakers"): "检查、匹配和标注项目 speaker。",
    ("project", "speakers", "inspect"): "打印诊断用 speaker 样例；只读，不应用姓名。",
    ("project", "speakers", "preview"): "用字幕打开源视频以辅助 speaker review。",
    ("project", "speakers", "apply"): "非交互式应用已知 speaker 映射；面向脚本或已确认姓名。",
    ("project", "speakers", "review"): "打开推荐的交互式 speaker 身份 review。",
    ("project", "speakers", "match"): "用全局声纹库匹配项目 speaker。",
    ("project", "speakers", "compare-srt"): "对比钉钉 SRT 和项目字幕。",
    ("project", "transcript"): "查看项目转写产物。",
    ("project", "transcript", "list"): "列出一个项目的转写产物。",
    ("project", "transcript", "show"): "显示一个项目转写产物的内容。",
    ("project", "trash"): "恢复或永久删除已删除项目。",
    ("project", "trash", "list"): "列出回收站里的项目。",
    ("project", "trash", "restore"): "从回收站恢复项目。",
    ("project", "trash", "purge"): "永久删除回收站里的项目。",
    ("project", "trash", "cleanup"): "清理超过指定天数的回收站项目。",
    ("project", "correct"): "检查并应用词汇纠错。",
    ("project", "correct", "edit"): "打开编辑器检查并生成词汇纠错建议。",
    ("project", "correct", "accept"): "接受并应用已生成的词汇纠错建议。",
    ("config", "path"): "打印全局配置文件路径。",
    ("config", "show"): "显示配置值，默认隐藏密钥。",
    ("config", "keys"): "列出支持的配置 key。",
    ("config", "set"): "设置一个全局配置值。",
    ("config", "unset"): "删除一个全局配置值。",
    ("config", "import-env"): "从 .env 文件导入配置。",
    ("voiceprint",): "管理跨项目声纹库。",
    ("voiceprint", "review"): "打开统一声纹 TUI，在项目候选样本和全局声纹库之间切换。",
    ("voiceprint", "capture"): "从已标注项目采集声纹样本。",
    ("voiceprint", "list"): "列出全局声纹库里的说话人。",
    ("voiceprint", "people"): "管理声纹库里的稳定人员 ID。",
    ("voiceprint", "people", "list"): "列出稳定人员 ID。",
    ("voiceprint", "people", "add"): "创建一个新的稳定人员 ID。",
    ("voiceprint", "people", "rename"): "按 ID 修改人员显示名称。",
    ("voiceprint", "people", "show"): "按 ID 显示人员详情。",
    ("voiceprint", "browse"): "打开声纹库浏览 TUI。",
    ("voiceprint", "embed"): "为声纹样本生成 embedding。",
    ("voiceprint", "show"): "显示某个说话人的声纹样本。",
    ("voiceprint", "play"): "播放某个声纹样本。",
    ("voiceprint", "delete-sample"): "删除指定声纹样本。",
    ("voiceprint", "delete-speaker"): "删除一个说话人及其声纹样本。",
    ("voiceprint", "path"): "显示声纹库路径。",
    ("lexicon",): "管理跨项目纠错词库。",
    ("lexicon", "list"): "列出本地词库词条。",
    ("lexicon", "show"): "显示一个词条的详情。",
    ("lexicon", "add"): "添加一个词库词条。",
    ("lexicon", "delete"): "删除一个词库词条。",
    ("lexicon", "stats"): "显示词库统计信息。",
    ("lexicon", "export"): "导出词库 JSON。",
    ("lexicon", "import"): "导入词库 JSON。",
    ("lexicon", "hotwords"): "从已接受纠错导出并同步 ASR 热词。",
    ("lexicon", "hotwords", "list"): "列出本地 ASR 热词候选。",
    ("lexicon", "hotwords", "export"): "导出本地 ASR 热词文件。",
    ("lexicon", "hotwords", "status"): "显示远端 ASR 热词同步状态。",
    ("lexicon", "hotwords", "sync"): "同步本地热词到远端 ASR 词表。",
    ("lexicon", "hotwords", "clear-cache"): "清除本地热词同步缓存。",
    ("lexicon", "hotwords", "remote-list"): "列出远端 ASR 词表。",
    ("lexicon", "hotwords", "remote-show"): "显示远端 ASR 词表详情。",
    ("lexicon", "hotwords", "remote-delete"): "删除远端 ASR 词表。",
    ("oss",): "上传、签名和配置 OSS 对象。",
    ("oss", "upload"): "上传本地文件到 OSS 并打印签名 URL。",
    ("oss", "presign"): "为已有 OSS 对象生成签名 URL。",
    ("oss", "lifecycle"): "配置 OSS 生命周期规则。",
    ("oss", "lifecycle", "set"): "设置或更新 OSS 生命周期规则。",
    ("completion",): "生成或安装 shell completion 脚本。",
    ("completion", "bash"): "输出 bash completion 脚本。",
    ("completion", "zsh"): "输出 zsh completion 脚本。",
    ("completion", "fish"): "输出 fish completion 脚本。",
    ("completion", "powershell"): "输出 PowerShell completion 脚本。",
    ("completion", "pwsh"): "输出 pwsh completion 脚本。",
    ("completion", "csh"): "输出 csh completion 脚本。",
    ("completion", "tcsh"): "输出 tcsh completion 脚本。",
    ("completion", "install"): "安装 shell completion 脚本。",
}
EXAMPLES = {
    ("doctor",): {
        "en": ("meeting-asr doctor", "meeting-asr doctor --full", "meeting-asr doctor --full --json"),
        "zh": ("meeting-asr doctor", "meeting-asr doctor --full", "meeting-asr doctor --full --json"),
    },
    ("paths",): {
        "en": ("meeting-asr paths", "meeting-asr paths --json"),
        "zh": ("meeting-asr paths", "meeting-asr paths --json"),
    },
    ("project", "create"): {
        "en": (
            "meeting-asr project create ~/Downloads/meeting.mp4",
            'meeting-asr project create ~/Downloads/meeting.mp4 --meeting-time "2026-05-02T10:00:00+08:00"',
        ),
        "zh": (
            "meeting-asr project create ~/Downloads/meeting.mp4",
            'meeting-asr project create ~/Downloads/meeting.mp4 --meeting-time "2026-05-02T10:00:00+08:00"',
        ),
    },
    ("project", "run"): {
        "en": (
            "meeting-asr project run ~/Downloads/meeting.mp4",
            "meeting-asr project run ~/Downloads/meeting.mp4 --no-summarize",
        ),
        "zh": (
            "meeting-asr project run ~/Downloads/meeting.mp4",
            "meeting-asr project run ~/Downloads/meeting.mp4 --no-summarize",
        ),
    },
    ("project", "list"): {
        "en": (
            "meeting-asr project list",
            "XDG_DATA_HOME=/path/to/data-home meeting-asr project list",
            "meeting-asr project list --json",
        ),
        "zh": (
            "meeting-asr project list",
            "XDG_DATA_HOME=/path/to/data-home meeting-asr project list",
            "meeting-asr project list --json",
        ),
    },
    ("project", "show"): {
        "en": ("meeting-asr project show p-292d10c1232b79a0", "meeting-asr project show --json"),
        "zh": ("meeting-asr project show p-292d10c1232b79a0", "meeting-asr project show --json"),
    },
    ("project", "update"): {
        "en": (
            'meeting-asr project update p-292d10c1232b79a0 --meeting-time "2026-05-02T10:00:00+08:00"',
            'meeting-asr project update p-292d10c1232b79a0 --title "新的会议标题"',
        ),
        "zh": (
            'meeting-asr project update p-292d10c1232b79a0 --meeting-time "2026-05-02T10:00:00+08:00"',
            'meeting-asr project update p-292d10c1232b79a0 --title "新的会议标题"',
        ),
    },
    ("project", "review"): {
        "en": ("meeting-asr project review", "meeting-asr project review p-292d10c1232b79a0"),
        "zh": ("meeting-asr project review", "meeting-asr project review p-292d10c1232b79a0"),
    },
    ("project", "delete"): {
        "en": ("meeting-asr project delete p-292d10c1232b79a0", "meeting-asr project delete p-292d10c1232b79a0 --permanent"),
        "zh": ("meeting-asr project delete p-292d10c1232b79a0", "meeting-asr project delete p-292d10c1232b79a0 --permanent"),
    },
    ("project", "trash", "list"): {
        "en": ("meeting-asr project trash list", "meeting-asr project trash list --plain"),
        "zh": ("meeting-asr project trash list", "meeting-asr project trash list --plain"),
    },
    ("project", "trash", "restore"): {
        "en": ("meeting-asr project trash restore <project-id>",),
        "zh": ("meeting-asr project trash restore <project-id>",),
    },
    ("project", "transcript", "list"): {
        "en": ("meeting-asr project transcript list <project-id>",),
        "zh": ("meeting-asr project transcript list <project-id>",),
    },
    ("project", "transcript", "show"): {
        "en": (
            "meeting-asr project transcript show <project-id> --kind corrected",
            "meeting-asr project transcript show <project-id> --kind srt",
        ),
        "zh": (
            "meeting-asr project transcript show <project-id> --kind corrected",
            "meeting-asr project transcript show <project-id> --kind srt",
        ),
    },
    ("project", "correct", "edit"): {
        "en": ("meeting-asr project correct edit <project-id>", "meeting-asr project correct edit <project-id> --no-ai"),
        "zh": ("meeting-asr project correct edit <project-id>", "meeting-asr project correct edit <project-id> --no-ai"),
    },
    ("project", "correct", "accept"): {
        "en": ("meeting-asr project correct accept <project-id>",),
        "zh": ("meeting-asr project correct accept <project-id>",),
    },
    ("project", "speakers", "review"): {
        "en": ("meeting-asr project speakers review <project-id>",),
        "zh": ("meeting-asr project speakers review <project-id>",),
    },
    ("project", "speakers", "match"): {
        "en": ("meeting-asr project speakers match <project-id>",),
        "zh": ("meeting-asr project speakers match <project-id>",),
    },
    ("voiceprint", "list"): {
        "en": ("meeting-asr voiceprint list", "meeting-asr voiceprint list --plain"),
        "zh": ("meeting-asr voiceprint list", "meeting-asr voiceprint list --plain"),
    },
    ("voiceprint", "review"): {
        "en": ("meeting-asr voiceprint review <project-id>", "meeting-asr voiceprint review"),
        "zh": ("meeting-asr voiceprint review <project-id>", "meeting-asr voiceprint review"),
    },
    ("voiceprint", "capture"): {
        "en": ("meeting-asr voiceprint capture <project-id>",),
        "zh": ("meeting-asr voiceprint capture <project-id>",),
    },
    ("voiceprint", "embed"): {
        "en": ("meeting-asr voiceprint embed",),
        "zh": ("meeting-asr voiceprint embed",),
    },
    ("voiceprint", "browse"): {
        "en": ("meeting-asr voiceprint browse",),
        "zh": ("meeting-asr voiceprint browse",),
    },
    ("voiceprint", "people", "list"): {
        "en": ("meeting-asr voiceprint people list",),
        "zh": ("meeting-asr voiceprint people list",),
    },
    ("voiceprint", "people", "add"): {
        "en": ("meeting-asr voiceprint people add NAME",),
        "zh": ("meeting-asr voiceprint people add NAME",),
    },
    ("lexicon", "list"): {
        "en": ("meeting-asr lexicon list", "meeting-asr lexicon list --query 术语"),
        "zh": ("meeting-asr lexicon list", "meeting-asr lexicon list --query 术语"),
    },
    ("lexicon", "add"): {
        "en": ('meeting-asr lexicon add "正确术语" --alias "常见错词" --category system',),
        "zh": ('meeting-asr lexicon add "正确术语" --alias "常见错词" --category system',),
    },
    ("lexicon", "hotwords", "export"): {
        "en": ("meeting-asr lexicon hotwords export --output hotwords.json",),
        "zh": ("meeting-asr lexicon hotwords export --output hotwords.json",),
    },
    ("lexicon", "hotwords", "sync"): {
        "en": ("meeting-asr lexicon hotwords sync --dry-run", "meeting-asr lexicon hotwords sync --force"),
        "zh": ("meeting-asr lexicon hotwords sync --dry-run", "meeting-asr lexicon hotwords sync --force"),
    },
    ("oss", "upload"): {
        "en": ("meeting-asr oss upload ./audio.wav",),
        "zh": ("meeting-asr oss upload ./audio.wav",),
    },
    ("oss", "lifecycle", "set"): {
        "en": ("meeting-asr oss lifecycle set --prefix meeting-asr/ --days 7",),
        "zh": ("meeting-asr oss lifecycle set --prefix meeting-asr/ --days 7",),
    },
    ("completion", "install"): {
        "en": ("meeting-asr completion install zsh",),
        "zh": ("meeting-asr completion install zsh",),
    },
}
OPTION_ZH = {
    "--version": "显示版本并退出。",
    "--no-color": "关闭 Rich 彩色输出。",
    "--verbose": "显示详细诊断日志。",
    "--lang": "设置 help 语言：auto、en 或 zh。",
    "--full": "运行完整检查：OSS 上传探测和声纹 embedding 严格检查。",
    "--require-oss": "要求 OSS 配置完整。",
    "--check-oss-access": "检查 OSS bucket 元数据访问。",
    "--oss-upload-probe": "上传、签名 GET 并删除一个很小的 OSS 探测对象。",
    "--require-voiceprint-embedding": "声纹 embedding 后端不可运行时返回失败。",
    "--project-dir": "指定项目目录。",
    "--hash-source": "兼容旧参数；项目身份始终基于内容 hash。",
    "--title": "设置会议标题；省略时可由回忆索引步骤自动生成。",
    "--meeting-time": "设置会议时间。",
    "--speaker-count": "指定预期 speaker 数量。",
    "--language": "指定 ASR 语言。",
    "--model": "指定模型 ID。",
    "--oss-upload": "控制是否上传到 OSS。",
    "--file-url": "直接使用已有 HTTP/HTTPS 音频 URL。",
    "--audio-format": "指定提取后的音频格式。",
    "--asr-hotwords": "指定 ASR 热词表 ID 或热词文件。",
    "--store-dir": "指定全局数据目录。",
    "--voiceprint-model": "指定声纹 embedding 模型。",
    "--match-threshold": "设置声纹自动接受阈值。",
    "--summarize": "ASR 后生成会议标题和回忆索引；可用 --no-summarize 关闭。",
    "--summary-model": "指定回忆索引使用的 DashScope 模型。",
    "--polish": "ASR 后生成转写润色建议；可用 --no-polish 关闭。",
    "--local-correction": "ASR 后应用已接受的本地词库订正规则；可用 --no-local-correction 关闭。",
    "--correction-model": "指定转写润色使用的 DashScope 模型。",
    "--polish-concurrency": "指定转写润色并发批次数。",
    "--progress": "在终端显示交互式进度；可用 --no-progress 关闭。",
    "--agent-log": "输出给 Agent/日志系统使用的 stage/heartbeat 结构化文本；可与 --no-progress 搭配。",
    "--identity-mode": "兼容旧参数；项目身份始终基于内容 hash。",
    "--yes": "跳过确认提示。",
    "--permanent": "物理删除，而不是移入回收站。",
    "--samples-per-page": "覆盖每页 sample 数；默认按 TUI 面板高度计算。",
    "--page-size": "覆盖每页 sample 数；默认按 TUI 面板高度计算。",
    "--summary": "只打印摘要，不打开 TUI。",
    "--map": "非交互式 speaker_id=name 映射。",
    "--sample-count": "每个 speaker 显示的样例数量。",
    "--editor": "指定编辑器命令；可用 {file} 作为文件占位符。",
    "--no-open": "只写 review 文件，不打开编辑器。",
    "--no-ai": "禁用 DashScope 纠错建议，只使用本地规则。",
    "--no-proposal-open": "不打开生成的全量修改建议文件。",
    "--category": "词条分类。",
    "--lexicon-db": "指定词库 SQLite 路径。",
    "--from-original": "忽略已有 corrected transcript，从原始转写开始。",
    "--reveal": "显示密钥明文。",
    "--overwrite": "覆盖已有配置值。",
    "--status": "按状态过滤。",
    "--query": "搜索标准词和别名。",
    "--limit": "限制返回数量。",
    "--context-limit": "限制显示的上下文数量。",
    "--description": "词条说明。",
    "--alias": "别名或常见 ASR 错词。",
    "--output": "输出文件路径。",
    "--active-only": "跳过 inactive 词条。",
    "--target-model": "指定 DashScope ASR 目标模型。",
    "--provider": "指定 provider。",
    "--prefix": "指定 DashScope 词表前缀。",
    "--force": "强制更新远端词表。",
    "--dry-run": "只预览本地热词，不写远端。",
    "--endpoint": "DashScope API 地址；默认读取配置。",
    "--vocabulary-id": "DashScope 词表 ID。",
    "--clear-cache": "同时清除匹配的本地缓存记录。",
    "--rebuild": "重新生成已有 embedding。",
    "--max-seconds": "限制每个声纹样本的最长秒数。",
    "--padding-seconds": "样本前后保留的音频秒数。",
    "--object-name": "指定 OSS object 名称。",
    "--expires-seconds": "指定签名 URL 有效秒数。",
    "--target": "指定 shell 类型。",
    "--bin-dir": "指定 completion 安装目录。",
    "--update-rc": "更新 shell rc 文件；可用 --no-update-rc 关闭。",
    "--days": "对象保留天数。",
    "--rule-id": "OSS 生命周期规则 ID。",
    "--json": "输出机器可读 JSON。",
    "--plain": "输出稳定的制表符分隔文本。",
    "--kind": "选择产物类型，例如 auto、plain、named、corrected、srt。",
    "--help": "显示帮助并退出。",
}
OPTION_ZH_BY_COMMAND = {
    ("project", "transcribe", "--model"): "指定 DashScope ASR 模型。",
    ("project", "summarize", "--model"): "指定回忆索引使用的 DashScope 模型。",
    ("project", "run", "--model"): "指定 DashScope ASR 模型。",
    ("project", "correct", "edit", "--model"): "指定 DashScope 纠错模型。",
    ("lexicon", "delete", "--permanent"): "物理删除词条及上下文。",
    ("lexicon", "delete", "--yes"): "跳过删除确认。",
    ("lexicon", "hotwords", "clear-cache", "--target-model"): "指定要清缓存的 ASR 目标模型。",
    ("lexicon", "hotwords", "remote-delete", "--target-model"): "指定清缓存使用的 ASR 目标模型。",
    ("lexicon", "hotwords", "remote-delete", "--yes"): "确认删除远端词表。",
}
OPTION_EN = {}


def render_help(command: click.Command, command_path: tuple[str, ...]) -> None:
    """
    Render localized help for a Click command with Rich panels.

    Args:
        command: Click command generated by Typer.
        command_path: Nested command path after ``meeting-asr help``.

    Returns:
        None.
    """
    lang = current_cli_language()
    ctx = click.Context(command, info_name=" ".join(("meeting-asr", *command_path)))
    console = cli_console(width=120)
    console.print(Padding(_usage_text(command, ctx, lang), 1))
    body = _help_body(command, command_path, lang)
    if body:
        console.print(Padding(body, (0, 1, 1, 1)))
    _print_examples_panel(command_path, lang, console)
    _print_arguments_panel(command, lang, console)
    _print_options_panel(command, command_path, lang, console)
    _print_commands_panel(command, command_path, lang, console)


def _usage_text(command: click.Command, ctx: click.Context, lang: str) -> Text:
    """Build the localized usage line."""
    pieces = " ".join(command.collect_usage_pieces(ctx))
    suffix = f" {pieces}" if pieces else ""
    text = Text(f"{LABELS[lang]['usage']}: ", style="bold")
    text.append(f"{ctx.info_name}{suffix}", style="bold")
    return text


def _help_body(command: click.Command, command_path: tuple[str, ...], lang: str) -> Text:
    """Build the localized command help body."""
    lines = [_command_description(command, command_path, lang)]
    if command_path == ():
        lines.extend(_root_help_lines(lang))
    return Text("\n".join(line for line in lines if line).strip())


def _root_help_lines(lang: str) -> list[str]:
    """Build root quick-start help lines."""
    lines: list[str] = []
    for title, commands in ROOT_SECTIONS[lang]:
        lines.extend(["", f"{title}:"])
        lines.extend(f"  {command}" for command in commands)
    return lines


def _print_arguments_panel(command: click.Command, lang: str, console: Console) -> None:
    """Print positional arguments when a command has them."""
    rows = [param for param in command.params if isinstance(param, click.Argument)]
    if not rows:
        return
    table = _table()
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="green", no_wrap=True)
    for argument in rows:
        table.add_row(argument.human_readable_name, _argument_required_text(argument, lang))
    console.print(Panel(table, title=LABELS[lang]["arguments"], border_style="dim"))


def _print_examples_panel(command_path: tuple[str, ...], lang: str, console: Console) -> None:
    """Print common invocations when examples are defined for a command."""
    examples = EXAMPLES.get(command_path, {}).get(lang)
    if not examples:
        return
    body = Text("\n".join(f"  {example}" for example in examples))
    console.print(Panel(body, title=LABELS[lang]["examples"], border_style="blue"))


def _print_options_panel(command: click.Command, command_path: tuple[str, ...], lang: str, console: Console) -> None:
    """Print option rows in a scan-friendly Rich panel."""
    options = [param for param in command.params if isinstance(param, click.Option) and not param.hidden]
    if not any("--help" in option.opts for option in options):
        options.append(_implicit_help_option(lang))
    if not options:
        return
    table = _table()
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="magenta", no_wrap=True)
    table.add_column(ratio=1)
    for option in options:
        table.add_row(_option_names(option), _option_metavar(option), _option_help(option, command_path, lang))
    console.print(Panel(table, title=LABELS[lang]["options"], border_style="dim"))


def _print_commands_panel(command: click.Command, command_path: tuple[str, ...], lang: str, console: Console) -> None:
    """Print child commands in a scan-friendly Rich panel."""
    if not isinstance(command, click.Group):
        return
    commands = [(name, child) for name, child in command.commands.items() if not child.hidden]
    if not commands:
        return
    table = _table()
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(ratio=1)
    for name, child in commands:
        table.add_row(name, _command_description(child, (*command_path, name), lang))
    console.print(Panel(table, title=LABELS[lang]["commands"], border_style="dim"))


def _table() -> Table:
    """Create a compact help table matching Typer's default visual density."""
    return Table(show_header=False, expand=True, box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 1))


def _implicit_help_option(lang: str) -> click.Option:
    """Build the implicit help option for groups that rely on Click defaults."""
    return click.Option(["--help", "-h"], is_flag=True, help=OPTION_ZH["--help"] if lang == "zh" else "Show this message and exit.")


def _option_names(option: click.Option) -> str:
    """Return display names for one option."""
    names = [*option.opts, *option.secondary_opts]
    if "--help" in names and "-h" not in names:
        names.append("-h")
    return ", ".join(names)


def _option_metavar(option: click.Option) -> str:
    """Return an option value placeholder when needed."""
    if getattr(option, "is_flag", False):
        return ""
    return option.metavar or option.type.name.upper()


def _option_help(option: click.Option, command_path: tuple[str, ...], lang: str) -> str:
    """Return localized option help."""
    if lang == "zh":
        for name in option.opts:
            path_specific = OPTION_ZH_BY_COMMAND.get((*command_path, name))
            if path_specific:
                return path_specific
            if name in OPTION_ZH:
                return OPTION_ZH[name]
    for name in option.opts:
        if name in OPTION_EN:
            return OPTION_EN[name]
    return option.help or ""


def _command_description(command: click.Command, command_path: tuple[str, ...], lang: str) -> str:
    """Return localized command description."""
    if lang == "zh":
        return COMMAND_ZH.get(command_path) or command.short_help or command.help or ""
    if command_path == ():
        return ROOT_INTRO["en"]
    return command.short_help or command.help or ""


def _argument_required_text(argument: click.Argument, lang: str) -> str:
    """Return a short required or optional label for one argument."""
    if argument.required:
        return "必填" if lang == "zh" else "required"
    return "可选" if lang == "zh" else "optional"
