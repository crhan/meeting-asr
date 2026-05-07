"""Localized text for the unified voiceprint review TUI."""

from __future__ import annotations

from app.presentation.tui.i18n import tr


def status_text() -> str:
    """Return the localized voiceprint review status line."""
    return tr(
        (
            "Voiceprint: Tab switch Project/Global | p project | g global | h/l columns | "
            "j/k rows | Space play/stop | x include/exclude | d exclude speaker | s save selected | e evaluate | ? help | q back/quit"
        ),
        (
            "声纹：Tab 切项目/全局 | p 项目 | g 全局 | h/l 切列 | j/k 移动 | "
            "Space 播放/停止 | x 选中/排除 | d 取消当前人全部样本 | s 保存已选 | e 评测 | ? 帮助 | q 返回/退出"
        ),
    )


def help_text() -> str:
    """Return localized voiceprint review shortcut help."""
    return tr(
        """\
[b]Voiceprint Review Shortcuts[/b]

[b]Views[/b]
Project candidates   Clips planned from the current project before they enter the global library
Global library       Stored WAV samples grouped by stable person id

[b]Navigation[/b]
tab                  Switch Project candidates / Global library
p / g                Jump to Project / Global library
h/l or left/right    Switch focused column
j/k or up/down       Move within focused column
PageUp/PageDown      Previous/next sample page
[ / ]                Previous/next sample page

[b]Project Actions[/b]
space                Play or stop selected source-media sample
x                    Include/exclude selected planned sample
a                    Include/exclude all planned samples for the selected speaker
d                    Exclude all planned samples for the selected speaker
s                    Save checked samples; from Project Review it also embeds and evaluates score impact
e                    Re-run voiceprint evaluation without adding new samples when opened from Project Review

[b]Library Actions[/b]
space                Play or stop selected stored WAV sample

[b]Exit[/b]
q / Esc              Return to caller without writing new samples
?                    Show or close this help
""",
        """\
[b]Voiceprint Review 快捷键[/b]

[b]视图[/b]
项目候选样本       当前项目中计划采集、尚未进入全局声纹库的片段
全局声纹库         按稳定人员 ID 分组保存的 WAV 样本

[b]导航[/b]
tab                  切换项目候选样本/全局声纹库
p / g                跳到项目/全局声纹库
h/l 或 ←/→           切换当前列
j/k 或 ↑/↓           在当前列内移动
PageUp/PageDown      上一页/下一页 sample
[ / ]                上一页/下一页 sample

[b]项目操作[/b]
space                播放或停止当前源媒体 sample
x                    选中/排除当前计划样本
a                    选中/排除当前 speaker 的全部计划样本
d                    取消当前 speaker 的全部计划样本
s                    保存勾选样本；从 Project Review 进入时会同时生成 embedding 并评测分数影响
e                    从 Project Review 进入时，不新增样本，只重新运行声纹评测

[b]全局声纹库操作[/b]
space                播放或停止当前已保存 WAV 样本

[b]退出[/b]
q / Esc              返回调用方，不写入新样本
?                    显示或关闭帮助
""",
    )
