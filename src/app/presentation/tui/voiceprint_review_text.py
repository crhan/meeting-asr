"""Localized text for the unified voiceprint review TUI."""

from __future__ import annotations

from app.presentation.tui.i18n import tr


def status_text() -> str:
    """Return the localized voiceprint review status line."""
    return tr(
        (
            "Voiceprint: Tab switch Project/Global/Quality | p project | g global | y quality | h/l columns | "
            "j/k rows | Space play/stop | x include/quarantine | r reassign/quarantine | v verify | s save/refresh | u refresh quality | e evaluate | ? help | q back/quit"
        ),
        (
            "声纹：Tab 切项目/全局/质量 | p 项目 | g 全局 | y 质量 | h/l 切列 | j/k 移动 | "
            "Space 播放/停止 | x 选择/隔离 | r 改归属/隔离 | v 人工确认 | s 保存/刷新 | u 刷新质量 | e 评测 | ? 帮助 | q 返回/退出"
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
Quality review       Stored samples scored against their person's active voiceprint cluster

[b]Navigation[/b]
tab                  Cycle Project candidates / Global library / Quality review
p / g / y            Jump to Project / Global library / Quality review
h/l or left/right    Switch focused column
j/k or up/down       Move within focused column
PageUp/PageDown      Previous/next sample page
[ / ]                Previous/next sample page

[b]Project Actions[/b]
space                Play or stop selected source-media sample
x                    Include/exclude selected planned sample
r                    Reassign selected planned sample to another speaker
a                    Include/exclude all planned samples for the selected speaker
d                    Exclude all planned samples for the selected speaker
s                    Save checked samples; from Project Review it also embeds and evaluates score impact
e                    Re-run voiceprint evaluation without adding new samples when opened from Project Review

[b]Library Actions[/b]
space                Play or stop selected stored WAV sample

[b]Quality Actions[/b]
space                Play or stop selected stored WAV sample
x                    Toggle selected sample active/quarantined
a                    Mark selected sample active
r                    Mark selected sample quarantined
v                    Mark selected sample human-verified active
s                    Save quality changes and refresh scores
u                    Refresh quality scores from SQLite

[b]Exit[/b]
q / Esc              Return to caller without writing new samples
?                    Show or close this help
""",
        """\
[b]Voiceprint Review 快捷键[/b]

[b]视图[/b]
项目候选样本       当前项目中计划采集、尚未进入全局声纹库的片段
全局声纹库         按稳定人员 ID 分组保存的 WAV 样本
质量检查           已保存样本相对其本人 active 声纹簇的离群评分

[b]导航[/b]
tab                  循环切换项目候选样本/全局声纹库/质量检查
p / g / y            跳到项目/全局声纹库/质量检查
h/l 或 ←/→           切换当前列
j/k 或 ↑/↓           在当前列内移动
PageUp/PageDown      上一页/下一页 sample
[ / ]                上一页/下一页 sample

[b]项目操作[/b]
space                播放或停止当前源媒体 sample
x                    选中/排除当前计划样本
r                    把当前计划样本改给其他 speaker
a                    选中/排除当前 speaker 的全部计划样本
d                    取消当前 speaker 的全部计划样本
s                    保存勾选样本；从 Project Review 进入时会同时生成 embedding 并评测分数影响
e                    从 Project Review 进入时，不新增样本，只重新运行声纹评测

[b]全局声纹库操作[/b]
space                播放或停止当前已保存 WAV 样本

[b]质量检查操作[/b]
space                播放或停止当前已保存 WAV 样本
x                    在 active/quarantined 之间切换
a                    保留当前样本，参与后续匹配
r                    隔离当前样本，不参与后续匹配
v                    人工确认当前样本，继续参与匹配且不再作为质量风险
s                    保存质量变更并刷新评分
u                    从 SQLite 刷新质量评分

[b]退出[/b]
q / Esc              返回调用方，不写入新样本
?                    显示或关闭帮助
""",
    )


def quality_reason_text(reason: str) -> str:
    """
    Return localized display text for a voiceprint quality reason.

    Args:
        reason: Stable internal quality reason.

    Returns:
        Human-facing localized reason.
    """
    if reason == "statistical outlier":
        return tr("statistical outlier: this sample is far from this person's voiceprint cluster", "统计离群：这段样本和此人的其他声纹样本差异明显")
    if reason == "cluster-consistent":
        return tr("cluster-consistent: this sample matches this person's voiceprint cluster", "声纹一致：这段样本和此人的声纹簇匹配")
    if reason == "human verified active":
        return tr("human verified: keep this sample active despite quality risk", "人工确认：这段样本保留参与匹配，不再作为质量风险")
    if reason == "need at least 3 active samples":
        return tr("need at least 3 active samples for quality scoring", "至少需要 3 个 active 样本才能计算质量评分")
    if reason.startswith("score<"):
        return tr(f"score below threshold ({reason.removeprefix('score<')})", f"分数低于阈值（{reason.removeprefix('score<')}）")
    if reason.startswith("status="):
        return tr(f"sample status is {reason.removeprefix('status=')}", f"样本状态为 {reason.removeprefix('status=')}")
    return reason
