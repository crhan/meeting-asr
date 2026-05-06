"""Static help text for the speaker review TUI."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Static

from app.presentation.tui.i18n import tr


def browse_status() -> str:
    """Return the localized browse-mode status line."""
    return tr(
        (
            "Browse: h/l or left/right choose column | j/k or up/down move | "
            "PgUp/PgDn page samples | Space play/stop | p project | / name | e edit text | "
            "v capture | m rematch | b embed | ? help | s save"
        ),
        (
            "浏览：h/l 或 ←/→ 切列 | j/k 或 ↑/↓ 移动 | PgUp/PgDn 翻页 | "
            "Space 播放/停止 | p 切项目 | / 改人名 | e 改文字 | v 声纹采样 | m 重新匹配 | b 生成 embedding | ? 帮助 | s 保存"
        ),
    )


def edit_status() -> str:
    """Return the localized identity-modal status line."""
    return tr(
        "Identity modal: type to filter people | Up/Down select | Enter choose | +Name create | Esc cancel",
        "身份选择：输入过滤人员 | ↑/↓ 选择 | Enter 确认 | +名字 新建 | Esc 取消",
    )


def shortcut_help() -> str:
    """Return localized speaker-review shortcut help."""
    return tr(
        """\
[b]Speaker Review Shortcuts[/b]

[b]Top status[/b]
Output               Final project files written by Save
Next/Done            Next command, or final preview/read commands
Steps 1 Match        Whether voiceprint matching has been run
Steps 2 Names        Saved speaker_map progress, named speakers, ignored speakers
Steps 3 Capture      Named speakers still missing voiceprint clips
Steps 4 Embed        Captured clips still missing embeddings
Auto                 Automatic match counts and score quality
Check                Conflicts, mismatches, and selected speaker state

[b]Navigation[/b]
h/l or left/right    Switch focused column
j/k or up/down       Move within focused column
PageUp/PageDown      Previous/next sample page
[ / ]                Previous/next sample page

[b]Actions[/b]
space                Play or stop selected sample
a                    Accept current voiceprint match
i                    Ignore this speaker: keep anonymous and skip capture
/                    Open identity modal
e                    Edit selected transcript text inside this TUI
c                    Same as e
v                    Open voiceprint review: project candidates and global library
m                    Rematch speakers against the current global voiceprint library
b                    Embed captured voiceprint samples
p                    Switch to another project from project history
s                    Save speaker mapping, then run staged text correction if present
q                    Quit without saving

[b]Transcript edit[/b]
Enter                Stage the edited sentence and show feedback
s                    Save review state and run the full-document correction proposal
Esc                  Cancel sentence edit

[b]Name edit[/b]
Type                 Filter stable voiceprint people in the modal
Up/Down              Move highlighted person
Enter                Select highlighted/exact person, or create when input starts with +
Tab                  Use highlighted suggestion
Esc                  Cancel edit

[dim]Press Esc, q, or ? to close this help.[/]
""",
        """\
[b]Speaker Review 快捷键[/b]

[b]顶部状态[/b]
Output               保存后写出的最终项目文件
Next/Done            下一步动作，或最终预览/查看命令
Steps 1 Match        是否已完成声纹匹配
Steps 2 Names        speaker_map 保存进度、已命名和已忽略 speaker
Steps 3 Capture      已命名但还缺声纹片段的 speaker
Steps 4 Embed        已采集但还缺 embedding 的声纹片段
Auto                 自动匹配数量和分数质量
Check                冲突、不一致和当前 speaker 状态

[b]导航[/b]
h/l 或 ←/→           切换当前列
j/k 或 ↑/↓           在当前列内移动
PageUp/PageDown      上一页/下一页 sample
[ / ]                上一页/下一页 sample

[b]操作[/b]
space                播放或停止当前 sample
a                    接受当前声纹匹配
i                    忽略当前 speaker：保持匿名，并跳过声纹采样
/                    打开身份选择弹窗
e                    在 TUI 内编辑当前转写句子
c                    等同于 e
v                    打开声纹 Review：项目候选样本和全局声纹库
m                    使用当前全局声纹库重新匹配 speaker
b                    为已采集声纹样本生成 embedding
p                    从历史项目中切换项目
s                    保存 speaker 映射；如有文字修正则继续生成全篇修正建议
q                    不保存退出

[b]文字编辑[/b]
Enter                暂存当前句子修改并展示反馈
s                    保存 review 状态并生成全篇修正建议
Esc                  取消句子编辑

[b]人名编辑[/b]
输入                 过滤全局声纹人员
↑/↓                  移动高亮人员
Enter                选择高亮/精确匹配人员，或在 + 开头时新建
Tab                  使用高亮建议
Esc                  取消编辑

[dim]按 Esc、q 或 ? 关闭帮助。[/]
""",
    )


class ShortcutHelpScreen(ModalScreen[None]):
    """Modal shortcut help for the speaker review TUI."""

    CSS = """
    ShortcutHelpScreen {
        align: center middle;
    }
    #shortcut-help {
        width: 76;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close_help", "Close", show=False),
        Binding("q", "close_help", "Close"),
        Binding("?", "close_help", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Build the help popup."""
        yield Static(shortcut_help(), id="shortcut-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)
