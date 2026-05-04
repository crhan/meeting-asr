"""Static help text for the speaker review TUI."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Static

BROWSE_STATUS = (
    "Browse: h/l or left/right choose column | j/k or up/down move | "
    "PgUp/PgDn page samples | Space play/stop | p project | / name | e edit text | v voiceprint | ? help | s save"
)
EDIT_STATUS = (
    "Identity modal: type to filter people | Up/Down select | Enter choose | "
    "+Name create | Esc cancel"
)
SHORTCUT_HELP = """\
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
"""


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
        yield Static(SHORTCUT_HELP, id="shortcut-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)
