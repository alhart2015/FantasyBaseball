"""Terminal colour, and width arithmetic that survives it.

Extracted from `streaks.reports.sunday`, which owned private copies of `visible_width`
and `pad`. A second caller (`scripts/player_trajectory.py`, painting years a comp spent
out of the league) made the duplication real, and width-under-escapes is exactly the
kind of thing that gets subtly re-derived wrong: an escape sequence has no width, so
padding on `len()` shifts every coloured row left of every plain one.

ASCII only, per the repo rule -- an escape sequence is `ESC [ ... m`, all of it in the
low range, so this stays cp1252-safe on Windows.
"""

from __future__ import annotations

from collections.abc import Sequence

GREEN = "\033[32m"
RED = "\033[31m"
#: Faint grey, for a value that is present and real but means something categorically
#: different from its neighbours -- currently "this comp was not in the league", which
#: on the VAR scale prints as an ordinary-looking negative.
DIM_GRAY = "\033[2;90m"
RESET = "\033[0m"


def paint(text: str, color: str, *, enabled: bool = True) -> str:
    """Wrap `text` in `color`, or return it untouched when colour is off.

    The `enabled` flag lives here rather than at each call site so that "am I a TTY"
    is asked once and answered the same way everywhere.
    """
    return f"{color}{text}{RESET}" if enabled else text


def visible_width(s: str) -> int:
    """Width of `s` as a reader sees it, ignoring ANSI escape sequences."""
    visible = []
    i = 0
    while i < len(s):
        if s[i] == "\033":
            # Skip to the `m` that closes the escape.
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        visible.append(s[i])
        i += 1
    return len("".join(visible))


def pad(cell: str, width: int, *, right: bool = False) -> str:
    """Pad `cell` to `width` VISIBLE characters."""
    fill = " " * max(0, width - visible_width(cell))
    return fill + cell if right else cell + fill


def column_widths(rows: Sequence[Sequence[str]]) -> list[int]:
    """Widest visible cell in each column."""
    widths: list[int] = []
    for row in rows:
        for c_idx, cell in enumerate(row):
            w = visible_width(cell)
            if c_idx >= len(widths):
                widths.append(w)
            elif w > widths[c_idx]:
                widths[c_idx] = w
    return widths
