"""How `--show-comps` renders a year the comp was out of the league.

On the VAR scale the frame is shifted, so a career ending prints as `-floor` -- at the
OF floor, -9.96, four hundredths from a real -10.00 season. The number alone cannot
carry the distinction, so the mask has to reach the page.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd

from fantasy_baseball.utils.ansi import DIM_GRAY, RESET, visible_width

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "player_trajectory", PROJECT_ROOT / "scripts" / "player_trajectory.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Four comps at the OF floor. "Gone" left the league and renders -9.96; "Bad" played
#: and was awful, landing at -10.00. Adjacent on the page, opposite in meaning, and
#: the pair every test here turns on.
#:
#: "Wide" exists so the h1 column is wider than a `{v:6.2f}` cell. Without him every
#: numeric cell is already exactly the column width, padding is a no-op, and the
#: alignment test below cannot fail however the padding is computed -- verified by
#: mutating `pad` to count escape bytes and watching it stay green.
COMPS = pd.DataFrame(
    {
        "player": ["Gone", "Bad", "Fine", "Wide"],
        "season": [2019, 2020, 2021, 2022],
        "sgp0": [6.00, 6.01, 5.98, 6.02],
        "h1": [-9.96, -10.00, 2.63, -123.45],
        "h2": [-9.96, 3.10, float("nan"), 5.00],
    }
)
DEPARTED = pd.DataFrame({"h1": [True, False, False, False], "h2": [True, False, False, False]})
COLS = ["player", "season", "sgp0", "h1", "h2"]


def _cells(line: str) -> list[str]:
    return line.split()


def test_a_year_out_of_the_league_is_painted_and_a_bad_season_is_not() -> None:
    """The two rows the reader cannot otherwise tell apart."""
    module = _script()
    lines = module.comp_table_lines(COMPS, COLS, DEPARTED, color=True)
    gone = next(ln for ln in lines if "Gone" in ln)
    bad = next(ln for ln in lines if "Bad" in ln)

    assert f"{DIM_GRAY}" in gone, "a departed year was rendered like every other number"
    assert "-9.96" in gone, "painting must not hide the value"
    assert DIM_GRAY not in bad, (
        "a real -10.00 season was painted as a career ending -- the exact confusion "
        "the mask exists to remove, now pointing the other way"
    )


def test_only_the_departed_cell_is_painted_not_the_whole_row() -> None:
    """Bad's h1 is a real season and his h2 is a real season; Gone has two dead years.

    A row-level paint would mark Gone's `sgp0` -- the season he was matched ON, which he
    plainly played -- and would say nothing about which YEAR the career ended.
    """
    module = _script()
    lines = module.comp_table_lines(COMPS, COLS, DEPARTED, color=True)
    gone = next(ln for ln in lines if "Gone" in ln)

    assert gone.count(DIM_GRAY) == 2, "expected exactly the two dead years painted"
    assert gone.index("Gone") < gone.index(DIM_GRAY), "the name was painted"
    assert f"{DIM_GRAY}   6.00" not in gone, "the matched season was painted"


def test_a_year_not_played_yet_is_not_painted() -> None:
    """`--` already means "has not happened yet" and is a different fact."""
    module = _script()
    lines = module.comp_table_lines(COMPS, COLS, DEPARTED, color=True)
    fine = next(ln for ln in lines if "Fine" in ln)

    assert "--" in fine
    assert DIM_GRAY not in fine


def test_piped_output_carries_no_escapes() -> None:
    """Redirected to a file the table still has to be readable, and diffable."""
    module = _script()
    lines = module.comp_table_lines(COMPS, COLS, DEPARTED, color=False)

    assert not any("\033" in ln for ln in lines), "escape codes leaked into piped output"
    assert any("-9.96" in ln for ln in lines), "the values did not survive"


def test_the_columns_still_line_up_under_the_escapes() -> None:
    """ANSI bytes have no width. Padding on `len()` shifts every painted row left.

    Asserted on VISIBLE width, which is the only width a reader sees.
    """
    module = _script()
    lines = module.comp_table_lines(COMPS, COLS, DEPARTED, color=True)
    widths = {visible_width(ln) for ln in lines}

    # Padding on `len()` would make each painted row 9 characters of escape narrower
    # than the plain ones, so this is exactly the assertion that catches it.
    assert len(widths) == 1, f"rows ended at different columns: {sorted(widths)}"
    assert all(ln.endswith(RESET) or "\033" not in ln for ln in lines), "an escape was left open"


def test_piped_output_still_marks_the_departed_years() -> None:
    """Colour is the only signal on a TTY; redirected, there has to be another.

    The legend this replaced said "0 = did not play" while the frame printed -9.96 --
    a legend describing something the page did not show. A legend saying "faint" into
    a file with no faint in it is the same defect facing the other way.
    """
    module = _script()
    lines = module.comp_table_lines(COMPS, COLS, DEPARTED, color=False)
    gone = next(ln for ln in lines if "Gone" in ln)
    bad = next(ln for ln in lines if "Bad" in ln)

    assert "-9.96*" in gone, "no way to tell a career ending from a bad year in a file"
    assert "-10.00*" not in bad, "a real season was marked as a departure"


def test_the_legend_names_whatever_the_table_actually_shows() -> None:
    """Whichever marker is in use, the header has to be the one describing it."""
    module = _script()
    assert module.comps_legend(scale="var", color=True).startswith("faint")
    assert "*" in module.comps_legend(scale="var", color=False)
    # On the raw scale nothing is shifted, so the original 0.0 is still on the page.
    assert "0 = did not play" in module.comps_legend(scale="sgp", color=True)
