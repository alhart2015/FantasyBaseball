"""What `trajectory_board`'s per-team headline actually sums.

The block displays the best `per_team` players and headlines a total. Those must
describe the same set: VAR is unclamped as of #331, so a whole-roster sum is dominated
by players nobody would keep.
"""

from __future__ import annotations

import importlib.util
import pathlib

from fantasy_baseball.data.rosters import RosterSpot

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "trajectory_board", PROJECT_ROOT / "scripts" / "trajectory_board.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(name: str, total: float, team: str) -> dict:
    return {
        "name": name,
        "team": team,
        "status": "",
        "total": total,
        "next": total / 3,
        "p10": total - 3,
        "p90": total + 3,
        "age": 27,
        "slot": "OF",
        "rank_total": 1,
        "rank_next": 1,
        "band_fell_back": False,
        "extrapolated": False,
    }


def _spot(name: str, team: str) -> RosterSpot:
    return RosterSpot(
        name=name,
        normalized=name.lower(),
        player_type="hitter",
        team=team,
        yahoo_id="0",
        status="",
    )


def _headline(out: str, team: str) -> float:
    """The VAR figure out of one team's header line."""
    line = next(ln for ln in out.splitlines() if ln.startswith(team))
    return float(line.split("scored, ")[1].split(" total")[0])


def test_the_team_headline_sums_per_team_not_the_whole_roster(capsys) -> None:
    """The header must total the same players the block lists.

    Measured on the live pool at the 2027-29 range: 93.5% of scored players carry a
    negative VAR, and each roster's tail runs -62 to -196 against a best-5 signal of
    15 to 73. Summing everything ranked Boston Estrellas last in the league on the
    strength of its tail while its best five were 4th -- a six-place error on the one
    comparison the block exists to support. The tail is not a keeper signal: only three
    players per team can be kept, and no one keeps the 20th.
    """
    module = _script()
    # Same five keepable players; DEEP differs only in fringe nobody would hold.
    scored = [_row(f"Good {i}", 12.0, "DEEP") for i in range(5)]
    scored += [_row(f"Fringe {i}", -20.0, "DEEP") for i in range(15)]
    scored += [_row(f"Fine {i}", 12.0, "THIN") for i in range(5)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(scored, spots, {}, "THIN", 5, 2026, (1, 2, 3), None)
    out = capsys.readouterr().out

    assert _headline(out, "DEEP") == 60.0, "opposing team headlined its whole roster"
    assert _headline(out, "THIN") == 60.0, "your own team headlined its whole roster"


def test_your_own_team_caps_the_headline_while_still_listing_everyone(capsys) -> None:
    """Your block is called with `limit=None` so it can list every player you own.

    The cap belongs to the headline, not the list: you still want to see your whole
    roster, and the number beside it still has to be the best five so it sits on the
    same scale as every opponent's. Asserted together because the obvious wrong fix --
    summing the DISPLAYED rows -- passes the headline half and silently truncates your
    own list to five.
    """
    module = _script()
    scored = [_row(f"Mine {i}", 10.0, "MINE") for i in range(5)]
    scored += [_row(f"Mine tail {i}", -30.0, "MINE") for i in range(9)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(scored, spots, {}, "MINE", 5, 2026, (1, 2, 3), None)
    out = capsys.readouterr().out

    assert _headline(out, "MINE") == 50.0, "your headline summed the tail"
    listed = [ln for ln in out.splitlines() if "Mine" in ln and ln.startswith("  #")]
    assert len(listed) == 14, f"your own block stopped listing every player: {len(listed)}"


def test_the_headline_says_how_many_players_it_counted(capsys) -> None:
    """A bare "total VAR" beside a five-player list reads as the team's whole value.

    Only three players can be kept, so a best-five sum is a scouting aggregate of the
    candidate pool, not a keeper total. The header has to say which it is.
    """
    module = _script()
    scored = [_row(f"P{i}", 10.0, "T") for i in range(8)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(scored, spots, {}, "OTHER", 5, 2026, (1, 2, 3), None)
    out = capsys.readouterr().out

    header = next(ln for ln in out.splitlines() if ln.startswith("T"))
    assert "best 5" in header, f"header does not name the counted set: {header!r}"
