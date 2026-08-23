"""What `trajectory_board`'s per-team headline actually sums.

The block displays the best `per_team` players and headlines a total. Those must
describe the same set: VAR is unclamped as of #331, so a whole-roster sum is dominated
by players nobody would keep.
"""

from __future__ import annotations

import importlib.util
import pathlib

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.roster_join import index_rosters

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
        "pool": "hitter",
        "team": team,
        # Ownership is MEMBERSHIP: a key rostered by two teams belongs to both, and
        # `block()` filters on this rather than on the single `team` above.
        "teams": frozenset({team}),
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


def test_the_team_headline_sums_the_keepers_not_the_whole_roster(capsys) -> None:
    """The header totals what a team may KEEP -- three -- not what the block lists.

    Measured on the live pool at the 2027-29 range: 93.5% of scored players carry a
    negative VAR, and each roster's tail runs -62 to -196 against a best-5 signal of
    15 to 73. Summing everything ranked Boston Estrellas last in the league on the
    strength of its tail while its best five were 4th -- a six-place error on the one
    comparison the block exists to support.

    It used to sum the best `per_team` (five). That is the scouting POOL, not the
    keeper set, and it is a DIFFERENT ORDERING rather than a noisier one: on
    2026-08-22 the best-5 leader trailed on best-3. The web view moved to the keep
    count, so this surface summing five meant one board ranked teams two ways
    depending on which of them you happened to read.
    """
    module = _script()
    # Same five keepable players; DEEP differs only in fringe nobody would hold.
    scored = [_row(f"Good {i}", 12.0, "DEEP") for i in range(5)]
    scored += [_row(f"Fringe {i}", -20.0, "DEEP") for i in range(15)]
    scored += [_row(f"Fine {i}", 12.0, "THIN") for i in range(5)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(
        scored,
        spots,
        index_rosters(scored, spots, "THIN"),
        "THIN",
        5,
        2026,
        (1, 2, 3),
        None,
        keep=3,
    )
    out = capsys.readouterr().out

    assert _headline(out, "DEEP") == 36.0, "opposing team headlined more than it may keep"
    assert _headline(out, "THIN") == 36.0, "your own team headlined more than it may keep"


def test_your_own_team_caps_the_headline_while_still_listing_everyone(capsys) -> None:
    """Your block is called with `limit=None` so it can list every player you own.

    The cap belongs to the headline, not the list: you still want to see your whole
    roster, and the number beside it still has to be the best THREE so it sits on the
    same scale as every opponent's. Asserted together because the obvious wrong fix --
    summing the DISPLAYED rows -- passes the headline half and silently truncates your
    own list.
    """
    module = _script()
    scored = [_row(f"Mine {i}", 10.0, "MINE") for i in range(5)]
    scored += [_row(f"Mine tail {i}", -30.0, "MINE") for i in range(9)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(
        scored,
        spots,
        index_rosters(scored, spots, "MINE"),
        "MINE",
        5,
        2026,
        (1, 2, 3),
        None,
        keep=3,
    )
    out = capsys.readouterr().out

    assert _headline(out, "MINE") == 30.0, "your headline summed past the keeper cut"
    listed = [ln for ln in out.splitlines() if "Mine" in ln and ln.startswith("  #")]
    assert len(listed) == 14, f"your own block stopped listing every player: {len(listed)}"


def test_the_headline_says_how_many_players_it_counted(capsys) -> None:
    """A bare "total VAR" beside a five-player list reads as the team's whole value.

    Only three players can be kept, so the header names three -- and says they are
    the ones a team MAY KEEP, which is what makes the number comparable across teams
    and distinguishes it from the longer list printed underneath.
    """
    module = _script()
    scored = [_row(f"P{i}", 10.0, "T") for i in range(8)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(
        scored,
        spots,
        index_rosters(scored, spots, "OTHER"),
        "OTHER",
        5,
        2026,
        (1, 2, 3),
        None,
        keep=3,
    )
    out = capsys.readouterr().out

    header = next(ln for ln in out.splitlines() if ln.startswith("T"))
    assert "best 3 they may keep" in header, f"header does not name the counted set: {header!r}"


def test_a_team_with_fewer_rows_than_the_rule_names_what_it_summed(capsys) -> None:
    """A "best 3" label over a two-player sum states a number the rows disprove."""
    module = _script()
    scored = [_row(f"P{i}", 10.0, "T") for i in range(2)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(
        scored,
        spots,
        index_rosters(scored, spots, "OTHER"),
        "OTHER",
        5,
        2026,
        (1, 2, 3),
        None,
        keep=3,
    )
    out = capsys.readouterr().out

    header = next(ln for ln in out.splitlines() if ln.startswith("T"))
    assert "best 2 they may keep" in header, header


def test_opposing_blocks_are_ordered_by_keeper_strength_not_alphabetically(capsys) -> None:
    """The per-team view exists to COMPARE teams; a list sorted by name buries that.

    It also has to agree with the web view, which orders on the same keeper sum.
    """
    module = _script()
    scored = [_row(f"A{i}", 1.0, "Aardvark") for i in range(3)]
    scored += [_row(f"Z{i}", 30.0, "Zebra") for i in range(3)]
    spots = [_spot(r["name"], r["team"]) for r in scored]

    module.by_team(
        scored,
        spots,
        index_rosters(scored, spots, "MINE"),
        "MINE",
        5,
        2026,
        (1, 2, 3),
        None,
        keep=3,
    )
    out = capsys.readouterr().out

    assert out.index("Zebra") < out.index("Aardvark")


def test_a_rostered_player_the_model_cannot_price_is_named(capsys) -> None:
    """The output `assign_teams` used to produce, driven end to end.

    All three tests above hand `by_team` an empty `missing` dict, so the map
    `assign_teams` returned -- the one thing being deleted and re-homed -- had
    ZERO coverage. Without this test the CLI could stop reporting unscored
    players entirely and the suite would stay green.
    """
    module = _script()
    scored = [_row("Scored Guy", 12.0, "T")]
    spots = [_spot("Scored Guy", "T"), _spot("Bench Rookie", "T")]

    module.by_team(
        scored, spots, index_rosters(scored, spots, "T"), "OTHER", 5, 2026, (1, 2, 3), None
    )
    out = capsys.readouterr().out

    assert "not scored: Bench Rookie" in out
    assert "Scored Guy" in out, "the scored player still renders"
