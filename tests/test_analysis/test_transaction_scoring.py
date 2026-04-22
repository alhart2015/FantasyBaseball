"""ΔRoto scoring for transactions.

``score_transaction`` replaces the prior wSGP per-side computation
with a single roto-point delta per transaction. Paired drop+add
txns are scored as one swap with the full ΔRoto attributed to the
drop side, so summing ``delta_roto`` across a team's list gives the
team's net roto impact.
"""

from datetime import date

import pandas as pd

from fantasy_baseball.analysis.transactions import score_transaction
from fantasy_baseball.models.league import League
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.roster import Roster, RosterEntry
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.models.team import Team
from fantasy_baseball.utils.name_utils import normalize_name

SEASON_START = date(2026, 3, 27)
SEASON_END = date(2026, 9, 28)
TXN_TS = "1775059200"  # 2026-03-28
TXN_DATE = date(2026, 3, 28)


def _hitter_row(name, **overrides):
    row = {
        "name": name,
        "player_type": "hitter",
        "pa": 500,
        "ab": 450,
        "h": 120,
        "r": 60,
        "hr": 15,
        "rbi": 60,
        "sb": 10,
        "avg": 0.267,
    }
    row.update(overrides)
    return row


def _pitcher_row(name, **overrides):
    row = {
        "name": name,
        "player_type": "pitcher",
        "ip": 180,
        "w": 12,
        "k": 190,
        "sv": 0,
        "er": 60,
        "bb": 50,
        "h_allowed": 155,
        "era": 3.00,
        "whip": 1.14,
    }
    row.update(overrides)
    return row


def _df(rows):
    df = pd.DataFrame(rows)
    if not df.empty and "name" in df.columns:
        df["_name_norm"] = df["name"].apply(normalize_name)
    return df


def _projected_standings(team_stats):
    """Build a ``ProjectedStandings`` from ``{team_name: stats_dict}``.

    Two-team league is enough to give score_roto a non-degenerate
    ranking.
    """
    return ProjectedStandings.from_json(
        {
            "effective_date": SEASON_START.isoformat(),
            "teams": [
                {"name": name, "team_key": "", "rank": 0, "stats": dict(stats)}
                for name, stats in team_stats.items()
            ],
        }
    )


def _baseline_stats():
    """A reasonable mid-pack projected stat line."""
    return {
        "R": 800,
        "HR": 200,
        "RBI": 800,
        "SB": 80,
        "AVG": 0.260,
        "W": 80,
        "K": 1400,
        "SV": 50,
        "ERA": 3.80,
        "WHIP": 1.20,
    }


def _empty_league():
    return League(season_year=2026, teams=[], standings=[])


def _league_with_roster(team_name, roster_names_positions):
    """Build a league where ``team_name`` has a single roster snapshot.

    ``roster_names_positions`` is ``[(name, [Position,...]), ...]``.
    The snapshot's effective_date precedes TXN_DATE so
    ``roster_as_of(TXN_DATE)`` resolves to it.
    """
    entries = [
        RosterEntry(
            name=name,
            positions=positions,
            selected_position=positions[0],
        )
        for name, positions in roster_names_positions
    ]
    roster = Roster(effective_date=date(2026, 3, 27), entries=entries)
    team = Team(name=team_name, team_key="", rosters=[roster])
    return League(season_year=2026, teams=[team], standings=[])


# ---------------------------------------------------------------------------
# Paired drop+add
# ---------------------------------------------------------------------------


class TestPairedScoring:
    def test_drop_side_carries_full_delta_add_side_is_zero(self):
        """Net ΔRoto of the pair must come from exactly one of the two
        legs — the drop side — so ``sum(delta_roto)`` across a team's
        list gives the team's net roto change.
        """
        hitters = _df(
            [
                _hitter_row("Otto Lopez", r=75, hr=18, rbi=70, sb=20, avg=0.290),
                _hitter_row("Marcus Semien", r=60, hr=12, rbi=55, sb=8, avg=0.240),
            ]
        )
        pitchers = _df([])
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )

        drop_txn = {
            "team": "Team A",
            "type": "drop",
            "timestamp": TXN_TS,
            "transaction_id": "d1",
            "add_name": None,
            "add_positions": None,
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        add_txn = {
            "team": "Team A",
            "type": "add",
            "timestamp": TXN_TS,
            "transaction_id": "a1",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": None,
            "drop_positions": None,
        }

        drop_result = score_transaction(
            _empty_league(),
            drop_txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            partner=add_txn,
            team_sds=None,
        )
        add_result = score_transaction(
            _empty_league(),
            add_txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            partner=drop_txn,
            team_sds=None,
        )
        assert add_result["delta_roto"] == 0.0
        assert drop_result["delta_roto"] > 0  # Otto is strictly better

    def test_cross_type_paired_swap_hitter_for_pitcher(self):
        """A dropped hitter paired with an added pitcher must still
        produce a finite delta — the cross-type swap loses hitter stats
        on the ``loses`` side and gains pitcher stats on the ``gains``
        side, with apply_swap_delta handling the category zeroes.
        """
        hitters = _df([_hitter_row("Hitter Dropped")])
        pitchers = _df([_pitcher_row("Pitcher Added")])
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )

        drop_txn = {
            "team": "Team A",
            "type": "drop",
            "timestamp": TXN_TS,
            "transaction_id": "d1",
            "drop_name": "Hitter Dropped",
            "drop_positions": "OF",
            "add_name": None,
            "add_positions": None,
        }
        add_txn = {
            "team": "Team A",
            "type": "add",
            "timestamp": TXN_TS,
            "transaction_id": "a1",
            "add_name": "Pitcher Added",
            "add_positions": "SP",
            "drop_name": None,
            "drop_positions": None,
        }

        result = score_transaction(
            _empty_league(),
            drop_txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            partner=add_txn,
            team_sds=None,
        )
        assert "delta_roto" in result
        assert isinstance(result["delta_roto"], float)


# ---------------------------------------------------------------------------
# Solo add — worst-at-position counterfactual
# ---------------------------------------------------------------------------


class TestSoloAdd:
    def test_adding_better_than_worst_same_position_is_positive(self):
        hitters = _df(
            [
                _hitter_row("Star Added", r=90, hr=30, rbi=100, sb=20, avg=0.310),
                _hitter_row("Weak Rostered", r=40, hr=5, rbi=30, sb=2, avg=0.220),
            ]
        )
        pitchers = _df([])
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )
        league = _league_with_roster(
            "Team A",
            [("Weak Rostered", [Position.OF])],
        )

        txn = {
            "team": "Team A",
            "type": "add",
            "timestamp": TXN_TS,
            "transaction_id": "a1",
            "add_name": "Star Added",
            "add_positions": "OF",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(
            league,
            txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        assert result["delta_roto"] > 0

    def test_add_unknown_player_returns_zero(self):
        hitters = _df([])
        pitchers = _df([])
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )
        txn = {
            "team": "Team A",
            "type": "add",
            "timestamp": TXN_TS,
            "transaction_id": "a1",
            "add_name": "Nobody",
            "add_positions": "OF",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(
            _empty_league(),
            txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        assert result["delta_roto"] == 0.0


# ---------------------------------------------------------------------------
# Solo drop — replacement-level counterfactual
# ---------------------------------------------------------------------------


class TestSoloDrop:
    def test_dropping_above_replacement_hitter_is_negative(self):
        hitters = _df(
            [
                _hitter_row("Solid Player", r=90, hr=30, rbi=100, sb=20, avg=0.310),
            ]
        )
        pitchers = _df([])
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )

        txn = {
            "team": "Team A",
            "type": "drop",
            "timestamp": TXN_TS,
            "transaction_id": "d1",
            "drop_name": "Solid Player",
            "drop_positions": "OF",
            "add_name": None,
            "add_positions": None,
        }
        result = score_transaction(
            _empty_league(),
            txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        assert result["delta_roto"] < 0

    def test_drop_unknown_player_returns_zero(self):
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )
        txn = {
            "team": "Team A",
            "type": "drop",
            "timestamp": TXN_TS,
            "transaction_id": "d1",
            "drop_name": "Nobody",
            "drop_positions": "OF",
            "add_name": None,
            "add_positions": None,
        }
        result = score_transaction(
            _empty_league(),
            txn,
            standings,
            _df([]),
            _df([]),
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        assert result["delta_roto"] == 0.0


# ---------------------------------------------------------------------------
# Defensive: team not in projected_standings
# ---------------------------------------------------------------------------


class TestMissingTeamStandings:
    def test_unknown_team_returns_zero(self):
        hitters = _df([_hitter_row("Star Added", r=90, hr=30, rbi=100)])
        pitchers = _df([])
        standings = _projected_standings({"Other Team": _baseline_stats()})

        txn = {
            "team": "Ghost Team",
            "type": "add",
            "timestamp": TXN_TS,
            "transaction_id": "a1",
            "add_name": "Star Added",
            "add_positions": "OF",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(
            _empty_league(),
            txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        assert result["delta_roto"] == 0.0


# ---------------------------------------------------------------------------
# Proration sanity
# ---------------------------------------------------------------------------


class TestProration:
    def test_late_season_drop_is_smaller_than_early_drop(self):
        """A solo drop late in the season costs less roto than the same
        drop early — the replacement floor is prorated by fraction
        remaining, and so is the dropped player's ROS (both shrink, but
        the gap shrinks with them).
        """
        hitters = _df(
            [
                _hitter_row("Solid Player", r=90, hr=30, rbi=100, sb=20, avg=0.310),
            ]
        )
        pitchers = _df([])
        standings = _projected_standings(
            {
                "Team A": _baseline_stats(),
                "Team B": _baseline_stats(),
            }
        )

        late_ts = "1788307200"  # 2026-08-30 - near season end
        early_ts = TXN_TS

        late_txn = {
            "team": "Team A",
            "type": "drop",
            "timestamp": late_ts,
            "transaction_id": "d2",
            "drop_name": "Solid Player",
            "drop_positions": "OF",
            "add_name": None,
            "add_positions": None,
        }
        early_txn = {
            "team": "Team A",
            "type": "drop",
            "timestamp": early_ts,
            "transaction_id": "d1",
            "drop_name": "Solid Player",
            "drop_positions": "OF",
            "add_name": None,
            "add_positions": None,
        }

        late = score_transaction(
            _empty_league(),
            late_txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        early = score_transaction(
            _empty_league(),
            early_txn,
            standings,
            hitters,
            pitchers,
            SEASON_START,
            SEASON_END,
            team_sds=None,
        )
        # Both negative; late-season drop is less negative (closer to 0).
        assert late["delta_roto"] >= early["delta_roto"]
