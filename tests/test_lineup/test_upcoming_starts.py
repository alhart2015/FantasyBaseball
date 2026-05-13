"""Unit tests for the rotation anchor + projection logic."""

from datetime import date as _date

import pandas as pd

from fantasy_baseball.lineup.upcoming_starts import (
    GameSlot,
    MatchupContext,
    StartEntry,
    build_team_game_index,
    compose_pitcher_entries,
    filter_starting_pitchers,
    find_anchor_index,
    project_start_indices,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position


def _build_ctx(team_stats: dict[str, dict[str, float]]) -> MatchupContext:
    """Build a MatchupContext from raw team stats.

    Mirrors what ``matchups.get_probable_starters`` does in production:
    park-neutralizes each team's season OPS/K% by their home park factor,
    builds the sorted league baseline, and computes raw season ranks
    for tooltip display.
    """
    from fantasy_baseball.data.park_factors import get_park_factor, park_neutral_value

    neutral_ops = {
        abbrev: park_neutral_value(s["ops"], get_park_factor(abbrev, "ops"))
        for abbrev, s in team_stats.items()
    }
    neutral_k = {
        abbrev: park_neutral_value(s["k_pct"], get_park_factor(abbrev, "k"))
        for abbrev, s in team_stats.items()
    }
    ops_ranked = sorted(team_stats.items(), key=lambda x: x[1]["ops"], reverse=True)
    k_ranked = sorted(team_stats.items(), key=lambda x: x[1]["k_pct"])
    return MatchupContext(
        team_stats=team_stats,
        neutral_ops=neutral_ops,
        neutral_k_pct=neutral_k,
        neutral_ops_sorted_desc=tuple(sorted(neutral_ops.values(), reverse=True)),
        neutral_k_pct_sorted_asc=tuple(sorted(neutral_k.values())),
        ops_rank_map={a: i + 1 for i, (a, _) in enumerate(ops_ranked)},
        k_rank_map={a: i + 1 for i, (a, _) in enumerate(k_ranked)},
    )


def _league_avg_stats() -> dict[str, dict[str, float]]:
    """30-team synthetic league with a spread of OPS / K% values.

    Spans ~.640 to .790 OPS (realistic) and ~18% to 27% K% (realistic).
    Includes the real abbreviations so park factor lookups hit the
    actual table, not the 1.00 fallback.
    """
    teams = [
        "LAD",
        "NYY",
        "ATL",
        "HOU",
        "BAL",
        "TEX",
        "BOS",
        "PHI",
        "TOR",
        "ARI",
        "SEA",
        "CHC",
        "MIN",
        "CLE",
        "MIL",
        "DET",
        "STL",
        "SDP",
        "TBR",
        "SFG",
        "NYM",
        "LAA",
        "CIN",
        "KCR",
        "WSN",
        "CHW",
        "PIT",
        "MIA",
        "ATH",
        "COL",
    ]
    out: dict[str, dict[str, float]] = {}
    for i, team in enumerate(teams):
        # OPS descends from .790 to .640 by 0.005 per step.
        # K% climbs from 0.18 to 0.27 by ~0.003 per step (low-K teams are
        # the toughest matchups, so they're at the top of the OPS list
        # too, which is realistic -- good contact hitters tend to score
        # more runs).
        out[team] = {
            "ops": round(0.790 - 0.005 * i, 3),
            "k_pct": round(0.18 + (0.27 - 0.18) * i / 29, 4),
        }
    return out


def test_game_slot_fields():
    slot = GameSlot(
        date="2026-05-05",
        game_number=1,
        opponent="LAD",
        indicator="@",
        announced_starter="Bryan Woo",
    )
    assert slot.date == "2026-05-05"
    assert slot.game_number == 1
    assert slot.opponent == "LAD"
    assert slot.indicator == "@"
    assert slot.announced_starter == "Bryan Woo"


def test_start_entry_announced_default_false():
    entry = StartEntry(
        date="2026-05-05",
        day="Mon",
        opponent="LAD",
        indicator="@",
    )
    assert entry.announced is False


def test_start_entry_with_detail():
    entry = StartEntry(
        date="2026-05-05",
        day="Mon",
        opponent="LAD",
        indicator="@",
        announced=True,
        matchup_quality="Tough",
        detail={"ops": 0.789, "ops_rank": 4, "k_pct": 22.1, "k_rank": 18},
    )
    assert entry.announced is True
    assert entry.matchup_quality == "Tough"
    assert entry.detail["ops_rank"] == 4


def _pp(date_, away, home, awp="", hwp="", num=1):
    return {
        "date": date_,
        "game_number": num,
        "away_team": away,
        "home_team": home,
        "away_pitcher": awp or "TBD",
        "home_pitcher": hwp or "TBD",
    }


class TestBuildTeamGameIndex:
    def test_filters_to_target_team(self):
        pps = [
            _pp("2026-05-05", "SEA", "LAD", awp="Woo"),
            _pp("2026-05-05", "NYY", "BOS", awp="Cole"),
            _pp("2026-05-06", "TEX", "SEA", hwp="Castillo"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert len(slots) == 2
        assert slots[0].opponent == "LAD"
        assert slots[0].indicator == "@"
        assert slots[0].announced_starter == "Woo"
        assert slots[1].opponent == "TEX"
        assert slots[1].indicator == "vs"
        assert slots[1].announced_starter == "Castillo"

    def test_chronological_ordering(self):
        pps = [
            _pp("2026-05-07", "SEA", "TEX"),
            _pp("2026-05-05", "SEA", "LAD"),
            _pp("2026-05-06", "SEA", "TEX"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert [s.date for s in slots] == ["2026-05-05", "2026-05-06", "2026-05-07"]

    def test_doubleheader_sorts_by_game_number(self):
        pps = [
            _pp("2026-05-05", "SEA", "LAD", num=2, awp="Gilbert"),
            _pp("2026-05-05", "SEA", "LAD", num=1, awp="Woo"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert [s.game_number for s in slots] == [1, 2]
        assert slots[0].announced_starter == "Woo"
        assert slots[1].announced_starter == "Gilbert"

    def test_tbd_announced_starter_becomes_empty(self):
        pps = [_pp("2026-05-05", "SEA", "LAD", awp="TBD")]
        slots = build_team_game_index(pps, "SEA")
        assert slots[0].announced_starter == ""

    def test_empty_when_team_not_in_schedule(self):
        pps = [_pp("2026-05-05", "NYY", "BOS")]
        assert build_team_game_index(pps, "SEA") == []


def _slot(d, opp, ann="", ind="@", num=1):
    return GameSlot(date=d, game_number=num, opponent=opp, indicator=ind, announced_starter=ann)


class TestFindAnchorIndex:
    def test_finds_most_recent_past_start(self):
        games = [
            _slot("2026-05-01", "TEX", ann="Bryan Woo"),
            _slot("2026-05-03", "LAD", ann="Castillo"),
            _slot("2026-05-06", "TEX", ann="Bryan Woo"),
        ]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx == 2  # the May 6 start

    def test_excludes_today_and_future(self):
        games = [
            _slot("2026-05-01", "TEX", ann="Bryan Woo"),
            _slot("2026-05-07", "LAD", ann="Bryan Woo"),  # today — excluded
            _slot("2026-05-08", "LAD", ann="Bryan Woo"),  # future — excluded
        ]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx == 0

    def test_returns_none_when_no_match(self):
        games = [_slot("2026-05-01", "TEX", ann="Castillo")]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx is None

    def test_returns_none_when_pitcher_has_only_future_starts(self):
        games = [_slot("2026-05-08", "TEX", ann="Bryan Woo")]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx is None

    def test_name_match_is_accent_insensitive(self):
        # normalize_name strips accents, so "José Berríos" and "Jose Berrios" match.
        games = [_slot("2026-05-01", "TEX", ann="José Berríos")]
        idx = find_anchor_index(games, "Jose Berrios", today=_date(2026, 5, 7))
        assert idx == 0


class TestProjectStartIndices:
    def test_simple_rotation_one_projection(self):
        # 10 games total, anchor at index 2 -> projections at 7
        assert project_start_indices(anchor_index=2, total_games=10, step=5) == [7]

    def test_two_projections_within_window(self):
        # anchor 0, total 12 -> 5, 10
        assert project_start_indices(anchor_index=0, total_games=12, step=5) == [5, 10]

    def test_no_projection_when_anchor_at_end(self):
        assert project_start_indices(anchor_index=7, total_games=10, step=5) == []

    def test_anchor_index_negative_returns_empty(self):
        assert project_start_indices(anchor_index=-1, total_games=10, step=5) == []

    def test_step_other_than_five(self):
        # 6-man rotation = step 6
        assert project_start_indices(anchor_index=0, total_games=20, step=6) == [6, 12, 18]


def _seq(*specs):
    """Build a team game index from compact specs: (date, opp, ann, indicator, num)."""
    out = []
    for spec in specs:
        d, opp, ann = spec[:3]
        ind = spec[3] if len(spec) > 3 else "@"
        num = spec[4] if len(spec) > 4 else 1
        out.append(
            GameSlot(date=d, game_number=num, opponent=opp, indicator=ind, announced_starter=ann)
        )
    return out


_LEAGUE = _league_avg_stats()
_CTX = _build_ctx(_LEAGUE)


class TestComposePitcherEntries:
    def test_simple_5_day_rotation_no_off_day(self):
        # Mon..Sun. Anchor: pitcher started Mon -> projected Sat.
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # past anchor (yesterday)
            ("2026-05-05", "TEX", "", "@"),
            ("2026-05-06", "TEX", "", "@"),
            ("2026-05-07", "TEX", "", "@"),
            ("2026-05-08", "TEX", "", "@"),
            ("2026-05-09", "TEX", "", "@"),  # +5 -> projected
            ("2026-05-10", "TEX", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(entries) == 1
        e = entries[0]
        assert e.date == "2026-05-09"
        assert e.day == "Sat"
        assert e.opponent == "TEX"
        assert e.announced is False
        # TEX sits at OPS index 5 in the synthetic league (top-6 offense)
        # and the venue is TEX's hitter-friendly park -- should color Tough.
        assert e.matchup_quality == "Tough"
        # Raw season rank is still preserved for tooltip context.
        assert e.detail["ops_rank"] == 6
        assert e.detail["venue"] == "TEX"

    def test_off_day_extends_calendar_gap(self):
        # Anchor Mon (idx 0). Team games after anchor: Tue, Wed, [off Thu],
        # Fri, Sat, Sun. Sun is the 5th team-game post-anchor (idx 5),
        # so the projected start lands on Sun = 2026-05-10. Compare with
        # test_simple_5_day_rotation_no_off_day: same anchor, but no off-day
        # so the 5th team-game is Sat = 2026-05-09. The off-day pushes the
        # next start one calendar day later -- that's the gap "extension."
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor (idx 0)
            ("2026-05-05", "TEX", "", "@"),  # idx 1
            ("2026-05-06", "TEX", "", "@"),  # idx 2
            # No 2026-05-07 entry -- off day
            ("2026-05-08", "TEX", "", "@"),  # idx 3
            ("2026-05-09", "TEX", "", "@"),  # idx 4
            ("2026-05-10", "TEX", "", "@"),  # idx 5 -- 5th team-game post-anchor
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(entries) == 1
        assert entries[0].date == "2026-05-10"

    def test_announced_start_takes_precedence_over_projection(self):
        # Anchor Mon, MLB announces same pitcher Sat. Result: 1 entry, announced=True.
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor
            *[("2026-05-0" + str(d), "TEX", "", "@") for d in range(5, 9)],
            ("2026-05-09", "TEX", "Bryan Woo", "@"),  # announced same date as projection
            ("2026-05-10", "TEX", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(entries) == 1
        assert entries[0].date == "2026-05-09"
        assert entries[0].announced is True

    def test_announced_other_pitcher_drops_projection(self):
        # Projection lands on a game where MLB has someone else announced.
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor
            *[("2026-05-0" + str(d), "TEX", "", "@") for d in range(5, 9)],
            ("2026-05-09", "TEX", "Castillo", "@"),  # someone else announced
            ("2026-05-10", "TEX", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert entries == []

    def test_two_start_week_from_old_anchor(self):
        # Build a 21-game team stream where the anchor sits at index 0
        # (2026-04-25, ten days before the window opens). Anchor + 5/10/15/20
        # land at indices 5, 10, 15, 20 -- i.e. dates 04-30, 05-05, 05-10, 05-15.
        # Window is 2026-05-05..2026-05-11, so projections at 05-05 and 05-10
        # both fall inside it; 04-30 (before) and 05-15 (after) are excluded.
        from datetime import timedelta as _td

        base = _date(2026, 4, 25)
        team_games: list[GameSlot] = []
        for i in range(21):
            d = (base + _td(days=i)).isoformat()
            ann = "Bryan Woo" if i == 0 else ""
            opp = "TEX" if i < 14 else "LAD"
            team_games.append(
                GameSlot(date=d, game_number=1, opponent=opp, indicator="@", announced_starter=ann)
            )

        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(entries) == 2
        assert entries[0].date == "2026-05-05"
        assert entries[1].date == "2026-05-10"
        assert all(e.announced is False for e in entries)

    def test_no_anchor_yields_only_announced(self):
        # No past start by this pitcher; MLB announces them mid-week.
        team_games = _seq(
            ("2026-05-04", "TEX", "OtherGuy", "@"),
            *[("2026-05-0" + str(d), "TEX", "", "@") for d in range(5, 9)],
            ("2026-05-09", "TEX", "Bryan Woo", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(entries) == 1
        assert entries[0].announced is True

    def test_empty_when_pitcher_has_no_starts(self):
        team_games = _seq(("2026-05-05", "TEX", "OtherGuy", "@"))
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert entries == []


class TestParkAdjustment:
    """Verify the park-adjusted ranking behaves correctly across venues."""

    def test_coors_makes_a_weak_offense_tougher(self):
        # COL in the synthetic league sits at the bottom of the OPS table
        # (a "weak" offense). But at Coors (the strongest hitter park),
        # the effective OPS jumps -- the start should color worse than
        # the same opponent played in a neutral park.
        weak_team_at_coors = _seq(
            ("2026-05-04", "COL", "Bryan Woo", "@"),
            *[("2026-05-0" + str(d), "COL", "", "@") for d in range(5, 9)],
            ("2026-05-09", "COL", "", "@"),  # projected start at Coors
            ("2026-05-10", "COL", "", "@"),
        )
        at_coors = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",  # pitcher's home park irrelevant for an @ game
            weak_team_at_coors,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        # Same opponent, but hosted at a neutral park (use indicator "vs"
        # with a pitcher whose home is neutral). LAA is ops factor 1.00.
        weak_team_at_neutral = _seq(
            ("2026-05-04", "COL", "Bryan Woo", "vs"),
            *[("2026-05-0" + str(d), "COL", "", "vs") for d in range(5, 9)],
            ("2026-05-09", "COL", "", "vs"),
            ("2026-05-10", "COL", "", "vs"),
        )
        at_neutral = compose_pitcher_entries(
            "Bryan Woo",
            "LAA",  # neutral home park
            weak_team_at_neutral,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(at_coors) == 1
        assert len(at_neutral) == 1
        # Park-adjusted rank at Coors must be tougher (lower rank number).
        assert at_coors[0].detail["effective_ops_rank"] < at_neutral[0].detail["effective_ops_rank"]

    def test_petco_eases_a_strong_offense(self):
        # LAD is the league's #1 OPS team in the synthetic distribution.
        # At Petco (the strongest pitcher park), they hit much less --
        # park-adjusted rank should be easier (higher number).
        strong_team_at_petco = _seq(
            ("2026-05-04", "LAD", "Bryan Woo", "@"),
            *[("2026-05-0" + str(d), "LAD", "", "@") for d in range(5, 9)],
            ("2026-05-09", "LAD", "", "@"),
            ("2026-05-10", "LAD", "", "@"),
        )
        # When LAD is the "opponent" and the pitcher's home park is Petco
        # (SDP), an @ LAD game has venue=LAD. We need the reverse: LAD
        # visiting Petco. Use indicator "vs" with our pitcher at SDP.
        strong_team_at_petco = _seq(
            ("2026-05-04", "LAD", "Bryan Woo", "vs"),
            *[("2026-05-0" + str(d), "LAD", "", "vs") for d in range(5, 9)],
            ("2026-05-09", "LAD", "", "vs"),
            ("2026-05-10", "LAD", "", "vs"),
        )
        at_petco = compose_pitcher_entries(
            "Bryan Woo",
            "SDP",  # pitcher home = Petco
            strong_team_at_petco,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        # Compare to LAD at LAD's home park.
        strong_team_at_home = _seq(
            ("2026-05-04", "LAD", "Bryan Woo", "@"),
            *[("2026-05-0" + str(d), "LAD", "", "@") for d in range(5, 9)],
            ("2026-05-09", "LAD", "", "@"),
            ("2026-05-10", "LAD", "", "@"),
        )
        at_home = compose_pitcher_entries(
            "Bryan Woo",
            "SDP",
            strong_team_at_home,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert len(at_petco) == 1
        assert len(at_home) == 1
        # Park-adjusted rank at Petco must be easier (higher rank number).
        assert at_petco[0].detail["effective_ops_rank"] > at_home[0].detail["effective_ops_rank"]

    def test_venue_resolves_to_opponent_for_away_games(self):
        team_games = _seq(
            ("2026-05-04", "COL", "Bryan Woo", "@"),
            *[("2026-05-0" + str(d), "COL", "", "@") for d in range(5, 9)],
            ("2026-05-09", "COL", "", "@"),
            ("2026-05-10", "COL", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert entries[0].detail["venue"] == "COL"

    def test_venue_resolves_to_pitcher_team_for_home_games(self):
        team_games = _seq(
            ("2026-05-04", "COL", "Bryan Woo", "vs"),
            *[("2026-05-0" + str(d), "COL", "", "vs") for d in range(5, 9)],
            ("2026-05-09", "COL", "", "vs"),
            ("2026-05-10", "COL", "", "vs"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            "SEA",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            ctx=_CTX,
        )
        assert entries[0].detail["venue"] == "SEA"

    def test_color_band_distribution_is_roughly_balanced(self):
        """Color bands should split the league into rough thirds when
        opponents are drawn evenly. Direct refutation of the original
        'all reds and yellows' bug: with the new logic, ~1/3 of starts
        against a balanced opponent sample should color Great."""
        from fantasy_baseball.lineup.upcoming_starts import _matchup_quality

        # Walk every team as opponent at their own home park (the most
        # common case for an away-pitcher start). With park-neutral
        # ranking the distribution should be roughly even across bands.
        bands = {"Tough": 0, "Fair": 0, "Great": 0}
        for team in _LEAGUE:
            quality, _, _ = _matchup_quality(team, team, _CTX)
            bands[quality] += 1
        # Each band should land within 7-13 of a perfect 10/10/10 split.
        # That gives the test some slack for K%/OPS not perfectly aligning
        # while still failing loudly if any band collapses to ~0 again.
        assert 7 <= bands["Tough"] <= 13, f"Tough bucket lopsided: {bands}"
        assert 7 <= bands["Fair"] <= 13, f"Fair bucket lopsided: {bands}"
        assert 7 <= bands["Great"] <= 13, f"Great bucket lopsided: {bands}"

    def test_missing_opponent_falls_back_to_fair(self):
        from fantasy_baseball.lineup.upcoming_starts import _matchup_quality

        quality, ops_rank, k_rank = _matchup_quality("XXX", "XXX", _CTX)
        assert quality == "Fair"
        assert ops_rank == 0
        assert k_rank == 0


def _player(name, positions):
    return Player(name=name, player_type=PlayerType.PITCHER, positions=positions)


class TestFilterStartingPitchers:
    def _proj(self, rows):
        df = pd.DataFrame(rows)
        # Tests provide _name_norm explicitly to avoid coupling to normalize_name internals
        return df

    def test_keeps_sp_eligible_with_positive_gs(self):
        roster = [
            _player("Bryan Woo", [Position.SP]),
            _player("Mason Miller", [Position.RP]),
        ]
        proj = self._proj(
            [
                {"_name_norm": "bryan woo", "gs": 28.0},
                {"_name_norm": "mason miller", "gs": 0.0},
            ]
        )
        kept = filter_starting_pitchers(roster, proj)
        assert [p.name for p in kept] == ["Bryan Woo"]

    def test_drops_sp_eligible_with_zero_gs(self):
        # Swingman: SP+RP eligible but projected as a reliever.
        roster = [_player("AJ Puk", [Position.SP, Position.RP])]
        proj = self._proj([{"_name_norm": "aj puk", "gs": 0.0}])
        assert filter_starting_pitchers(roster, proj) == []

    def test_drops_sp_eligible_missing_from_projections(self):
        # No projection row at all -> excluded (can't verify gs > 0).
        roster = [_player("Unknown Guy", [Position.SP])]
        proj = self._proj([])
        assert filter_starting_pitchers(roster, proj) == []

    def test_drops_pure_rp(self):
        roster = [_player("Mason Miller", [Position.RP])]
        proj = self._proj([{"_name_norm": "mason miller", "gs": 0.0}])
        assert filter_starting_pitchers(roster, proj) == []

    def test_handles_missing_gs_column(self):
        # Older projection blob without a gs column -> exclude all (defensive).
        roster = [_player("Bryan Woo", [Position.SP])]
        proj = self._proj([{"_name_norm": "bryan woo", "ip": 180.0}])
        assert filter_starting_pitchers(roster, proj) == []

    def test_p_only_yahoo_league_keeps_starters(self):
        # Yahoo "P-only" leagues (no SP/RP distinction) tag every pitcher as
        # just [Position.P]. The filter must still keep starters in this case;
        # gs > 0 does the actual starter-vs-reliever separation.
        roster = [
            _player("Bryan Woo", [Position.P]),
            _player("Mason Miller", [Position.P]),  # closer in a P-only league
        ]
        proj = self._proj(
            [
                {"_name_norm": "bryan woo", "gs": 29.9},
                {"_name_norm": "mason miller", "gs": 0.0},
            ]
        )
        kept = filter_starting_pitchers(roster, proj)
        assert [p.name for p in kept] == ["Bryan Woo"]
