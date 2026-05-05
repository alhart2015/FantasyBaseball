"""Unit tests for the rotation anchor + projection logic."""

from datetime import date as _date

from fantasy_baseball.lineup.upcoming_starts import (
    GameSlot,
    StartEntry,
    build_team_game_index,
    compose_pitcher_entries,
    find_anchor_index,
    project_start_indices,
)


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


_FACTORS = {
    "TEX": {"era_whip_factor": 0.90, "k_factor": 1.05},  # Great
    "LAD": {"era_whip_factor": 1.10, "k_factor": 0.95},  # Tough
    "HOU": {"era_whip_factor": 1.00, "k_factor": 1.00},  # Fair
}
_TEAM_STATS = {
    "TEX": {"ops": 0.700, "k_pct": 0.24},
    "LAD": {"ops": 0.800, "k_pct": 0.20},
    "HOU": {"ops": 0.750, "k_pct": 0.22},
}
_OPS_RANK = {"TEX": 25, "LAD": 4, "HOU": 14}
_K_RANK = {"TEX": 8, "LAD": 26, "HOU": 14}


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
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 1
        e = entries[0]
        assert e.date == "2026-05-09"
        assert e.day == "Sat"
        assert e.opponent == "TEX"
        assert e.announced is False
        assert e.matchup_quality == "Great"
        assert e.detail["ops_rank"] == 25

    def test_off_day_extends_calendar_gap(self):
        # Anchor Mon (idx 0). Team games after anchor: Tue, Wed, [off Thu],
        # Fri, Sat, Sun. Sun is the 5th team-game post-anchor (idx 5),
        # so the projected start lands on Sun = 2026-05-10. Compare with
        # test_simple_5_day_rotation_no_off_day: same anchor, but no off-day
        # so the 5th team-game is Sat = 2026-05-09. The off-day pushes the
        # next start one calendar day later — that's the gap "extension."
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor (idx 0)
            ("2026-05-05", "TEX", "", "@"),  # idx 1
            ("2026-05-06", "TEX", "", "@"),  # idx 2
            # No 2026-05-07 entry — off day
            ("2026-05-08", "TEX", "", "@"),  # idx 3
            ("2026-05-09", "TEX", "", "@"),  # idx 4
            ("2026-05-10", "TEX", "", "@"),  # idx 5 — 5th team-game post-anchor
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
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
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
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
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert entries == []

    def test_two_start_week_from_old_anchor(self):
        # Build a 21-game team stream where the anchor sits at index 0
        # (2026-04-25, ten days before the window opens). Anchor + 5/10/15/20
        # land at indices 5, 10, 15, 20 — i.e. dates 04-30, 05-05, 05-10, 05-15.
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
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
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
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 1
        assert entries[0].announced is True

    def test_empty_when_pitcher_has_no_starts(self):
        team_games = _seq(("2026-05-05", "TEX", "OtherGuy", "@"))
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert entries == []
