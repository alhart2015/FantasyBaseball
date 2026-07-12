from fantasy_baseball.summary.builders import build_standings_delta


def _standings_json(effective_date, teams):
    return {"effective_date": effective_date, "teams": teams}


def _team(name, rank, stats, yahoo_points=None, team_key=None):
    return {
        "name": name,
        "team_key": team_key or f"k.{name}",
        "rank": rank,
        "yahoo_points_for": yahoo_points,
        "stats": stats,
        "extras": {},
    }


# Two teams. HIGH strictly beats LOW in BOTH set categories (HR and SB), so the
# team holding HIGH leads both and the team holding LOW trails both. The 8 unset
# categories default to 0.0 for both teams -> tie -> equal points -> they cancel
# in the delta. That makes the movement below unambiguous.
_STATS_HIGH = {"HR": 100.0, "SB": 60.0}
_STATS_LOW = {"HR": 80.0, "SB": 50.0}


def test_first_run_yields_baseline():
    current = _standings_json(
        "2026-07-14", [_team("My Team", 1, _STATS_HIGH), _team("Rival", 2, _STATS_LOW)]
    )
    delta = build_standings_delta(current, None, "My Team")
    assert delta.is_first_run is True
    assert delta.teams == []


def test_delta_computes_rank_and_category_movement():
    # Prior: My Team holds LOW (trails both cats). Current: My Team holds HIGH
    # (leads both cats). Trailing both -> leading both is +1 point per category
    # in each of HR and SB = +2.0 total.
    prior = _standings_json(
        "2026-07-14", [_team("My Team", 2, _STATS_LOW), _team("Rival", 1, _STATS_HIGH)]
    )
    current = _standings_json(
        "2026-07-14", [_team("My Team", 1, _STATS_HIGH), _team("Rival", 2, _STATS_LOW)]
    )
    snapshot = {"written_at": "2026-07-10T12:00:00+00:00", "standings": prior}

    delta = build_standings_delta(current, snapshot, "My Team")

    assert delta.is_first_run is False
    assert delta.user_team_name == "My Team"
    mine = next(t for t in delta.teams if t.name == "My Team")
    assert mine.rank_prev == 2
    assert mine.rank_now == 1
    assert mine.points_now - mine.points_prev == 2.0


def test_delta_joins_on_team_key_across_a_rename():
    # Same team (team_key "k.me") is renamed between the snapshot and today.
    # Joining on team_key (not the mutable name) keeps the team in the delta;
    # a name-join would drop it (prior name != current name).
    prior = _standings_json(
        "2026-07-14",
        [
            _team("Old Name", 2, _STATS_LOW, team_key="k.me"),
            _team("Rival", 1, _STATS_HIGH, team_key="k.rival"),
        ],
    )
    current = _standings_json(
        "2026-07-14",
        [
            _team("New Name", 1, _STATS_HIGH, team_key="k.me"),
            _team("Rival", 2, _STATS_LOW, team_key="k.rival"),
        ],
    )
    snapshot = {"written_at": "2026-07-10T12:00:00+00:00", "standings": prior}

    delta = build_standings_delta(current, snapshot, "New Name")

    mine = next(t for t in delta.teams if t.name == "New Name")
    assert mine.rank_prev == 2
    assert mine.rank_now == 1
    assert mine.points_now - mine.points_prev == 2.0
