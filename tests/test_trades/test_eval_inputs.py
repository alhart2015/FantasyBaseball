from fantasy_baseball.trades.eval_inputs import load_trade_eval_context


def _fixture_blobs():
    roster_raw = [{"name": "A B", "player_type": "hitter", "hr": 20, "ab": 400}]
    opp_raw = {"Opp": [{"name": "C D", "player_type": "pitcher", "ip": 100, "w": 8}]}
    proj_cache = {
        # ProjectedStandings.from_json shape: effective_date + teams[{name, stats}];
        # stats {} -> CategoryStats defaults (models/standings.py:363-381).
        "projected_standings": {
            "effective_date": "2026-07-23",
            "teams": [
                {"name": "Hart of the Order", "stats": {}},
                {"name": "Opp", "stats": {}},
            ],
        },
        "team_sds": None,
        "fraction_remaining": 0.4,
    }
    ros_cache = {"hitters": [], "pitchers": []}
    return roster_raw, opp_raw, proj_cache, ros_cache


def test_load_trade_eval_context_shapes_inputs():
    roster_raw, opp_raw, proj_cache, ros_cache = _fixture_blobs()
    ctx = load_trade_eval_context(
        hart_name="Hart of the Order",
        roster_raw=roster_raw,
        opp_rosters_raw=opp_raw,
        proj_cache=proj_cache,
        ros_cache=ros_cache,
        roster_slots={"OF": 4, "P": 9, "BN": 2},
    )
    assert ctx.hart_name == "Hart of the Order"
    assert [p.name for p in ctx.hart_roster] == ["A B"]
    assert set(ctx.opp_rosters) == {"Opp"}
    assert ctx.fraction_remaining == 0.4
    assert isinstance(ctx.waiver_pool, dict)  # keyed by player_key
    assert ctx.roster_slots["OF"] == 4
    assert any(e.team_name == "Hart of the Order" for e in ctx.projected_standings.entries)
