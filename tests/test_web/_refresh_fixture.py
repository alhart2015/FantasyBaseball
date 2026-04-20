"""Minimal-but-realistic fixture data for the run_full_refresh
integration test. Returns plain dicts/lists matching the shapes
produced by Yahoo (post-parse) and the projection CSVs (post-blend).
"""
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

TEAM_NAMES = [f"Team {i:02d}" for i in range(1, 13)]  # 12 teams
USER_TEAM_NAME = "Team 01"


def _hitter_proj_row(name: str, fg_id: str, **stats) -> dict:
    """One row of a blended hitter projection table."""
    base = {
        "name": name, "fg_id": fg_id, "team": "TBD",
        "positions": "OF", "ab": 500, "pa": 580,
        "r": 80, "hr": 25, "rbi": 80, "sb": 8, "h": 145, "avg": 0.290,
        "player_type": "hitter",
    }
    base.update(stats)
    return base


def _pitcher_proj_row(name: str, fg_id: str, **stats) -> dict:
    base = {
        "name": name, "fg_id": fg_id, "team": "TBD",
        "positions": "SP", "ip": 180.0,
        "w": 12, "k": 200, "sv": 0, "era": 3.50, "whip": 1.15,
        "er": 70, "bb": 50, "h_allowed": 160,
        "player_type": "pitcher",
    }
    base.update(stats)
    return base


def hitter_projections() -> list[dict]:
    """120 hitters — enough to cover 12 rosters x ~6 hitters + spares."""
    rows = []
    for i in range(120):
        rows.append(_hitter_proj_row(
            name=f"Hitter{i:03d}", fg_id=f"fg_h_{i:03d}",
            r=70 + (i % 30), hr=15 + (i % 20), rbi=60 + (i % 30),
            sb=2 + (i % 15), avg=0.250 + (i % 50) / 1000,
        ))
    return rows


def pitcher_projections() -> list[dict]:
    """80 pitchers — covers 12 rosters x ~5 pitchers + spares + closers."""
    rows = []
    for i in range(80):
        is_closer = i < 12  # First 12 are closers
        rows.append(_pitcher_proj_row(
            name=f"Pitcher{i:03d}", fg_id=f"fg_p_{i:03d}",
            positions="RP" if is_closer else "SP",
            ip=70.0 if is_closer else 180.0,
            sv=25 if is_closer else 0,
            w=4 if is_closer else 10 + (i % 8),
            k=80 if is_closer else 180 + (i % 40),
            era=3.0 + (i % 10) / 10, whip=1.10 + (i % 10) / 100,
        ))
    return rows


def standings() -> list[dict]:
    """12 teams with all 10 categories populated."""
    out = []
    for i, name in enumerate(TEAM_NAMES, start=1):
        out.append({
            "name": name,
            "team_key": f"458.l.123.t.{i}",
            "rank": i,
            "stats": {
                "R": 100 + i * 10, "HR": 20 + i, "RBI": 100 + i * 8,
                "SB": 10 + i, "AVG": 0.250 + i / 1000,
                "W": 10 + i, "K": 150 + i * 5, "SV": 5 + i,
                "ERA": 3.50 + i / 100, "WHIP": 1.15 + i / 1000,
            },
        })
    return out


def roster_for_team(team_index: int) -> list[dict]:
    """One team's roster: 6 hitters, 5 pitchers (1 closer + 4 starters)."""
    base_h = team_index * 6
    base_p = team_index * 5
    out = []
    # Hitters in OF / Util / BN slots
    slot_cycle = ["OF", "OF", "OF", "Util", "BN", "BN"]
    for i in range(6):
        idx = base_h + i
        out.append({
            "name": f"Hitter{idx:03d}",
            "positions": ["OF", "Util"] if i < 4 else ["OF"],
            "selected_position": slot_cycle[i],
            "player_id": f"yh_h_{idx:03d}",
            "status": "",
        })
    # Pitchers — first one is a closer (RP), rest are SP
    for i in range(5):
        idx = base_p + i
        is_closer = i == 0
        out.append({
            "name": f"Pitcher{idx:03d}",
            "positions": ["RP", "P"] if is_closer else ["SP", "P"],
            "selected_position": "P",
            "player_id": f"yh_p_{idx:03d}",
            "status": "",
        })
    return out


def all_rosters() -> dict[str, list[dict]]:
    """All 12 teams' rosters by team name."""
    return {name: roster_for_team(i) for i, name in enumerate(TEAM_NAMES)}


def hitter_game_logs() -> dict[str, dict]:
    """Mid-season actuals keyed by normalized name. ~half the league."""
    out = {}
    for i in range(60):
        name = f"hitter{i:03d}"
        out[name] = {
            "r": 40 + (i % 20), "hr": 10 + (i % 12), "rbi": 40 + (i % 20),
            "sb": 1 + (i % 8), "avg": 0.260 + (i % 30) / 1000,
        }
    return out


def pitcher_game_logs() -> dict[str, dict]:
    out = {}
    for i in range(40):
        name = f"pitcher{i:03d}"
        out[name] = {
            "w": 5 + (i % 6), "k": 90 + (i % 40),
            "sv": 12 if i < 12 else 0,
            "era": 3.20 + (i % 10) / 10, "whip": 1.10 + (i % 10) / 100,
        }
    return out


def free_agents() -> list[dict]:
    """20 free agents — players NOT on any roster."""
    out = []
    # Hitters 72..91 (past the 72 used by 12 teams x 6 hitters)
    for i in range(72, 92):
        out.append({
            "name": f"Hitter{i:03d}",
            "positions": ["OF"],
            "selected_position": "BN",
            "player_id": f"yh_h_{i:03d}",
            "status": "",
        })
    return out


def transactions() -> list[dict]:
    """At least one transaction so transaction_analyzer.json gets written."""
    return [
        {
            "transaction_id": "tx-1",
            "type": "add/drop",
            "timestamp": 1744636800,  # 2026-04-14 epoch
            "team": USER_TEAM_NAME,
            "add_name": "Hitter072",
            "add_positions": "OF",
            "drop_name": "Hitter071",
            "drop_positions": "OF",
        },
    ]


def scoring_period() -> tuple[str, str]:
    """Sunday-ending scoring week."""
    return ("2026-04-13", "2026-04-19")  # Mon-Sun


def _mock_league(team_keys_to_names: dict[str, str]) -> MagicMock:
    """Yahoo league mock that returns canned teams() and supports the
    attribute access patterns used in run_full_refresh."""
    mock = MagicMock(name="MockLeague")
    mock.teams.return_value = {
        team_key: {"name": tname, "team_key": team_key}
        for team_key, tname in team_keys_to_names.items()
    }
    return mock


def _team_keys_to_names() -> dict[str, str]:
    return {f"458.l.123.t.{i}": name for i, name in enumerate(TEAM_NAMES, start=1)}


def seed_redis(client) -> None:
    """Write the projection blobs into fake Redis so
    redis_get_blended() reads them back. Uses the keys that
    fantasy_baseball.data.redis_store expects."""
    # Match the keys used by data.redis_store.get_blended_projections.
    # The reader inspects keys like "blended_projections:hitters" and
    # "blended_projections:pitchers". Encode as JSON list of dicts.
    client.set("blended_projections:hitters", json.dumps(hitter_projections()))
    client.set("blended_projections:pitchers", json.dumps(pitcher_projections()))


@contextmanager
def patched_refresh_environment(
    fake_redis,
    *,
    has_rest_of_season: bool = True,
    cache_dir,
):
    """Patch every external dependency of run_full_refresh and yield.

    - Yahoo session/league: MagicMock returning canned teams()
    - fetch_roster, fetch_standings, fetch_scoring_period: canned data
    - fetch_all_transactions: returns transactions() (empty by default)
    - fetch_and_match_free_agents: returns ([Player...], None)
    - fetch_game_log_totals: seeds canned game logs into fake Redis
    - get_week_schedule, get_team_batting_stats: return canned/empty
    - run_ros_monte_carlo: 10 iters instead of 1000; preseason baseline seeded in Redis
    - get_kv (data.kv_store): returns fake_redis
    - read_cache("ros_projections"): returns ROS proj rows or None
    """
    from fantasy_baseball.models.player import HitterStats, Player, PlayerType
    rosters = all_rosters()
    team_keys = _team_keys_to_names()

    league_mock = _mock_league(team_keys)

    # Seed Redis with projections + League data
    seed_redis(fake_redis)

    # Build League dataclass from rosters by writing snapshot keys
    from fantasy_baseball.data.redis_store import (
        write_roster_snapshot,
        write_standings_snapshot,
    )
    snapshot_date = "2026-04-21"  # next_tuesday after 2026-04-19
    for tname, team_roster in rosters.items():
        entries = [
            {
                "slot": r["selected_position"],
                "player_name": r["name"],
                "positions": ", ".join(r.get("positions", [])),
                "status": r.get("status") or "",
                "yahoo_id": r.get("player_id") or "",
            }
            for r in team_roster
        ]
        write_roster_snapshot(fake_redis, snapshot_date, tname, entries)
    write_standings_snapshot(
        fake_redis, snapshot_date,
        {"teams": [
            {
                "team": s["name"],
                "team_key": s["team_key"],
                "rank": s["rank"],
                **{k.lower(): v for k, v in s["stats"].items()},
            }
            for s in standings()
        ]},
    )

    # FA players — attach ROS stats so audit_roster's SGP calculation
    # sees real numbers instead of None.
    from fantasy_baseball.utils.positions import Position
    fa_player_objs = []
    for fa in free_agents():
        stats = HitterStats(
            pa=580, ab=500, h=145, r=80, hr=20, rbi=75, sb=6, avg=0.280,
        )
        fa_player_objs.append(Player(
            name=fa["name"],
            positions=[Position.parse(p) for p in fa["positions"]],
            player_type=PlayerType.HITTER,
            selected_position=Position.parse(fa.get("selected_position", "BN")),
            yahoo_id=fa.get("player_id", ""),
            rest_of_season=stats,
        ))

    def _fetch_roster(league, team_key, day=None):
        tname = team_keys.get(team_key)
        return rosters.get(tname, [])

    def _fetch_standings(league):
        return standings()

    def _fetch_scoring_period(league):
        return scoring_period()

    def _fetch_all_transactions(league):
        return transactions()

    def _fetch_and_match_fa(league, hitters_proj, pitchers_proj):
        return (fa_player_objs, None)

    def _fetch_game_logs(season_year, progress_cb=None):
        # Seed game logs into Redis using the data.redis_store API.
        # set_game_log_totals signature: (client, player_type, totals)
        from fantasy_baseball.data.redis_store import set_game_log_totals
        set_game_log_totals(fake_redis, "hitters", hitter_game_logs())
        set_game_log_totals(fake_redis, "pitchers", pitcher_game_logs())

    # Refresh no longer calls blend_and_cache_ros — it only READS
    # cache:ros_projections from Redis (the admin fetch is the sole
    # authoritative writer on Render). Seed the cache up front so the
    # refresh can find it, or leave it empty when has_rest_of_season
    # is False.
    if has_rest_of_season:
        from fantasy_baseball.web.season_data import CacheKey, write_cache
        write_cache(
            CacheKey.ROS_PROJECTIONS,
            {"hitters": hitter_projections(), "pitchers": pitcher_projections()},
            cache_dir,
        )

    # Capture the real Monte Carlo function BEFORE patching it,
    # otherwise the scaled wrapper calls itself recursively.
    # Seed a canned preseason baseline so refresh reads it from Redis
    # instead of running the (now-deleted) preseason MC live.
    from fantasy_baseball.data.redis_store import set_preseason_baseline
    from fantasy_baseball.simulation import (
        run_ros_monte_carlo as _real_ros_mc,
    )
    _canned_mc = {
        "team_results": {
            tname: {
                "median_pts": 70.0, "p10": 60.0, "p90": 80.0,
                "first_pct": 8.0, "top3_pct": 25.0,
            }
            for tname in rosters
        },
        "category_risk": {
            cat: {"median_pts": 7.0, "p10": 4.0, "p90": 10.0,
                  "top3_pct": 25.0, "bot3_pct": 20.0}
            for cat in ("R", "HR", "RBI", "SB", "AVG",
                        "W", "K", "SV", "ERA", "WHIP")
        },
    }
    set_preseason_baseline(fake_redis, 2026, {
        "base": _canned_mc,
        "with_management": _canned_mc,
        "meta": {
            "frozen_at": "2026-04-17T00:00:00Z",
            "season_year": 2026,
            "roster_date": "2026-03-27",
            "projections_source": "blended",
        },
    })

    def _scaled_ros_mc(*, team_rosters, actual_standings, fraction_remaining,
                       h_slots, p_slots, user_team_name,
                       n_iterations=1000, use_management=False, progress_cb=None):
        return _real_ros_mc(
            team_rosters=team_rosters, actual_standings=actual_standings,
            fraction_remaining=fraction_remaining,
            h_slots=h_slots, p_slots=p_slots, user_team_name=user_team_name,
            n_iterations=10, use_management=use_management,
            progress_cb=progress_cb,
        )

    # Build a test LeagueConfig so team_name/num_teams match the fixture data.
    from fantasy_baseball.config import LeagueConfig
    test_config = LeagueConfig(
        league_id=123,
        num_teams=12,
        game_code="mlb",
        team_name=USER_TEAM_NAME,
        draft_position=1,
        keepers=[],
        roster_slots={
            "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
            "OF": 3, "Util": 1, "P": 9, "BN": 3, "IL": 2,
        },
        projection_systems=["atc"],
        projection_weights={"atc": 1.0},
        sgp_overrides={},
        teams={i: name for i, name in enumerate(TEAM_NAMES, start=1)},
        strategy="no_punt_opp",
        scoring_mode="var",
        season_year=2026,
        season_start="2026-03-27",
        season_end="2026-09-28",
    )

    patches = [
        # Yahoo auth is imported lazily inside run_full_refresh from auth.yahoo_auth
        patch("fantasy_baseball.auth.yahoo_auth.get_yahoo_session", return_value=MagicMock()),
        patch("fantasy_baseball.auth.yahoo_auth.get_league", return_value=league_mock),
        # load_config is lazy-imported from fantasy_baseball.config
        patch("fantasy_baseball.config.load_config", return_value=test_config),
        # Roster/standings fetchers come from lineup.yahoo_roster
        patch("fantasy_baseball.lineup.yahoo_roster.fetch_standings", side_effect=_fetch_standings),
        patch("fantasy_baseball.lineup.yahoo_roster.fetch_scoring_period", side_effect=_fetch_scoring_period),
        patch("fantasy_baseball.lineup.yahoo_roster.fetch_roster", side_effect=_fetch_roster),
        patch("fantasy_baseball.lineup.yahoo_roster.fetch_all_transactions", side_effect=_fetch_all_transactions),
        # Free agents live in lineup.waivers
        patch("fantasy_baseball.lineup.waivers.fetch_and_match_free_agents", side_effect=_fetch_and_match_fa),
        # MLB game logs + schedule
        patch("fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals", side_effect=_fetch_game_logs),
        patch("fantasy_baseball.data.mlb_schedule.get_week_schedule", return_value={}),
        # Matchups (team batting stats)
        patch("fantasy_baseball.lineup.matchups.get_team_batting_stats", return_value={}),
        # Monte Carlo: patch at the source module
        patch("fantasy_baseball.simulation.run_ros_monte_carlo", side_effect=_scaled_ros_mc),
        # Redis clients. The central singleton owner is ``kv_store.get_kv``;
        # patching it covers every call site that does ``from kv_store
        # import get_kv`` inside a function body (most refresh_pipeline
        # entry points). ``season_data._get_redis`` is a thin wrapper that
        # also has to be patched because tests don't set RENDER=true.
        patch("fantasy_baseball.data.kv_store.get_kv", return_value=fake_redis),
        patch("fantasy_baseball.web.season_data._get_redis", return_value=fake_redis),
        # refresh_pipeline imports _get_redis at module level, so we also
        # have to patch the local name there (``_write_spoe_snapshot`` uses it).
        patch("fantasy_baseball.web.refresh_pipeline._get_redis", return_value=fake_redis),
    ]

    started = []
    try:
        for p in patches:
            started.append(p.start())
        yield
    finally:
        for p in patches:
            p.stop()
