"""Tests for lineup.waivers.fetch_and_match_free_agents."""

import pandas as pd

from fantasy_baseball.lineup import waivers


def _pitchers_df(rows):
    defaults = {
        "w": 0,
        "k": 0,
        "sv": 0,
        "ip": 0,
        "er": 0,
        "bb": 0,
        "h_allowed": 0,
        "era": 0.0,
        "whip": 0.0,
        "player_type": "pitcher",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _empty():
    return pd.DataFrame(columns=["name", "_name_norm", "player_type"])


def test_fa_players_get_preseason_when_frames_provided(monkeypatch):
    """Free-agent stash candidates must carry ``.preseason`` so the stash board
    scores them with the same remaining-season slot-share model as owned IL arms
    (not the legacy direct-IP fallback). Preseason follows the matched ROS row's
    identity (mlbam_id), per type.
    """

    def fake_fetch(league, pos, count):
        if pos == "RP":
            return [{"name": "Joe Reliever", "positions": ["RP"], "status": ""}]
        return []

    monkeypatch.setattr(waivers, "fetch_free_agents", fake_fetch)

    ros_p = _pitchers_df(
        [{"name": "Joe Reliever", "_name_norm": "joe reliever", "mlbam_id": 555, "ip": 30, "k": 40}]
    )
    pre_p = _pitchers_df(
        [{"name": "Joe Reliever", "_name_norm": "joe reliever", "mlbam_id": 555, "ip": 65, "k": 80}]
    )

    fa_players, _ = waivers.fetch_and_match_free_agents(
        league=None,
        hitters_proj=_empty(),
        pitchers_proj=ros_p,
        preseason_hitters_proj=_empty(),
        preseason_pitchers_proj=pre_p,
    )

    assert len(fa_players) == 1
    assert fa_players[0].rest_of_season.ip == 30  # ROS frame
    assert fa_players[0].preseason is not None
    assert fa_players[0].preseason.ip == 65  # preseason (healthy full-season) frame


def test_fa_players_no_preseason_when_frames_omitted(monkeypatch):
    """Backward-compatible: without preseason frames, FA players have no
    .preseason (and the displacement falls back to the legacy path)."""

    def fake_fetch(league, pos, count):
        if pos == "RP":
            return [{"name": "Joe Reliever", "positions": ["RP"], "status": ""}]
        return []

    monkeypatch.setattr(waivers, "fetch_free_agents", fake_fetch)
    ros_p = _pitchers_df(
        [{"name": "Joe Reliever", "_name_norm": "joe reliever", "mlbam_id": 555, "ip": 30, "k": 40}]
    )
    fa_players, _ = waivers.fetch_and_match_free_agents(
        league=None, hitters_proj=_empty(), pitchers_proj=ros_p
    )
    assert len(fa_players) == 1
    assert fa_players[0].preseason is None


def test_fa_same_name_collision_resolves_by_volume(monkeypatch):
    """FAs now route through match_roster_to_projections, so a same-name
    collision resolves to the high-playing-time row (the real player) instead of
    whichever row is first -- the same _pick_best_match guard the roster path
    uses."""

    def fake_fetch(league, pos, count):
        if pos == "RP":
            return [{"name": "Mason Miller", "positions": ["RP"], "status": ""}]
        return []

    monkeypatch.setattr(waivers, "fetch_free_agents", fake_fetch)
    ros_p = _pitchers_df(
        [
            {"name": "Mason Miller", "_name_norm": "mason miller", "mlbam_id": 1, "ip": 2, "k": 3},
            {
                "name": "Mason Miller",
                "_name_norm": "mason miller",
                "mlbam_id": 2,
                "ip": 60,
                "k": 90,
            },
        ]
    )
    fa_players, _ = waivers.fetch_and_match_free_agents(
        league=None, hitters_proj=_empty(), pitchers_proj=ros_p
    )
    assert len(fa_players) == 1
    assert fa_players[0].rest_of_season.ip == 60  # high-volume real arm, not the 2-IP first row
