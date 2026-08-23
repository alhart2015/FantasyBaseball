"""Tests for lineup.waivers.fetch_and_match_free_agents."""

import dataclasses
import inspect

import pandas as pd
import pytest

from fantasy_baseball.lineup import waivers
from fantasy_baseball.models.player import Player


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


# ---------------------------------------------------------------------------
# The free-agent-source injection seam.
#
# FreeAgentRequest / FreeAgentSource were added so a caller with no Yahoo access
# can supply the pool from elsewhere (fantasy_baseball.manual.free_agents). The
# tests below are the revert-path guardrail: they pin that the addition is
# strictly additive and that the Yahoo path is untouched by it.
# ---------------------------------------------------------------------------


def _hitters_df(rows):
    defaults = {
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "h": 0,
        "ab": 0,
        "pa": 0,
        "g": 0,
        "avg": 0.0,
        "player_type": "hitter",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_yahoo_fetch_signature_is_unchanged_by_the_injection_seam():
    """Every existing caller passes these names positionally or by keyword; the
    seam must not have reordered, renamed, or re-defaulted any of them."""
    params = inspect.signature(waivers.fetch_and_match_free_agents).parameters

    assert list(params) == [
        "league",
        "hitters_proj",
        "pitchers_proj",
        "fa_per_position",
        "on_position_loaded",
        "preseason_hitters_proj",
        "preseason_pitchers_proj",
    ]
    assert params["fa_per_position"].default == 100
    assert params["on_position_loaded"].default is None
    assert params["preseason_hitters_proj"].default is None
    assert params["preseason_pitchers_proj"].default is None
    # The preseason frames stayed keyword-only, so no caller can pass one
    # positionally into the wrong slot.
    assert params["preseason_hitters_proj"].kind is inspect.Parameter.KEYWORD_ONLY


def test_yahoo_path_never_references_the_injection_seam():
    """Default-off is structural here, not a runtime flag: the Yahoo function
    does not read the new names at all, so there is no branch for a future edit
    to get wrong."""
    referenced = set(waivers.fetch_and_match_free_agents.__code__.co_names)

    assert "FreeAgentRequest" not in referenced
    assert "FreeAgentSource" not in referenced


def test_yahoo_path_still_queries_the_same_eight_positions_and_dedupes_by_name(monkeypatch):
    """Behavioral proof the seam changed nothing: same eight position queries,
    same per-position count, same normalized-name dedupe, and the returned
    fetched-count is still the RAW total (pre-dedupe), which is what the refresh
    progress line reports."""
    calls = []

    def fake_fetch(league, pos, count):
        calls.append((pos, count))
        return [{"name": "Utility Man", "positions": [pos], "status": ""}]

    monkeypatch.setattr(waivers, "fetch_free_agents", fake_fetch)
    ros_h = _hitters_df([{"name": "Utility Man", "_name_norm": "utility man", "pa": 500, "hr": 20}])

    fa_players, fetched = waivers.fetch_and_match_free_agents(
        league=None, hitters_proj=ros_h, pitchers_proj=_empty()
    )

    assert [pos for pos, _ in calls] == ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    assert {count for _, count in calls} == {100}
    assert fetched == 8
    assert len(fa_players) == 1


def test_free_agent_request_splits_rostered_names_by_player_type():
    """The field names and order downstream sources construct against.

    The hitter/pitcher split is the whole point of the object: a single pooled
    set of bare names would delete the pitcher Shohei Ohtani because the hitter
    Shohei Ohtani is rostered.
    """
    fields = [f.name for f in dataclasses.fields(waivers.FreeAgentRequest)]

    assert fields == [
        "hitters_proj",
        "pitchers_proj",
        "preseason_hitters_proj",
        "preseason_pitchers_proj",
        "rostered_hitters",
        "rostered_pitchers",
        "rankings_lookup",
    ]

    req = waivers.FreeAgentRequest(
        hitters_proj=_empty(),
        pitchers_proj=_empty(),
        preseason_hitters_proj=None,
        preseason_pitchers_proj=None,
        rostered_hitters=frozenset({"shohei ohtani"}),
        rostered_pitchers=frozenset(),
    )

    assert req.rankings_lookup is None  # the only field with a default
    assert "shohei ohtani" in req.rostered_hitters
    assert "shohei ohtani" not in req.rostered_pitchers
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.rostered_hitters = frozenset()


def test_free_agent_source_accepts_a_plain_callable():
    """The seam is a callable type, not a class to subclass, so a functools.partial
    over the manual builder satisfies it without importing anything from here."""
    players: list[Player] = []

    def source(req: waivers.FreeAgentRequest) -> list[Player]:
        assert isinstance(req, waivers.FreeAgentRequest)
        return players

    fn: waivers.FreeAgentSource = source
    req = waivers.FreeAgentRequest(
        hitters_proj=_empty(),
        pitchers_proj=_empty(),
        preseason_hitters_proj=None,
        preseason_pitchers_proj=None,
        rostered_hitters=frozenset(),
        rostered_pitchers=frozenset(),
    )

    assert fn(req) is players
