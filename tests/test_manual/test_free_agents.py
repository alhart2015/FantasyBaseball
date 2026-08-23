"""Tests for manual.free_agents.build_manual_free_agents.

The pool this module builds is the substitute for the one Yahoo used to return,
and ``lineup.roster_audit.audit_roster`` takes the MAXIMUM-DeltaRoto free agent
per slot -- so every filter here is load-bearing on the headline "drop X, add Y"
recommendation. These tests pin the derivation, not the audit math.
"""

import logging

import pandas as pd
import pytest

from fantasy_baseball.lineup.waivers import FreeAgentRequest
from fantasy_baseball.manual.free_agents import build_manual_free_agents
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.name_utils import normalize_name

# --------------------------------------------------------------------------
# Frame builders. Column sets mirror the real blended_projections:* blobs.
# --------------------------------------------------------------------------

_HITTER_DEFAULTS = {
    "r": 0.0,
    "hr": 0.0,
    "rbi": 0.0,
    "sb": 0.0,
    "h": 0.0,
    "ab": 0.0,
    "pa": 600.0,
    "g": 0.0,
    "avg": 0.0,
    "player_type": "hitter",
    "team": "XXX",
    "fg_id": None,
}

_PITCHER_DEFAULTS = {
    "w": 0.0,
    "k": 0.0,
    "sv": 0.0,
    "ip": 180.0,
    "er": 0.0,
    "bb": 0.0,
    "h_allowed": 0.0,
    "gs": 0.0,
    "g": 0.0,
    "era": 0.0,
    "whip": 0.0,
    "player_type": "pitcher",
    "team": "XXX",
    "fg_id": None,
}


def _frame(rows, defaults):
    if not rows:
        return pd.DataFrame(columns=[*defaults, "name", "_name_norm"])
    built = []
    for row in rows:
        merged = {**defaults, **row}
        merged["_name_norm"] = normalize_name(merged["name"])
        built.append(merged)
    return pd.DataFrame(built)


def _hitters(rows):
    return _frame(rows, _HITTER_DEFAULTS)


def _pitchers(rows):
    return _frame(rows, _PITCHER_DEFAULTS)


def _ranks(*names_and_types, start=1):
    """Build a rankings_lookup giving each ``(name, player_type)`` a rank.

    Keyed the way ``sgp.rankings.rank_key`` keys them -- ``name::player_type``,
    normalized -- because that is the fallback ``lookup_rank`` uses when a row
    carries no fg_id, which is the case for every fixture row here unless it
    sets one explicitly.
    """
    return {
        f"{normalize_name(name)}::{ptype}": {
            "rest_of_season": i,
            "preseason": None,
            "current": None,
            "total": None,
        }
        for i, (name, ptype) in enumerate(names_and_types, start=start)
    }


def _request(hitters, pitchers, *, rostered_hitters=(), rostered_pitchers=(), rankings=None):
    return FreeAgentRequest(
        hitters_proj=hitters,
        pitchers_proj=pitchers,
        preseason_hitters_proj=None,
        preseason_pitchers_proj=None,
        rostered_hitters=frozenset(normalize_name(n) for n in rostered_hitters),
        rostered_pitchers=frozenset(normalize_name(n) for n in rostered_pitchers),
        rankings_lookup=rankings,
    )


def _by_key(players):
    return {(normalize_name(p.name), p.player_type) for p in players}


# --------------------------------------------------------------------------
# The subtraction: per (normalized name, player_type), never a bare name.
# --------------------------------------------------------------------------


def test_rostered_players_excluded_per_player_type():
    """Rostering Ohtani-the-hitter must not remove Ohtani-the-pitcher.

    This league keeps Shohei Ohtani as a batter only (``config/league.yaml``:
    ``note: "batter only"``), so his arm is genuinely a free agent while his bat
    is not. He is one row in each projection frame under one name. A bare-name
    subtraction -- the obvious implementation -- deletes both rows and silently
    removes the single most valuable pitcher available. The same bug hides the
    pitcher Will Smith behind the catcher Will Smith.
    """
    hitters = _hitters([{"name": "Shohei Ohtani"}, {"name": "Will Smith"}])
    pitchers = _pitchers([{"name": "Shohei Ohtani"}, {"name": "Will Smith"}])
    positions = {
        "shohei ohtani": ["UTIL", "OF"],
        "will smith": ["C", "UTIL"],
    }
    req = _request(
        hitters,
        pitchers,
        rostered_hitters=["Shohei Ohtani", "Will Smith"],
        rostered_pitchers=[],
        rankings=_ranks(
            ("Shohei Ohtani", PlayerType.HITTER),
            ("Will Smith", PlayerType.HITTER),
            ("Shohei Ohtani", PlayerType.PITCHER),
            ("Will Smith", PlayerType.PITCHER),
        ),
    )

    players = build_manual_free_agents(req, positions_by_name=positions)

    assert _by_key(players) == {
        ("shohei ohtani", PlayerType.PITCHER),
        ("will smith", PlayerType.PITCHER),
    }
    # And the surviving rows carry the PITCHER projection, not the hitter one.
    for player in players:
        assert player.player_type == PlayerType.PITCHER
        assert player.rest_of_season is not None
        assert hasattr(player.rest_of_season, "ip")


def test_rostered_pitcher_does_not_remove_the_same_named_hitter():
    """The mirror image, so the subtraction cannot be right by accident in only
    one direction."""
    hitters = _hitters([{"name": "Shohei Ohtani"}])
    pitchers = _pitchers([{"name": "Shohei Ohtani"}])
    req = _request(
        hitters,
        pitchers,
        rostered_pitchers=["Shohei Ohtani"],
        rankings=_ranks(
            ("Shohei Ohtani", PlayerType.HITTER),
            ("Shohei Ohtani", PlayerType.PITCHER),
        ),
    )

    players = build_manual_free_agents(req, positions_by_name={"shohei ohtani": ["OF", "UTIL"]})

    assert _by_key(players) == {("shohei ohtani", PlayerType.HITTER)}


def test_two_way_pitcher_with_hitter_only_eligibility_is_still_a_pitcher():
    """The frozen ``cache:positions`` blob stores Ohtani as ``["UTIL"]`` -- the
    slot he is ROSTERED at, which says nothing about his arm.

    Filtering eligibility by the frame's player type (and falling back to
    ``["P"]``) is what stops that hitter-only entry from routing the pitcher row
    into ``match_roster_to_projections``' hitter branch, where it would come
    back carrying the wrong player's stat line entirely.
    """
    hitters = _hitters([{"name": "Shohei Ohtani", "hr": 47.0}])
    pitchers = _pitchers([{"name": "Shohei Ohtani", "k": 130.0, "ip": 107.0}])
    req = _request(
        hitters,
        pitchers,
        rostered_hitters=["Shohei Ohtani"],
        rankings=_ranks(("Shohei Ohtani", PlayerType.PITCHER)),
    )

    players = build_manual_free_agents(req, positions_by_name={"shohei ohtani": ["UTIL"]})

    assert len(players) == 1
    assert players[0].player_type == PlayerType.PITCHER
    assert [p.value for p in players[0].positions] == ["P"]
    assert players[0].rest_of_season.k == 130.0


# --------------------------------------------------------------------------
# The volume floor.
# --------------------------------------------------------------------------


def test_volume_floor_drops_low_playing_time_rows():
    hitters = _hitters(
        [
            {"name": "Regular Starter", "pa": 400.0},
            {"name": "Depth Bat", "pa": 12.0},
        ]
    )
    pitchers = _pitchers(
        [
            {"name": "Real Arm", "ip": 55.0},
            {"name": "Aaa Filler", "ip": 4.0},
        ]
    )
    positions = {"regular starter": ["OF"], "depth bat": ["OF"]}
    req = _request(
        hitters,
        pitchers,
        rankings=_ranks(
            ("Regular Starter", PlayerType.HITTER),
            ("Depth Bat", PlayerType.HITTER),
            ("Real Arm", PlayerType.PITCHER),
            ("Aaa Filler", PlayerType.PITCHER),
        ),
    )

    players = build_manual_free_agents(
        req, positions_by_name=positions, min_ros_pa=40.0, min_ros_ip=12.0
    )

    assert _by_key(players) == {
        ("regular starter", PlayerType.HITTER),
        ("real arm", PlayerType.PITCHER),
    }


def test_zero_valued_volume_column_is_not_treated_as_missing():
    """A projected 0.0 PA is a real number and is compared as one.

    ``row.get("pa") or 0.0`` -- the idiom CLAUDE.md forbids -- maps a genuine
    0.0 and an absent cell onto the same value. At a floor of 0.0 the correct
    reading keeps the row (0.0 >= 0.0); at any positive floor it drops it. Both
    ends are asserted so the comparison cannot quietly become a truthiness test.
    """
    hitters = _hitters([{"name": "Zero Pa Bat", "pa": 0.0}])
    positions = {"zero pa bat": ["1B"]}
    ranks = _ranks(("Zero Pa Bat", PlayerType.HITTER))

    kept = build_manual_free_agents(
        _request(hitters, _pitchers([]), rankings=ranks),
        positions_by_name=positions,
        min_ros_pa=0.0,
    )
    assert _by_key(kept) == {("zero pa bat", PlayerType.HITTER)}

    dropped = build_manual_free_agents(
        _request(hitters, _pitchers([]), rankings=ranks),
        positions_by_name=positions,
        min_ros_pa=0.1,
    )
    assert dropped == []


def test_missing_volume_column_skips_the_floor_instead_of_emptying_the_pool(caplog):
    """A projection frame with no ``pa`` column is a schema change, not a pool of
    zero-PA players. Reading every row as 0.0 and dropping the lot would hand the
    audit an empty hitter pool, which renders as "no upgrades available" -- a
    claim, not the gap it actually is."""
    hitters = _hitters([{"name": "Regular Starter"}]).drop(columns=["pa"])
    positions = {"regular starter": ["OF"]}
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Regular Starter", PlayerType.HITTER)))

    with caplog.at_level(logging.WARNING, logger="fantasy_baseball.manual.free_agents"):
        players = build_manual_free_agents(req, positions_by_name=positions, min_ros_pa=40.0)

    assert _by_key(players) == {("regular starter", PlayerType.HITTER)}
    assert any("volume floor" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# The rank cap.
# --------------------------------------------------------------------------


def test_per_position_cap_keeps_best_by_ros_rank():
    """Yahoo returned a pre-ranked, ownership-filtered slice; without a cap every
    projected AAA bat becomes an audit candidate."""
    names = [f"Outfielder {i:02d}" for i in range(6)]
    hitters = _hitters([{"name": n} for n in names])
    positions = {normalize_name(n): ["OF"] for n in names}
    # Rank 1 is the best; assign in list order so the first two are the keepers.
    ranks = _ranks(*[(n, PlayerType.HITTER) for n in names])
    req = _request(hitters, _pitchers([]), rankings=ranks)

    players = build_manual_free_agents(req, positions_by_name=positions, per_position_cap=2)

    assert _by_key(players) == {
        ("outfielder 00", PlayerType.HITTER),
        ("outfielder 01", PlayerType.HITTER),
    }


def test_cap_is_applied_per_bucket_so_a_thin_position_is_not_crowded_out():
    """The scarce catcher must survive a cap of one even though six outfielders
    outrank him -- the same reason Yahoo is queried position by position."""
    hitters = _hitters([{"name": f"Outfielder {i}"} for i in range(6)] + [{"name": "Some Catcher"}])
    positions = {normalize_name(f"Outfielder {i}"): ["OF"] for i in range(6)}
    positions["some catcher"] = ["C"]
    ranks = _ranks(*[(f"Outfielder {i}", PlayerType.HITTER) for i in range(6)])
    ranks.update(_ranks(("Some Catcher", PlayerType.HITTER), start=99))
    req = _request(hitters, _pitchers([]), rankings=ranks)

    players = build_manual_free_agents(req, positions_by_name=positions, per_position_cap=1)

    assert _by_key(players) == {
        ("outfielder 0", PlayerType.HITTER),
        ("some catcher", PlayerType.HITTER),
    }


def test_pitchers_share_one_bucket_at_twice_the_cap():
    """Mirrors the Yahoo path, which fetches SP and RP separately -- two
    position-sized slices for one undifferentiated pitcher slot."""
    names = [f"Arm {i:02d}" for i in range(6)]
    pitchers = _pitchers([{"name": n} for n in names])
    ranks = _ranks(*[(n, PlayerType.PITCHER) for n in names])
    req = _request(_hitters([]), pitchers, rankings=ranks)

    players = build_manual_free_agents(req, positions_by_name={}, per_position_cap=2)

    assert len(players) == 4  # 2 x per_position_cap
    assert _by_key(players) == {(normalize_name(n), PlayerType.PITCHER) for n in names[:4]}


def test_unranked_candidates_are_dropped():
    hitters = _hitters([{"name": "Ranked Bat"}, {"name": "Unranked Bat"}])
    positions = {"ranked bat": ["OF"], "unranked bat": ["OF"]}
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Ranked Bat", PlayerType.HITTER)))

    players = build_manual_free_agents(req, positions_by_name=positions)

    assert _by_key(players) == {("ranked bat", PlayerType.HITTER)}


def test_rank_zero_is_a_real_rank_not_a_missing_one():
    """Rank is an ordinal where SMALLER is better, so 0 would be the best player
    there is. ``if not rank`` -- or a ``rank or BIG_NUMBER`` sort key -- discards
    or last-places exactly that player. The presence test is ``is None``."""
    hitters = _hitters([{"name": "Rank Zero Bat"}, {"name": "Rank One Bat"}])
    positions = {"rank zero bat": ["OF"], "rank one bat": ["OF"]}
    ranks = {
        "rank zero bat::hitter": {"rest_of_season": 0},
        "rank one bat::hitter": {"rest_of_season": 1},
    }
    req = _request(hitters, _pitchers([]), rankings=ranks)

    players = build_manual_free_agents(req, positions_by_name=positions, per_position_cap=1)

    assert _by_key(players) == {("rank zero bat", PlayerType.HITTER)}


def test_missing_rankings_lookup_raises_instead_of_returning_a_junk_pool():
    req = _request(_hitters([{"name": "Some Bat"}]), _pitchers([]), rankings=None)
    with pytest.raises(ValueError, match="rankings_lookup"):
        build_manual_free_agents(req, positions_by_name={"some bat": ["OF"]})


def test_fg_id_is_preferred_over_the_name_key_for_the_rank_lookup():
    """``lookup_rank`` tries ``fg_id::player_type`` first. Reading the fg_id off
    the row -- rather than hand-typing one -- is what keeps the two same-named
    pitchers from sharing a rank."""
    pitchers = _pitchers(
        [{"name": "Mason Miller", "fg_id": "31757"}, {"name": "Other Arm", "fg_id": "1"}]
    )
    ranks = {
        "31757::pitcher": {"rest_of_season": 1},
        # Name key deliberately absent for Mason Miller: if the fg_id key were
        # ignored he would be treated as unranked and dropped.
        "other arm::pitcher": {"rest_of_season": 2},
    }
    req = _request(_hitters([]), pitchers, rankings=ranks)

    players = build_manual_free_agents(req, positions_by_name={}, per_position_cap=1)

    assert ("mason miller", PlayerType.PITCHER) in _by_key(players)


# --------------------------------------------------------------------------
# Eligibility.
# --------------------------------------------------------------------------


def test_players_without_eligible_positions_are_dropped(caplog):
    """A hitter we have no position data for is dropped, never defaulted to UTIL.

    A synthetic UTIL bat is eligible for a real lineup slot on the strength of a
    projection alone, so it can out-score a rostered player and surface as an
    upgrade that does not exist. The drop is warned about, not silent.
    """
    hitters = _hitters([{"name": "Known Bat"}, {"name": "Unknown Bat"}])
    req = _request(
        hitters,
        _pitchers([]),
        rankings=_ranks(("Known Bat", PlayerType.HITTER), ("Unknown Bat", PlayerType.HITTER)),
    )

    with caplog.at_level(logging.WARNING, logger="fantasy_baseball.manual.free_agents"):
        players = build_manual_free_agents(req, positions_by_name={"known bat": ["1B"]})

    assert _by_key(players) == {("known bat", PlayerType.HITTER)}
    messages = [r.getMessage() for r in caplog.records]
    assert any("Unknown Bat" in m for m in messages)
    assert any("no position data" in m for m in messages)


def test_a_hitter_is_never_given_a_synthetic_util_slot():
    """The specific shape of the previous failure: nothing in the output may
    carry UTIL-only eligibility that the position source did not supply."""
    hitters = _hitters([{"name": "Unknown Bat"}])
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Unknown Bat", PlayerType.HITTER)))

    players = build_manual_free_agents(req, positions_by_name={})

    assert players == []


def test_pitchers_without_position_data_default_to_p():
    """Yahoo reports every pitcher in this league as ``["P"]`` and
    ``roster_audit`` splits SP/RP on projected saves, so the default is the real
    answer rather than a guess."""
    pitchers = _pitchers([{"name": "Unknown Arm"}])
    req = _request(_hitters([]), pitchers, rankings=_ranks(("Unknown Arm", PlayerType.PITCHER)))

    players = build_manual_free_agents(req, positions_by_name={})

    assert len(players) == 1
    assert [p.value for p in players[0].positions] == ["P"]


def test_stale_il_tokens_in_the_positions_blob_are_not_carried_as_positions():
    """The frozen blob stores ``["OF", "UTIL", "IL"]`` for a player who was hurt
    when it was written. IL is a roster slot, not eligibility, and the blob is a
    month stale -- carrying it onto a free agent would launder an old injury
    into a current claim. Real unavailability belongs in fa_exclusions.yaml."""
    hitters = _hitters([{"name": "Sometime Injured"}])
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Sometime Injured", PlayerType.HITTER)))

    players = build_manual_free_agents(
        req, positions_by_name={"sometime injured": ["OF", "UTIL", "IL"]}
    )

    assert len(players) == 1
    assert [p.value for p in players[0].positions] == ["OF", "UTIL"]
    assert players[0].status == ""


def test_unparseable_position_tokens_are_skipped_not_fatal():
    hitters = _hitters([{"name": "Odd Tokens"}])
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Odd Tokens", PlayerType.HITTER)))

    players = build_manual_free_agents(req, positions_by_name={"odd tokens": ["not-a-slot", "OF"]})

    assert [p.value for p in players[0].positions] == ["OF"]


def test_positions_are_looked_up_by_normalized_name():
    """Accented and unaccented spellings must resolve to the same entry -- the
    same normalization the rest of the repo joins on."""
    hitters = _hitters([{"name": "Yoan Moncada"}])
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Yoan Moncada", PlayerType.HITTER)))

    players = build_manual_free_agents(req, positions_by_name={"yoan moncada": ["3B", "IF"]})

    assert [p.value for p in players[0].positions] == ["3B", "IF"]


# --------------------------------------------------------------------------
# Exclusions.
# --------------------------------------------------------------------------


def test_exclusions_applied():
    hitters = _hitters([{"name": "Available Bat"}, {"name": "Injured Bat"}])
    positions = {"available bat": ["OF"], "injured bat": ["OF"]}
    req = _request(
        hitters,
        _pitchers([]),
        rankings=_ranks(("Injured Bat", PlayerType.HITTER), ("Available Bat", PlayerType.HITTER)),
    )

    players = build_manual_free_agents(
        req,
        positions_by_name=positions,
        excluded_names=frozenset({"injured bat"}),
    )

    assert _by_key(players) == {("available bat", PlayerType.HITTER)}


def test_exclusions_do_not_consume_a_cap_slot():
    """An excluded name must not push a usable candidate out of the pool: the cap
    exists to bound pool SIZE, and spending a slot on someone who can never be
    recommended silently shrinks it."""
    hitters = _hitters([{"name": "Injured Bat"}, {"name": "Available Bat"}])
    positions = {"injured bat": ["OF"], "available bat": ["OF"]}
    req = _request(
        hitters,
        _pitchers([]),
        rankings=_ranks(("Injured Bat", PlayerType.HITTER), ("Available Bat", PlayerType.HITTER)),
    )

    players = build_manual_free_agents(
        req,
        positions_by_name=positions,
        excluded_names=frozenset({"injured bat"}),
        per_position_cap=1,
    )

    assert _by_key(players) == {("available bat", PlayerType.HITTER)}


def test_exclusions_remove_both_types_of_a_shared_name():
    """Documented behavior of ``data/manual/fa_exclusions.yaml``: the match is by
    name only, so an entry removes the hitter and the pitcher alike."""
    hitters = _hitters([{"name": "Will Smith"}])
    pitchers = _pitchers([{"name": "Will Smith"}])
    req = _request(
        hitters,
        pitchers,
        rankings=_ranks(
            ("Will Smith", PlayerType.HITTER),
            ("Will Smith", PlayerType.PITCHER),
        ),
    )

    players = build_manual_free_agents(
        req,
        positions_by_name={"will smith": ["C", "UTIL"]},
        excluded_names=frozenset({"will smith"}),
    )

    assert players == []


# --------------------------------------------------------------------------
# Output shape: identical to what the Yahoo path produces.
# --------------------------------------------------------------------------


def test_output_matches_the_yahoo_paths_player_shape():
    """The audit must not be able to tell the two pools apart. ``.preseason`` is
    attached from the preseason frames by the matched row's identity, exactly as
    ``fetch_and_match_free_agents`` does it."""
    pitchers = _pitchers([{"name": "Joe Reliever", "mlbam_id": 555, "ip": 30.0, "k": 40.0}])
    preseason = _pitchers([{"name": "Joe Reliever", "mlbam_id": 555, "ip": 65.0, "k": 80.0}])
    req = FreeAgentRequest(
        hitters_proj=_hitters([]),
        pitchers_proj=pitchers,
        preseason_hitters_proj=_hitters([]),
        preseason_pitchers_proj=preseason,
        rostered_hitters=frozenset(),
        rostered_pitchers=frozenset(),
        rankings_lookup=_ranks(("Joe Reliever", PlayerType.PITCHER)),
    )

    players = build_manual_free_agents(req, positions_by_name={})

    assert len(players) == 1
    player = players[0]
    assert player.name == "Joe Reliever"
    assert player.player_type == PlayerType.PITCHER
    assert player.rest_of_season.ip == 30.0  # ROS frame
    assert player.preseason is not None
    assert player.preseason.ip == 65.0  # preseason frame
    assert player.status == ""
    assert player.selected_position is None


def test_same_name_collision_resolves_by_playing_time_via_the_shared_matcher():
    """Delegating to ``match_roster_to_projections`` means the two Mason Millers
    resolve the same way here as on the Yahoo path -- to the high-volume row, not
    to whichever came first."""
    pitchers = _pitchers(
        [
            {"name": "Mason Miller", "mlbam_id": 1, "ip": 20.0, "k": 3.0, "fg_id": "aaa"},
            {"name": "Mason Miller", "mlbam_id": 2, "ip": 60.0, "k": 90.0, "fg_id": "bbb"},
        ]
    )
    ranks = {
        "aaa::pitcher": {"rest_of_season": 5},
        "bbb::pitcher": {"rest_of_season": 1},
    }
    req = _request(_hitters([]), pitchers, rankings=ranks)

    players = build_manual_free_agents(req, positions_by_name={})

    # Two candidate rows collapse to one player: the pool de-duplicates on
    # (normalized name, player_type), and the matcher picks the real arm.
    assert len(players) == 1
    assert players[0].rest_of_season.ip == 60.0


def test_empty_frames_produce_an_empty_pool_without_raising():
    req = _request(_hitters([]), _pitchers([]), rankings=_ranks(("Nobody", PlayerType.HITTER)))
    assert build_manual_free_agents(req, positions_by_name={}) == []


def test_frame_without_name_norm_is_rejected_loudly():
    hitters = _hitters([{"name": "Some Bat"}]).drop(columns=["_name_norm"])
    req = _request(hitters, _pitchers([]), rankings=_ranks(("Some Bat", PlayerType.HITTER)))
    with pytest.raises(ValueError, match="_name_norm"):
        build_manual_free_agents(req, positions_by_name={"some bat": ["OF"]})
