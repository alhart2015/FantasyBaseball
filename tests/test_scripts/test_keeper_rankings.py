"""Unit tests for the pure pricing helpers in the keeper-rankings script.

`_slots_for` exists because a pitcher was being priced against a hitter floor.
"""

import pandas as pd
import pytest

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import HITTER_ELIGIBLE, Position
from fantasy_baseball.sgp.var import calculate_var
from scripts import keeper_rankings as module
from scripts.keeper_rankings import FALLBACK_POS, _dedupe_two_way, _slots_for

POSITIONS = {
    "shohei ohtani": ["UTIL"],
    "tarik skubal": ["P"],
    "ivan herrera": ["C", "UTIL"],
    "otto lopez": ["2B", "SS", "IF", "UTIL"],
    "someone hurt": ["OF", "IL"],
}


def test_a_util_only_pitcher_is_still_priced_as_a_pitcher():
    """Yahoo lists Ohtani as UTIL. In the pitcher pool that sent his pitching
    projection to the UTIL floor via calculate_var's hitter branch."""
    assert _slots_for(POSITIONS, "Shohei Ohtani", PlayerType.PITCHER) == FALLBACK_POS["pitcher"]


def test_a_pitcher_only_player_is_not_priced_as_a_hitter():
    assert _slots_for(POSITIONS, "Tarik Skubal", PlayerType.HITTER) == FALLBACK_POS["hitter"]


def test_a_hitter_keeps_only_his_batting_slots():
    assert _slots_for(POSITIONS, "Otto Lopez", PlayerType.HITTER) == ["2B", "SS", "IF", "UTIL"]


def test_a_bench_or_il_token_is_not_a_position_to_price_against():
    """An allowlist, not a denylist: IL is not in HITTER_ELIGIBLE."""
    assert _slots_for(POSITIONS, "Someone Hurt", PlayerType.HITTER) == ["OF"]


def test_an_unknown_player_falls_back_to_the_deepest_floor():
    assert _slots_for(POSITIONS, "Nobody At All", PlayerType.HITTER) == FALLBACK_POS["hitter"]


def _board(ids, names, proj_var):
    return pd.DataFrame({"name": names, "proj_var": proj_var}, index=pd.Index(ids, name="mlbam_id"))


def test_a_two_way_player_collapses_to_his_better_side():
    board = _board([660271, 660271, 592450], ["Ohtani", "Ohtani", "Judge"], [11.8, 7.8, 10.0])
    out = _dedupe_two_way(board)
    assert len(out) == 2
    assert out.loc[out["name"] == "Ohtani", "proj_var"].tolist() == [11.8]


def test_two_different_players_sharing_a_name_both_survive():
    """2022 had two Will Smiths, two Diego Castillos and two Luis Garcias. A
    name-keyed drop deletes a real rival, and probability_top_n then spreads the
    same slot mass over fewer people -- inflating everyone while the sum-to-slots
    check still passes."""
    board = _board([519293, 669257], ["Will Smith", "Will Smith"], [9.0, 8.0])
    assert len(_dedupe_two_way(board)) == 2


def test_dedupe_leaves_a_board_with_no_duplicates_alone():
    board = _board([1, 2, 3], ["A", "B", "C"], [3.0, 2.0, 1.0])
    assert _dedupe_two_way(board)["name"].tolist() == ["A", "B", "C"]


def test_dedupe_refuses_a_frame_that_lost_its_id_index():
    """The dedupe's correctness rests on the index being mlbam_id, and a synthetic
    index passes the tests above just as well -- so a reset_index upstream would
    silently restore the double-count with everything green."""
    board = pd.DataFrame({"name": ["A", "A"], "proj_var": [2.0, 1.0]})
    with pytest.raises(ValueError, match="mlbam_id"):
        _dedupe_two_way(board)


def test_every_real_hitter_slot_has_a_floor():
    """Drop or rename a key in `keepers.scarcity.NATIVE_CREDITS` and
    `calculate_var` silently falls back for that slot rather than raising, so every
    player eligible there is quietly repriced against UTIL with the suite green.
    DH and IF are excluded deliberately: they are Yahoo aggregates, not scarcity
    positions, and have no floor by design."""
    _, floors = module.pricing_table()
    real = set(HITTER_ELIGIBLE) - {Position.DH, Position.IF}
    missing = sorted(str(slot) for slot in real - set(floors))
    assert not missing, f"_slots_for can emit {missing}, which no floor prices"
    assert set(FALLBACK_POS["hitter"]) <= set(floors)


def test_an_aggregate_only_slot_is_charged_the_deepest_floor_not_zero():
    """IF and DH have no floor, so they must fall through to UTIL -- the harshest
    hitter floor -- rather than scoring as a free 0.0 credit, which would hand an
    unmapped slot the best price on the board."""
    _, floors = module.pricing_table()
    var, pos = calculate_var(
        pd.Series({"total_sgp": 0.0, "positions": [Position.IF], "ip": 0.0}), floors, True
    )
    assert pos == "UTIL"
    assert var == pytest.approx(-floors["UTIL"])
    assert var < 0.0


class _FakeKV:
    """Stands in for the live KV, keyed the way `redis_key` spells it."""

    def __init__(self, values):
        self.values = values
        self.asked = []

    def get(self, key):
        self.asked.append(key)
        return self.values.get(key)


def _fake_kv(monkeypatch, values):
    kv = _FakeKV(values)
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: kv)
    return kv


def test_the_league_loader_unions_my_roster_with_the_opponents(monkeypatch):
    """`cache:roster` holds only my team and `cache:opp_rosters` only the other
    nine, so a league report that read either alone would silently drop a team.
    Also pins the KEYS: reading one wrong would return None and lose a roster
    without raising."""
    kv = _fake_kv(
        monkeypatch,
        {
            "cache:roster": [{"name": "Juan Soto"}],
            "cache:opp_rosters": {"Rivals": [{"name": "Bobby Witt Jr."}]},
        },
    )
    rosters = module.load_league_rosters("Mine")
    assert rosters == {"Rivals": {"bobby witt jr."}, "Mine": {"juan soto"}}
    assert set(kv.asked) == {"cache:roster", "cache:opp_rosters"}


def test_an_unreachable_kv_yields_no_league_rather_than_raising(monkeypatch):
    """The rest of the script runs offline; only this report needs the network, so
    a dead KV has to degrade to an empty league instead of taking the run down."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", boom)
    assert module.load_league_rosters("Mine") == {}


def test_the_data_envelope_is_unwrapped_but_a_bare_payload_survives(monkeypatch):
    """A cache value arrives as a JSON string, a dict wrapped in `_data`, or the
    bare value. All three have to reach the caller as the same thing."""
    for stored, expected in (
        ('{"_data": [{"name": "A"}]}', [{"name": "A"}]),
        ({"_data": [{"name": "A"}]}, [{"name": "A"}]),
        ([{"name": "A"}], [{"name": "A"}]),
        ({"Team": [{"name": "A"}]}, {"Team": [{"name": "A"}]}),
        (None, None),
        ("", None),
    ):
        _fake_kv(monkeypatch, {"cache:roster": stored})
        assert module._kv_payload(module.CacheKey.ROSTER) == expected


def test_names_are_normalized_so_accents_join_the_position_map():
    assert module._normalized_names([{"name": "Julio Rodríguez"}]) == {"julio rodriguez"}
    assert module._normalized_names([{"name": ""}, {}]) == set()
    assert module._normalized_names("not a list") == set()
