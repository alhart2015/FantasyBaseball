"""Tests for refresh_steps.py — pure helpers extracted from
run_full_refresh that are specific to the refresh orchestration
(not general enough to push into a domain module)."""

from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter
from fantasy_baseball.models.player import (
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
)
from fantasy_baseball.models.positions import Position
from fantasy_baseball.web.refresh_steps import (
    build_positions_map,
    compute_lineup_moves,
    merge_matched_and_raw_roster,
)

_UNSET = object()


def _player(
    name, player_type=PlayerType.HITTER, positions=_UNSET, selected_position=_UNSET, ros=None
):
    """Build a Player fixture.

    Accepts either ``HitterStats`` / ``PitcherStats`` for ``ros`` (the
    real stat-bag types in this codebase — there is no ``RosterStats``).
    Positions default to ``["OF"]`` for hitters and ``["SP"]`` for
    pitchers; pass an explicit list (including ``[]``) to override.
    ``selected_position`` defaults to the first position when not
    provided; pass ``None`` explicitly to leave it unset (required when
    ``positions=[]`` so we don't index into an empty list).
    """
    if positions is _UNSET:
        positions = ["OF"] if player_type == PlayerType.HITTER else ["SP"]
    if selected_position is _UNSET:
        selected_position = positions[0]
    p = Player(
        name=name,
        positions=positions,
        player_type=player_type,
        selected_position=selected_position,
        yahoo_id=f"{name}::{player_type.value}",
    )
    p.rest_of_season = ros
    return p


class TestMergeMatchedAndRawRoster:
    def test_matched_players_get_preseason_attached(self):
        soto = _player("Soto", ros=HitterStats(r=80))
        soto_pre = _player("Soto", ros=HitterStats(r=100, hr=35))
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[
                {
                    "name": "Soto",
                    "positions": ["OF"],
                    "selected_position": "OF",
                    "player_id": "1",
                    "status": "",
                }
            ],
            preseason_lookup={"soto": soto_pre},
        )
        assert len(result) == 1
        assert result[0].preseason is soto_pre.rest_of_season

    def test_matched_player_without_preseason_entry(self):
        soto = _player("Soto", ros=HitterStats(r=80))
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[
                {
                    "name": "Soto",
                    "positions": ["OF"],
                    "selected_position": "OF",
                    "player_id": "1",
                    "status": "",
                }
            ],
            preseason_lookup={},  # no preseason match
        )
        assert len(result) == 1
        # No preseason attached (attribute not set or stays as default)
        assert result[0].preseason is None

    def test_unmatched_raw_player_added_as_hitter(self):
        # Raw player not in matched list — should be added with
        # player_type inferred from positions (OF → HITTER).
        result = merge_matched_and_raw_roster(
            matched=[],
            roster_raw=[
                {
                    "name": "Newbie",
                    "positions": ["OF"],
                    "selected_position": "OF",
                    "player_id": "99",
                    "status": "",
                }
            ],
            preseason_lookup={},
        )
        assert len(result) == 1
        assert result[0].name == "Newbie"
        assert result[0].player_type == PlayerType.HITTER

    def test_unmatched_raw_player_added_as_pitcher(self):
        # SP positions → PITCHER
        result = merge_matched_and_raw_roster(
            matched=[],
            roster_raw=[
                {
                    "name": "RookiePitcher",
                    "positions": ["SP"],
                    "selected_position": "P",
                    "player_id": "100",
                    "status": "",
                }
            ],
            preseason_lookup={},
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.PITCHER

    def test_matched_player_skipped_in_raw_iteration(self):
        # When a player is in BOTH matched and raw, only one entry should
        # appear in the result (the matched one).
        soto = _player("Soto")
        result = merge_matched_and_raw_roster(
            matched=[soto],
            roster_raw=[
                {
                    "name": "Soto",
                    "positions": ["OF"],
                    "selected_position": "OF",
                    "player_id": "1",
                    "status": "",
                },
                {
                    "name": "Newbie",
                    "positions": ["OF"],
                    "selected_position": "BN",
                    "player_id": "99",
                    "status": "",
                },
            ],
            preseason_lookup={},
        )
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"Soto", "Newbie"}


class TestComputeLineupMoves:
    def test_no_moves_when_lineup_already_optimal(self):
        # All starters in active slots, all bench players already on bench.
        soto = _player("Soto", selected_position="OF", ros=HitterStats(sgp=10.0))
        nola = _player(
            "Nola",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=8.0),
        )
        result = compute_lineup_moves(
            optimal_hitters=[
                HitterAssignment(slot=Position.OF, name="Soto", player=soto, roto_delta=0.5)
            ],
            optimal_pitchers=[PitcherStarter(name="Nola", player=nola, roto_delta=0.4)],
            pitcher_bench=[],
            roster_players=[soto, nola],
        )
        assert result == {"swaps": [], "unpaired_starts": [], "unpaired_benches": []}

    def test_single_pitcher_swap_pairs_by_type(self):
        # Nola on BN, Strider on P; optimizer wants Nola active, Strider benched.
        nola = _player(
            "Nola",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=8.0),
        )
        strider = _player(
            "Strider",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=4.0),
        )
        result = compute_lineup_moves(
            optimal_hitters=[],
            optimal_pitchers=[PitcherStarter(name="Nola", player=nola, roto_delta=0.42)],
            pitcher_bench=[strider],
            roster_players=[nola, strider],
        )
        assert len(result["swaps"]) == 1
        swap = result["swaps"][0]
        assert swap["start"] == {"player": "Nola", "from": "BN", "to": "P", "roto_delta": 0.42}
        assert swap["bench"] == {"player": "Strider", "from": "P", "to": "BN"}
        assert result["unpaired_starts"] == []
        assert result["unpaired_benches"] == []

    def test_hitter_swap_pairs_by_exact_slot(self):
        # Judge on BN (OF-eligible) replacing Acuna at OF — same base slot,
        # pass 1 should pair them.
        acuna = _player("Acuna", positions=["OF"], selected_position="OF", ros=HitterStats(sgp=6.0))
        judge = _player(
            "Judge", positions=["OF"], selected_position="BN", ros=HitterStats(sgp=11.0)
        )
        soto = _player("Soto", positions=["OF"], selected_position="OF", ros=HitterStats(sgp=12.0))
        result = compute_lineup_moves(
            optimal_hitters=[
                HitterAssignment(slot=Position.OF, name="Soto", player=soto, roto_delta=0.6),
                HitterAssignment(slot=Position.OF, name="Judge", player=judge, roto_delta=0.55),
            ],
            optimal_pitchers=[],
            pitcher_bench=[],
            roster_players=[acuna, judge, soto],
        )
        assert len(result["swaps"]) == 1
        swap = result["swaps"][0]
        assert swap["start"]["player"] == "Judge"
        assert swap["start"]["from"] == "BN"
        assert swap["start"]["to"] == "OF"
        assert swap["start"]["roto_delta"] == 0.55
        assert swap["bench"]["player"] == "Acuna"
        assert swap["bench"]["from"] == "OF"
        assert swap["bench"]["to"] == "BN"

    def test_pitcher_swap_pairs_by_descending_roto_then_sgp(self):
        # Two starts (high-ΔRoto + low-ΔRoto) and two benches (high-SGP +
        # low-SGP) should pair high-with-high, low-with-low — pass 2 ordering.
        nola = _player(
            "Nola",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=8.0),
        )
        skenes = _player(
            "Skenes",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=2.0),
        )
        strider = _player(
            "Strider",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=4.0),
        )
        gausman = _player(
            "Gausman",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=1.0),
        )
        # Inputs are deliberately ordered AGAINST the expected pairing
        # (low-ΔRoto start first, low-SGP bench first) so the assertion
        # only holds because _pair_swaps pre-sorts both lists.
        result = compute_lineup_moves(
            optimal_hitters=[],
            optimal_pitchers=[
                PitcherStarter(name="Skenes", player=skenes, roto_delta=0.10),
                PitcherStarter(name="Nola", player=nola, roto_delta=0.50),
            ],
            pitcher_bench=[gausman, strider],
            roster_players=[nola, skenes, strider, gausman],
        )
        assert len(result["swaps"]) == 2
        # Highest ΔRoto start (Nola) pairs with highest SGP bench (Strider, sgp=4)
        # Lowest ΔRoto start (Skenes) pairs with lowest SGP bench (Gausman, sgp=1)
        nola_swap = next(s for s in result["swaps"] if s["start"]["player"] == "Nola")
        skenes_swap = next(s for s in result["swaps"] if s["start"]["player"] == "Skenes")
        assert nola_swap["bench"]["player"] == "Strider"
        assert skenes_swap["bench"]["player"] == "Gausman"

    def test_mixed_hitter_and_pitcher_swaps_dont_cross_types(self):
        judge = _player(
            "Judge", positions=["OF"], selected_position="BN", ros=HitterStats(sgp=11.0)
        )
        acuna = _player("Acuna", positions=["OF"], selected_position="OF", ros=HitterStats(sgp=6.0))
        nola = _player(
            "Nola",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=8.0),
        )
        strider = _player(
            "Strider",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=4.0),
        )
        result = compute_lineup_moves(
            optimal_hitters=[
                HitterAssignment(slot=Position.OF, name="Judge", player=judge, roto_delta=0.55),
            ],
            optimal_pitchers=[
                PitcherStarter(name="Nola", player=nola, roto_delta=0.42),
            ],
            pitcher_bench=[strider],
            roster_players=[judge, acuna, nola, strider],
        )
        assert len(result["swaps"]) == 2
        for swap in result["swaps"]:
            if swap["start"]["player"] == "Judge":
                assert swap["bench"]["player"] == "Acuna"
            else:
                assert swap["start"]["player"] == "Nola"
                assert swap["bench"]["player"] == "Strider"

    def test_asymmetric_more_starts_than_benches_emits_unpaired(self):
        # Player just returned from IL, opening a 2nd P slot; only 1 of the
        # 2 currently-active pitchers should be benched, but both bench
        # pitchers should be activated → 2 starts, 1 bench → 1 swap + 1
        # unpaired_start.
        nola = _player(
            "Nola",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=8.0),
        )
        skenes = _player(
            "Skenes",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=7.0),
        )
        strider = _player(
            "Strider",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=4.0),
        )
        # Inputs put the LOWER-ΔRoto start (Skenes) first; without the
        # pre-sort, pass 1 would greedily pair Skenes with Strider and
        # leave Nola unpaired. The assertions below only hold because
        # _pair_swaps sorts starts by descending roto_delta first.
        result = compute_lineup_moves(
            optimal_hitters=[],
            optimal_pitchers=[
                PitcherStarter(name="Skenes", player=skenes, roto_delta=0.30),
                PitcherStarter(name="Nola", player=nola, roto_delta=0.50),
            ],
            pitcher_bench=[strider],
            roster_players=[nola, skenes, strider],
        )
        # Higher ΔRoto start pairs with the bench; lower ΔRoto goes unpaired.
        assert len(result["swaps"]) == 1
        assert result["swaps"][0]["start"]["player"] == "Nola"
        assert result["swaps"][0]["bench"]["player"] == "Strider"
        assert len(result["unpaired_starts"]) == 1
        assert result["unpaired_starts"][0]["player"] == "Skenes"
        assert result["unpaired_starts"][0]["roto_delta"] == 0.30
        assert result["unpaired_benches"] == []

    def test_partial_counterfactual_zero_roto_delta_still_renders(self):
        # If optimizer returns roto_delta=0.0 (partial counterfactual case
        # in optimizer.py), the value still propagates as 0.0 — not None.
        nola = _player(
            "Nola",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="BN",
            ros=PitcherStats(sgp=8.0),
        )
        strider = _player(
            "Strider",
            player_type=PlayerType.PITCHER,
            positions=["SP", "P"],
            selected_position="P",
            ros=PitcherStats(sgp=4.0),
        )
        result = compute_lineup_moves(
            optimal_hitters=[],
            optimal_pitchers=[PitcherStarter(name="Nola", player=nola, roto_delta=0.0)],
            pitcher_bench=[strider],
            roster_players=[nola, strider],
        )
        assert result["swaps"][0]["start"]["roto_delta"] == 0.0

    def test_il_player_in_optimal_treated_as_bench_crossing(self):
        # Player on IL shouldn't normally appear in optimal_pitchers, but
        # if a roster has IL-but-now-active state during transition, we
        # still surface them.
        soto = _player("Soto", selected_position="IL", ros=HitterStats(sgp=12.0))
        result = compute_lineup_moves(
            optimal_hitters=[
                HitterAssignment(slot=Position.OF, name="Soto", player=soto, roto_delta=0.6),
            ],
            optimal_pitchers=[],
            pitcher_bench=[],
            roster_players=[soto],
        )
        # No bench partner → unpaired start.
        assert result["swaps"] == []
        assert len(result["unpaired_starts"]) == 1
        assert result["unpaired_starts"][0]["player"] == "Soto"
        assert result["unpaired_starts"][0]["from"] == "IL"
        assert result["unpaired_starts"][0]["to"] == "OF"


class TestBuildPositionsMap:
    def test_includes_roster_players(self):
        roster = [_player("Soto", positions=["OF"])]
        result = build_positions_map(roster, opp_rosters={}, fa_players=[])
        assert result["soto"] == ["OF"]

    def test_includes_opponent_players(self):
        opp = {"OtherTeam": [_player("Trout", positions=["OF", "Util"])]}
        result = build_positions_map([], opp_rosters=opp, fa_players=[])
        assert result["trout"] == ["OF", "Util"]

    def test_includes_free_agents(self):
        fas = [_player("Acuna", positions=["OF"])]
        result = build_positions_map([], opp_rosters={}, fa_players=fas)
        assert result["acuna"] == ["OF"]

    def test_free_agent_with_empty_positions_skipped(self):
        # FAs with no positions data shouldn't pollute the map
        fa = _player("Mystery", positions=[], selected_position=None)
        result = build_positions_map([], opp_rosters={}, fa_players=[fa])
        assert "mystery" not in result

    def test_normalizes_keys(self):
        # Accents and case should be normalized
        roster = [_player("José Ramírez", positions=["3B"])]
        result = build_positions_map(roster, opp_rosters={}, fa_players=[])
        # normalize_name strips accents and lowercases
        assert "jose ramirez" in result

    def test_combines_all_three_sources(self):
        roster = [_player("A", positions=["OF"])]
        opp = {"T2": [_player("B", positions=["1B"])]}
        fas = [_player("C", positions=["SS"])]
        result = build_positions_map(roster, opp, fas)
        assert set(result.keys()) == {"a", "b", "c"}
