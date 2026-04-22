"""Tests for refresh_steps.py — pure helpers extracted from
run_full_refresh that are specific to the refresh orchestration
(not general enough to push into a domain module)."""

from fantasy_baseball.models.player import (
    HitterStats,
    Player,
    PlayerType,
)
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
    def test_bench_to_starter_emits_start_move(self):
        # Player on BN; optimizer wants them at OF
        ros = HitterStats(sgp=12.5)
        p = _player("Soto", selected_position="BN", ros=ros)
        optimal = {"OF_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1
        assert moves[0]["action"] == "START"
        assert moves[0]["player"] == "Soto"
        assert moves[0]["slot"] == "OF"
        assert "12.5" in moves[0]["reason"]

    def test_starter_to_starter_emits_no_move(self):
        # Player already at OF; optimizer keeps them at OF — no move
        p = _player("Soto", selected_position="OF")
        optimal = {"OF_1": "Soto"}
        assert compute_lineup_moves(optimal, [p]) == []

    def test_il_to_starter_emits_start_move(self):
        # IL counts as bench-like
        p = _player("Soto", selected_position="IL")
        optimal = {"OF_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1
        assert moves[0]["action"] == "START"

    def test_starter_to_bench_emits_start_move(self):
        # Optimizer demoting a starter to bench also counts
        # (loop only iterates optimal slots, so this case fires when
        # the same player appears in optimal under a BN_x slot).
        p = _player("Soto", selected_position="OF")
        optimal = {"BN_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1
        assert moves[0]["slot"] == "BN"

    def test_player_not_on_roster_skipped(self):
        # Defensive: optimizer references a name not in roster_players
        p = _player("Other", selected_position="OF")
        optimal = {"OF_1": "Ghost"}
        assert compute_lineup_moves(optimal, [p]) == []

    def test_player_with_no_selected_position_treated_as_bench(self):
        # selected_position is None → falls back to "BN"
        p = _player("Soto", selected_position=None)
        # With no current slot and optimizer wanting OF, it's bench→starter
        optimal = {"OF_1": "Soto"}
        moves = compute_lineup_moves(optimal, [p])
        assert len(moves) == 1

    def test_slot_suffix_stripped(self):
        # OF_1 vs OF_2 — both should be treated as OF
        p = _player("Soto", selected_position="OF")
        optimal = {"OF_2": "Soto"}
        # Current is OF, target is OF (after stripping _2) → no move
        assert compute_lineup_moves(optimal, [p]) == []


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
