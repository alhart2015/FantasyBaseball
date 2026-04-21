"""Integration tests for the draft board and draft strategy pipelines.

These tests use the ACTUAL projection files in data/projections/ and the
ACTUAL config/league.yaml to exercise the full draft pipeline end-to-end.
"""
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from fantasy_baseball.config import LeagueConfig, load_config
from fantasy_baseball.data.db import (
    create_tables,
    get_connection,
    load_blended_projections,
    load_positions,
)
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.board import apply_keepers, build_draft_board
from fantasy_baseball.draft.recommender import get_recommendations
from fantasy_baseball.draft.strategy import (
    STRATEGIES,
    build_player_lookup,
)
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

# ---------------------------------------------------------------------------
# Paths — resolved once, reused across fixtures
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "league.yaml"
_PROJECTIONS_DIR = _PROJECT_ROOT / "data" / "projections"
_POSITIONS_PATH = _PROJECT_ROOT / "data" / "player_positions.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> LeagueConfig:
    """Load the real league config."""
    return load_config(_CONFIG_PATH)


@pytest.fixture(scope="module")
def full_board(config: LeagueConfig) -> pd.DataFrame:
    """Build a draft board from real projections via SQLite."""
    conn = get_connection(":memory:")
    create_tables(conn)

    load_blended_projections(conn, _PROJECTIONS_DIR, config.projection_systems, config.projection_weights)

    if _POSITIONS_PATH.exists():
        positions = load_positions_cache(_POSITIONS_PATH)
        load_positions(conn, positions)

    board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides,
        roster_slots=config.roster_slots,
        num_teams=config.num_teams,
    )
    conn.close()
    return board


@pytest.fixture(scope="module")
def board_after_keepers(full_board: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    """Board with keepers removed."""
    return apply_keepers(full_board, config.keepers)


def _make_config_for_strategy() -> LeagueConfig:
    """Load real config for strategy tests (function-scoped helper)."""
    return load_config(_CONFIG_PATH)


def _simulate_n_picks(
    board: pd.DataFrame,
    strategy_name: str,
    config: LeagueConfig,
    n_picks: int,
    scoring_mode: str = "var",
) -> list[dict]:
    """Simulate n user picks using a given strategy, returning picked players.

    Opponents draft by ADP between user picks (simple simulation).
    """
    tracker = DraftTracker(
        num_teams=config.num_teams,
        user_position=config.draft_position,
        rounds=22,
    )
    balance = CategoryBalance()
    strategy_fn = STRATEGIES[strategy_name]
    picks = []

    # Pre-draft keepers: mark all keeper players as drafted by their teams
    for keeper in config.keepers:
        keeper_rows = board[
            board["name"].str.lower() == keeper["name"].lower()
        ]
        if not keeper_rows.empty:
            row = keeper_rows.iloc[0]
            pid = row["player_id"]
            is_user_keeper = keeper.get("team", "") == config.team_name
            tracker.draft_player(
                row["name"], is_user=is_user_keeper, player_id=pid,
            )
            if is_user_keeper:
                balance.add_player(row)
            # Advance past keeper slot
            tracker.advance()

    user_picks_made = 0
    max_iterations = 500  # safety valve

    while user_picks_made < n_picks and max_iterations > 0:
        max_iterations -= 1

        if tracker.current_pick > tracker.total_picks:
            break

        if tracker.is_user_pick:
            # Build player lookup for strategy
            player_lookup = build_player_lookup(board, board)
            name, pid = strategy_fn(
                board=board,
                full_board=board,
                tracker=tracker,
                balance=balance,
                config=config,
                team_filled={},
                player_lookup=player_lookup,
                scoring_mode=scoring_mode,
                total_rounds=22,
            )
            if name is None:
                break

            # Look up the player row
            rows = board[board["player_id"] == pid]
            if rows.empty:
                rows = board[board["name"] == name]
            if not rows.empty:
                player_row = rows.iloc[0]
                tracker.draft_player(name, is_user=True, player_id=pid)
                balance.add_player(player_row)
                picks.append({
                    "name": name,
                    "player_id": pid,
                    "player_type": player_row["player_type"],
                    "sv": player_row.get("sv", 0),
                    "positions": player_row.get("positions", []),
                    "best_position": player_row.get("best_position", ""),
                    "ip": player_row.get("ip", 0),
                })
                user_picks_made += 1
            tracker.advance()
        else:
            # Opponent pick: draft by ADP from remaining pool
            available = board[~board["player_id"].isin(tracker.drafted_ids)]
            if "adp" in available.columns:
                available = available.sort_values("adp", ascending=True)
            else:
                available = available.sort_values("var", ascending=False)
            if not available.empty:
                opp_pick = available.iloc[0]
                tracker.draft_player(
                    opp_pick["name"], is_user=False,
                    player_id=opp_pick["player_id"],
                )
            tracker.advance()

    return picks


# ===========================================================================
# DRAFT BOARD TESTS
# ===========================================================================


class TestBoardBuildsFromRealProjections:
    """build_draft_board with actual projection files produces a valid board."""

    def test_board_is_non_empty(self, full_board: pd.DataFrame):
        assert len(full_board) > 100, (
            f"Board should have hundreds of players, got {len(full_board)}"
        )

    def test_board_has_player_id(self, full_board: pd.DataFrame):
        assert "player_id" in full_board.columns

    def test_board_has_var(self, full_board: pd.DataFrame):
        assert "var" in full_board.columns

    def test_board_has_total_sgp(self, full_board: pd.DataFrame):
        assert "total_sgp" in full_board.columns

    def test_board_has_name(self, full_board: pd.DataFrame):
        assert "name" in full_board.columns

    def test_board_has_positions(self, full_board: pd.DataFrame):
        assert "positions" in full_board.columns

    def test_board_has_best_position(self, full_board: pd.DataFrame):
        assert "best_position" in full_board.columns


class TestKeepersRemovedFromBoard:
    """After apply_keepers, keeper players should not appear in the board."""

    def test_keepers_not_in_filtered_board(
        self, full_board: pd.DataFrame, board_after_keepers: pd.DataFrame,
        config: LeagueConfig,
    ):
        # apply_keepers removes the highest-VAR entry per keeper name.
        # Players with both hitter and pitcher entries (e.g. Ohtani) will
        # have one entry removed; the other legitimately remains on the
        # board.  So we check that each keeper lost at least one entry.
        for keeper in config.keepers:
            name_lower = keeper["name"].lower()
            entries_before = len(
                full_board[full_board["name"].str.lower() == name_lower]
            )
            entries_after = len(
                board_after_keepers[
                    board_after_keepers["name"].str.lower() == name_lower
                ]
            )
            if entries_before > 0:
                assert entries_after < entries_before, (
                    f"Keeper '{keeper['name']}' was not removed: "
                    f"{entries_before} entries before, {entries_after} after"
                )

    def test_board_shrunk_by_keeper_count(
        self, full_board: pd.DataFrame, board_after_keepers: pd.DataFrame,
        config: LeagueConfig,
    ):
        # Some keepers might not be in projections; count how many actually matched
        removed = len(full_board) - len(board_after_keepers)
        assert removed > 0, "No keepers were removed from the board"
        assert removed <= len(config.keepers), (
            f"Removed {removed} players but only {len(config.keepers)} keepers configured"
        )


class TestVarIsPositiveForTopPlayers:
    """The top 50 players by VAR should all have VAR > 0."""

    def test_top_50_var_positive(self, full_board: pd.DataFrame):
        top50 = full_board.nlargest(50, "var")
        assert len(top50) == 50, "Board has fewer than 50 players"
        negatives = top50[top50["var"] <= 0]
        assert negatives.empty, (
            f"{len(negatives)} of the top 50 have VAR <= 0: "
            f"{negatives[['name', 'var']].to_string()}"
        )


class TestNoDuplicatePlayerIds:
    """Every player_id in the board should be unique."""

    def test_unique_player_ids(self, full_board: pd.DataFrame):
        duplicated = full_board[full_board["player_id"].duplicated(keep=False)]
        assert duplicated.empty, (
            f"Found {len(duplicated)} duplicated player_ids: "
            f"{duplicated[['name', 'player_id']].drop_duplicates().to_string()}"
        )


# ===========================================================================
# DRAFT STRATEGY TESTS
# ===========================================================================


class TestAllStrategiesRegistered:
    """Every strategy name in STRATEGIES dict should map to a callable."""

    EXPECTED_STRATEGIES: ClassVar[list[str]] = [
        "default", "nonzero_sv", "avg_hedge", "two_closers",
        "three_closers", "four_closers", "no_punt", "no_punt_opp",
        "no_punt_stagger", "no_punt_cap3", "avg_anchor", "closers_avg",
        "balanced", "anti_fragile",
    ]

    def test_all_expected_strategies_present(self):
        for name in self.EXPECTED_STRATEGIES:
            assert name in STRATEGIES, f"Strategy '{name}' not found in STRATEGIES"

    def test_all_strategies_are_callable(self):
        for name, fn in STRATEGIES.items():
            assert callable(fn), f"Strategy '{name}' is not callable: {type(fn)}"


class TestDefaultStrategyProducesValidPick:
    """Running recommend_pick with default strategy on a fresh board returns a
    player that exists on the board."""

    def test_default_pick_exists_on_board(
        self, board_after_keepers: pd.DataFrame, config: LeagueConfig,
    ):
        picks = _simulate_n_picks(
            board=board_after_keepers,
            strategy_name="default",
            config=config,
            n_picks=1,
        )
        assert len(picks) == 1, "Default strategy failed to produce a pick"
        pid = picks[0]["player_id"]
        assert pid in board_after_keepers["player_id"].values, (
            f"Picked player_id '{pid}' not found on board"
        )

    def test_default_pick_is_a_real_player(
        self, board_after_keepers: pd.DataFrame, config: LeagueConfig,
    ):
        picks = _simulate_n_picks(
            board=board_after_keepers,
            strategy_name="default",
            config=config,
            n_picks=1,
        )
        name = picks[0]["name"]
        assert name and name != "unknown", f"Default strategy returned bad name: {name}"


class TestStrategyDoesNotDraftFiveSPInFiveRounds:
    """No strategy should draft 5 SPs in 5 picks (pitcher overvaluation guard)."""

    @pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
    def test_not_five_sp_in_five_picks(
        self, board_after_keepers: pd.DataFrame, config: LeagueConfig,
        strategy_name: str,
    ):
        picks = _simulate_n_picks(
            board=board_after_keepers,
            strategy_name=strategy_name,
            config=config,
            n_picks=5,
        )
        sp_count = sum(
            1 for p in picks
            if p["player_type"] == "pitcher"
            and p.get("sv", 0) < CLOSER_SV_THRESHOLD
        )
        assert sp_count < 5, (
            f"Strategy '{strategy_name}' drafted {sp_count} SPs in 5 picks: "
            f"{[p['name'] for p in picks]}"
        )


class TestTwoClosersStrategyDraftsClosers:
    """The two_closers strategy should draft at least 1 closer in the first 15 picks."""

    def test_at_least_one_closer_in_fifteen_picks(
        self, board_after_keepers: pd.DataFrame, config: LeagueConfig,
    ):
        picks = _simulate_n_picks(
            board=board_after_keepers,
            strategy_name="two_closers",
            config=config,
            n_picks=15,
        )
        closers = [p for p in picks if p.get("sv", 0) >= CLOSER_SV_THRESHOLD]
        assert len(closers) >= 1, (
            f"two_closers strategy drafted 0 closers in {len(picks)} picks: "
            f"{[p['name'] for p in picks]}"
        )


class TestVonaModeProducesDifferentRankingThanVar:
    """VONA scoring should produce a different #1 recommendation than VAR
    for at least some board states."""

    def test_vona_differs_from_var(
        self, board_after_keepers: pd.DataFrame, config: LeagueConfig,
    ):
        # Test across several board states: fresh, and after some drafting
        board = board_after_keepers
        found_difference = False

        for n_drafted in [0, 30, 60, 90]:
            # Create a set of "drafted" player IDs (top-N by ADP)
            if n_drafted > 0 and "adp" in board.columns:
                drafted_ids = list(
                    board.nsmallest(n_drafted, "adp")["player_id"]
                )
            else:
                drafted_ids = []

            var_recs = get_recommendations(
                board, drafted=drafted_ids, user_roster=[], n=1,
                roster_slots=config.roster_slots,
                num_teams=config.num_teams,
                scoring_mode="var",
            )
            vona_recs = get_recommendations(
                board, drafted=drafted_ids, user_roster=[], n=1,
                roster_slots=config.roster_slots,
                num_teams=config.num_teams,
                scoring_mode="vona",
                picks_until_next=12,
            )

            if var_recs and vona_recs and var_recs[0]["name"] != vona_recs[0]["name"]:
                found_difference = True
                break

        assert found_difference, (
            "VONA and VAR produced the same #1 recommendation across all "
            "tested board states -- they should diverge in at least one state"
        )
