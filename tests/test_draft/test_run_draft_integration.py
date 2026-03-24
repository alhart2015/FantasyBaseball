import json
import threading
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fantasy_baseball.draft.state import read_state, write_state


class TestWriteStateIntegration:
    """Verify that write_state produces a file readable by the Flask API."""

    def test_state_file_roundtrip_via_read(self, tmp_path):
        state = {
            "current_pick": 10,
            "current_round": 1,
            "picking_team": 10,
            "is_user_pick": False,
            "picks_until_user_turn": 2,
            "user_roster": ["Juan Soto"],
            "drafted_players": ["Juan Soto", "Shohei Ohtani"],
            "recommendations": [],
            "balance": {"totals": {"R": 110}, "warnings": []},
            "available_players": [],
            "filled_positions": {},
        }
        path = tmp_path / "draft_state.json"
        write_state(state, path)
        loaded = read_state(path)
        assert loaded["current_pick"] == 10
        assert loaded["user_roster"] == ["Juan Soto"]

    def test_concurrent_reads_do_not_crash(self, tmp_path):
        """Simulate Flask thread reading while CLI writes."""
        path = tmp_path / "draft_state.json"
        errors = []

        def writer():
            for i in range(20):
                write_state({"current_pick": i}, path)

        def reader():
            for _ in range(20):
                try:
                    result = read_state(path)
                    # Should always be a dict (possibly empty)
                    assert isinstance(result, dict)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == [], f"Concurrent read errors: {errors}"


class TestFlaskBackgroundThread:
    """Verify the Flask server can start in a background thread."""

    def test_flask_app_starts_in_daemon_thread(self, tmp_path):
        from fantasy_baseball.web.app import create_app

        state_path = tmp_path / "draft_state.json"
        write_state({"current_pick": 1}, state_path)
        app = create_app(state_path=state_path)

        # Start in background thread (like run_draft.py will)
        server_thread = threading.Thread(
            target=lambda: app.run(port=5099, use_reloader=False),
            daemon=True,
        )
        server_thread.start()

        # Give server a moment to start, then test it's alive
        import time
        time.sleep(0.5)
        assert server_thread.is_alive()
        # Thread is daemon, will be cleaned up when test exits


# ---------------------------------------------------------------------------
# Import helpers for the simulation roundtrip test
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from simulate_draft import build_board_and_context, run_simulation


class TestDraftSimulationRoundtrip:
    """End-to-end integration test: build a real board, run a full draft,
    and verify key invariants of the resulting state."""

    @pytest.fixture(scope="class")
    def sim_result(self):
        """Run the simulation once and share the result across all tests."""
        ctx = build_board_and_context()
        result = run_simulation(
            ctx,
            strategy_name="default",
            scoring_mode="var",
            seed=42,
        )
        # Attach context so tests can inspect config, board, etc.
        result["ctx"] = ctx
        return result

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _expected_counts(config):
        """Derive expected totals from config."""
        num_keepers = len(config.keepers)
        user_keepers = sum(
            1 for k in config.keepers if k.get("team") == config.team_name
        )
        draftable_slots = sum(
            v for k, v in config.roster_slots.items() if k != "IL"
        )
        rounds = draftable_slots - user_keepers
        total_draft_picks = config.num_teams * rounds
        total_drafted = total_draft_picks + num_keepers
        user_roster_size = rounds + user_keepers
        return {
            "num_keepers": num_keepers,
            "rounds": rounds,
            "total_draft_picks": total_draft_picks,
            "total_drafted": total_drafted,
            "user_roster_size": user_roster_size,
        }

    # 1. No double-drafted players
    def test_no_double_drafted_players(self, sim_result):
        tracker = sim_result["tracker"]
        ids = tracker.drafted_ids
        assert len(ids) == len(set(ids)), (
            f"Duplicate player_ids found: "
            f"{[pid for pid in ids if ids.count(pid) > 1]}"
        )

    # 2. All roster slots legal
    def test_all_roster_slots_legal(self, sim_result):
        config = sim_result["config"]
        team_players = sim_result["team_players"]
        roster_slots = config.roster_slots

        for team_num, players in team_players.items():
            total_allowed = sum(
                v for k, v in roster_slots.items() if k != "IL"
            )
            assert len(players) <= total_allowed, (
                f"Team {team_num} has {len(players)} players but only "
                f"{total_allowed} draftable slots"
            )

    # 3. Standings sum correctly
    def test_standings_sum_correctly(self, sim_result):
        config = sim_result["config"]
        results = sim_result["results"]
        n = config.num_teams
        expected_per_cat = n * (n + 1) / 2  # 1+2+...+10 = 55
        all_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]

        for cat in all_cats:
            total = sum(t[f"{cat}_p"] for t in results)
            assert abs(total - expected_per_cat) < 0.01, (
                f"Category {cat} roto points sum to {total}, "
                f"expected {expected_per_cat}"
            )

        grand_total = sum(t["tot"] for t in results)
        expected_total = expected_per_cat * len(all_cats)  # 550
        assert abs(grand_total - expected_total) < 0.1, (
            f"Grand total roto points = {grand_total}, expected {expected_total}"
        )

    # 4. All picks assigned
    def test_all_picks_assigned(self, sim_result):
        config = sim_result["config"]
        tracker = sim_result["tracker"]
        counts = self._expected_counts(config)
        assert len(tracker.drafted_ids) == counts["total_drafted"], (
            f"Expected {counts['total_drafted']} total drafted entries "
            f"({counts['num_keepers']} keepers + {counts['total_draft_picks']} "
            f"picks), got {len(tracker.drafted_ids)}"
        )

    # 5. User roster is correct size
    def test_user_roster_correct_size(self, sim_result):
        config = sim_result["config"]
        counts = self._expected_counts(config)
        assert len(sim_result["user_roster"]) == counts["user_roster_size"], (
            f"User roster has {len(sim_result['user_roster'])} players, "
            f"expected {counts['user_roster_size']}"
        )

    # 6. Results include all teams
    def test_results_include_all_teams(self, sim_result):
        config = sim_result["config"]
        results = sim_result["results"]
        teams_in_standings = {t["team"] for t in results}
        expected_teams = set(config.teams.values())
        assert teams_in_standings == expected_teams, (
            f"Missing teams in standings: {expected_teams - teams_in_standings}"
        )
