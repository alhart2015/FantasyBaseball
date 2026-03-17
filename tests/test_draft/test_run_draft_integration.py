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
