import json
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.state import serialize_state, write_state, read_state


def _make_hitter(name, positions, var, best_position, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name,
        "positions": positions,
        "var": var,
        "best_position": best_position,
        "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb, "avg": avg,
        "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, positions, var, best_position, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name,
        "positions": positions,
        "var": var,
        "best_position": best_position,
        "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip,
        "ip": ip, "er": era * ip / 9,
        "bb": int(whip * ip * 0.3),
        "h_allowed": int(whip * ip * 0.7),
    })


@pytest.fixture
def sample_board():
    hitter = _make_hitter("Juan Soto", ["OF"], 12.5, "OF", 110, 35, 100, 10, .290, 550)
    pitcher = _make_pitcher("Gerrit Cole", ["SP"], 8.2, "P", 16, 250, 0, 2.80, 1.05, 200)
    return pd.DataFrame([hitter, pitcher])


@pytest.fixture
def sample_tracker():
    t = DraftTracker(num_teams=10, user_position=8, rounds=22)
    t.draft_player("Elly De La Cruz", is_user=False)
    t.draft_player("Juan Soto", is_user=True)
    return t


@pytest.fixture
def sample_balance(sample_board):
    bal = CategoryBalance()
    bal.add_player(sample_board.iloc[0])  # Soto (hitter)
    return bal


@pytest.fixture
def sample_recs():
    return [
        {
            "name": "Gerrit Cole",
            "var": 8.2,
            "best_position": "P",
            "positions": ["SP"],
            "need_flag": True,
            "note": "fills P need",
        },
    ]


class TestSerializeState:
    def test_returns_dict(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(
            tracker=sample_tracker,
            balance=sample_balance,
            board=sample_board,
            recommendations=sample_recs,
            filled_positions={"OF": 1},
        )
        assert isinstance(state, dict)

    def test_contains_pick_info(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        assert state["current_pick"] == sample_tracker.current_pick
        assert state["current_round"] == sample_tracker.current_round
        assert state["picking_team"] == sample_tracker.picking_team
        assert state["is_user_pick"] == sample_tracker.is_user_pick
        assert "picks_until_user_turn" in state

    def test_contains_rosters(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        assert state["user_roster"] == ["Juan Soto"]
        assert "Elly De La Cruz" in state["drafted_players"]
        assert "Juan Soto" in state["drafted_players"]

    def test_contains_recommendations(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        assert len(state["recommendations"]) == 1
        assert state["recommendations"][0]["name"] == "Gerrit Cole"

    def test_contains_balance(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        assert "totals" in state["balance"]
        assert "warnings" in state["balance"]
        assert state["balance"]["totals"]["HR"] == 35

    def test_available_players_excludes_drafted(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        names = [p["name"] for p in state["available_players"]]
        # Soto and De La Cruz are drafted; Soto is on the board but should be excluded
        # De La Cruz is not on the board so irrelevant
        # Only Cole should remain
        assert "Juan Soto" not in names
        assert "Gerrit Cole" in names

    def test_hitter_fields(self, sample_tracker, sample_balance, sample_board, sample_recs):
        # Add a non-drafted hitter to the board so it appears in available
        extra_hitter = _make_hitter("Pete Alonso", ["1B"], 5.0, "1B", 88, 35, 95, 2, .254, 520)
        board = pd.concat([sample_board, pd.DataFrame([extra_hitter])], ignore_index=True)
        state = serialize_state(sample_tracker, sample_balance, board, sample_recs, {})
        alonso = [p for p in state["available_players"] if p["name"] == "Pete Alonso"][0]
        assert alonso["player_type"] == "hitter"
        assert alonso["hr"] == 35
        assert alonso["r"] == 88
        assert "w" not in alonso

    def test_pitcher_fields(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        cole = [p for p in state["available_players"] if p["name"] == "Gerrit Cole"][0]
        assert cole["player_type"] == "pitcher"
        assert cole["k"] == 250
        assert cole["w"] == 16
        assert "hr" not in cole

    def test_filled_positions_passthrough(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {"OF": 2, "SS": 1})
        assert state["filled_positions"] == {"OF": 2, "SS": 1}

    def test_is_json_serializable(self, sample_tracker, sample_balance, sample_board, sample_recs):
        state = serialize_state(sample_tracker, sample_balance, sample_board, sample_recs, {})
        # Should not raise
        json.dumps(state)


class TestWriteState:
    def test_writes_valid_json(self, tmp_path):
        state = {"current_pick": 1, "user_roster": []}
        path = tmp_path / "draft_state.json"
        write_state(state, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == state

    def test_atomic_write_does_not_leave_tmp(self, tmp_path):
        state = {"current_pick": 1}
        path = tmp_path / "draft_state.json"
        write_state(state, path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "draft_state.json"
        write_state({"pick": 1}, path)
        write_state({"pick": 2}, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["pick"] == 2


class TestReadState:
    def test_reads_written_state(self, tmp_path):
        path = tmp_path / "draft_state.json"
        state = {"current_pick": 5, "user_roster": ["Soto"]}
        write_state(state, path)
        loaded = read_state(path)
        assert loaded == state

    def test_returns_empty_dict_on_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert read_state(path) == {}

    def test_returns_empty_dict_on_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json!!")
        assert read_state(path) == {}
