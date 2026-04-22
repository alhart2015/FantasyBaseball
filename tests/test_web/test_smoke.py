"""End-to-end smoke test: serialize state -> write file -> Flask reads it -> API returns it."""
import json

import pandas as pd

from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.recommender import Recommendation
from fantasy_baseball.draft.state import serialize_state, write_state
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.web.app import create_app


def _make_hitter(name, positions, var, best_position, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name,
        "positions": positions,
        "var": var,
        "best_position": best_position,
        "player_type": "hitter",
        "player_id": f"{name}::hitter",
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
        "player_id": f"{name}::pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip,
        "ip": ip, "er": era * ip / 9,
        "bb": int(whip * ip * 0.3),
        "h_allowed": int(whip * ip * 0.7),
    })


class TestEndToEndSmoke:
    """Full pipeline: draft objects -> serialize -> write file -> Flask API -> verify."""

    def test_full_pipeline(self, tmp_path):
        # 1. Set up draft objects
        tracker = DraftTracker(num_teams=10, user_position=8, rounds=22)
        balance = CategoryBalance()
        board = pd.DataFrame([
            _make_hitter("Juan Soto", ["OF"], 12.5, "OF", 110, 35, 100, 10, .290, 550),
            _make_hitter("Julio Rodriguez", ["OF"], 9.8, "OF", 95, 30, 90, 25, .275, 580),
            _make_pitcher("Gerrit Cole", ["SP"], 8.2, "P", 16, 250, 0, 2.80, 1.05, 200),
        ])

        # Simulate a pick: team 1 drafts Soto
        tracker.draft_player("Juan Soto", is_user=False, player_id="Juan Soto::hitter")

        # User has Julio (pretend)
        tracker.draft_player("Julio Rodriguez", is_user=True, player_id="Julio Rodriguez::hitter")
        balance.add_player(board.iloc[1])  # Julio

        recs = [Recommendation(
            name="Gerrit Cole",
            var=8.2,
            score=8.2,
            best_position="P",
            positions=["SP"],
            player_type=PlayerType.PITCHER,
            need_flag=True,
            note="fills P need",
        )]
        filled = {"OF": 1}

        # 2. Serialize
        state = serialize_state(tracker, balance, board, recs, filled)

        # 3. Write to file
        state_path = tmp_path / "draft_state.json"
        write_state(state, state_path)

        # 4. Create Flask app pointing at that file
        app = create_app(state_path=state_path)
        app.config["TESTING"] = True

        with app.test_client() as client:
            # 5. Hit the API
            resp = client.get("/api/state")
            assert resp.status_code == 200
            data = json.loads(resp.data)

            # 6. Verify key fields survived the full pipeline
            assert data["current_pick"] == tracker.current_pick
            assert data["current_round"] == tracker.current_round
            assert "Julio Rodriguez" in data["user_roster"]
            assert "Juan Soto" in data["drafted_players"]
            assert len(data["recommendations"]) == 1
            assert data["recommendations"][0]["name"] == "Gerrit Cole"
            assert data["balance"]["totals"]["HR"] == 30
            # Soto and Julio are drafted, only Cole should be available
            available_names = [p["name"] for p in data["available_players"]]
            assert "Gerrit Cole" in available_names
            assert "Juan Soto" not in available_names
            assert data["filled_positions"] == {"OF": 1}

            # 7. Dashboard HTML also works
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"Draft Dashboard" in resp.data
