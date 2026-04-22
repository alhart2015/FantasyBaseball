"""Tests for custom draft order with traded picks."""

import json
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.config import LeagueConfig, load_config
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.state import serialize_state
from fantasy_baseball.draft.tracker import DraftTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_with_strategy(tmp_path):
    """Config file with strategy and scoring_mode fields."""
    config_file = tmp_path / "league.yaml"
    config_file.write_text("""
league:
  id: 5652
  num_teams: 10
  game_code: mlb
  team_name: "Hart of the Order"

draft:
  strategy: two_closers
  scoring_mode: var
  position: 8
  teams:
    1: "Send in the Cavalli"
    2: "SkeleThor"
    3: "Work in Progress"
    4: "Jon's Underdogs"
    5: "Boston Estrellas"
    6: "Spacemen"
    7: "Springfield Isotopes"
    8: "Hart of the Order"
    9: "Tortured Baseball Department"
    10: "Hello Peanuts!"

keepers:
  - name: "Player A"
    team: "Hart of the Order"
  - name: "Player B"
    team: "SkeleThor"

roster_slots:
  C: 1
  1B: 1
  2B: 1
  3B: 1
  SS: 1
  IF: 1
  OF: 4
  UTIL: 2
  P: 9
  BN: 2
  IL: 2

projections:
  systems:
    - steamer
  weights:
    steamer: 1.0
""")
    return config_file


@pytest.fixture
def draft_order_file(tmp_path):
    """Draft order JSON with a traded pick."""
    order = {
        "description": "Test draft order",
        "trades": [
            {"round": 2, "slot": 1, "from": "SkeleThor", "to": "Hart of the Order"},
        ],
        "rounds": [
            # R1 (keeper round): standard order
            [
                "Send in the Cavalli",
                "SkeleThor",
                "Work in Progress",
                "Jon's Underdogs",
                "Boston Estrellas",
                "Spacemen",
                "Springfield Isotopes",
                "Hart of the Order",
                "Tortured Baseball Department",
                "Hello Peanuts!",
            ],
            # R2: Hart gets SkeleThor's pick (slot 1), SkeleThor has no pick
            [
                "Hart of the Order",
                "Work in Progress",
                "Jon's Underdogs",
                "Boston Estrellas",
                "Spacemen",
                "Springfield Isotopes",
                "Hart of the Order",
                "Send in the Cavalli",
                "Tortured Baseball Department",
                "Hello Peanuts!",
            ],
        ],
    }
    order_file = tmp_path / "draft_order.json"
    order_file.write_text(json.dumps(order))
    return order_file


@pytest.fixture
def small_draft_order_json():
    """Draft order data with 3 teams, 1 keeper round, 2 draft rounds, 1 trade."""
    return {
        "trades": [
            {"round": 2, "slot": 2, "from": "Team B", "to": "Team A"},
        ],
        "rounds": [
            # R1 (keepers)
            ["Team A", "Team B", "Team C"],
            # R2: Team A gets Team B's slot 2
            ["Team C", "Team A", "Team A"],
            # R3: standard
            ["Team A", "Team B", "Team C"],
        ],
    }


# ---------------------------------------------------------------------------
# Config strategy/scoring tests
# ---------------------------------------------------------------------------


class TestConfigStrategy:
    def test_strategy_loaded_from_config(self, config_with_strategy):
        config = load_config(config_with_strategy)
        assert config.strategy == "two_closers"

    def test_scoring_mode_loaded_from_config(self, config_with_strategy):
        config = load_config(config_with_strategy)
        assert config.scoring_mode == "var"

    def test_strategy_defaults(self, tmp_path):
        config_file = tmp_path / "league.yaml"
        config_file.write_text("""
league:
  id: 1
  num_teams: 10
  game_code: mlb
  team_name: "Test"
draft:
  position: 1
keepers: []
roster_slots:
  C: 1
projections:
  systems: [steamer]
  weights: {steamer: 1.0}
""")
        config = load_config(config_file)
        assert config.strategy == "no_punt_opp"
        assert config.scoring_mode == "var"


# ---------------------------------------------------------------------------
# Draft order loading tests (run_draft._load_draft_order)
# ---------------------------------------------------------------------------


class TestLoadDraftOrder:
    def test_load_draft_order_returns_picks(self, tmp_path, draft_order_file):
        """_load_draft_order returns a list of pick dicts."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_draft import _load_draft_order

        picks = _load_draft_order(draft_order_file, num_teams=10)
        assert picks is not None
        # 2 rounds x 10 picks = 20, minus 10 keepers = 10 post-keeper picks
        # But _load_draft_order returns ALL picks including keepers
        assert len(picks) == 20

    def test_load_draft_order_marks_trades(self, tmp_path, draft_order_file):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_draft import _load_draft_order

        picks = _load_draft_order(draft_order_file, num_teams=10)
        traded_picks = [p for p in picks if p["traded"]]
        assert len(traded_picks) == 1
        assert traded_picks[0]["round"] == 2
        assert traded_picks[0]["slot"] == 1
        assert traded_picks[0]["team"] == "Hart of the Order"
        assert traded_picks[0]["original_team"] == "SkeleThor"

    def test_load_draft_order_missing_file(self, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_draft import _load_draft_order

        result = _load_draft_order(tmp_path / "nonexistent.json", num_teams=10)
        assert result is None

    def test_load_draft_order_team_names(self, tmp_path, draft_order_file):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_draft import _load_draft_order

        picks = _load_draft_order(draft_order_file, num_teams=10)
        # R2 slot 1 should be Hart (traded from SkeleThor)
        r2_picks = [p for p in picks if p["round"] == 2]
        assert r2_picks[0]["team"] == "Hart of the Order"
        # R2 slot 7 should also be Hart (standard pick)
        assert r2_picks[6]["team"] == "Hart of the Order"


# ---------------------------------------------------------------------------
# Simulation pick order tests (simulate_draft._load_pick_order)
# ---------------------------------------------------------------------------


class TestSimulationPickOrder:
    def _make_config(self, teams, keepers, num_teams=3):
        return LeagueConfig(
            league_id=1,
            num_teams=num_teams,
            game_code="mlb",
            team_name="Team A",
            draft_position=1,
            keepers=keepers,
            roster_slots={"C": 1},
            projection_systems=["steamer"],
            projection_weights={"steamer": 1.0},
            teams=teams,
        )

    def test_pick_order_skips_keeper_rounds(self, tmp_path, small_draft_order_json):
        """Post-keeper pick order should not include keeper round picks."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

        order_file = tmp_path / "draft_order.json"
        order_file.write_text(json.dumps(small_draft_order_json))

        # Monkey-patch the path
        import simulate_draft

        orig_path = simulate_draft.DRAFT_ORDER_PATH
        simulate_draft.DRAFT_ORDER_PATH = order_file

        try:
            config = self._make_config(
                teams={1: "Team A", 2: "Team B", 3: "Team C"},
                keepers=[
                    {"name": "K1", "team": "Team A"},
                    {"name": "K2", "team": "Team B"},
                    {"name": "K3", "team": "Team C"},
                ],
            )
            pick_order = simulate_draft._load_pick_order(config)
            # 1 keeper round (3 keepers / 3 teams), so 2 post-keeper rounds
            assert len(pick_order) == 6  # 2 rounds x 3 teams
        finally:
            simulate_draft.DRAFT_ORDER_PATH = orig_path

    def test_pick_order_maps_team_names_to_numbers(self, tmp_path, small_draft_order_json):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import simulate_draft

        order_file = tmp_path / "draft_order.json"
        order_file.write_text(json.dumps(small_draft_order_json))

        orig_path = simulate_draft.DRAFT_ORDER_PATH
        simulate_draft.DRAFT_ORDER_PATH = order_file

        try:
            config = self._make_config(
                teams={1: "Team A", 2: "Team B", 3: "Team C"},
                keepers=[
                    {"name": "K1", "team": "Team A"},
                    {"name": "K2", "team": "Team B"},
                    {"name": "K3", "team": "Team C"},
                ],
            )
            pick_order = simulate_draft._load_pick_order(config)
            # R2: ["Team C", "Team A", "Team A"]
            assert pick_order[0] == 3  # Team C
            assert pick_order[1] == 1  # Team A (traded from Team B)
            assert pick_order[2] == 1  # Team A (standard)
            # R3: ["Team A", "Team B", "Team C"]
            assert pick_order[3] == 1  # Team A
            assert pick_order[4] == 2  # Team B
            assert pick_order[5] == 3  # Team C
        finally:
            simulate_draft.DRAFT_ORDER_PATH = orig_path

    def test_traded_pick_gives_extra_pick_to_user(self, tmp_path, small_draft_order_json):
        """User should appear more times when they have a traded pick."""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import simulate_draft

        order_file = tmp_path / "draft_order.json"
        order_file.write_text(json.dumps(small_draft_order_json))

        orig_path = simulate_draft.DRAFT_ORDER_PATH
        simulate_draft.DRAFT_ORDER_PATH = order_file

        try:
            config = self._make_config(
                teams={1: "Team A", 2: "Team B", 3: "Team C"},
                keepers=[
                    {"name": "K1", "team": "Team A"},
                    {"name": "K2", "team": "Team B"},
                    {"name": "K3", "team": "Team C"},
                ],
            )
            pick_order = simulate_draft._load_pick_order(config)
            # Team A has 3 picks (R2.2 traded + R2.3 standard + R3.1)
            # Team B has 1 pick (R3.2 only, lost R2.2)
            # Team C has 2 picks (R2.1 + R3.3)
            assert pick_order.count(1) == 3  # Team A
            assert pick_order.count(2) == 1  # Team B
            assert pick_order.count(3) == 2  # Team C
        finally:
            simulate_draft.DRAFT_ORDER_PATH = orig_path


# ---------------------------------------------------------------------------
# Serialize state with keeper offset tests
# ---------------------------------------------------------------------------


class TestSerializeStateKeepers:
    def _make_board(self):
        return pd.DataFrame(
            [
                {
                    "name": "P1",
                    "player_id": "p1",
                    "positions": ["OF"],
                    "var": 5.0,
                    "best_position": "OF",
                    "player_type": "hitter",
                    "total_sgp": 5.0,
                    "adp": 10,
                },
            ]
        )

    def test_pick_number_includes_keepers(self):
        tracker = DraftTracker(num_teams=10, user_position=8, rounds=20)
        balance = CategoryBalance()
        board = self._make_board()

        state = serialize_state(
            tracker=tracker,
            balance=balance,
            board=board,
            recommendations=[],
            filled_positions={},
            num_keepers=30,
        )
        # tracker.current_pick=1, plus 30 keepers = overall pick 31
        assert state["current_pick"] == 31

    def test_round_includes_keeper_rounds(self):
        tracker = DraftTracker(num_teams=10, user_position=8, rounds=20)
        balance = CategoryBalance()
        board = self._make_board()

        state = serialize_state(
            tracker=tracker,
            balance=balance,
            board=board,
            recommendations=[],
            filled_positions={},
            num_keepers=30,
        )
        # tracker.current_round=1, plus 3 keeper rounds = round 4
        assert state["current_round"] == 4

    def test_zero_keepers_unchanged(self):
        tracker = DraftTracker(num_teams=10, user_position=8, rounds=20)
        balance = CategoryBalance()
        board = self._make_board()

        state = serialize_state(
            tracker=tracker,
            balance=balance,
            board=board,
            recommendations=[],
            filled_positions={},
            num_keepers=0,
        )
        assert state["current_pick"] == 1
        assert state["current_round"] == 1

    def test_mid_draft_pick_number(self):
        tracker = DraftTracker(num_teams=10, user_position=8, rounds=20)
        # Advance to pick 28 (R3 in tracker = R6 overall)
        for _ in range(27):
            tracker.advance()
        balance = CategoryBalance()
        board = self._make_board()

        state = serialize_state(
            tracker=tracker,
            balance=balance,
            board=board,
            recommendations=[],
            filled_positions={},
            num_keepers=30,
        )
        # pick 28 + 30 keepers = 58
        assert state["current_pick"] == 58
        # round 3 + 3 keeper rounds = 6
        assert state["current_round"] == 6


# ---------------------------------------------------------------------------
# Strategy registration tests
# ---------------------------------------------------------------------------


class TestStrategyRegistration:
    def test_two_closers_registered(self):
        from fantasy_baseball.draft.strategy import STRATEGIES

        assert "two_closers" in STRATEGIES

    def test_four_closers_registered(self):
        from fantasy_baseball.draft.strategy import STRATEGIES

        assert "four_closers" in STRATEGIES

    def test_all_strategies_callable(self):
        from fantasy_baseball.draft.strategy import STRATEGIES

        for name, fn in STRATEGIES.items():
            assert callable(fn), f"Strategy '{name}' is not callable"

    def test_closer_threshold_consistent(self):
        from fantasy_baseball.draft.recommender import CLOSER_SV_THRESHOLD as rec_threshold
        from fantasy_baseball.draft.strategy import CLOSER_SV_THRESHOLD as strat_threshold
        from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD

        assert strat_threshold == CLOSER_SV_THRESHOLD == rec_threshold == 20


# ---------------------------------------------------------------------------
# Integration: real draft order file
# ---------------------------------------------------------------------------


class TestRealDraftOrder:
    """Tests against the actual config/draft_order.json."""

    @pytest.fixture
    def real_order(self):
        path = Path(__file__).resolve().parents[2] / "config" / "draft_order.json"
        if not path.exists():
            pytest.skip("config/draft_order.json not found")
        with open(path) as f:
            return json.load(f)

    def test_23_rounds(self, real_order):
        assert len(real_order["rounds"]) == 23

    def test_10_picks_per_round(self, real_order):
        for i, rnd in enumerate(real_order["rounds"]):
            assert len(rnd) == 10, f"Round {i + 1} has {len(rnd)} picks"

    def test_hart_has_20_post_keeper_picks(self, real_order):
        post_keeper = real_order["rounds"][3:]  # skip 3 keeper rounds
        hart_count = sum(1 for rnd in post_keeper for team in rnd if team == "Hart of the Order")
        assert hart_count == 20

    def test_hart_has_no_r18_pick(self, real_order):
        r18 = real_order["rounds"][17]  # 0-indexed
        assert "Hart of the Order" not in r18

    def test_hart_has_two_r5_picks(self, real_order):
        r5 = real_order["rounds"][4]  # 0-indexed
        hart_in_r5 = sum(1 for team in r5 if team == "Hart of the Order")
        assert hart_in_r5 == 2

    def test_six_traded_picks(self, real_order):
        assert len(real_order["trades"]) == 6

    def test_hart_gained_r5_from_tbd(self, real_order):
        hart_gains = [t for t in real_order["trades"] if t["to"] == "Hart of the Order"]
        assert len(hart_gains) == 1
        assert hart_gains[0]["round"] == 5
        assert hart_gains[0]["from"] == "Tortured Baseball Department"

    def test_hart_lost_r18_to_tbd(self, real_order):
        hart_losses = [t for t in real_order["trades"] if t["from"] == "Hart of the Order"]
        assert len(hart_losses) == 1
        assert hart_losses[0]["round"] == 18
        assert hart_losses[0]["to"] == "Tortured Baseball Department"

    def test_every_team_has_20_post_keeper_picks(self, real_order):
        """All teams should have exactly 20 draftable picks."""
        from collections import Counter

        post_keeper = real_order["rounds"][3:]
        counts = Counter()
        for rnd in post_keeper:
            for team in rnd:
                counts[team] += 1
        for team, count in counts.items():
            assert count == 20, f"{team} has {count} picks, expected 20"
