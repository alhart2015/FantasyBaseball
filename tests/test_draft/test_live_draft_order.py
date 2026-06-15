"""Live-draft custom pick-order tests.

The web dashboard controller historically generated a pure snake order, ignoring
``config/draft_order.json`` (traded picks + keeper rounds) that the CLI and
simulator already honor. These tests pin that the dashboard path uses the real
order: the loader extracts the post-keeper order with trades baked in, and the
controller advances ``on_the_clock`` through it -- including consecutive same-team
picks a pure snake can never produce.
"""

from pathlib import Path

REAL_ORDER = Path(__file__).resolve().parents[2] / "config" / "draft_order.json"

TEAMS = {i: f"T{i}" for i in range(1, 11)}


def _state(on_clock):
    return {
        "version": 0,
        "keepers": [],
        "picks": [],
        "undo_stack": [],
        "on_the_clock": on_clock,
        "projected_standings_cache": {},
    }


# --- loader -----------------------------------------------------------------


def test_load_post_keeper_order_starts_at_first_live_round():
    from fantasy_baseball.draft.draft_order import load_post_keeper_pick_order

    order = load_post_keeper_pick_order(REAL_ORDER, keeper_rounds=3)
    # Round 4 (first live round) leads; its first pick is Send in the Cavalli,
    # NOT the keeper-round-1 / pure-snake leader (Hello Peanuts!).
    assert order[0] == "Send in the Cavalli"


def test_load_post_keeper_order_bakes_in_traded_back_to_back_picks():
    from fantasy_baseball.draft.draft_order import load_post_keeper_pick_order

    order = load_post_keeper_pick_order(REAL_ORDER, keeper_rounds=3)
    # R5 occupies indices 10-19. Hart acquired Tortured's slot-2 pick and also
    # holds the natural slot-3 snake pick -> two consecutive Hart picks, which a
    # pure snake can never produce.
    assert order[11] == "Hart of the Order"
    assert order[12] == "Hart of the Order"


def test_load_post_keeper_order_drops_only_keeper_rounds():
    from fantasy_baseball.draft.draft_order import load_post_keeper_pick_order

    order = load_post_keeper_pick_order(REAL_ORDER, keeper_rounds=3)
    # 23 rounds total - 3 keeper rounds = 20 live rounds * 10 teams.
    assert len(order) == 20 * 10


def test_load_post_keeper_order_missing_file_returns_none(tmp_path):
    from fantasy_baseball.draft.draft_order import load_post_keeper_pick_order

    assert load_post_keeper_pick_order(tmp_path / "nope.json", keeper_rounds=3) is None


def test_load_post_keeper_order_empty_result_returns_none(tmp_path):
    # keeper_rounds >= number of rounds -> empty post-keeper slice. Must return
    # None (not []), so both consumers fall back to snake consistently rather than
    # _compute_on_the_clock seating no one (on_the_clock None on pick 1).
    import json

    from fantasy_baseball.draft.draft_order import load_post_keeper_pick_order

    p = tmp_path / "draft_order.json"
    p.write_text(json.dumps({"rounds": [["A", "B"], ["B", "A"]]}))
    assert load_post_keeper_pick_order(p, keeper_rounds=3) is None


# --- controller -------------------------------------------------------------


def test_apply_pick_follows_pick_order_including_back_to_back():
    from fantasy_baseball.draft.draft_controller import apply_pick

    pick_order = ["A", "B", "B", "C"]  # B twice in a row (traded-pick shape)
    state = _state("A")
    state = apply_pick(
        state,
        player_id="p1::hitter",
        player_name="P1",
        position="OF",
        team="A",
        teams_by_position=TEAMS,
        pick_order=pick_order,
    )
    assert state["on_the_clock"] == "B"
    state = apply_pick(
        state,
        player_id="p2::hitter",
        player_name="P2",
        position="OF",
        team="B",
        teams_by_position=TEAMS,
        pick_order=pick_order,
    )
    # Back-to-back: a pure snake could never leave the same team on the clock.
    assert state["on_the_clock"] == "B"


def test_apply_pick_returns_none_when_pick_order_exhausted():
    from fantasy_baseball.draft.draft_controller import apply_pick

    state = _state("A")
    state = apply_pick(
        state,
        player_id="p1::hitter",
        player_name="P1",
        position="OF",
        team="A",
        teams_by_position=TEAMS,
        pick_order=["A"],
    )
    assert state["on_the_clock"] is None


def test_start_new_draft_uses_pick_order_for_first_pick():
    from fantasy_baseball.draft.draft_controller import start_new_draft

    league_yaml = {
        "league": {"team_name": "A"},
        "draft": {"position": 1, "teams": {1: "A", 2: "B", 3: "C"}},
        "keepers": [],
    }
    state = start_new_draft(
        league_yaml,
        resolve_keeper=lambda n, t: (n, n, "OF"),
        pick_order=["C", "B", "A"],
    )
    assert state["on_the_clock"] == "C"


def test_undo_pick_with_pick_order_rolls_back():
    from fantasy_baseball.draft.draft_controller import apply_pick, undo_pick

    pick_order = ["A", "B", "B", "C"]
    state = _state("A")
    state = apply_pick(
        state,
        player_id="p1::hitter",
        player_name="P1",
        position="OF",
        team="A",
        teams_by_position=TEAMS,
        pick_order=pick_order,
    )
    state = undo_pick(state, teams_by_position=TEAMS, pick_order=pick_order)
    assert state["on_the_clock"] == "A"
