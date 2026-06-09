"""Tests for simulate_draft.py -- integration tests for the simulation harness.

These tests exercise run_simulation end-to-end to verify that the unified
recommend() seam works for all scoring modes. Goldens (test_parity_sim_golden.py)
pin exact pick sequences; these tests verify that a full draft completes.
"""

from fantasy_baseball.draft.strategy import STRATEGIES
from scripts.sim_deltaroto import DELTAROTO
from scripts.simulate_draft import build_board_and_context, run_simulation


def test_deltaroto_immediate_draft_completes():
    """A deltaroto_immediate draft runs end-to-end and populates all rosters.

    Regression guard: the recommend() seam must handle the deltaRoto path
    (RecInputs per pick, scoring_mode="deltaroto_immediate") without crashing
    and must return a complete roster for every team.
    """
    saved = dict(STRATEGIES)
    try:
        for name, attr in DELTAROTO.items():
            from scripts.sim_deltaroto import make_deltaroto_pick

            STRATEGIES[name] = make_deltaroto_pick(attr)

        ctx = build_board_and_context()
        config = ctx["config"]

        result = run_simulation(
            ctx,
            strategy_name="deltaroto_immediate",
            scoring_mode="var",  # sim_deltaroto runs under "var" scoring_mode
            adp_noise=0.0,
            strategy_noise=0.0,
            seed=42,
            field_noise=False,
        )
    finally:
        STRATEGIES.clear()
        STRATEGIES.update(saved)

    # Every team must have drafted at least one player
    team_players = result["team_players"]
    for team_num in range(1, config.num_teams + 1):
        assert team_players[team_num], (
            f"Team {team_num} ({config.teams.get(team_num)}) has no players"
        )

    # User team roster must be non-empty
    assert result["user_roster_ids"], "User roster_ids is empty"
    assert result["user_roster"], "User roster is empty"

    # Standings must be computed and contain all teams
    assert len(result["results"]) == config.num_teams

    # Score must be a positive number
    assert result["pts"] > 0, f"User points {result['pts']} <= 0"


def test_var_draft_completes():
    """A var-mode draft runs end-to-end and populates all rosters."""
    ctx = build_board_and_context()
    config = ctx["config"]

    result = run_simulation(
        ctx,
        strategy_name="default",
        scoring_mode="var",
        adp_noise=0.0,
        strategy_noise=0.0,
        seed=42,
        field_noise=False,
    )

    team_players = result["team_players"]
    for team_num in range(1, config.num_teams + 1):
        assert team_players[team_num], (
            f"Team {team_num} ({config.teams.get(team_num)}) has no players"
        )

    assert result["user_roster_ids"], "User roster_ids is empty"
    assert len(result["results"]) == config.num_teams
    assert result["pts"] > 0
