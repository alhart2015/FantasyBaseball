"""Tests for simulate_draft.py -- integration tests for the simulation harness.

These tests exercise run_simulation end-to-end to verify that the unified
recommend() seam works for all scoring modes. Goldens (test_parity_sim_golden.py)
pin exact pick sequences; these tests verify that a full draft completes and that
the variance model behaves correctly.

Variance model
--------------
With all noise off (adp_noise=0.0, strategy_noise=0.0, field_noise=False) the
draft is fully deterministic -- the seed has no effect and every call with any
seed produces identical picks.  Enabling adp_noise jitters each team's per-player
ADP draw independently, so different seeds yield different opponent selections and
therefore different available pools for the user, producing different user rosters.
"""

from scripts.simulate_draft import build_board_and_context, run_simulation, run_user_pick_sequence


def test_deltaroto_immediate_draft_completes():
    """A deltaroto_immediate draft runs end-to-end and populates all rosters.

    Task 14: uses the consolidated harness directly -- no sim_deltaroto strategy
    injection needed.  The recommend() seam handles the deltaRoto path natively
    (RecInputs per pick, scoring_mode="deltaroto_immediate").
    """
    ctx = build_board_and_context()
    config = ctx["config"]

    result = run_simulation(
        ctx,
        strategy_name="deltaroto_immediate",
        scoring_mode="deltaroto_immediate",
        adp_noise=0.0,
        strategy_noise=0.0,
        seed=42,
        field_noise=False,
    )

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


def test_noise_off_is_fully_deterministic():
    """With all noise off, different seeds produce identical pick sequences.

    This pins the variance model: adp_noise=0.0, strategy_noise=0.0,
    field_noise=False means the RNG is never consulted.  The seed argument is
    accepted for traceability but has zero effect on the output.  Tested for
    both var and deltaroto_immediate modes.
    """
    seq_var_42 = run_user_pick_sequence(scoring_mode="var", strategy="default", seed=42)
    seq_var_99 = run_user_pick_sequence(scoring_mode="var", strategy="default", seed=99)
    assert seq_var_42 == seq_var_99, (
        "var mode: noise-off draft must be identical across seeds; "
        "seed 42 and seed 99 diverged at first differing position"
    )

    seq_dr_42 = run_user_pick_sequence(
        scoring_mode="deltaroto_immediate", strategy="deltaroto_immediate", seed=42
    )
    seq_dr_99 = run_user_pick_sequence(
        scoring_mode="deltaroto_immediate", strategy="deltaroto_immediate", seed=99
    )
    assert seq_dr_42 == seq_dr_99, (
        "deltaroto_immediate mode: noise-off draft must be identical across seeds; "
        "seed 42 and seed 99 diverged at first differing position"
    )


def test_adp_noise_on_different_seeds_diverge():
    """With adp_noise enabled, different seeds produce different user rosters.

    Each team draws its own per-player ADP jitter from the seeded RNG, so
    opponent draft order varies across seeds.  The user's available pool changes
    accordingly, producing a different pick sequence.  This pins the variance
    model: noise ON + different seeds -> different outcomes.
    """
    ctx = build_board_and_context()

    r42 = run_simulation(
        ctx,
        strategy_name="default",
        scoring_mode="var",
        adp_noise=20.0,
        strategy_noise=0.0,
        seed=42,
        field_noise=False,
    )
    r99 = run_simulation(
        ctx,
        strategy_name="default",
        scoring_mode="var",
        adp_noise=20.0,
        strategy_noise=0.0,
        seed=99,
        field_noise=False,
    )

    ids42 = list(r42["user_roster_ids"])
    ids99 = list(r99["user_roster_ids"])
    assert ids42 != ids99, (
        "adp_noise=20.0 with different seeds must produce different user rosters; "
        "seeds 42 and 99 produced identical picks -- noise may not be seeded correctly"
    )


def test_deltaroto_immediate_with_strategic_opponents_completes():
    """A deltaroto_immediate draft with strategic opponents runs without raising.

    Regression test for the opponent reroute crash: when scoring_mode is a
    deltaRoto mode and opponent_strategies_str is non-empty, the opponent pick
    path must build RecInputs (not fall through to the board-only ctx, which
    raises ValueError: requires inputs).  Uses a minimal opponent string that
    assigns two_closers strategy to one opponent team (team 1).
    """
    ctx = build_board_and_context()
    config = ctx["config"]

    # Route one opponent team through the strategy path so the deltaRoto
    # opponent reroute fires at least once.
    user_pos = config.draft_position
    opp_team = 1 if user_pos != 1 else 2
    opp_str = f"{opp_team}:two_closers"

    result = run_simulation(
        ctx,
        strategy_name="deltaroto_immediate",
        scoring_mode="deltaroto_immediate",
        adp_noise=0.0,
        strategy_noise=0.0,
        seed=42,
        field_noise=False,
        opponent_strategies_str=opp_str,
    )

    assert len(result["results"]) == config.num_teams
    assert result["pts"] > 0
