"""Per-team ADP draft boards (`simulate_draft._build_adp_boards`).

Each team must draft off its OWN noised view of ADP, so an elite player one team
undervalues is still grabbed near his true ADP by another team. The old single
shared reshuffle let one unlucky draw push a low-ADP player down for ALL teams at
once -- so he'd fall league-wide (e.g. an ADP-33 bat surviving to pick ~60).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _sim():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import simulate_draft

    return simulate_draft


def _board(n=20):
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "name": [f"P{i}" for i in range(n)],
            "adp": list(range(1, n + 1)),
            "positions": [["OF"] for _ in range(n)],
        }
    )


def test_per_team_boards_have_same_players_but_differ_in_order():
    sim = _sim()
    boards = sim._build_adp_boards(
        _board(), num_teams=3, adp_noise=7.0, rng=np.random.default_rng(0)
    )

    assert set(boards) == {1, 2, 3}
    for tn in boards:
        assert set(boards[tn]["player_id"]) == {f"p{i}" for i in range(20)}  # same players
    # Distinct per-team opinions -> different orderings.
    assert list(boards[1]["player_id"]) != list(boards[2]["player_id"])
    assert list(boards[2]["player_id"]) != list(boards[3]["player_id"])


def test_no_noise_gives_identical_true_adp_order():
    sim = _sim()
    boards = sim._build_adp_boards(
        _board(), num_teams=3, adp_noise=0.0, rng=np.random.default_rng(0)
    )
    expected = [f"p{i}" for i in range(20)]  # already sorted by adp 1..20
    for tn in boards:
        assert list(boards[tn]["player_id"]) == expected


def test_elite_player_not_buried_league_wide():
    """An elite (low-ADP) player stays near the top of SOME team's board -- he
    can't be pushed down for everyone the way the shared-reshuffle model allowed.
    """
    sim = _sim()
    boards = sim._build_adp_boards(
        _board(), num_teams=10, adp_noise=7.0, rng=np.random.default_rng(1)
    )
    # The true #1 ADP player ("p0") should sit in the top few of at least one
    # team's board -> he won't slide past the early picks across the whole league.
    best_ranks = [list(boards[tn]["player_id"]).index("p0") for tn in boards]
    assert min(best_ranks) <= 2
