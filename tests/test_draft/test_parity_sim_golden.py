"""Pre-refactor simulator golden. Pins the user team's pick sequence for a
fixed seed WITH team_sds so P3/P4 prove the seam reproduces pre-refactor picks.
Generated against simulate_draft.py (var) + sim_deltaroto.py (deltaRoto) before
Phase 3; re-pointed at the consolidated sim in Task 14.

Determinism guarantee
---------------------
Both wrappers set adp_noise=0.0, strategy_noise=0.0, field_noise=False.
All randomness is disabled -- opponent teams draft in clean-ADP order and the
user always picks the top recommendation.  The seed parameter is threaded
through for traceability but has no effect when all noise is zero.

Golden capture
--------------
First run writes fixtures/sim_golden_var.json and
fixtures/sim_golden_deltaroto.json from the pre-refactor pick path; every
subsequent run compares byte-for-byte.  Delete the JSON files to re-capture
(only do this intentionally, after verifying the new output is correct).
"""

import json
from pathlib import Path

from scripts.simulate_draft import run_user_pick_sequence as varvona_seq

FIX = Path(__file__).parent / "fixtures"

SEED = 7


def _assert_golden(seq, name):
    """Write golden on first run; compare byte-for-byte on subsequent runs."""
    assert isinstance(seq, list), f"expected list[str], got {type(seq)}"
    assert seq, f"pick sequence is empty -- golden {name!r} would be useless"
    for item in seq:
        assert isinstance(item, str), f"player_id must be str, got {type(item)}: {item!r}"

    g = FIX / name
    if not g.exists():
        g.write_text(json.dumps(seq, indent=2, ensure_ascii=True))
        return

    expected = json.loads(g.read_text())
    assert seq == expected, (
        f"Golden-master mismatch for {name}.\n"
        "The simulator pick path changed.  If this is intentional (a bug was fixed,\n"
        f"or the board changed), delete {g} and re-run to capture a new baseline."
    )


def test_sim_var_picks_match_golden():
    """Pin the VAR-mode user pick sequence."""
    seq = varvona_seq(scoring_mode="var", strategy="default", seed=SEED)
    _assert_golden(seq, "sim_golden_var.json")


def test_sim_deltaroto_picks_match_golden():
    """Pin the deltaRoto-immediate user pick sequence."""
    from scripts.sim_deltaroto import run_user_pick_sequence as dr_seq

    seq = dr_seq(scoring_mode="deltaroto_immediate", strategy="deltaroto_immediate", seed=SEED)
    _assert_golden(seq, "sim_golden_deltaroto.json")
