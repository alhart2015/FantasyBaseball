"""Golden-master parity guard.

Pins the pre-refactor /api/recs payload so every later phase proves the
deltaRoto path through recommend() reproduces it byte-for-byte.  Runs with
team_sds active (the production path) per the standing meta-lesson that
variance-free scoring flips verdicts.

Fixture provenance
------------------
recs_golden_state_board.json
    Built by rebuild_board() (build_draft_board + serialize_board) against
    the blended_projections SQLite table at commit 342a068 on branch
    feat/unify-draft-engine.  Contains 3695 rows with all stat columns
    (total_sgp, var, adp, and per-type volume stats) required by
    compute_rec_inputs.

recs_golden_state.json
    Hand-built dashboard-schema state (keepers=[], 35 fixed picks against
    top-35 board players by var-desc, snake-order assigned).  Hart of the
    Order holds pick 8 (Cal Raleigh/C), pick 13 (Gunnar Henderson/SS),
    pick 28 (Cristopher Sanchez/P) and pick 33 (Edwin Diaz/P); on_the_clock
    is Boston Estrellas (pick 36).  Timestamps are frozen at 1717977600.0
    for strict reproducibility.

recs_golden.json
    Written automatically on first run by this test; subsequent runs
    compare against it byte-for-byte to guard the deltaRoto refactor.
"""

import json
from pathlib import Path

from fantasy_baseball.web.app import create_app

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "recs_golden.json"
STATE = FIXTURES / "recs_golden_state.json"
TEAM = "Hart of the Order"


def _get_recs() -> list[dict]:
    app = create_app(state_path=STATE)
    app.config["TESTING"] = True
    client = app.test_client()
    resp = client.get(f"/api/recs?team={TEAM.replace(' ', '%20')}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()  # type: ignore[return-value]


def test_recs_match_golden() -> None:
    """First run writes the golden; subsequent runs assert byte-for-byte equality."""
    rows = _get_recs()
    assert isinstance(rows, list), f"expected list, got {type(rows)}"
    assert rows, "/api/recs returned an empty list -- no candidates for the chosen team"

    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=True))

    expected = json.loads(GOLDEN.read_text())
    assert rows == expected, (
        "Golden-master mismatch: the deltaRoto path changed /api/recs output.\n"
        "If this is intentional (the refactor improved correctness), delete\n"
        f"{GOLDEN} and re-run to capture a new baseline."
    )


def test_recs_has_required_fields() -> None:
    """Every row must contain the six fields RecRow.__dict__ exposes."""
    rows = _get_recs()
    required = {
        "player_id",
        "name",
        "positions",
        "immediate_delta",
        "value_of_picking_now",
        "per_category",
    }
    for row in rows:
        missing = required - set(row.keys())
        assert not missing, f"Row {row.get('name')} missing fields: {missing}"


def test_recs_team_sds_active() -> None:
    """Verify team_sds is non-trivially populated (not empty dicts).

    An empty team_sds produces variance-free scoring -- the meta-lesson
    from PR #127 is that this silently flips rankings. The golden is only
    trustworthy when SDs are live.
    """
    from fantasy_baseball.draft.draft_controller import resume_or_init
    from fantasy_baseball.web.app import _build_rec_inputs, _load_league_yaml

    app = create_app(state_path=STATE)
    state = resume_or_init(STATE)
    league_yaml = _load_league_yaml()
    inputs = _build_rec_inputs(app, state, league_yaml)

    assert inputs.team_sds, "team_sds is empty -- variance-free scoring active"
    non_zero_teams = sum(
        1 for sds in inputs.team_sds.values() if any(v != 0.0 for v in sds.values())
    )
    assert non_zero_teams > 0, "all team SDs are zero -- scoring is variance-free"
