"""Head-to-head: the (strategy, scoring) combo Hart ACTUALLY drafted with
(three_closers + var) vs deltaroto_immediate and friends, from Hart's real
keeper seat, scored by MC win%.

The standard bake-off only varied the SCORING mode under the default overlay, so
three_closers+var -- what Hart really used -- was never measured against
immediate. This fills that gap. Same MC machinery as mc_eval (perturb team
category totals by build_team_sds, rank per cat, 1000 draws -> P(Hart 1st)),
keeper config, all 10 seats (seat-swap), vs ADP opponents, gated.

  python scripts/bakeoff_actual_vs_immediate.py
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mc_eval import _mc_win_rank

# (label, strategy_overlay, scoring_mode)
COMBOS = [
    ("three_closers + var  (ACTUAL)", "three_closers", "var"),
    ("default + immediate  (current default)", "default", "deltaroto_immediate"),
    ("default + var", "default", "var"),
    ("three_closers + immediate", "three_closers", "deltaroto_immediate"),
    ("default + vopn", "default", "deltaroto_vopn"),
    ("default + finalslate", "default", "deltaroto_finalslate"),
]
POSITIONS = list(range(1, 11))
ITERS = 10
N_MC = 1000
MAX_WORKERS = 8
REAL_CONFIG = PROJECT_ROOT / "config" / "league.yaml"


def _cell(label, strategy, scoring, position, seed_base):
    import simulate_draft as sd

    cap = {}
    _orig = sd.score_roto_dict

    def _patched(team_stats, team_sds=None, **kw):
        cap["stats"] = team_stats
        cap["sds"] = team_sds
        return _orig(team_stats, team_sds=team_sds, **kw)

    sd.score_roto_dict = _patched
    sd.DRAFT_ORDER_PATH = Path("__cleanroom_no_draft_order__")  # standard snake, no trades
    ctx = sd.build_board_and_context(config_path=REAL_CONFIG)
    cfg = ctx["config"]
    hart_num = next(k for k, v in cfg.teams.items() if v == cfg.team_name)
    teams = dict(cfg.teams)
    teams[hart_num] = teams[position]
    teams[position] = cfg.team_name
    cfg.teams = teams
    cfg.draft_position = position

    rng = np.random.default_rng(seed_base)
    wins = []
    for i in range(ITERS):
        sd.run_simulation(
            ctx,
            strategy_name=strategy,
            scoring_mode=scoring,
            adp_noise=15.0,
            strategy_noise=0.0,
            seed=seed_base + i,
            opponent_strategies_str="",
            position_aware=True,
        )
        wp, _ = _mc_win_rank(cap["stats"], cap["sds"], cfg.team_name, N_MC, rng)
        wins.append(wp)
    return {"label": label, "wins": wins}


def main():
    t0 = time.perf_counter()
    print(
        f"Combo bake-off (KEEPERS, all seats, {ITERS} drafts/cell x {N_MC} MC, vs ADP, gated).\n"
        "Question: does the three_closers+var combo Hart actually used beat immediate?\n"
    )
    cells = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        for label, strat, scor in COMBOS:
            for pos in POSITIONS:
                futs[pool.submit(_cell, label, strat, scor, pos, 7000 + pos * 100)] = label
        for f in as_completed(futs):
            cells.append(f.result())

    print(f"{'combo':<40}{'MC win%':>9}")
    print("-" * 50)
    rows = []
    for label, _s, _c in COMBOS:
        w = np.mean([x for cc in cells if cc["label"] == label for x in cc["wins"]]) * 100
        rows.append((label, w))
    for label, w in sorted(rows, key=lambda r: -r[1]):
        print(f"{label:<40}{w:>8.1f}%")
    print(f"\nNull (no-edge) ~10%.  Total: {(time.perf_counter() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
