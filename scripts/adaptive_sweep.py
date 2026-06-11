"""Sweep the adaptive flip-round K, scored by MC win%.

deltaroto_adaptive uses VOPN for rounds < K and immediate for rounds >= K.
K=1 == fixed immediate; K=24 (> rounds) == fixed VOPN. Same seeds across K (paired).

  python scripts/adaptive_sweep.py            # level field (no keepers)
  python scripts/adaptive_sweep.py keepers    # Hart's real keepers
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mc_eval import _mc_win_rank

KS = [1, 4, 7, 9, 11, 13, 16, 24]
POSITIONS = list(range(1, 11))
ITERS = 8
N_MC = 1000
MAX_WORKERS = 8
REAL_CONFIG = PROJECT_ROOT / "config" / "league.yaml"
TMP_CONFIG = PROJECT_ROOT / "config" / "_cleanroom_nokeepers.yaml"


def _write_clean_config():
    raw = yaml.safe_load(open(REAL_CONFIG))
    raw["keepers"] = []
    yaml.safe_dump(raw, open(TMP_CONFIG, "w"))


def _cell(k, position, seed_base, config_path):
    import simulate_draft as sd

    os.environ["ADAPTIVE_K"] = str(k)
    cap = {}
    _orig = sd.score_roto_dict

    def _patched(team_stats, team_sds=None, **kw):
        cap["stats"] = team_stats
        cap["sds"] = team_sds
        return _orig(team_stats, team_sds=team_sds, **kw)

    sd.score_roto_dict = _patched
    sd.DRAFT_ORDER_PATH = Path("__no_draft_order__")
    ctx = sd.build_board_and_context(config_path=config_path)
    cfg = ctx["config"]
    hart_num = next(kk for kk, v in cfg.teams.items() if v == cfg.team_name)
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
            strategy_name="default",
            scoring_mode="deltaroto_adaptive",
            adp_noise=15.0,
            strategy_noise=0.0,
            seed=seed_base + i,
            opponent_strategies_str="",
            position_aware=True,
        )
        wp, _ = _mc_win_rank(cap["stats"], cap["sds"], cfg.team_name, N_MC, rng)
        wins.append(wp)
    return {"k": k, "wins": wins}


def main():
    keep = "keepers" in sys.argv
    config_path = REAL_CONFIG if keep else TMP_CONFIG
    if not keep:
        _write_clean_config()
    t0 = time.perf_counter()
    label = "KEEPERS" if keep else "level field"
    print(
        f"Adaptive K-sweep ({label}, all seats, {ITERS} drafts/cell x {N_MC} MC). K=1==immediate, K=24==VOPN.\n"
    )
    cells = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        for k in KS:
            for pos in POSITIONS:
                futs[pool.submit(_cell, k, pos, 7000 + pos * 100, config_path)] = (k, pos)
        for f in as_completed(futs):
            cells.append(f.result())

    print(f"{'K (flip round)':<16}{'MC win%':>9}   note")
    print("-" * 46)
    best_k, best_w = None, -1.0
    for k in KS:
        w = np.mean([x for c in cells if c["k"] == k for x in c["wins"]]) * 100
        note = "fixed immediate" if k == 1 else ("fixed VOPN" if k == 24 else "")
        if w > best_w:
            best_w, best_k = w, k
        print(f"{k:<16}{w:>8.1f}%   {note}")
    imm = np.mean([x for c in cells if c["k"] == 1 for x in c["wins"]]) * 100
    vop = np.mean([x for c in cells if c["k"] == 24 for x in c["wins"]]) * 100
    print(
        f"\nbest K={best_k} at {best_w:.1f}%  vs  fixed immediate {imm:.1f}%  /  fixed VOPN {vop:.1f}%"
    )
    print(
        f"adaptive {'BEATS' if best_w > max(imm, vop) + 0.3 else 'does NOT clearly beat'} both fixed modes"
    )
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
