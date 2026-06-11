"""Monte-Carlo evaluation of draft modes.

The clean-room used SINGLE-SHOT analytic eRoto (one deterministic roto score per
roster) -- the same metric the strategies optimize, so margins can be circular
and ranks saturate (a strong keeper roster ranks 1st every time). This re-scores
each final draft by Monte-Carlo over PROJECTION UNCERTAINTY: perturb every team's
category totals by their projection SDs (the codebase's own build_team_sds /
monte_carlo_roto_totals model), rank per category, sum to roto, repeat N_MC draws
-> Hart's WIN PROBABILITY and mean rank. This converts margin into "how often do
you actually win," and de-saturates the keeper case.

  python scripts/mc_eval.py            # level field (no keepers)
  python scripts/mc_eval.py keepers    # Hart's real keepers
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

MODES = ["var", "vona", "deltaroto_immediate", "deltaroto_vopn", "deltaroto_finalslate"]
POSITIONS = list(range(1, 11))
ITERS = 10
N_MC = 1000
MAX_WORKERS = 8
REAL_CONFIG = PROJECT_ROOT / "config" / "league.yaml"
TMP_CONFIG = PROJECT_ROOT / "config" / "_cleanroom_nokeepers.yaml"


def _write_clean_config():
    raw = yaml.safe_load(open(REAL_CONFIG))
    raw["keepers"] = []
    yaml.safe_dump(raw, open(TMP_CONFIG, "w"))


def _mc_win_rank(team_stats, team_sds, hart, n, rng):
    """(win_prob, mean_rank) for `hart` over n MC draws under projection SDs."""
    teams = list(team_stats.keys())
    hi = teams.index(hart)
    totals = np.zeros((n, len(teams)))
    for cat in ALL_CATEGORIES:
        means = np.array([float(team_stats[t].get(cat.value, 0.0)) for t in teams])
        sds = np.array([float(team_sds.get(t, {}).get(cat, 0.0)) for t in teams])
        samples = rng.normal(loc=means, scale=sds, size=(n, len(teams)))
        if cat in INVERSE_STATS:
            samples = -samples
        totals += rankdata(samples, axis=1)
    hart_tot = totals[:, hi]
    rank = (totals >= hart_tot[:, None]).sum(axis=1)  # rank 1 = highest total
    return float(np.mean(rank == 1)), float(np.mean(rank))


def _cell(mode, position, seed_base, config_path):
    import simulate_draft as sd

    cap = {}
    _orig = sd.score_roto_dict

    def _patched(team_stats, team_sds=None, **kw):
        cap["stats"] = team_stats
        cap["sds"] = team_sds
        return _orig(team_stats, team_sds=team_sds, **kw)

    sd.score_roto_dict = _patched
    sd.DRAFT_ORDER_PATH = Path("__cleanroom_no_draft_order__")
    ctx = sd.build_board_and_context(config_path=config_path)
    cfg = ctx["config"]
    hart_num = next(k for k, v in cfg.teams.items() if v == cfg.team_name)
    teams = dict(cfg.teams)
    teams[hart_num] = teams[position]
    teams[position] = cfg.team_name
    cfg.teams = teams
    cfg.draft_position = position

    rng = np.random.default_rng(seed_base)
    win_probs, mc_ranks, ss_ranks = [], [], []
    for i in range(ITERS):
        r = sd.run_simulation(
            ctx,
            strategy_name="default",
            scoring_mode=mode,
            adp_noise=15.0,
            strategy_noise=0.0,
            seed=seed_base + i,
            opponent_strategies_str="",
            position_aware=True,
        )
        wp, mr = _mc_win_rank(cap["stats"], cap["sds"], cfg.team_name, N_MC, rng)
        win_probs.append(wp)
        mc_ranks.append(mr)
        ss_ranks.append(r["rank"])  # single-shot rank, for comparison
    return {"mode": mode, "win": win_probs, "mc_rank": mc_ranks, "ss_rank": ss_ranks}


def main():
    keep = "keepers" in sys.argv
    config_path = REAL_CONFIG if keep else TMP_CONFIG
    if not keep:
        _write_clean_config()
    t0 = time.perf_counter()
    label = "KEEPERS ON" if keep else "NO keepers (level field)"
    print(
        f"MC eval: {label}, NO trades, all seats, {ITERS} drafts/cell x {N_MC} MC draws, vs ADP, gated."
    )
    if not keep:
        print("Single-shot null was rank 5.5 / 10% win; MC win-prob null ~10%.\n")
    else:
        print(
            "(keeper field asymmetric; compare modes -- MC win% de-saturates the single-shot 100%s)\n"
        )

    cells = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        for mode in MODES:
            for pos in POSITIONS:
                futs[pool.submit(_cell, mode, pos, 7000 + pos * 100, config_path)] = (mode, pos)
        for f in as_completed(futs):
            cells.append(f.result())

    print(f"{'mode':<22}{'MC win%':>9}{'MC rank':>10}{'single-shot rank':>18}")
    print("-" * 60)
    rows = []
    for mode in MODES:
        win = np.mean([w for c in cells if c["mode"] == mode for w in c["win"]]) * 100
        mcr = np.mean([r for c in cells if c["mode"] == mode for r in c["mc_rank"]])
        ssr = np.mean([r for c in cells if c["mode"] == mode for r in c["ss_rank"]])
        rows.append((mode, win, mcr, ssr))
    for mode, win, mcr, ssr in sorted(rows, key=lambda r: -r[1]):
        print(f"{mode:<22}{win:>8.1f}%{mcr:>10.2f}{ssr:>18.2f}")
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
