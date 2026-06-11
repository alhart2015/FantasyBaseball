"""Characterize the adaptive tipping point: what state does Hart's keeper start
(round 1) share with the from-scratch tipping point (~round 13)?

For each of Hart's picks we record his roster STATE before the pick and test
three candidate triggers:
  * eRoto rank      -- Hart's projected roto rank among the 10 teams (1=best)
  * dominant cats   -- # of the 10 categories Hart already leads
  * SGP vs field    -- Hart's roster total SGP minus the other-9 average

If one metric is ~equal at (level field, round 13) and (keepers, round 1), that
is the situation-independent trigger for VOPN->immediate.

  python scripts/state_diagnostic.py            # level field (build trajectory)
  python scripts/state_diagnostic.py keepers     # Hart's keepers (read round 1)
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

POSITIONS = list(range(1, 11))
ITERS = 8
MAX_WORKERS = 8
REAL_CONFIG = PROJECT_ROOT / "config" / "league.yaml"
TMP_CONFIG = PROJECT_ROOT / "config" / "_cleanroom_nokeepers.yaml"


def _write_clean_config():
    with open(REAL_CONFIG) as f:
        raw = yaml.safe_load(f)
    raw["keepers"] = []
    with open(TMP_CONFIG, "w") as f:
        yaml.safe_dump(raw, f)


def _roto_state(ps, hart):
    """(eRoto rank 1=best, dominant-cat count) for hart from a ProjectedStandings."""
    teams = [e.team_name for e in ps.entries]
    hi = teams.index(hart)
    n = len(teams)
    totals = np.zeros(n)
    dom = 0
    for cat in ALL_CATEGORIES:
        sv = [dict(e.stats.items()) for e in ps.entries]
        vals = np.array([float(s.get(cat, 0.0)) for s in sv])
        pts = rankdata(-vals) if cat in INVERSE_STATS else rankdata(vals)  # n = best
        totals += pts
        if pts[hi] == pts.max():
            dom += 1
    rank = int((totals > totals[hi]).sum()) + 1
    return rank, dom


def _cell(position, seed_base, config_path, num_teams):
    import simulate_draft as sd

    from fantasy_baseball.draft import recommend as rec
    from fantasy_baseball.draft.tracker import DraftTracker

    # capture (round, projected_standings) at each Hart pick
    picks_state = []
    _orig_rank = rec.rank_for_mode

    def _prank(ctx):
        if ctx.inputs is not None:
            picks_state.append((ctx.current_round, ctx.inputs.projected_standings))
        return _orig_rank(ctx)

    rec.rank_for_mode = _prank
    sd.rank_for_mode = _prank
    # capture ordered pick log (is_user, player_id) for SGP
    log = []
    _orig_dp = DraftTracker.draft_player

    def _dp(self, name, is_user=False, player_id=None):
        log.append((is_user, player_id or name))
        return _orig_dp(self, name, is_user=is_user, player_id=player_id)

    DraftTracker.draft_player = _dp
    sd.DRAFT_ORDER_PATH = Path("__no_draft_order__")
    ctx = sd.build_board_and_context(config_path=config_path)
    cfg = ctx["config"]
    sgp = {
        str(p): float(s)
        for p, s in zip(
            ctx["full_board"]["player_id"], ctx["full_board"]["total_sgp"], strict=False
        )
    }
    hart_num = next(k for k, v in cfg.teams.items() if v == cfg.team_name)
    teams = dict(cfg.teams)
    teams[hart_num] = teams[position]
    teams[position] = cfg.team_name
    cfg.teams = teams
    cfg.draft_position = position

    rows = []  # (round, erank, dom, sgp_vs_field)
    for i in range(ITERS):
        picks_state.clear()
        log.clear()
        sd.run_simulation(
            ctx,
            strategy_name="default",
            scoring_mode="deltaroto_vopn",
            adp_noise=15.0,
            strategy_noise=0.0,
            seed=seed_base + i,
            opponent_strategies_str="",
            position_aware=True,
        )
        # keeper boundary in log
        n_keepers = len(cfg.keepers)
        keepers = log[:n_keepers]
        live = log[n_keepers:]
        hart_keeper_sgp = sum(sgp.get(pid, 0.0) for u, pid in keepers if u)
        hart_live_idx = [j for j, (u, _) in enumerate(live) if u]
        for r_idx, (rnd, ps) in enumerate(picks_state):
            erank, dom = _roto_state(ps, cfg.team_name)
            # SGP state at this pick: everything drafted before Hart's r_idx-th live pick
            cut = hart_live_idx[r_idx] if r_idx < len(hart_live_idx) else len(live)
            before = keepers + live[:cut]
            field_total = sum(sgp.get(pid, 0.0) for _, pid in before)
            hart_sgp = hart_keeper_sgp + sum(sgp.get(pid, 0.0) for u, pid in live[:cut] if u)
            others_avg = (field_total - hart_sgp) / (num_teams - 1)
            rows.append((rnd, erank, dom, hart_sgp - others_avg))
    return rows


def main():
    keep = "keepers" in sys.argv
    config_path = REAL_CONFIG if keep else TMP_CONFIG
    if not keep:
        _write_clean_config()
    with open(REAL_CONFIG) as f:
        raw = yaml.safe_load(f)
    num_teams = raw.get("num_teams", raw.get("league", {}).get("num_teams", 10))
    t0 = time.perf_counter()
    print(
        f"State diagnostic: {'KEEPERS' if keep else 'level field'}, VOPN build trajectory, {ITERS} drafts/seat.\n"
    )
    cells = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(_cell, p, 7000 + p * 100, config_path, num_teams) for p in POSITIONS]
        for f in as_completed(futs):
            cells.extend(f.result())

    by_round = {}
    for rnd, erank, dom, sgpf in cells:
        by_round.setdefault(rnd, []).append((erank, dom, sgpf))
    print(f"{'round':>6}{'eRoto rank':>12}{'dom cats':>10}{'SGP vs field':>14}")
    print("-" * 44)
    for rnd in sorted(by_round):
        arr = np.array(by_round[rnd])
        mark = (
            "  <== tipping"
            if rnd == 13
            else (
                "  <== keeper start"
                if (rnd == 1 and __import__("sys").argv[1:2] == ["keepers"])
                else ""
            )
        )
        print(
            f"{rnd:>6}{arr[:, 0].mean():>12.2f}{arr[:, 1].mean():>10.2f}{arr[:, 2].mean():>14.1f}{mark}"
        )
    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
