"""Clean-room draft: isolate the SCORING MODE's edge from situational factors.

Always removes traded picks (standard snake) and averages over all 10 draft
seats (relocating the user CORRECTLY by swapping team names -- setting
draft_position alone is a bug: the strategy follows draft_position but the
measured/recommender team stays pinned to the user's original team_num, so you
end up scoring an ADP team).

Keeper toggle (argv):
  python scripts/cleanroom_draft.py            # NO keepers  -> level field
  python scripts/cleanroom_draft.py keepers    # real keepers -> Hart's situation

Level field has a clean symmetric NULL (no-edge ADP drafter = rank 5.5 / 10%
win). With keepers the field is asymmetric (keeper quality varies), so there is
no clean null -- compare the modes head-to-head instead.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

MODES = ["var", "vona", "deltaroto_immediate", "deltaroto_vopn", "deltaroto_finalslate"]
POSITIONS = list(range(1, 11))
ITERS = 20
MAX_WORKERS = 8
REAL_CONFIG = PROJECT_ROOT / "config" / "league.yaml"
TMP_CONFIG = PROJECT_ROOT / "config" / "_cleanroom_nokeepers.yaml"


def _write_clean_config():
    raw = yaml.safe_load(open(REAL_CONFIG))
    raw["keepers"] = []
    yaml.safe_dump(raw, open(TMP_CONFIG, "w"))


def _cell(mode, position, seed_base, config_path):
    import simulate_draft as sd

    sd.DRAFT_ORDER_PATH = Path("__cleanroom_no_draft_order__")  # standard snake
    ctx = sd.build_board_and_context(config_path=config_path)
    cfg = ctx["config"]
    # Relocate user to `position` by swapping team names so cfg.team_name (and
    # its keepers, which are keyed by name) sit at team_num == position.
    hart_num = next(k for k, v in cfg.teams.items() if v == cfg.team_name)
    teams = dict(cfg.teams)
    teams[hart_num] = teams[position]
    teams[position] = cfg.team_name
    cfg.teams = teams
    cfg.draft_position = position

    pts, ranks, wins = [], [], 0
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
        h = next(t for t in r["results"] if t["team"] == cfg.team_name)
        pts.append(h["tot"])
        ranks.append(r["rank"])
        wins += int(r["rank"] == 1)
    return {"mode": mode, "position": position, "pts": pts, "ranks": ranks, "wins": wins}


def main():
    keep = "keepers" in sys.argv
    config_path = REAL_CONFIG if keep else TMP_CONFIG
    if not keep:
        _write_clean_config()

    t0 = time.perf_counter()
    label = "KEEPERS ON (Hart's real keepers)" if keep else "NO keepers (level field)"
    print(
        f"Clean-room draft: {label}, NO trades (snake), all seats, {ITERS} iters/cell, vs ADP, gated."
    )
    if keep:
        print(
            "Field is asymmetric (keeper quality varies) -> no clean null; compare modes head-to-head.\n"
        )
    else:
        print("NULL (no-edge ADP drafter, symmetric): rank 5.50 / win 10%\n")

    cells = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        for mode in MODES:
            for pos in POSITIONS:
                futs[pool.submit(_cell, mode, pos, 7000 + pos * 100, config_path)] = (mode, pos)
        for f in as_completed(futs):
            cells.append(f.result())

    print(f"{'mode':<22}{'avg pts':>9}{'avg rank':>10}{'win%':>8}")
    print("-" * 50)
    rows = []
    for mode in MODES:
        pts = [p for c in cells if c["mode"] == mode for p in c["pts"]]
        ranks = [r for c in cells if c["mode"] == mode for r in c["ranks"]]
        wins = sum(c["wins"] for c in cells if c["mode"] == mode)
        n = len(ranks)
        rows.append((mode, float(np.mean(pts)), float(np.mean(ranks)), wins / n * 100))
    for mode, ap, ar, wp in sorted(rows, key=lambda r: r[2]):
        print(f"{mode:<22}{ap:>9.1f}{ar:>10.2f}{wp:>7.0f}%")

    print(f"\nTotal: {(time.perf_counter() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
