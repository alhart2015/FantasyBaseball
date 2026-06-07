"""Read-only: empirically cross-check the team_sds the fragility analysis uses.

Three independent angles on each category's REMAINING-season SD (the persisted,
fraction_remaining-scaled team_sds from cache:projections):

  1. Sampling-noise floor: pure Poisson (counting) / binomial (AVG) SD of the
     projected total. The model SD MUST exceed this -- a real projection has
     talent + playing-time uncertainty on top of sampling. Model SD < floor =>
     definitely overconfident.

  2. Weekly-increment estimate: from the 8 weekly YTD snapshots, the SD of each
     team's week-to-week production, extrapolated to the remaining weeks. This
     captures realized game-to-game noise only (NOT talent/injury uncertainty),
     so it is a LOWER BOUND on the true remaining SD.

  3. The model's own SD, for Hart and the league median.

Nothing is written.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path
from statistics import median, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.data.redis_store import get_standings_history  # noqa: E402
from fantasy_baseball.scoring import team_sds_from_json  # noqa: E402
from fantasy_baseball.utils.constants import (  # noqa: E402
    COUNTING_STATS,
    Category,
)

TEAM = "Hart of the Order"
COUNTING_ORDER = [Category.R, Category.HR, Category.RBI, Category.SB, Category.W, Category.K, Category.SV]


def _read_prod_cache(kv, key):
    raw = kv.get(redis_key(key))
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict) and "_data" in obj:
        return obj["_data"]
    return obj


def main() -> None:
    kv = build_explicit_upstash_kv()
    proj = _read_prod_cache(kv, CacheKey.PROJECTIONS) or {}
    fr = float(proj["fraction_remaining"])
    ps_rows = proj["projected_standings"]["teams"]
    proj_means = {r["name"]: {Category(k): float(v) for k, v in r["stats"].items()} for r in ps_rows}
    team_sds = team_sds_from_json(proj.get("team_sds") or {})
    teams = [r["name"] for r in ps_rows]

    std_hist = get_standings_history(kv)
    dates = sorted(std_hist)
    print(f"fraction_remaining={fr:.4f} (played ~{(1 - fr) * 100:.0f}%)")
    print(f"YTD snapshots: {dates}\n")

    # League-wide fraction covered by the snapshot window (via R), to convert
    # weekly increments into a remaining-weeks count.
    by = {d: std_hist[d].by_team() for d in dates}
    first, last = dates[0], dates[-1]

    def lg(cat, d):
        return sum(e.stats[cat] for e in std_hist[d].entries)

    lg_proj_R = sum(proj_means[t][Category.R] for t in teams)
    window_frac = (lg(Category.R, last) - lg(Category.R, first)) / lg_proj_R
    n_weeks = len(dates) - 1
    weeks_remaining = fr * n_weeks / window_frac if window_frac > 0 else float("nan")
    print(f"snapshot window covered ~{window_frac:.3f} of season over {n_weeks} steps; "
          f"=> ~{weeks_remaining:.1f} remaining steps\n")

    print(f"{'cat':<5} {'projMean(Hart)':>14} | {'model SD':>9} {'modelCV':>8} | "
          f"{'sampling floor':>14} | {'wkly-incr SD':>13} {'(x floor)':>9} | flag")
    print("-" * 95)

    for cat in COUNTING_ORDER:
        mean_h = proj_means[TEAM][cat]
        model_sd_h = team_sds.get(TEAM, {}).get(cat, 0.0)
        model_cv = model_sd_h / mean_h if mean_h else float("nan")

        # 1) sampling floor: Poisson on the projected total, scaled to remaining.
        floor = sqrt(mean_h) * sqrt(fr)

        # 2) weekly-increment empirical SD for Hart, extrapolated to remaining.
        incs = []
        for i in range(1, len(dates)):
            a = by[dates[i - 1]].get(TEAM)
            b = by[dates[i]].get(TEAM)
            if a and b:
                incs.append(b.stats[cat] - a.stats[cat])
        wk_sd = pstdev(incs) if len(incs) > 1 else float("nan")
        emp_remaining = wk_sd * sqrt(weeks_remaining) if wk_sd == wk_sd else float("nan")
        x_floor = model_sd_h / floor if floor else float("nan")

        flag = ""
        if model_sd_h < floor:
            flag = "BELOW SAMPLING FLOOR (overconfident)"
        elif emp_remaining == emp_remaining and model_sd_h < emp_remaining * 0.8:
            flag = "below realized game-noise (overconfident)"
        elif emp_remaining == emp_remaining and model_sd_h > emp_remaining * 3:
            flag = "far above game-noise (check talent/PT term)"

        print(f"{cat.value:<5} {mean_h:>14.1f} | {model_sd_h:>9.2f} {model_cv:>8.3f} | "
              f"{floor:>14.2f} | {emp_remaining:>13.2f} {x_floor:>8.2f}x | {flag}")

    # Rate stats: binomial floor for AVG; report model SD + implied full-season.
    print()
    print("RATE STATS (model SD is remaining-season; full-season = /sqrt(fr)):")
    inv_fr = 1.0 / sqrt(fr)
    avg_mean = proj_means[TEAM][Category.AVG]
    total_ab = next(r["total_ab"] for r in ps_rows if r["name"] == TEAM)
    avg_floor = sqrt(avg_mean * (1 - avg_mean) / total_ab) * sqrt(fr)
    for cat in (Category.AVG, Category.ERA, Category.WHIP):
        sd = team_sds.get(TEAM, {}).get(cat, 0.0)
        line = f"  {cat.value:<5} model_remaining_SD={sd:.4f}  full_season_SD={sd * inv_fr:.4f}"
        if cat == Category.AVG:
            line += f"  binomial floor(remaining)={avg_floor:.4f}  (x floor={sd / avg_floor:.2f})"
        print(line)

    # League-median model CV per counting cat (is Hart typical?)
    print("\nLeague model CV by counting cat (median across teams):")
    for cat in COUNTING_ORDER:
        cvs = []
        for t in teams:
            m = proj_means[t][cat]
            s = team_sds.get(t, {}).get(cat, 0.0)
            if m:
                cvs.append(s / m)
        print(f"  {cat.value:<5} median CV={median(cvs):.3f}  (Hart={team_sds.get(TEAM, {}).get(cat, 0.0) / proj_means[TEAM][cat]:.3f})")

    _ = COUNTING_STATS  # keep import explicit


if __name__ == "__main__":
    main()
