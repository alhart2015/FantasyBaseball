"""Measure the realized top-10 / top-30 / top-100 VAR bars and write them to data/.

These are what the board's probabilities are computed AGAINST. They must be realized
values, not quantiles of the projected pool -- see `trajectory.value_bars` for why, and
for the measurement showing the projected #30 bar sits at roughly the realized #100.

    python scripts/build_value_bars.py              # measure and write
    python scripts/build_value_bars.py --dry-run    # print, write nothing

Fast (~30s): this reads the panel directly rather than running a held-out sweep, so it is
not the 35-minute job `build_band_calibration.py` is.

WHEN TO RE-RUN: whenever `data/trajectory/` is rebuilt, or when a new season's fielding
data lands in `data/cache/keeper_skills/` -- the second one MATTERS, because eligibility
coverage is what bounds how many windows each span has, and a new year of it extends the
longest measurable span. The artifact carries the panel filenames and `ValueBars.load`
refuses a mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from fantasy_baseball.config import load_config
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.trajectory.board import (
    _collapse,
    _pitcher_slots,
    best_floor,
    season_slots,
)
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.value import resolve_slots
from fantasy_baseball.trajectory.value_bars import (
    VALUE_BARS_PATH,
    build_bars,
    eligible_seasons,
    league_ranks,
    panel_vintage_of,
)

KEEPER_CACHE = PROJECT_ROOT / "data" / "cache" / "keeper_skills"


def floors_for(live: pd.DataFrame, kind: str, season: int, levels: dict) -> dict[int, float]:
    """Each player's replacement floor in `season`, by the board's own rule.

    Reuses `_pitcher_slots` / `resolve_slots` / `best_floor` rather than re-deriving
    eligibility: a bar computed against a different floor rule than the board projects
    against is not comparable to it, and the difference would be invisible on screen.
    """
    if kind == "pitcher":
        slots = _pitcher_slots(live, season)
    else:
        eligible = season_slots(KEEPER_CACHE, season)
        slots = {
            pid: resolve_slots(set(eligible.get(pid, frozenset())), kind)
            for pid in live[live["season"] == season]["mlbam_id"].astype(int)
        }
    return {
        pid: best_floor(slots.get(pid, set()), levels)[1]
        for pid in live[live["season"] == season]["mlbam_id"].astype(int)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir
    out = args.out or (PROJECT_ROOT / VALUE_BARS_PATH)

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    levels = position_aware_replacement_levels(get_sgp_denominators(config.sgp_overrides))
    ranks = league_ranks(config.num_teams, config.keepers_per_team)

    panels: dict[str, pd.DataFrame] = {}
    for kind in ("hitter", "pitcher"):
        raw = load_scored_panel(kind, panel_dir=args.panel_dir, sgp_overrides=config.sgp_overrides)
        panels[kind] = _collapse(era_normalize(raw, kind, sgp_overrides=config.sgp_overrides))
    last_complete = int(min(int(p["season"].max()) for p in panels.values()))
    eligible = eligible_seasons(KEEPER_CACHE)
    print(f"  panel complete through {last_complete}; eligibility cached for {eligible}")
    print(
        f"  ranks: {ranks}  (elite and keeper derived from {config.num_teams} teams x "
        f"{config.keepers_per_team} keepers)"
    )

    floors_by_season = {
        season: {kind: floors_for(panels[kind], kind, season, levels) for kind in panels}
        for season in eligible
        if season <= last_complete
    }

    bars = build_bars(
        panels,
        floors_by_season,
        panel_vintage=panel_vintage_of(args.panel_dir),
        ranks=ranks,
        cache_dir=KEEPER_CACHE,
        last_complete=last_complete,
    )

    print(f"\n{'=' * 62}\nREALIZED VAR BARS\n{'=' * 62}")
    print(f"  {'span':<6}{'windows':>9}  {'starts':<18}" + "".join(f"{n:>10}" for n in ranks))
    for span, starts in bars.windows.items():
        row = "".join(
            f"{bars.bar(span, n):>10.1f}" if bars.bar(span, n) is not None else f"{'--':>10}"
            for n in ranks
        )
        note = "  <-- no complete window" if not starts else ""
        print(f"  {span:<6}{len(starts):>9}  {starts!s:<18}{row}{note}")
    print(
        "\n  A span with no window carries no bar, so the board will not headline a "
        "probability\n  for it. Bars are not linear in span (the 3-year keeper bar is "
        "2.2x the 1-year, not 3x),\n  so a missing span cannot be extrapolated from a "
        "shorter one."
    )

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    bars.save(out)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
