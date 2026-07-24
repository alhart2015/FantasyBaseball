"""Breakout/mirage keeper report: ranks board players by surface-believed keeper
value (today's --anchor current number) vs a skill-adjusted keeper value (the
current anchor regressed toward Statcast xStats/FanGraphs rates via
fantasy_baseball.analysis.breakout.adjust_line). Reuses scripts/keeper_value.py's
board/scale/anchor loading; all computation lives in the tested
fantasy_baseball.analysis.breakout.breakout_rows -- this script is I/O glue only.

PROVISIONAL -- pending backtest (Task 10, Phase 4): the w-mapping (reliability x
confirmation) weights are seed defaults, not yet validated against realized
year-over-year outcomes. Treat labels/deltas as a diagnostic, not a verdict.

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keeper_value as kv_script  # scripts/keeper_value.py: board/scale/anchor loading

from fantasy_baseball.analysis.breakout import breakout_rows
from fantasy_baseball.analysis.keeper_value import DEFAULT_DISCOUNT
from fantasy_baseball.data.skill_luck import build_hitter_skill_luck, build_pitcher_skill_luck
from fantasy_baseball.draft.board import build_board_from_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_LUCK_CACHE_DIR = REPO_ROOT / "data" / "skill_luck"
OUT_DIR = REPO_ROOT / "data" / "analysis"
BASE_YEAR = kv_script.BASE_YEAR

PROVISIONAL_BANNER = (
    "PROVISIONAL -- pending backtest (Task 10): adjustment weights are seed "
    "defaults, not yet validated against realized year-over-year outcomes."
)


def _preseason_projection_index(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> dict:
    """Preseason blended lines keyed 'fg_id::player_type' -- the projection_line
    baseline adjust_line compares the current-season surface line against. Read
    from the pre-overlay frames (overlay_current_anchors copies, never mutates
    its inputs, but this keeps the intent explicit)."""
    idx: dict[str, dict] = {}
    for df, ptype in [(hitters, "hitter"), (pitchers, "pitcher")]:
        for _, row in df.iterrows():
            fg = kv_script._fg_id(row)
            if fg is None:
                continue
            idx[f"{fg}::{ptype}"] = row.to_dict()
    return idx


def _merged_skill_luck(cache_dir: Path, year: int) -> dict:
    """Hitter + pitcher SkillLuckRow, keyed by fg_id (matching breakout_rows'
    ``skill_luck.get(fgid)`` lookup).

    KNOWN LIMITATION: a two-way player who shares one FanGraphs id across
    batting_stats/pitching_stats (e.g. Ohtani) collides on this int key -- the
    hitter row wins the merge, so ``skill_luck.get(fgid)`` returns the hitter's
    row for BOTH of that player's board rows. breakout_rows guards this: it
    compares the looked-up row's player_type against the board row's and
    discards a mismatch, so the pitcher board row degrades gracefully to "no
    skill/luck data" instead of being corrupted with the hitter's Statcast
    numbers. No board-side fix is possible without namespacing this dict by
    player_type too, which would let the pitcher board row get its OWN
    skill/luck data (full two-way valuation); deferred (rare in practice: one
    player leaguewide).
    """
    hitters, _ = build_hitter_skill_luck(cache_dir, year)
    pitchers, _ = build_pitcher_skill_luck(cache_dir, year)
    merged = dict(hitters)
    for fgid, row in pitchers.items():
        merged.setdefault(fgid, row)
    return merged


def load_inputs(*, horizon: int, pt_heal_cap: float, skill_luck_year: int):
    """Board (current-anchored, matching --anchor current), scale, ZiPS out-year
    indices, merged skill/luck rows, and preseason projections -- the five
    already-loaded inputs breakout_rows is pure over. Mirrors
    scripts/keeper_value.py:build_results up to (not including) the per-row
    keeper_value loop, which breakout_rows performs itself (twice per player).
    """
    conn = kv_script.get_connection()
    try:
        hitters, pitchers = kv_script.get_blended_projections(conn)
        positions = kv_script.get_positions(conn)
    finally:
        conn.close()
    projections = _preseason_projection_index(hitters, pitchers)

    config = kv_script.load_config(kv_script.CONFIG_PATH)
    by_name = kv_script.load_current_full_season_lines()
    hitters, pitchers, _current_keys = kv_script.overlay_current_anchors(
        hitters, pitchers, by_name, heal_cap=pt_heal_cap
    )
    board, scale = build_board_from_frames(
        hitters,
        pitchers,
        positions,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
        sgp_overrides=config.sgp_overrides,
    )
    indices = {
        year: kv_script.zips_index(*kv_script.load_zips_year(kv_script.PROJECTIONS_ROOT, year))
        for year in range(BASE_YEAR, BASE_YEAR + horizon)
    }
    skill_luck = _merged_skill_luck(SKILL_LUCK_CACHE_DIR, skill_luck_year)
    return board, scale, indices, skill_luck, projections


def _fmt(x: float | None) -> str:
    return "  N/A" if x is None else f"{x:7.1f}"


def render_markdown(rows: list[dict], limit: int) -> str:
    lines = ["# Breakout/Mirage Keeper Report", "", PROVISIONAL_BANNER, ""]
    shown = rows if limit <= 0 else rows[:limit]
    if 0 < limit < len(rows):
        lines.append(f"(showing top {limit} of {len(rows)})")
        lines.append("")
    lines.append("| Player | Type | Surface | Adjusted | Delta | Label | Conf | Reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in shown:
        lines.append(
            f"| {r['name']} | {r['player_type']} | {_fmt(r['surface_value'])} "
            f"| {_fmt(r['adjusted_value'])} | {_fmt(r['delta'])} | {r['label']} "
            f"| {r['confidence']} | {r['reason']} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rank board players by surface vs skill-adjusted keeper value."
    )
    ap.add_argument("--horizon", type=int, default=kv_script.DEFAULT_HORIZON)
    ap.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT)
    ap.add_argument(
        "--out-year-regression", type=float, default=kv_script.DEFAULT_OUT_YEAR_REGRESSION
    )
    ap.add_argument("--pt-heal-cap", type=float, default=kv_script.DEFAULT_PT_HEAL_CAP)
    ap.add_argument(
        "--skill-luck-year",
        type=int,
        default=BASE_YEAR,
        help="season to pull Statcast xStats/FanGraphs rates for (default: base year).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=25,
        help="show only the top N players by adjusted value (default 25; 0 = all).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(argv)
    board, scale, indices, skill_luck, projections = load_inputs(
        horizon=args.horizon,
        pt_heal_cap=args.pt_heal_cap,
        skill_luck_year=args.skill_luck_year,
    )
    rows = breakout_rows(
        board,
        scale,
        indices,
        skill_luck,
        projections,
        base_year=BASE_YEAR,
        horizon=args.horizon,
        discount=args.discount,
        out_year_regression=args.out_year_regression,
    )
    report = render_markdown(rows, args.limit)
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "breakout_report.md"
    csv_path = OUT_DIR / "breakout_report.csv"
    md_path.write_text(report, encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nWrote {md_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
