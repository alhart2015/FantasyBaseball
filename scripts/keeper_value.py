"""Rank players by keeper-asset value (discounted multi-year VAR), swept across
discount rates. Default --anchor current reads cache:full_season_projections fresh
from Upstash; --anchor preseason reads only SQLite + manual ZiPS out-year CSV
exports (no network I/O). See docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.analysis.draft_value import parse_full_season_lines
from fantasy_baseball.analysis.keeper_value import (
    DEFAULT_HORIZON,
    DEFAULT_OUT_YEAR_REGRESSION,
    DEFAULT_PT_HEAL_CAP,
    discounted_total,
    keeper_value,
    mark_preseason_fallback,
    out_year_share,
    overlay_current_anchors,
)
from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.db import (
    get_blended_projections,
    get_connection,
    get_positions,
)
from fantasy_baseball.data.fangraphs import load_projection_set
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
from fantasy_baseball.draft.board import build_board_from_frames
from fantasy_baseball.draft.keepers import find_keeper_match, index_by_normalized_name
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.rankings import fg_key, lookup_rank, rank_key
from fantasy_baseball.web.season_data import unwrap_cache_envelope

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS_ROOT = REPO_ROOT / "data" / "projections"
CONFIG_PATH = REPO_ROOT / "config" / "league.yaml"
BASE_YEAR = 2026  # the anchor year: the DB's blended board + the ZiPS trajectory denominator
DISCOUNTS = [0.60, 0.70, 0.80, 0.90]
CANDIDATES = [
    "Juan Soto",
    "Julio Rodriguez",
    "Junior Caminero",
    "CJ Abrams",
    "Mason Miller",
    "Kyle Tucker",
]
_ZIPS_URL = "https://www.fangraphs.com/projections?type=zips&stats={t}&pos=all"


def load_zips_year(projections_root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hitters, pitchers = load_projection_set(projections_root / str(year), "zips")
    if hitters.empty or pitchers.empty:
        raise FileNotFoundError(
            f"Missing ZiPS {year} export. Download the {year} ZiPS projections from "
            f"{_ZIPS_URL.format(t='bat')} (hitters) and {_ZIPS_URL.format(t='pit')} "
            f"(pitchers), save as data/projections/{year}/zips-hitters.csv and "
            f"data/projections/{year}/zips-pitchers.csv."
        )
    return hitters, pitchers


def _fg_id(row: pd.Series) -> str | None:
    """Row's FanGraphs id as a str, or None when absent/NaN (blend rows may lack it)."""
    fg = row.get("fg_id")
    return str(fg) if fg is not None and pd.notna(fg) else None


def load_current_full_season_lines() -> dict:
    """Fresh Upstash read of cache:full_season_projections (YTD+ROS blend), parsed to
    the by-name map. Fails loud if the blob is missing/empty -- never silently serve
    preseason under a `current` label."""
    kv = build_explicit_upstash_kv()
    raw = kv.get(redis_key(CacheKey.FULL_SEASON_PROJECTIONS))
    if raw is None:
        raise SystemExit("cache:full_season_projections missing in Upstash; run a refresh first.")
    payload = unwrap_cache_envelope(json.loads(raw) if isinstance(raw, str) else raw)
    if not isinstance(payload, dict) or not (payload.get("hitters") or payload.get("pitchers")):
        raise SystemExit("cache:full_season_projections is empty; run a refresh first.")
    _by_mlbam, by_name = parse_full_season_lines(payload)
    return by_name


def zips_index(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Index ZiPS lines by fg_id (primary) and normalized-name (fallback), both
    namespaced by player_type -- so same-name/same-type players (Max Muncy LAD vs
    ATH) resolve by fg_id and don't collapse last-write-wins. Read via lookup_rank.

    Inherent residual: a same-name/same-type pair can only be disambiguated when
    both sides carry a real fg_id. A board row whose fg_id was fillna'd to its name
    (no id in any blended system), or a ZiPS row lacking an id, falls to the name
    key -- last-write-wins -- and can't be told apart from its namesake. There is no
    code fix without a shared id; the common (real-fg_id) collision is resolved.
    """
    idx: dict[str, dict[str, Any]] = {}
    for df, ptype in [(hitters, PlayerType.HITTER), (pitchers, PlayerType.PITCHER)]:
        for _, row in df.iterrows():
            line = row.to_dict()
            fg = _fg_id(row)
            if fg is not None:
                idx[fg_key(fg, ptype)] = line
            idx[rank_key(row["name"], ptype)] = line  # name fallback (no-fg_id players)
    return idx


def resolve_candidate_ids(board: pd.DataFrame, candidates: list[str]) -> set[str]:
    """Map each candidate name to the player_id of its best-VAR board match, so the
    highlight is collision-safe (VAR tie-break via find_keeper_match), never bare-name.
    """
    by_norm = index_by_normalized_name(board.to_dict("records"))
    ids: set[str] = set()
    for name in candidates:
        match = find_keeper_match(name, by_norm)
        if match is not None:
            ids.add(match["player_id"])
    return ids


def _zips_by_year(
    fg_id: str | None,
    name: str,
    player_type: str,
    indices: dict[int, dict[str, dict[str, Any]]],
) -> dict[int, dict[str, Any] | None]:
    # lookup_rank tries fg_id (fg_key) first, then name (rank_key); {} miss -> None.
    return {
        year: (lookup_rank(idx, fg_id, name, player_type) or None) for year, idx in indices.items()
    }


def build_results(
    base_year: int,
    horizon: int,
    *,
    anchor: str = "current",
    out_year_regression: float = DEFAULT_OUT_YEAR_REGRESSION,
    pt_heal_cap: float = DEFAULT_PT_HEAL_CAP,
):
    conn = get_connection()
    try:
        hitters, pitchers = get_blended_projections(conn)
        positions = get_positions(conn)
    finally:
        conn.close()
    config = load_config(CONFIG_PATH)
    current_keys: set[str] = set()
    if anchor == "current":
        by_name = load_current_full_season_lines()
        hitters, pitchers, current_keys = overlay_current_anchors(
            hitters, pitchers, by_name, heal_cap=pt_heal_cap
        )
        board_keys = {
            rank_key(str(n), pt)
            for df, pt in ((hitters, "hitter"), (pitchers, "pitcher"))
            for n in df["name"]
        }
        skipped = sum(1 for k in by_name if k not in board_keys)
        if skipped:
            print(
                f"[keeper-value] {skipped} current-blob players absent from the "
                f"preseason board (skipped; see spec follow-up)",
                file=sys.stderr,
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
        year: zips_index(*load_zips_year(PROJECTIONS_ROOT, year))
        for year in range(base_year, base_year + horizon)
    }
    candidate_ids = resolve_candidate_ids(board, CANDIDATES)
    results = []
    for _, row in board.iterrows():
        results.append(
            keeper_value(
                row["player_id"],
                row["name"],
                row.to_dict(),
                list(row["positions"]),
                str(row["player_type"]),
                _zips_by_year(_fg_id(row), row["name"], row["player_type"], indices),
                scale,
                base_year=base_year,
                horizon=horizon,
                out_year_regression=out_year_regression,
            )
        )
    if anchor == "current":
        results = mark_preseason_fallback(results, current_keys)
    return results, candidate_ids


def render(results, discounts: list[float], candidate_ids: set[str], limit: int = 0) -> str:
    """Render the ranked table. ``limit > 0`` shows only the top ``limit`` players
    (the leaguewide rank column still reflects the full pool)."""
    if not results:
        return "No players scored."
    first = next(iter(results))
    base_year = min(first.per_year_var)
    horizon = len(first.per_year_var)

    # Total for every (player, discount), then rank per discount so the ranking
    # slide across discounts is explicit (spec: "total and rank at each discount").
    totals = {
        d: {r.player_id: discounted_total(r.per_year_var, base_year, d, horizon) for r in results}
        for d in discounts
    }
    ranks: dict[float, dict[str, int]] = {}
    for d in discounts:
        order = sorted(results, key=lambda r: totals[d][r.player_id], reverse=True)
        ranks[d] = {r.player_id: i + 1 for i, r in enumerate(order)}

    # Order rows (and thus which top-N --limit keeps) by the most dynasty-weighted
    # discount -- the LARGEST rate, regardless of --discount input order.
    primary = max(discounts)
    ranked = sorted(results, key=lambda r: totals[primary][r.player_id], reverse=True)
    shown = ranked if limit <= 0 else ranked[:limit]

    lines = [
        "Keeper-asset value (discounted multi-year VAR); cell = total(#rank at that discount)",
    ]
    if 0 < limit < len(ranked):
        lines.append(f"(showing top {limit} of {len(ranked)})")
    lines.append("")
    header = f"{'':1} {'Player':22} " + " ".join(
        f"{'d=' + format(d, '.2f'):>13}" for d in discounts
    )
    header += (
        "  perYr(" + "/".join(str(base_year + k) for k in range(horizon)) + ")  %out  %sv  flags"
    )
    lines.append(header)
    for r in shown:
        mark = "*" if r.player_id in candidate_ids else " "
        cells = " ".join(
            f"{totals[d][r.player_id]:7.1f}(#{ranks[d][r.player_id]:>3})" for d in discounts
        )
        per = "/".join(f"{r.per_year_var[base_year + k]:.0f}" for k in range(horizon))
        # %out shown at the primary (row-ordering) discount so it matches the ranking.
        pct_out = out_year_share(r.per_year_var, base_year, totals[primary][r.player_id])
        pout = "N/A " if pct_out is None else f"{pct_out * 100:3.0f}%"
        psv = "N/A " if r.pct_from_saves is None else f"{r.pct_from_saves * 100:3.0f}%"
        lines.append(
            f"{mark} {r.name[:22]:22} {cells}  {per:>10}  {pout} {psv}  {','.join(r.flags)}"
        )
    return "\n".join(lines)


def _discounts_arg(s: str) -> list[float]:
    """Parse a comma-separated discount list, e.g. '0.6,0.8,0.9'. Each must be in (0, 1]."""
    try:
        vals = [float(x) for x in s.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid discount list {s!r}: {exc}") from exc
    if not vals or any(not (0.0 < v <= 1.0) for v in vals):
        raise argparse.ArgumentTypeError(
            f"discounts must be comma-separated values in (0, 1], e.g. 0.6,0.8,0.9; got {s!r}"
        )
    return vals


def _nonneg_int(s: str) -> int:
    """Parse a non-negative int for --limit (0 = show all)."""
    try:
        v = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int {s!r}: {exc}") from exc
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0 (0 shows all); got {v}")
    return v


def _unit_float(s: str) -> float:
    """Parse a fraction in [0, 1] for --out-year-regression."""
    try:
        v = float(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float {s!r}: {exc}") from exc
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0, 1]; got {v}")
    return v


def _min_one_float(s: str) -> float:
    """Parse a float >= 1.0 for --pt-heal-cap (1.0 disables the heal)."""
    try:
        v = float(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float {s!r}: {exc}") from exc
    if v < 1.0:
        raise argparse.ArgumentTypeError(f"must be >= 1.0 (1.0 disables); got {v}")
    return v


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rank players by keeper-asset value (discounted multi-year VAR)."
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="years to score, including the base year (default 3 = ZiPS availability; "
        ">3 requires the matching ZiPS out-year exports).",
    )
    ap.add_argument(
        "--discount",
        type=_discounts_arg,
        default=list(DISCOUNTS),
        metavar="R[,R,...]",
        help="comma-separated discount rates to sweep, each in (0, 1] "
        "(default 0.60,0.70,0.80,0.90). 1.0 = no discount (out-years count fully).",
    )
    ap.add_argument(
        "--limit",
        type=_nonneg_int,
        default=100,
        help="show only the top N players by keeper value (default 100; 0 = all). "
        "Skips the long tail no one would keep.",
    )
    ap.add_argument(
        "--anchor",
        choices=["current", "preseason"],
        default="current",
        help="anchor the 2026 base on current-season talent (YTD+ROS, default) or the "
        "preseason blend. current requires a synced cache:full_season_projections.",
    )
    ap.add_argument(
        "--out-year-regression",
        type=_unit_float,
        default=DEFAULT_OUT_YEAR_REGRESSION,
        metavar="F",
        help=f"regress 2027+ toward ZiPS's forward projection, fraction in [0,1] "
        f"(default {DEFAULT_OUT_YEAR_REGRESSION}). 0 = pure current-anchor x aging "
        f"(over-indexes on this year); 1 = pure ZiPS out-year (ignores this year).",
    )
    ap.add_argument(
        "--pt-heal-cap",
        type=_min_one_float,
        default=DEFAULT_PT_HEAL_CAP,
        metavar="X",
        help=f"heal injury-shortened anchors: scale counting stats up toward healthy "
        f"PT by min(X, preseason_PT/current_PT), X >= 1 (default {DEFAULT_PT_HEAL_CAP}). "
        f"1.0 disables. Rates (talent) are never scaled.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(argv)
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    results, candidate_ids = build_results(
        base_year=BASE_YEAR,
        horizon=args.horizon,
        anchor=args.anchor,
        out_year_regression=args.out_year_regression,
        pt_heal_cap=args.pt_heal_cap,
    )
    print(render(results, args.discount, candidate_ids, limit=args.limit))


if __name__ == "__main__":
    main()
