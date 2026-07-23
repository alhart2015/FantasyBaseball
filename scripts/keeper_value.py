"""Rank players by keeper-asset value (discounted multi-year VAR), swept across
discount rates. Reads projections/positions from SQLite and manual ZiPS out-year
CSV exports; does no network I/O. See
docs/superpowers/specs/2026-07-22-keeper-value-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.analysis.keeper_value import (  # noqa: E402
    discounted_total,
    keeper_value,
)
from fantasy_baseball.config import load_config  # noqa: E402
from fantasy_baseball.data.db import (  # noqa: E402
    get_blended_projections,
    get_connection,
    get_positions,
)
from fantasy_baseball.data.fangraphs import load_projection_set  # noqa: E402
from fantasy_baseball.draft.board import build_board_from_frames  # noqa: E402
from fantasy_baseball.models.player import PlayerType  # noqa: E402
from fantasy_baseball.utils.name_utils import normalize_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS_ROOT = REPO_ROOT / "data" / "projections"
CONFIG_PATH = REPO_ROOT / "config" / "league.yaml"
DISCOUNTS = [0.60, 0.70, 0.80, 0.90]
CANDIDATES = [
    "Juan Soto", "Julio Rodriguez", "Junior Caminero",
    "CJ Abrams", "Mason Miller", "Kyle Tucker",
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


def zips_index(hitters: pd.DataFrame, pitchers: pd.DataFrame) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for df, ptype in [(hitters, PlayerType.HITTER), (pitchers, PlayerType.PITCHER)]:
        for _, row in df.iterrows():
            idx[f"{normalize_name(row['name'])}::{ptype}"] = row.to_dict()
    return idx


def is_candidate(name: str, candidate_norms: set[str]) -> bool:
    return normalize_name(name) in candidate_norms


def _zips_by_year(player_key: str, indices: dict[int, dict[str, dict[str, Any]]]) -> dict:
    return {year: idx.get(player_key) for year, idx in indices.items()}


def build_results(base_year: int, horizon: int):
    conn = get_connection()
    try:
        hitters, pitchers = get_blended_projections(conn)
        positions = get_positions(conn)
    finally:
        conn.close()
    config = load_config(CONFIG_PATH)
    board, scale = build_board_from_frames(
        hitters, pitchers, positions,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
        sgp_overrides=config.sgp_overrides,
    )
    indices = {
        year: zips_index(*load_zips_year(PROJECTIONS_ROOT, year))
        for year in range(base_year, base_year + horizon)
    }
    results = []
    for _, row in board.iterrows():
        key = f"{row['name_normalized']}::{row['player_type']}"
        results.append(
            keeper_value(
                row["player_id"], row["name"], row.to_dict(), list(row["positions"]),
                str(row["player_type"]), _zips_by_year(key, indices), scale,
                base_year=base_year, horizon=horizon,
            )
        )
    return results


def render(results, discounts: list[float]) -> str:
    candidate_norms = {normalize_name(c) for c in CANDIDATES}
    if not results:
        return "No players scored."
    base_year = min(next(iter(results)).per_year_var)
    horizon = len(next(iter(results)).per_year_var)

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

    primary = discounts[-1]  # order rows by the most dynasty-weighted discount
    ranked = sorted(results, key=lambda r: totals[primary][r.player_id], reverse=True)

    lines = [
        "Keeper-asset value (discounted multi-year VAR); cell = total(#rank at that discount)",
        "",
    ]
    header = f"{'':1} {'Player':22} " + " ".join(f"{'d=' + format(d, '.2f'):>13}" for d in discounts)
    header += "  perYr(" + "/".join(str(base_year + k) for k in range(horizon)) + ")  %out  %sv  flags"
    lines.append(header)
    for r in ranked:
        mark = "*" if is_candidate(r.name, candidate_norms) else " "
        cells = " ".join(
            f"{totals[d][r.player_id]:7.1f}(#{ranks[d][r.player_id]:>3})" for d in discounts
        )
        per = "/".join(f"{r.per_year_var[base_year + k]:.0f}" for k in range(horizon))
        pout = "N/A " if r.pct_from_out_years is None else f"{r.pct_from_out_years * 100:3.0f}%"
        psv = "N/A " if r.pct_from_saves is None else f"{r.pct_from_saves * 100:3.0f}%"
        lines.append(f"{mark} {r.name[:22]:22} {cells}  {per:>10}  {pout} {psv}  {','.join(r.flags)}")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = build_results(base_year=2026, horizon=3)
    print(render(results, DISCOUNTS))


if __name__ == "__main__":
    main()
