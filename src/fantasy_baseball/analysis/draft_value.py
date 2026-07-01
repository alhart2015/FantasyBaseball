"""Draft-value metric: realized VAR vs draft-slot par expectation."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.draft.board import build_board_from_frames

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_BOARD = _REPO_ROOT / "data" / "draft_state_board.json"
_PRESEASON_CSVS = _REPO_ROOT / "data" / "projections" / "2026"
_POSITIONS_JSON = _REPO_ROOT / "data" / "player_positions.json"
_CONFIG = _REPO_ROOT / "config" / "league.yaml"


@dataclass(frozen=True)
class ScaleInputs:
    denoms: dict[str, Any]
    repl_rates: dict[str, Any]
    replacement_levels: dict[str, Any]
    team_ab: int
    team_ip: int


def reproduce_draft_day_board() -> tuple[pd.DataFrame, ScaleInputs]:
    """Rebuild the preseason board from the Apr-1 projection CSVs -- pure, no DB/KV.

    blend_projections is deterministic over data/projections/2026/*.csv; positions
    come from data/player_positions.json. The board core is the shared
    build_board_from_frames, so the scale (rates/floors/denoms/volumes) is identical
    to the real draft board's, just computed off the preserved draft-day CSVs.
    """
    config = load_config(_CONFIG)
    hitters, pitchers, _quality = blend_projections(
        _PRESEASON_CSVS, config.projection_systems, config.projection_weights
    )
    positions = load_positions_cache(_POSITIONS_JSON)
    board, scale_d = build_board_from_frames(
        hitters,
        pitchers,
        positions,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
        return_scale=True,
    )
    scale = ScaleInputs(
        denoms=scale_d["denoms"],
        repl_rates=scale_d["repl_rates"],
        replacement_levels=scale_d["replacement_levels"],
        team_ab=scale_d["team_ab"],
        team_ip=scale_d["team_ip"],
    )
    return board, scale


def frozen_drift_summary(
    board_df: pd.DataFrame,
    frozen_path: Path | str | None = None,
    tol: float = 0.05,
) -> dict[str, float]:
    """SOFT cross-check vs the frozen draft-day board. Reports drift; never raises.

    The frozen board (draft_state_board.json) was built at draft time with
    possibly-since-churned code, so exact reproduction is not expected. A large
    systematic drift is worth surfacing (wrong config/vintage) but is not a stop.
    """
    frozen_path = Path(frozen_path) if frozen_path else _FROZEN_BOARD
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_var = {
        row["player_id"]: float(row["var"])
        for row in frozen
        if row.get("player_id") is not None and row.get("var") is not None
    }
    diffs = []
    for _, row in board_df.iterrows():
        pid = row["player_id"]
        if pid in frozen_var:
            diffs.append(abs(float(row["var"]) - frozen_var[pid]))
    over = sum(1 for d in diffs if d > tol)
    summary: dict[str, float] = {
        "joined": len(diffs),
        "over_tol": over,
        "max": max(diffs) if diffs else 0.0,
        "median": statistics.median(diffs) if diffs else 0.0,
    }
    if diffs and over > 0.5 * len(diffs):
        logger.warning(
            "draft-value: rebuilt board drifts from frozen draft_state_board.json "
            "(%d/%d players > %.2f VAR, max %.2f). Expected some drift from code "
            "churn since the freeze; investigate only if this looks systematic.",
            over,
            len(diffs),
            tol,
            summary["max"],
        )
    return summary
