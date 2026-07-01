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
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION, Category

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_BOARD = _REPO_ROOT / "data" / "draft_state_board.json"
_PRESEASON_CSVS = _REPO_ROOT / "data" / "projections" / "2026"
_POSITIONS_JSON = _REPO_ROOT / "data" / "player_positions.json"
_CONFIG = _REPO_ROOT / "config" / "league.yaml"


@dataclass(frozen=True)
class ScaleInputs:
    denoms: dict[Category, float]
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


_COUNTING_HIT = ("r", "hr", "rbi", "sb", "ab")  # ab is volume (scales)
_COUNTING_PIT = ("w", "k", "sv", "ip")  # ip is volume (scales)


def _to_date_floors(scale: ScaleInputs, fraction: float) -> dict[str, float]:
    """Recompute position floors on a to-date scale (NOT scale.replacement_levels * f).

    Floor SGP is NOT linear in f: its rate component is f-invariant while only the
    counting component scales. So rebuild each floor from REPLACEMENT_BY_POSITION with
    counting+volume * f, rates held, team volumes * f. At f=1 this reproduces the
    board's position_aware_replacement_levels floors.
    """
    if fraction == 1.0:
        return scale.replacement_levels
    floors: dict[str, float] = {}
    team_ab = scale.team_ab * fraction
    team_ip = scale.team_ip * fraction
    for pos, raw in REPLACEMENT_BY_POSITION.items():
        line: dict[str, Any] = dict(raw)
        for k in ("r", "hr", "rbi", "sb", "ab", "w", "k", "sv", "ip"):
            if k in line:
                line[k] = line[k] * fraction
        is_pitcher = "ip" in raw
        # derive rate stats the floor line implies, held constant vs f
        if is_pitcher:
            ip = raw["ip"] or 1.0
            line["era"] = (raw["er"] / ip) * 9.0
            line["whip"] = (raw["bb"] + raw["h_allowed"]) / ip
            line["player_type"] = (
                "pitcher"  # StrEnum-compatible; required by calculate_player_sgp dispatch
            )
        else:
            ab = raw["ab"] or 1.0
            line["avg"] = raw["h"] / ab
            line["player_type"] = "hitter"
        floors[pos] = calculate_player_sgp(
            pd.Series(line),
            denoms=scale.denoms,
            team_ab=int(team_ab),
            team_ip=int(team_ip),
            replacement_avg=scale.repl_rates["avg"],
            replacement_era=scale.repl_rates["era"],
            replacement_whip=scale.repl_rates["whip"],
        )
    # UTIL floor mirrors the board: max of the hitter floors (see replacement.py).
    hitter_floors = [floors[p] for p in ("C", "1B", "2B", "3B", "SS", "OF") if p in floors]
    if hitter_floors:
        floors["UTIL"] = max(hitter_floors)
    return floors


def score_var(
    line: dict[str, Any],
    positions: list[str],
    player_type: str,
    scale: ScaleInputs,
    fraction: float = 1.0,
) -> float:
    """Score a stat line into VAR on the board scale (projected or YTD-scaled).

    ``fraction < 1.0`` applies the YTD to-date scaling: counting + volume + team
    volumes scale by ``fraction`` while rates (AVG/ERA/WHIP) are held, and the
    position floors are recomputed on the same to-date scale. ``player_type`` is
    ``"hitter"`` or ``"pitcher"``.
    """
    # player_type must be "hitter"/"pitcher" (StrEnum-compatible) so calculate_player_sgp
    # dispatches (player.get("player_type") == PlayerType.HITTER/PITCHER, player_value.py:104,121).
    scaled: dict[str, Any] = dict(line)
    scaled["player_type"] = player_type
    counting = _COUNTING_HIT if player_type == "hitter" else _COUNTING_PIT
    if fraction != 1.0:
        for k in counting:
            if scaled.get(k) is not None:
                scaled[k] = scaled[k] * fraction
    team_ab = scale.team_ab * fraction
    team_ip = scale.team_ip * fraction
    total_sgp = calculate_player_sgp(
        pd.Series(scaled),
        denoms=scale.denoms,
        team_ab=int(team_ab),
        team_ip=int(team_ip),
        replacement_avg=scale.repl_rates["avg"],
        replacement_era=scale.repl_rates["era"],
        replacement_whip=scale.repl_rates["whip"],
    )
    floors = _to_date_floors(scale, fraction)
    # calculate_var needs total_sgp + positions + ip (pitcher floor routing reads
    # player.get("ip", 0.0) via _pitcher_floor_key -> role_from_ip, var.py:18).
    series = pd.Series(
        {
            "total_sgp": total_sgp,
            "positions": list(positions),
            "player_type": player_type,
            "ip": scaled.get("ip", 0.0),
        }
    )
    return calculate_var(series, floors)
