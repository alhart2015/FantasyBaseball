"""Draft-value metric: realized VAR vs draft-slot par expectation."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.draft.board import build_board_from_frames
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION, Category
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.web.season_data import (
    _load_game_log_totals,
    read_cache_dict,
    read_cache_list,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_BOARD = _REPO_ROOT / "data" / "draft_state_board.json"
_PRESEASON_CSVS = _REPO_ROOT / "data" / "projections" / "2026"
_POSITIONS_JSON = _REPO_ROOT / "data" / "player_positions.json"
_CONFIG_DIR = _REPO_ROOT / "config"
_CONFIG = _CONFIG_DIR / "league.yaml"
_DRAFT_ORDER = _CONFIG_DIR / "draft_order.json"
_DRAFT_STATE = _REPO_ROOT / "data" / "draft_state.json"


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


@dataclass(frozen=True)
class DraftPick:
    slot: int | None  # 1..200 live-pick ordinal; None for keepers
    round: int  # 0 for keepers; else absolute draft round (4..23)
    team: str
    player_name: str
    is_keeper: bool


def _load_league() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    return data


def reconstruct_draft() -> list[DraftPick]:
    """Reconstruct (team, slot) per 2026 pick from draft_order + draft_state + keepers.

    ``draft_order.json`` ``rounds`` is the full 23x10 snake order; rounds 1-3 (the
    first 30 slots) are consumed by the 30 keepers, so the 200 live picks map to the
    remaining slots. ``drafted_players[0:30]`` are the keepers in league.yaml order;
    ``drafted_players[30:230]`` are the live picks in snake order. Trades are applied
    on the absolute ``[round-1][slot-1]`` cell before the keeper-round slots are
    skipped.
    """
    order = json.loads(_DRAFT_ORDER.read_text(encoding="utf-8"))
    state = json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    league = _load_league()
    drafted: list[str] = state["drafted_players"]
    keeper_defs: list[dict[str, Any]] = league["keepers"]

    picks: list[DraftPick] = []
    n_keep = len(keeper_defs)
    for i, kd in enumerate(keeper_defs):
        picks.append(DraftPick(None, 0, kd["team"], drafted[i], True))

    # live picks: flatten ALL rounds (absolute round numbers), apply trades on the
    # absolute cell, then SKIP the first n_keep keeper-round slots and zip to live picks.
    rounds: list[list[str]] = [list(r) for r in order["rounds"]]
    for tr in order.get("trades", []):
        rounds[tr["round"] - 1][tr["slot"] - 1] = tr["to"]
    flat_teams = [(rnd_i + 1, team) for rnd_i, rnd in enumerate(rounds) for team in rnd]
    live_teams = flat_teams[n_keep:]  # drop the 30 keeper-round slots (rounds 1-3)
    live = drafted[n_keep:]
    for slot, (name, (rnd, team)) in enumerate(zip(live, live_teams, strict=False), start=1):
        picks.append(DraftPick(slot, rnd, team, name, False))
    return picks


def validate_reconstruction(
    picks: list[DraftPick],
    known_team: str | None = None,
    known_roster: list[str] | None = None,
) -> list[str]:
    """Runtime gate: every keeper maps to its owning team, and (optionally) a known
    team's roster reconstructs as a superset. Returns a list of problems ([] == pass).
    """
    problems: list[str] = []
    league = _load_league()
    keeper_norm = {(normalize_name(k["name"]), k["team"]) for k in league["keepers"]}
    recon_keeper_norm = {(normalize_name(p.player_name), p.team) for p in picks if p.is_keeper}
    missing = keeper_norm - recon_keeper_norm
    if missing:
        problems.append(f"keeper mismatch (missing {len(missing)}): {sorted(missing)[:5]}")
    if known_team and known_roster:
        recon = {normalize_name(p.player_name) for p in picks if p.team == known_team}
        want = {normalize_name(n) for n in known_roster}
        if not want <= recon:
            problems.append(
                f"known-team {known_team} roster mismatch: missing {sorted(want - recon)[:5]}"
            )
    return problems


def _board_index(board: pd.DataFrame) -> dict[str, Any]:
    """Map ``name_normalized::player_type`` -> board row (VAR tie-break on collisions)."""
    idx: dict[str, Any] = {}
    for _, row in board.iterrows():
        key = f"{row['name_normalized']}::{row['player_type']}"
        cur = idx.get(key)
        if cur is None or float(row["var"]) > float(cur["var"]):
            idx[key] = row
    return idx


def preseason_var_lookup(board: pd.DataFrame) -> dict[str, float]:
    """Preseason VAR keyed by ``name_normalized::player_type`` (collision-resolved)."""
    return {k: float(v["var"]) for k, v in _board_index(board).items()}


def _match_board_row(name: str, bindex: dict[str, Any]) -> Any:
    """Join a pick name to its board row across both player types; None if off-board."""
    norm = normalize_name(name)
    for ptype in ("hitter", "pitcher"):
        row = bindex.get(f"{norm}::{ptype}")
        if row is not None:
            return row
    return None


@dataclass
class ParCurve:
    """Draft-slot par expectation: sorted on-board drafted VAR plus keeper flat par."""

    drafted_pars: list[float]
    keeper_par: float

    def par_for_slot(self, ordinal: int) -> float:
        # ordinal is 1-based among ON-BOARD drafted picks (sorted descending)
        return self.drafted_pars[ordinal - 1]


def _var_for_row(row: Any, scale: ScaleInputs | None, fraction: float) -> float:
    """VAR for a board row: preseason VAR at f=1, else rescored on the to-date scale."""
    if fraction == 1.0:
        return float(row["var"])
    if scale is None:
        raise ValueError("scale is required when fraction != 1.0")
    ptype = str(row["player_type"])
    keys = (
        ("r", "hr", "rbi", "sb", "avg", "ab")
        if ptype == "hitter"
        else ("w", "k", "sv", "era", "whip", "ip")
    )
    line = {k: row[k] for k in keys}
    return score_var(line, list(row["positions"]), ptype, scale, fraction)


def build_par_curve(
    picks: list[DraftPick],
    board: pd.DataFrame,
    fraction: float = 1.0,
    scale: ScaleInputs | None = None,
) -> ParCurve:
    """Build the par curve from reconstructed picks joined to the preseason board.

    On-board drafted players contribute their (optionally to-date rescored) preseason
    VAR to a descending par curve; off-board fliers are skipped so the curve shrinks.
    Keeper par is the flat mean of the keeper VARs (keepers are elite, always on-board).
    ``fraction < 1.0`` requires ``scale`` so ``_var_for_row`` can rescore to the
    to-date scale.
    """
    bindex = _board_index(board)
    drafted_vars: list[float] = []
    keeper_vars: list[float] = []
    for p in picks:
        row = _match_board_row(p.player_name, bindex)
        if row is None:
            continue  # off-board flier: excluded from par curve
        v = _var_for_row(row, scale, fraction)
        if p.is_keeper:
            keeper_vars.append(v)
        else:
            drafted_vars.append(v)
    drafted_vars.sort(reverse=True)
    keeper_par = sum(keeper_vars) / len(keeper_vars) if keeper_vars else float("nan")
    return ParCurve(drafted_vars, keeper_par)


def _hit_line_from(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "r": rec.get("r", 0),
        "hr": rec.get("hr", 0),
        "rbi": rec.get("rbi", 0),
        "sb": rec.get("sb", 0),
        "ab": rec.get("ab", 0),
        "avg": (rec["h"] / rec["ab"]) if rec.get("ab") else 0.0,
    }


def _pit_line_from(rec: dict[str, Any]) -> dict[str, Any]:
    ip = rec.get("ip") or 0.0
    return {
        "w": rec.get("w", 0),
        "k": rec.get("k", 0),
        "sv": rec.get("sv", 0),
        "ip": ip,
        "era": (rec.get("er", 0) / ip * 9.0) if ip else 0.0,
        "whip": ((rec.get("bb", 0) + rec.get("h_allowed", 0)) / ip) if ip else 0.0,
    }


def load_full_season_lines() -> dict[str, Any]:
    """Full-season projection lines keyed ``name_normalized::player_type``.

    Reads ``CacheKey.FULL_SEASON_PROJECTIONS`` from the KV store (Upstash on
    Render, SQLite locally). Records are MLBAM-keyed with a ``name`` field, so
    normalize the name and tag the player type explicitly. Returns ``{}`` when
    the KV store lacks the blob (unsynced local runtime).
    """
    payload = read_cache_dict(CacheKey.FULL_SEASON_PROJECTIONS) or {}
    out: dict[str, Any] = {}
    for rec in payload.get("hitters", []):
        out[f"{normalize_name(rec['name'])}::hitter"] = _hit_line_from(rec)
    for rec in payload.get("pitchers", []):
        out[f"{normalize_name(rec['name'])}::pitcher"] = _pit_line_from(rec)
    return out


def load_actual_to_date_lines() -> dict[str, Any]:
    """Actual season-to-date lines keyed ``name_normalized::player_type``.

    Reads aggregated game-log totals via ``_load_game_log_totals`` (already
    keyed by normalized name in separate hitter/pitcher dicts, which gives the
    player type directly). Returns ``{}`` when the KV store has no game logs.
    """
    hitter_logs, pitcher_logs = _load_game_log_totals()
    out: dict[str, Any] = {}
    for norm, rec in hitter_logs.items():
        out[f"{norm}::hitter"] = _hit_line_from(rec)
    for norm, rec in pitcher_logs.items():
        out[f"{norm}::pitcher"] = _pit_line_from(rec)
    return out


@dataclass
class PlayerValue:
    """Per-player realized value with a projected skill/luck decomposition.

    ``value_proj`` subtracts the full-season ``baseline_proj``; ``value_ytd``
    subtracts the to-date ``baseline_ytd`` -- the two horizons have distinct pars
    and must never be conflated. ``skill``/``luck`` are only defined when the player
    was on the preseason board (``preseason_var is not None``); off-board fliers get
    ``None`` for both (value vs replacement is still computed).
    """

    team: str
    name: str
    player_type: str
    slot: int | None
    baseline_kind: str
    preseason_var: float | None
    est_var_proj: float | None
    est_var_ytd: float | None
    value_proj: float | None
    value_ytd: float | None
    skill: float | None
    luck: float | None


def compute_player_value(
    team: str,
    name: str,
    player_type: str,
    positions: list[str],
    baseline_proj: float,
    baseline_ytd: float,
    baseline_kind: str,
    preseason_var: float | None,
    full_line: dict[str, Any] | None,
    todate_line: dict[str, Any] | None,
    scale: ScaleInputs,
    fraction: float,
) -> PlayerValue:
    """Score a player's projected and YTD VAR and decompose the projected value.

    ``full_line`` is scored at ``fraction=1.0`` (full-season projection); ``todate_line``
    is scored at the elapsed ``fraction`` (to-date scale). ``value_proj``/``value_ytd``
    subtract the full-season and to-date baselines respectively (never conflate them).
    ``skill``/``luck`` split the projected value only when both a preseason VAR and a
    projected estimate exist; otherwise both are ``None``.
    """
    est_proj = (
        score_var(full_line, positions, player_type, scale, 1.0) if full_line is not None else None
    )
    est_ytd = (
        score_var(todate_line, positions, player_type, scale, fraction)
        if todate_line is not None
        else None
    )
    value_proj = (est_proj - baseline_proj) if est_proj is not None else None
    value_ytd = (est_ytd - baseline_ytd) if est_ytd is not None else None
    if preseason_var is not None and est_proj is not None:
        skill: float | None = preseason_var - baseline_proj
        luck: float | None = est_proj - preseason_var
    else:
        skill = luck = None
    return PlayerValue(
        team,
        name,
        player_type,
        None,
        baseline_kind,
        preseason_var,
        est_proj,
        est_ytd,
        value_proj,
        value_ytd,
        skill,
        luck,
    )


def season_fraction() -> float:
    """League games played / full schedule. v1: date-based fraction of the MLB season.

    Read the elapsed fraction from the standings snapshot game count if available;
    otherwise fall back to a date-based fraction. Pin the exact source when wiring
    the CLI (Task 9) against real standings; keep this helper the single source.
    """
    season_start = date(2026, 3, 26)
    season_end = date(2026, 9, 28)
    today = date.today()
    total = (season_end - season_start).days
    done = max(0, min(total, (today - season_start).days))
    return done / total


def normalize_name_team(team: str) -> str:
    """Normalize a fantasy team name for keying: trimmed and lowercased."""
    return team.strip().lower()


def load_add_txns_by_team() -> dict[str, set[str]]:
    """Map fantasy team (normalized) -> set of normalized add-txn player names.

    Reads the raw ``CacheKey.TRANSACTIONS`` LIST payload from the KV store and
    keeps only successful adds (``status`` unset or ``"successful"``). Used by the
    elimination-model classifier to distinguish waiver pickups from trade
    acquisitions. Returns ``{}`` when the KV store lacks the blob.
    """
    txns = read_cache_list(CacheKey.TRANSACTIONS) or []
    by_team: dict[str, set[str]] = {}
    for t in txns:
        if t.get("status") not in (None, "successful"):
            continue
        add_name = t.get("add_name")
        team = t.get("team")
        if add_name and team:
            by_team.setdefault(normalize_name_team(team), set()).add(normalize_name(add_name))
    return by_team


def classify_acquisition(
    team: str,
    norm_name: str,
    drafted_by_team: dict[str, set[str]],
    kept_by_team: dict[str, set[str]],
    add_by_team: dict[str, set[str]],
) -> str:
    """Classify how ``team`` acquired ``norm_name`` via elimination precedence.

    Precedence: drafted -> keeper -> waiver -> trade_excluded. A rostered player
    with no draft/keep/add record was trade-acquired, which the draft-value metric
    excludes. Draft/keep beat a later same-team re-add of the same player.
    """
    tkey = normalize_name_team(team)
    if norm_name in drafted_by_team.get(tkey, set()):
        return "drafted"
    if norm_name in kept_by_team.get(tkey, set()):
        return "keeper"
    if norm_name in add_by_team.get(tkey, set()):
        return "waiver"
    return "trade_excluded"


@dataclass
class TeamRollup:
    """Per-team draft-value roll-up: sum, per-player average, and credited counts.

    ``sum_value`` sums the chosen-horizon per-player values (``value_proj`` when
    ``horizon == "proj"``, else ``value_ytd``); ``avg_value`` divides by the number
    of credited players (``NaN`` when none are credited). ``credited_count`` is that
    number; ``case3_count`` is passed through from the caller's classification.
    """

    team: str
    sum_value: float
    avg_value: float
    credited_count: int
    case3_count: int


def roll_up_team(
    team: str,
    player_values: list[PlayerValue],
    case3_count: int,
    horizon: str = "proj",
) -> TeamRollup:
    """Roll a team's per-player values into sum, per-player average, and counts.

    ``horizon`` picks ``value_proj`` (``"proj"``) or ``value_ytd`` (otherwise). Only
    non-``None`` values are credited (values can be ``0.0`` or negative, so filter on
    ``is not None``, never truthiness). ``avg_value`` is ``NaN`` when no player is
    credited. ``case3_count`` is passed through unchanged.
    """
    attr = "value_proj" if horizon == "proj" else "value_ytd"
    vals = [getattr(pv, attr) for pv in player_values if getattr(pv, attr) is not None]
    n = len(vals)
    total = sum(vals)
    avg = total / n if n else float("nan")
    return TeamRollup(team, total, avg, n, case3_count)
