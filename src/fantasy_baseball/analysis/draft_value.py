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
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.redis_store import (
    get_game_log_totals,
)
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.draft.board import build_board_from_frames
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION, Category
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip
from fantasy_baseball.utils.time_utils import compute_fraction_remaining, local_today
from fantasy_baseball.web.season_data import (
    read_cache_dict,
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


def _pkey(norm: str, ptype: str) -> str:
    """Cross-source join key: ``name_normalized::player_type``."""
    return f"{norm}::{ptype}"


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
    scale = ScaleInputs(**scale_d)
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


def _sgp(line: dict[str, Any], scale: ScaleInputs, team_ab: float, team_ip: float) -> float:
    """Shared player-SGP call: score ``line`` on the board scale's denoms/repl rates."""
    return calculate_player_sgp(
        pd.Series(line),
        denoms=scale.denoms,
        team_ab=int(team_ab),
        team_ip=int(team_ip),
        replacement_avg=scale.repl_rates["avg"],
        replacement_era=scale.repl_rates["era"],
        replacement_whip=scale.repl_rates["whip"],
    )


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
            line["era"] = calculate_era(raw["er"], raw["ip"], default=0.0)
            line["whip"] = calculate_whip(raw["bb"], raw["h_allowed"], raw["ip"], default=0.0)
            line["player_type"] = (
                "pitcher"  # StrEnum-compatible; required by calculate_player_sgp dispatch
            )
        else:
            line["avg"] = calculate_avg(raw["h"], raw["ab"], default=0.0)
            line["player_type"] = "hitter"
        floors[pos] = _sgp(line, scale, team_ab, team_ip)
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
    ``"hitter"`` or ``"pitcher"``. The floors are always recomputed here via
    ``_to_date_floors`` (which early-returns the board floors at ``fraction==1.0``,
    so the projected path stays cheap).
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
    total_sgp = _sgp(scaled, scale, team_ab, team_ip)
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
        picks.append(DraftPick(None, kd["team"], drafted[i], True))

    # live picks: flatten ALL rounds, apply trades on the absolute cell, then SKIP the
    # first n_keep keeper-round slots and zip to live picks.
    rounds: list[list[str]] = [list(r) for r in order["rounds"]]
    for tr in order.get("trades", []):
        rounds[tr["round"] - 1][tr["slot"] - 1] = tr["to"]
    flat_teams = [team for rnd in rounds for team in rnd]
    live_teams = flat_teams[n_keep:]  # drop the 30 keeper-round slots (rounds 1-3)
    live = drafted[n_keep:]
    assert len(live) == len(live_teams), (
        f"draft reconstruction mismatch: {len(live)} live picks vs "
        f"{len(live_teams)} live slots (both should be 200)"
    )
    for slot, (name, team) in enumerate(zip(live, live_teams, strict=True), start=1):
        picks.append(DraftPick(slot, team, name, False))
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
    n_live = sum(1 for p in picks if not p.is_keeper)
    if n_live != 200:
        problems.append(f"non-keeper pick count {n_live} != 200")
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
        key = _pkey(row["name_normalized"], row["player_type"])
        cur = idx.get(key)
        if cur is None or float(row["var"]) > float(cur["var"]):
            idx[key] = row
    return idx


def _match_board_row(name: str, bindex: dict[str, Any]) -> Any:
    """Join a pick name to its board row across both player types; None if off-board."""
    norm = normalize_name(name)
    for ptype in ("hitter", "pitcher"):
        row = bindex.get(_pkey(norm, ptype))
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


def _var_for_row(
    row: Any,
    scale: ScaleInputs | None,
    fraction: float,
) -> float:
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
    bindex: dict[str, Any] | None = None,
) -> ParCurve:
    """Build the par curve from reconstructed picks joined to the preseason board.

    On-board drafted players contribute their (optionally to-date rescored) preseason
    VAR to a descending par curve; off-board fliers are skipped so the curve shrinks.
    Keeper par is the flat mean of the keeper VARs (keepers are elite, always on-board).
    ``fraction < 1.0`` requires ``scale`` so ``_var_for_row`` can rescore to the
    to-date scale. Pass ``bindex`` to reuse a prebuilt board index instead of
    rebuilding it from ``board``.
    """
    if bindex is None:
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
    # A JSON-null on a KV round-trip can land ab=None; calculate_avg guards on
    # denom > 0, which raises TypeError against a literal None, so coerce None->0.
    ab = rec.get("ab", 0)
    if ab is None:
        ab = 0
    return {
        "r": rec.get("r", 0),
        "hr": rec.get("hr", 0),
        "rbi": rec.get("rbi", 0),
        "sb": rec.get("sb", 0),
        "ab": ab,
        "avg": calculate_avg(rec.get("h", 0), ab, default=0.0),
    }


def _pit_line_from(rec: dict[str, Any]) -> dict[str, Any]:
    # ip=None (JSON-null on a KV round-trip) would raise TypeError in the rate-stat
    # denom guards (denom > 0), so coerce None->0 before scoring.
    ip = rec.get("ip", 0.0)
    if ip is None:
        ip = 0.0
    return {
        "w": rec.get("w", 0),
        "k": rec.get("k", 0),
        "sv": rec.get("sv", 0),
        "ip": ip,
        "era": calculate_era(rec.get("er", 0), ip, default=0.0),
        "whip": calculate_whip(rec.get("bb", 0), rec.get("h_allowed", 0), ip, default=0.0),
    }


def _row_mlbam(row: Any) -> int | None:
    """Extract a real integer mlbam id from a board row or projection record.

    Board rows and projection records carry ``mlbam_id`` as a float that may be
    NaN (an id-less player). Returns the int id when present and finite, else
    ``None`` (also None for a ``None`` row, so callers can pass a missing row).
    """
    try:
        v = row["mlbam_id"]
    except (KeyError, IndexError, TypeError):
        return None
    if v is None or v != v:  # None or NaN (NaN != NaN)
        return None
    return int(v)


def _insert_by_name(
    by_name: dict[str, Any],
    name_mlbam: dict[str, int | None],
    key: str,
    line: dict[str, Any],
    volume_key: str,
    mlbam: int | None,
) -> None:
    """Insert ``line`` into ``by_name`` keeping the higher-volume namesake.

    When two different players normalize to the same ``name::player_type`` key,
    the real MLB player (more ``ab`` for hitters / ``ip`` for pitchers) wins over
    a scrub namesake, so the fallback name-join scores the intended player. The
    collision is logged (not swallowed), naming both mlbam ids. The mlbam-keyed
    map is the authoritative join; this name map is only a fallback for rows that
    lack an id on either side.
    """
    cur = by_name.get(key)
    if cur is None:
        by_name[key] = line
        name_mlbam[key] = mlbam
        return
    cur_vol = float(cur[volume_key])
    new_vol = float(line[volume_key])
    new_wins = new_vol > cur_vol
    logger.warning(
        "draft-value: namesake collision on %s (%s=%.1f); keeping mlbam=%s over "
        "mlbam=%s (kept %.1f vs dropped %.1f)",
        key,
        volume_key,
        max(cur_vol, new_vol),
        mlbam if new_wins else name_mlbam.get(key),
        name_mlbam.get(key) if new_wins else mlbam,
        max(cur_vol, new_vol),
        min(cur_vol, new_vol),
    )
    if new_wins:
        by_name[key] = line
        name_mlbam[key] = mlbam


def load_full_season_lines() -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Full-season projection lines, keyed by mlbam id AND by ``name::player_type``.

    Reads ``CacheKey.FULL_SEASON_PROJECTIONS`` from the KV store (Upstash on
    Render, SQLite locally). Each record carries a real ``mlbam_id`` and a
    ``name``. Returns ``(by_mlbam, by_name)``: ``by_mlbam`` is the authoritative
    join (immune to namesake collisions like the two Mason Millers), keyed by int
    mlbam id; ``by_name`` is a fallback keyed ``name_normalized::player_type`` that
    keeps the higher-volume namesake so the real MLB player wins over a scrub.
    Returns ``({}, {})`` when the KV store lacks the blob (unsynced local runtime).
    """
    payload = read_cache_dict(CacheKey.FULL_SEASON_PROJECTIONS) or {}
    by_mlbam: dict[tuple[int, str], dict[str, Any]] = {}
    by_name: dict[str, Any] = {}
    name_mlbam: dict[str, int | None] = {}
    for ptype, recs, builder, vol in (
        ("hitter", payload.get("hitters", []), _hit_line_from, "ab"),
        ("pitcher", payload.get("pitchers", []), _pit_line_from, "ip"),
    ):
        for rec in recs:
            line = builder(rec)
            mlbam = _row_mlbam(rec)
            if mlbam is not None:
                # Key by (mlbam, player_type): a two-way player (e.g. Ohtani) has a
                # hitter AND a pitcher record under ONE mlbam id; keying by id alone
                # would let one overwrite the other.
                by_mlbam[(mlbam, ptype)] = line
            key = _pkey(normalize_name(rec["name"]), ptype)
            _insert_by_name(by_name, name_mlbam, key, line, vol, mlbam)
    return by_mlbam, by_name


def load_actual_to_date_lines() -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Actual season-to-date lines, keyed by mlbam id AND by ``name::player_type``.

    Reads aggregated game-log totals via ``get_game_log_totals`` (keyed by string
    mlbam id, with a ``name`` field). Returns ``(by_mlbam, by_name)``: ``by_mlbam``
    is the authoritative int-mlbam join; ``by_name`` is a fallback keyed
    ``name_normalized::player_type`` keeping the higher-volume namesake. Returns
    ``({}, {})`` when the KV store has no game logs.
    """
    client = get_kv()
    by_mlbam: dict[tuple[int, str], dict[str, Any]] = {}
    by_name: dict[str, Any] = {}
    name_mlbam: dict[str, int | None] = {}
    for ptype, logs, builder, vol in (
        ("hitter", get_game_log_totals(client, "hitters"), _hit_line_from, "ab"),
        ("pitcher", get_game_log_totals(client, "pitchers"), _pit_line_from, "ip"),
    ):
        for mid_str, rec in logs.items():
            line = builder(rec)
            try:
                mlbam: int | None = int(mid_str)
            except (TypeError, ValueError):
                mlbam = None
            if mlbam is not None:
                by_mlbam[(mlbam, ptype)] = line
            name = rec.get("name") or ""
            if not name:
                continue
            key = _pkey(normalize_name(name), ptype)
            _insert_by_name(by_name, name_mlbam, key, line, vol, mlbam)
    return by_mlbam, by_name


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
    slot: int | None = None,
    missing_line_est: float | None = None,
) -> PlayerValue:
    """Score a player's projected and YTD VAR and decompose the projected value.

    ``full_line`` is scored at ``fraction=1.0`` (full-season projection); ``todate_line``
    is scored at the elapsed ``fraction`` (to-date scale). ``value_proj``/``value_ytd``
    subtract the full-season and to-date baselines respectively (never conflate them).
    ``skill``/``luck`` split the projected value only when both a preseason VAR and a
    projected estimate exist; otherwise both are ``None``.

    ``missing_line_est`` is the estimated VAR used when a stat line is absent. Default
    ``None`` yields ``None`` estimates (skip). Pass ``0.0`` to score a drafted/kept
    player who never played at replacement level, so value == ``0 - par == -par`` and a
    wasted pick is penalized rather than silently dropped.
    """
    est_proj = (
        score_var(full_line, positions, player_type, scale, 1.0)
        if full_line is not None
        else missing_line_est
    )
    est_ytd = (
        score_var(todate_line, positions, player_type, scale, fraction)
        if todate_line is not None
        else missing_line_est
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
        slot,
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
    """League-wide elapsed fraction of the MLB season, computed purely from the date.

    Reads ``season_start``/``season_end`` from ``config/league.yaml`` and uses the
    user's local ``today`` (``local_today``), deriving the elapsed fraction as
    ``1 - compute_fraction_remaining(...)`` clamped to ``[0, 1]``. This is a
    date-based approximation of games played (it does NOT read any standings
    snapshot or per-team game count); it is the single source of the to-date
    scaling fraction ``f``.
    """
    config = load_config(_CONFIG)
    season_start = date.fromisoformat(config.season_start)
    season_end = date.fromisoformat(config.season_end)
    elapsed = 1.0 - compute_fraction_remaining(season_start, season_end, local_today())
    return max(0.0, min(1.0, elapsed))


@dataclass
class TeamRollup:
    """Per-team draft-value roll-up: sum, per-player average, and credited count.

    ``sum_value`` sums the chosen-horizon per-player values (``value_proj`` when
    ``horizon == "proj"``, else ``value_ytd``); ``avg_value`` divides by the number
    of credited players (``NaN`` when none are credited). ``credited_count`` is that
    number: every drafted pick + keeper credited to the team that drafted/kept the
    player, regardless of a later drop or trade.
    """

    team: str
    sum_value: float
    avg_value: float
    credited_count: int


def roll_up_team(
    team: str,
    player_values: list[PlayerValue],
    horizon: str = "proj",
) -> TeamRollup:
    """Roll a team's per-player values into sum, per-player average, and count.

    ``horizon`` picks ``value_proj`` (``"proj"``) or ``value_ytd`` (otherwise). Only
    non-``None`` values are credited (values can be ``0.0`` or negative, so filter on
    ``is not None``, never truthiness). ``avg_value`` is ``NaN`` when no player is
    credited. ``player_values`` covers every drafted pick + keeper credited to the
    team, so the sum, average, and credited count grade the full draft.
    """
    attr = "value_proj" if horizon == "proj" else "value_ytd"
    vals = [getattr(pv, attr) for pv in player_values if getattr(pv, attr) is not None]
    n = len(vals)
    total = sum(vals)
    avg = total / n if n else float("nan")
    return TeamRollup(team, total, avg, n)


def _default_positions(player_type: str) -> list[str]:
    """Off-board default positions matching the board's default-attach convention."""
    return ["OF"] if player_type == "hitter" else ["SP"]


def _assign_pick_types(
    picks: list[DraftPick],
    board: pd.DataFrame,
    league: dict[str, Any],
) -> list[tuple[DraftPick, str]]:
    """Assign a player_type to every pick, two-way aware.

    Draft-pick names are bare (no Yahoo " (Batter)"/" (Pitcher)" suffix), so a
    two-way player (e.g. Shohei Ohtani) appears as two picks under one name -- once
    as a keeper and once as a drafted pick. Iterating picks IN ORDER and tracking
    which board types each normalized name has already claimed lets the second pick
    take the remaining type: Ohtani's keeper claims "hitter" (his "batter only"
    keeper note), so the drafted Ohtani takes "pitcher".

    A KEEPER honors its league.yaml note ("batter only" -> hitter, "pitcher only" ->
    pitcher); otherwise it takes the sole board type, else defaults to hitter (or the
    lone pitcher type, or hitter off-board). A DRAFTED pick prefers an unclaimed board
    type (pitcher first, so a two-way second pick becomes pitcher), else the sole
    board type, else hitter.
    """
    bindex = _board_index(board)
    avail: dict[str, set[str]] = {}
    for key in bindex:
        norm, ptype = key.rsplit("::", 1)
        avail.setdefault(norm, set()).add(ptype)
    keeper_notes: dict[str, str] = {}
    for k in league["keepers"]:
        note = k.get("note")
        if note:
            keeper_notes[normalize_name(k["name"])] = str(note).lower()

    claimed: dict[str, set[str]] = {}
    result: list[tuple[DraftPick, str]] = []
    for p in picks:
        norm = normalize_name(p.player_name)
        avail_types = avail.get(norm, set())
        if p.is_keeper:
            note = keeper_notes.get(norm, "")
            if "batter only" in note:
                ptype = "hitter"
            elif "pitcher only" in note:
                ptype = "pitcher"
            elif len(avail_types) == 1:
                ptype = next(iter(avail_types))
            elif "hitter" in avail_types:
                ptype = "hitter"
            elif avail_types:  # board has only pitcher
                ptype = "pitcher"
            else:  # off-board keeper
                ptype = "hitter"
        else:  # drafted
            unclaimed = avail_types - claimed.get(norm, set())
            if unclaimed:
                ptype = "pitcher" if "pitcher" in unclaimed else next(iter(unclaimed))
            elif len(avail_types) == 1:
                ptype = next(iter(avail_types))
            elif "hitter" in avail_types:
                ptype = "hitter"
            else:  # off-board flier
                ptype = "hitter"
        claimed.setdefault(norm, set()).add(ptype)
        result.append((p, ptype))
    return result


def run_draft_value(
    fraction: float | None = None,
) -> tuple[list[PlayerValue], list[TeamRollup]]:
    """Orchestrate the draft-value metric end-to-end against the synced KV store.

    Reproduces the preseason board (+ soft frozen drift check), reconstructs the
    draft and enforces the reconstruction gate, builds projected and to-date par
    curves, then scores EVERY drafted pick + keeper (30 keepers + 200 drafted) by
    full-season realized value vs par, crediting the team that drafted/kept the
    player -- regardless of a later drop or trade. Waivers stay out (deferred to the
    transaction analyzer). A drafted/kept player who never played is scored at
    replacement (value == -par), so a wasted pick is penalized. Returns
    ``(player_values, team_rollups)``.
    """
    board, scale = reproduce_draft_day_board()
    frozen_drift_summary(board)  # soft: logs a warning on large drift, never raises
    picks = reconstruct_draft()
    # enforce the reconstruction gate against the user's known roster (spec oracle 6b)
    state = json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    user_roster: list[str] = state.get("user_roster") or []
    league = _load_league()
    keeper_team = {normalize_name(k["name"]): k["team"] for k in league["keepers"]}
    user_team = next(
        (keeper_team[normalize_name(n)] for n in user_roster if normalize_name(n) in keeper_team),
        None,
    )
    gate = validate_reconstruction(picks, known_team=user_team, known_roster=user_roster)
    if gate:
        raise RuntimeError(f"Draft reconstruction gate failed: {gate}")
    f = season_fraction() if fraction is None else fraction

    bindex = _board_index(board)
    preseason = {k: float(v["var"]) for k, v in bindex.items()}
    par_proj = build_par_curve(picks, board, fraction=1.0, bindex=bindex)
    par_ytd = build_par_curve(picks, board, fraction=f, scale=scale, bindex=bindex)
    full_by_mlbam, full_by_name = load_full_season_lines()
    td_by_mlbam, td_by_name = load_actual_to_date_lines()

    # DRAFT-ORDER ordinal among on-board drafted picks.
    # par(slot) = drafted_pars[ordinal-1] (drafted_pars is VAR-sorted desc): the k-th
    # on-board drafted pick is measured against the k-th-best available VAR. This is the
    # spec's par(slot). Do NOT index by the player's OWN VAR rank -- that makes
    # par == own VAR, so skill == preseason_var - par == 0 for every drafted player.
    slot_by_name: dict[str, int] = {}  # norm_name -> draft-order ordinal among on-board drafted
    onboard_ordinal = 0
    for p in sorted((pk for pk in picks if not pk.is_keeper), key=lambda x: x.slot or 0):
        if _match_board_row(p.player_name, bindex) is None:
            continue  # off-board flier: excluded from par curve and slot indexing
        onboard_ordinal += 1
        slot_by_name[normalize_name(p.player_name)] = onboard_ordinal

    # Score every pick (keepers + drafted), two-way aware, crediting pick.team.
    typed_picks = _assign_pick_types(picks, board, league)
    players: list[PlayerValue] = []
    for pick, ptype in typed_picks:
        norm = normalize_name(pick.player_name)
        key = _pkey(norm, ptype)
        row = bindex.get(key)  # may be None: off-board flier
        pre = preseason.get(key)
        mlbam = _row_mlbam(row)
        positions = list(row["positions"]) if row is not None else _default_positions(ptype)
        # Join lines by mlbam id first (immune to namesake collisions like the two
        # Mason Millers), falling back to the name key only when the board row has
        # no id. Explicit None checks -- never `x or fallback` (a 0.0-valued line
        # dict is falsy but valid).
        full_line = full_by_mlbam.get((mlbam, ptype)) if mlbam is not None else None
        if full_line is None:
            full_line = full_by_name.get(key)
        todate_line = td_by_mlbam.get((mlbam, ptype)) if mlbam is not None else None
        if todate_line is None:
            todate_line = td_by_name.get(key)
        # draft-order ordinal among on-board drafted picks; also the compute slot arg
        # (slot == rank for drafted, None otherwise), so compute it once and reuse.
        rank = slot_by_name.get(norm) if not pick.is_keeper else None
        if pick.is_keeper:
            base_proj, base_ytd = par_proj.keeper_par, par_ytd.keeper_par
        else:  # drafted
            base_proj = par_proj.par_for_slot(rank) if rank else 0.0
            base_ytd = par_ytd.par_for_slot(rank) if rank else 0.0
        pv = compute_player_value(
            team=pick.team,
            name=pick.player_name,
            player_type=ptype,
            positions=positions,
            baseline_proj=base_proj,
            baseline_ytd=base_ytd,
            baseline_kind=("keeper" if pick.is_keeper else "drafted"),
            preseason_var=pre,
            full_line=full_line,
            todate_line=todate_line,
            scale=scale,
            fraction=f,
            slot=(rank if not pick.is_keeper else None),
            # A drafted/kept player with NO stat line (never played) is scored at
            # replacement (0.0), so value == 0 - par == -par (a wasted pick is penalized).
            missing_line_est=0.0,
        )
        players.append(pv)

    # group by team (PlayerValue carries .team) and roll up
    by_team: dict[str, list[PlayerValue]] = {}
    for pv in players:
        by_team.setdefault(pv.team, []).append(pv)
    teams = [roll_up_team(t, pvs) for t, pvs in sorted(by_team.items())]
    return players, teams
