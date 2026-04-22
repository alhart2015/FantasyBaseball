"""Transaction analysis — pairing, ΔRoto scoring, and aggregation.

Each transaction (or paired drop+add) is scored by the roto-point
impact on the team's projected end-of-season standings. ΔRoto is
non-linear in the swap — a drop and its paired add MUST be scored as
one swap, not two independently-scored sides that sum. Accordingly
``score_transaction`` takes an optional ``partner`` so callers can
surface the pairing up front and score each paired move exactly once,
attributing the full delta to the drop side (add side gets 0).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

from fantasy_baseball.models.league import League
from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType
from fantasy_baseball.models.roster import Roster
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.trades.evaluate import apply_swap_delta
from fantasy_baseball.utils.constants import (
    REPLACEMENT_HITTER,
    REPLACEMENT_RP,
    REPLACEMENT_SP,
    Category,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_era, calculate_whip

HITTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "Util", "UTIL", "DH", "IF"}
PITCHER_POSITIONS = {"SP", "RP", "P"}

PAIRING_WINDOW_SECONDS = 86400  # 24 hours


def _parse_positions(pos_str: str | None) -> set[str]:
    if not pos_str:
        return set()
    return {p.strip() for p in pos_str.split(",")}


def _is_hitter(positions: set[str]) -> bool:
    return bool(positions & HITTER_POSITIONS)


def _is_pitcher(positions: set[str]) -> bool:
    return bool(positions & PITCHER_POSITIONS)


def pair_standalone_moves(transactions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Find pairs of standalone drop + add from the same team within 24h.

    Only considers transactions with type "drop" or "add" (not "add/drop").
    Skips transactions that already have a paired_with value.

    Matching priority:
    1. Exact position overlap between dropped and added player
    2. Same player type (hitter/pitcher)
    3. No match — left unpaired
    """
    drops: list[dict[str, Any]] = []
    adds: list[dict[str, Any]] = []
    for txn in transactions:
        if txn.get("paired_with"):
            continue
        if txn["type"] == "drop":
            drops.append(txn)
        elif txn["type"] == "add":
            adds.append(txn)

    paired_drop_ids: set[str] = set()
    paired_add_ids: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for drop in drops:
        drop_ts = int(drop.get("timestamp", 0) or 0)
        drop_pos = _parse_positions(drop.get("drop_positions"))
        drop_is_hitter = _is_hitter(drop_pos)
        drop_is_pitcher = _is_pitcher(drop_pos)

        candidates: list[dict[str, Any]] = []
        for add in adds:
            if add["transaction_id"] in paired_add_ids:
                continue
            if add["team"] != drop["team"]:
                continue
            add_ts = int(add.get("timestamp", 0) or 0)
            if abs(add_ts - drop_ts) > PAIRING_WINDOW_SECONDS:
                continue
            candidates.append(add)

        if not candidates:
            continue

        best: dict[str, Any] | None = None
        best_score = 0
        for add in candidates:
            add_pos = _parse_positions(add.get("add_positions"))
            overlap = drop_pos & add_pos
            if overlap:
                score = 2 + len(overlap)
            elif (drop_is_hitter and _is_hitter(add_pos)) or (
                drop_is_pitcher and _is_pitcher(add_pos)
            ):
                score = 1
            else:
                score = 0

            if score > best_score:
                best_score = score
                best = add

        if best and best_score > 0:
            pairs.append((drop["transaction_id"], best["transaction_id"]))
            paired_drop_ids.add(drop["transaction_id"])
            paired_add_ids.add(best["transaction_id"])

    return pairs


# --------------------------------------------------------------------------
# ΔRoto scoring
# --------------------------------------------------------------------------

# Empty ROS dict shared by the rare case where both sides are missing.
_ZERO_ROS = {
    "R": 0.0,
    "HR": 0.0,
    "RBI": 0.0,
    "SB": 0.0,
    "AVG": 0.0,
    "W": 0.0,
    "K": 0.0,
    "SV": 0.0,
    "ERA": 0.0,
    "WHIP": 0.0,
    "ab": 0.0,
    "ip": 0.0,
}


def _txn_date(ts) -> date:
    try:
        return datetime.fromtimestamp(int(ts)).date()
    except (ValueError, TypeError, OSError):
        return date.today()


def _frac_remaining(txn_date: date, season_start: date, season_end: date) -> float:
    """Return the fraction of the season remaining at ``txn_date``.

    Clamped to ``[0, 1]``. A transaction at or after ``season_end``
    returns 0.0 (replacement contributes nothing); before
    ``season_start`` returns 1.0 (full-season replacement).
    """
    total = (season_end - season_start).days
    if total <= 0:
        return 1.0
    if txn_date <= season_start:
        return 1.0
    if txn_date >= season_end:
        return 0.0
    remaining = (season_end - txn_date).days
    return max(0.0, min(1.0, remaining / total))


def _hitter_ros(stats: HitterStats) -> dict[str, float]:
    return {
        "R": stats.r,
        "HR": stats.hr,
        "RBI": stats.rbi,
        "SB": stats.sb,
        "AVG": stats.avg,
        "W": 0.0,
        "K": 0.0,
        "SV": 0.0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "ab": stats.ab,
        "ip": 0.0,
    }


def _pitcher_ros(stats: PitcherStats) -> dict[str, float]:
    return {
        "R": 0.0,
        "HR": 0.0,
        "RBI": 0.0,
        "SB": 0.0,
        "AVG": 0.0,
        "W": stats.w,
        "K": stats.k,
        "SV": stats.sv,
        "ERA": stats.era,
        "WHIP": stats.whip,
        "ab": 0.0,
        "ip": stats.ip,
    }


def _prorated_replacement_hitter(frac: float) -> dict[str, float]:
    """Full-season ``REPLACEMENT_HITTER`` counting stats scaled by ``frac``.

    Rate stats (AVG) stay at their full-season value because
    ``apply_swap_delta`` multiplies AVG by AB to back out hits — a
    prorated AB already shrinks the hits contribution correctly.
    """
    h = REPLACEMENT_HITTER
    avg = h["h"] / h["ab"] if h["ab"] else 0.0
    return {
        "R": h["r"] * frac,
        "HR": h["hr"] * frac,
        "RBI": h["rbi"] * frac,
        "SB": h["sb"] * frac,
        "AVG": avg,
        "W": 0.0,
        "K": 0.0,
        "SV": 0.0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "ab": h["ab"] * frac,
        "ip": 0.0,
    }


def _prorated_replacement_pitcher(positions: set[str], frac: float) -> dict[str, float]:
    """Replacement pitcher ROS scaled by ``frac``.

    Pulls from ``REPLACEMENT_SP`` when the dropped player had SP
    eligibility, otherwise ``REPLACEMENT_RP`` (more favorable SV/ERA
    profile). Rate stats stay at full-season values — see
    :func:`_prorated_replacement_hitter` for the reasoning.
    """
    p = REPLACEMENT_SP if "SP" in positions else REPLACEMENT_RP
    era = calculate_era(p["er"], p["ip"], default=0.0)
    whip = calculate_whip(p["bb"], p["h_allowed"], p["ip"], default=0.0)
    return {
        "R": 0.0,
        "HR": 0.0,
        "RBI": 0.0,
        "SB": 0.0,
        "AVG": 0.0,
        "W": p["w"] * frac,
        "K": p["k"] * frac,
        "SV": p["sv"] * frac,
        "ERA": era,
        "WHIP": whip,
        "ab": 0.0,
        "ip": p["ip"] * frac,
    }


def _lookup_player(
    name: str | None,
    positions: set[str],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
) -> tuple[dict[str, float] | None, PlayerType | None, float]:
    """Find a player's ROS stats in the projection DataFrames.

    Checks hitter DF if the player has hitter-eligible positions, pitcher
    DF if pitcher-eligible. Returns ``(ros_dict, player_type, sgp)`` or
    ``(None, None, 0.0)`` when the name isn't in either DF.
    """
    if not name:
        return None, None, 0.0
    norm = normalize_name(name)

    if _is_hitter(positions) and not hitters_proj.empty:
        matches = hitters_proj[hitters_proj["_name_norm"] == norm]
        if not matches.empty:
            h_stats = HitterStats.from_dict(matches.iloc[0].to_dict())
            return _hitter_ros(h_stats), PlayerType.HITTER, calculate_player_sgp(h_stats)

    if _is_pitcher(positions) and not pitchers_proj.empty:
        matches = pitchers_proj[pitchers_proj["_name_norm"] == norm]
        if not matches.empty:
            p_stats = PitcherStats.from_dict(matches.iloc[0].to_dict())
            return _pitcher_ros(p_stats), PlayerType.PITCHER, calculate_player_sgp(p_stats)

    return None, None, 0.0


def _worst_at_position(
    roster: Roster | None,
    add_positions: set[str],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
) -> dict[str, float] | None:
    """ROS stats of the worst rostered player with overlapping positions.

    Ranks candidates by ``calculate_player_sgp`` and picks the lowest.
    Returns ``None`` when the roster has no same-position candidate or
    when none of them are in the projections.
    """
    if roster is None or not add_positions:
        return None

    worst_ros: dict[str, float] | None = None
    worst_sgp = float("inf")
    for entry in roster.entries:
        entry_positions = {p.value for p in entry.positions}
        if not (entry_positions & add_positions):
            continue
        ros, _, sgp = _lookup_player(
            entry.name,
            entry_positions,
            hitters_proj,
            pitchers_proj,
        )
        if ros is None:
            continue
        if sgp < worst_sgp:
            worst_sgp = sgp
            worst_ros = ros

    return worst_ros


def _delta_roto(
    team_name: str,
    loses_ros: dict[str, float],
    gains_ros: dict[str, float],
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> float:
    """Return the team's total-ΔRoto from swapping ``loses_ros`` → ``gains_ros``.

    Returns 0.0 when ``team_name`` isn't in ``projected_standings`` — a
    cold-start condition (e.g. the very first refresh before standings
    exist) where scoring can't be computed meaningfully.

    When ``team_sds`` is supplied, ``score_roto`` returns fractional
    ERoto points; otherwise it falls back to integer rank-based
    scoring.
    """
    all_before = {e.team_name: e.stats.to_dict() for e in projected_standings.entries}
    if team_name not in all_before:
        return 0.0

    all_after = dict(all_before)
    all_after[team_name] = apply_swap_delta(
        all_before[team_name],
        loses_ros,
        gains_ros,
    )

    roto_before = score_roto_dict(all_before, team_sds=team_sds)
    roto_after = score_roto_dict(all_after, team_sds=team_sds)
    return roto_after[team_name]["total"] - roto_before[team_name]["total"]


def score_transaction(
    league: League,
    txn: dict[str, Any],
    projected_standings: ProjectedStandings,
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    season_start: date,
    season_end: date,
    *,
    partner: dict[str, Any] | None = None,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> dict[str, float]:
    """Compute ΔRoto for a transaction.

    Contract: a paired drop+add is scored as one swap; the full ΔRoto
    is attributed to the drop-side call and the add-side call returns
    0.0 so ``sum(delta_roto)`` across a team's transaction list equals
    the team's net ΔRoto.

    Counterfactuals for solo moves:
    - Solo add: the added player displaces the worst-rostered player at
      any of the added player's positions (falls back to prorated
      replacement when no same-position teammate is in projections).
    - Solo drop: the dropped player is replaced with a prorated
      replacement-level hitter/SP/RP.

    Args:
        league: League model with standings + roster history.
        txn: Transaction dict (team, type, timestamp, add_name,
            add_positions, drop_name, drop_positions).
        projected_standings: end-of-season standings baseline
            (typed :class:`ProjectedStandings`).
        hitters_proj / pitchers_proj: ROS projection DataFrames with a
            ``_name_norm`` column (added by
            :func:`_load_projections_for_date_redis`).
        season_start / season_end: used to prorate replacement stats.
        partner: the paired transaction, when this txn is part of a
            drop+add pair. Presence is what makes this a "paired" score.
        team_sds: per-team per-category standard deviations (``{team:
            {Category: sd}}``) for fractional ERoto scoring, or ``None``
            for integer rank-based roto. Required keyword — no default,
            so callers can't silently get integer roto by forgetting the
            argument.

    Returns:
        ``{"delta_roto": float}``.
    """
    team_name = txn["team"]
    txn_d = _txn_date(txn.get("timestamp"))
    frac = _frac_remaining(txn_d, season_start, season_end)

    txn_type = txn.get("type")

    # --- Paired: drop side computes the full delta; add side returns 0. ---
    if partner is not None:
        if txn_type == "add":
            return {"delta_roto": 0.0}
        drop_name = txn.get("drop_name")
        drop_positions = _parse_positions(txn.get("drop_positions"))
        add_name = partner.get("add_name")
        add_positions = _parse_positions(partner.get("add_positions"))

        loses_ros, _, _ = _lookup_player(
            drop_name,
            drop_positions,
            hitters_proj,
            pitchers_proj,
        )
        gains_ros, _, _ = _lookup_player(
            add_name,
            add_positions,
            hitters_proj,
            pitchers_proj,
        )
        if loses_ros is None:
            loses_ros = (
                _prorated_replacement_pitcher(drop_positions, frac)
                if _is_pitcher(drop_positions)
                else _prorated_replacement_hitter(frac)
            )
        if gains_ros is None:
            gains_ros = dict(_ZERO_ROS)

        delta = _delta_roto(team_name, loses_ros, gains_ros, projected_standings, team_sds)
        return {"delta_roto": round(delta, 2)}

    # --- Solo add: counterfactual is the worst same-position roster slot. ---
    if txn_type == "add":
        add_name = txn.get("add_name")
        add_positions = _parse_positions(txn.get("add_positions"))
        gains_ros, _, _ = _lookup_player(
            add_name,
            add_positions,
            hitters_proj,
            pitchers_proj,
        )
        if gains_ros is None:
            return {"delta_roto": 0.0}

        try:
            team = league.team_by_name(team_name)
        except KeyError:
            team = None
        roster = team.roster_as_of(txn_d) if team else None
        loses_ros = _worst_at_position(
            roster,
            add_positions,
            hitters_proj,
            pitchers_proj,
        )
        if loses_ros is None:
            loses_ros = (
                _prorated_replacement_pitcher(add_positions, frac)
                if _is_pitcher(add_positions)
                else _prorated_replacement_hitter(frac)
            )

        delta = _delta_roto(team_name, loses_ros, gains_ros, projected_standings, team_sds)
        return {"delta_roto": round(delta, 2)}

    # --- Solo drop: counterfactual is the replacement-level floor. ---
    if txn_type == "drop":
        drop_name = txn.get("drop_name")
        drop_positions = _parse_positions(txn.get("drop_positions"))
        loses_ros, _, _ = _lookup_player(
            drop_name,
            drop_positions,
            hitters_proj,
            pitchers_proj,
        )
        if loses_ros is None:
            return {"delta_roto": 0.0}

        gains_ros = (
            _prorated_replacement_pitcher(drop_positions, frac)
            if _is_pitcher(drop_positions)
            else _prorated_replacement_hitter(frac)
        )

        delta = _delta_roto(team_name, loses_ros, gains_ros, projected_standings, team_sds)
        return {"delta_roto": round(delta, 2)}

    # --- Already-paired "add/drop" txn (Yahoo's native pair). Score directly. ---
    add_name = txn.get("add_name")
    add_positions = _parse_positions(txn.get("add_positions"))
    drop_name = txn.get("drop_name")
    drop_positions = _parse_positions(txn.get("drop_positions"))

    loses_ros, _, _ = _lookup_player(
        drop_name,
        drop_positions,
        hitters_proj,
        pitchers_proj,
    )
    gains_ros, _, _ = _lookup_player(
        add_name,
        add_positions,
        hitters_proj,
        pitchers_proj,
    )
    if loses_ros is None:
        loses_ros = (
            _prorated_replacement_pitcher(drop_positions, frac)
            if _is_pitcher(drop_positions)
            else _prorated_replacement_hitter(frac)
        )
    if gains_ros is None:
        gains_ros = dict(_ZERO_ROS)

    delta = _delta_roto(team_name, loses_ros, gains_ros, projected_standings, team_sds)
    return {"delta_roto": round(delta, 2)}


def _load_projections_for_date_redis(client):
    """Load the latest ROS projection DataFrames from Redis.

    Redis holds only the freshest snapshot (no per-date history), so
    every transaction — new or historical — is scored against the
    same current ROS projections. The safety invariant that makes
    this OK: ``run_full_refresh`` invokes ``score_transaction`` only
    for newly-discovered transactions; previously-scored ones keep
    their cached ΔRoto and are NOT re-derived, so later projection
    shifts cannot retroactively change historical scores.

    Falls back to preseason blended projections when ROS is empty —
    useful for the first refresh of a new season before any ROS
    snapshot has been written.
    """
    from fantasy_baseball.data.redis_store import (
        get_blended_projections,
        get_ros_projections,
    )

    ros = get_ros_projections(client) or {}
    hitters_rows: list[dict[str, Any]] = list(ros.get("hitters") or [])
    pitchers_rows: list[dict[str, Any]] = list(ros.get("pitchers") or [])

    if not hitters_rows and not pitchers_rows:
        hitters_rows = get_blended_projections(client, "hitters") or []
        pitchers_rows = get_blended_projections(client, "pitchers") or []

    hitters_df = pd.DataFrame(hitters_rows)
    pitchers_df = pd.DataFrame(pitchers_rows)

    if not hitters_df.empty and "name" in hitters_df.columns:
        hitters_df["_name_norm"] = hitters_df["name"].apply(normalize_name)
    if not pitchers_df.empty and "name" in pitchers_df.columns:
        pitchers_df["_name_norm"] = pitchers_df["name"].apply(normalize_name)

    return hitters_df, pitchers_df


# --------------------------------------------------------------------------
# Display-cache builder
# --------------------------------------------------------------------------


def build_cache_output(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the JSON cache structure for the Transactions tab.

    Groups transactions by team, computes per-team net ΔRoto, and
    sorts teams by net value descending. Paired standalone moves merge
    into a single display entry whose ``delta_roto`` is the drop-side
    score (the add side stored 0.0 by construction).
    """
    by_id = {t["transaction_id"]: t for t in transactions}
    rendered = set()
    teams: dict[str, dict[str, Any]] = {}

    for txn in transactions:
        tid = txn["transaction_id"]
        if tid in rendered:
            continue

        team = txn["team"]
        if team not in teams:
            teams[team] = {"team": team, "transactions": [], "net_value": 0.0}

        paired_id = txn.get("paired_with")
        paired = by_id.get(paired_id) if paired_id else None

        if paired and paired["transaction_id"] not in rendered:
            drop_txn = txn if txn["type"] == "drop" else paired
            add_txn = paired if txn["type"] == "drop" else txn
            entry = {
                "transaction_id": drop_txn["transaction_id"],
                "date": _ts_to_date(drop_txn.get("timestamp")),
                "type": "add/drop",
                "add_name": add_txn.get("add_name"),
                "add_positions": _split_positions(add_txn.get("add_positions")),
                "drop_name": drop_txn.get("drop_name"),
                "drop_positions": _split_positions(drop_txn.get("drop_positions")),
                "delta_roto": round(drop_txn.get("delta_roto", 0) or 0, 2),
                "paired": True,
            }
            rendered.add(drop_txn["transaction_id"])
            rendered.add(add_txn["transaction_id"])
        else:
            entry = {
                "transaction_id": tid,
                "date": _ts_to_date(txn.get("timestamp")),
                "type": txn["type"],
                "add_name": txn.get("add_name"),
                "add_positions": _split_positions(txn.get("add_positions")),
                "drop_name": txn.get("drop_name"),
                "drop_positions": _split_positions(txn.get("drop_positions")),
                "delta_roto": round(txn.get("delta_roto", 0) or 0, 2),
                "paired": False,
            }
            rendered.add(tid)

        teams[team]["transactions"].append(entry)
        teams[team]["net_value"] = round(teams[team]["net_value"] + entry["delta_roto"], 2)

    team_list = sorted(teams.values(), key=lambda t: t["net_value"], reverse=True)
    for t in team_list:
        t["transaction_count"] = len(t["transactions"])
        t["transactions"].sort(key=lambda x: x.get("date", ""))

    return {"teams": team_list}


def _ts_to_date(timestamp):
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return None


def _split_positions(pos_str):
    if not pos_str:
        return []
    return [p.strip() for p in pos_str.split(",")]
