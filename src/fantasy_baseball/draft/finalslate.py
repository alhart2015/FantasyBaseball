"""Final-slate field projection for the draft recommender.

The live recommender (``build_team_rosters``) pads every team's empty roster
slots with *replacement-level* players, then scores a candidate against that
field. Early in the draft every opponent therefore looks like a replacement
team, and their projected category totals climb pick-by-pick as real players
displace the padding -- a target that shifts under the recommender's feet.

``deltaroto_finalslate`` replaces that padding with a *realistic* end state:
forward-simulate the rest of the draft by ADP so every team (the picking team
included) ends with a full, real roster, and score the candidate's marginal
roto against that stable field. The candidate displaces the picking team's own
*marginal ADP filler* (the worst player it would otherwise end up with at that
slot) rather than a replacement line -- so the swap math stays consistent with
the realistic baseline.

This module builds the field + the picking team's per-bucket marginal fillers;
:func:`eroto_recs.rank_candidates_finalslate` consumes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fantasy_baseball.draft.adp import ADPTable
from fantasy_baseball.draft.eroto_recs import pitcher_role
from fantasy_baseball.draft.recs_integration import build_projected_standings
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.scoring import build_team_sds
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.positions import can_fill_slot

# Buckets a candidate (and a filler) is valued against, mirroring
# eroto_recs._pick_replacement: hitters share one bucket (all hitters
# contribute the same five categories, so the displaced filler's *type* is what
# matters for the stat delta), pitchers split SP/RP because the RP line carries
# saves.
_HITTER = "HITTER"


def _positions_for(
    pid: str, player_lookup: Mapping[str, Any], board_by_id: Mapping[str, Player]
) -> list[str]:
    """Position-code strings for a player id (board row first, Player fallback)."""
    row = player_lookup.get(pid)
    if row is not None:
        positions = row["positions"]
        return [str(p) for p in positions]
    player = board_by_id.get(pid)
    if player is not None:
        return [p.value if hasattr(p, "value") else str(p) for p in player.positions]
    return []


def _assign_to_slot(
    positions: list[str],
    filled: dict[str, int],
    roster_slots: Mapping[str, int],
    scarcity_order: list[str],
) -> str | None:
    """Greedily seat a player in the scarcest open slot; mutate ``filled``.

    Mirrors ``simulate_draft._assign_slot``: specific slots (scarcest first),
    then flex (IF/UTIL), then bench/IL overflow. Returns the slot taken, or
    ``None`` if the player fits nowhere (roster full).
    """
    if scarcity_order:
        active = [s for s in scarcity_order if s not in ("BN", "IL", "IF", "UTIL")]
        flex = [s for s in scarcity_order if s in ("IF", "UTIL")]
    else:
        active = [p for p in roster_slots if p not in ("BN", "IL", "IF", "UTIL")]
        flex = ["IF", "UTIL"]
    for slot in active + flex + ["BN", "IL"]:
        if slot not in roster_slots:
            continue
        if filled.get(slot, 0) < roster_slots[slot] and can_fill_slot(positions, slot):
            filled[slot] = filled.get(slot, 0) + 1
            return slot
    return None


def _snake_team(pick_index_1based: int, num_teams: int) -> int:
    """1-indexed team on the clock at global pick ``pick_index_1based`` (snake)."""
    rnd = (pick_index_1based - 1) // num_teams + 1
    pos = (pick_index_1based - 1) % num_teams + 1
    return pos if rnd % 2 == 1 else num_teams - pos + 1


def forward_fill_rosters(
    *,
    team_rosters: Mapping[int, list[str]],
    board_by_id: Mapping[str, Player],
    player_lookup: Mapping[str, Any],
    adp_table: ADPTable,
    roster_slots: Mapping[str, int],
    scarcity_order: list[str],
    num_teams: int,
    current_pick: int,
    total_picks: int,
    drafted: set[str],
) -> dict[int, list[str]]:
    """Project each team's final roster by ADP-greedy snake fill.

    From global pick ``current_pick`` to ``total_picks``, the team on the clock
    takes the lowest-ADP available board player that fits one of its open slots.
    Keepers/picks already in ``team_rosters`` are kept; the returned rosters add
    the projected future picks. Returns ``{team_num: [pid, ...]}``.
    """
    final: dict[int, list[str]] = {num: list(pids) for num, pids in team_rosters.items()}
    # Replay current rosters through the one slot model so future fills see a
    # consistent filled-slot view (avoids drift vs get_filled_positions).
    filled: dict[int, dict[str, int]] = {num: {} for num in team_rosters}
    for num, pids in final.items():
        for pid in pids:
            _assign_to_slot(
                _positions_for(pid, player_lookup, board_by_id),
                filled[num],
                roster_slots,
                scarcity_order,
            )

    taken = set(drafted)
    # ADP-sorted available pool (cheapest pointer-free scan: skip taken as we go).
    available = sorted(
        (pid for pid in board_by_id if pid not in taken),
        key=lambda pid: adp_table.get(pid),
    )

    for pick in range(current_pick, total_picks + 1):
        team_num = _snake_team(pick, num_teams)
        if team_num not in final:
            continue
        for pid in available:
            if pid in taken:
                continue
            slot = _assign_to_slot(
                _positions_for(pid, player_lookup, board_by_id),
                filled[team_num],
                roster_slots,
                scarcity_order,
            )
            if slot is not None:
                final[team_num].append(pid)
                taken.add(pid)
                break
    return final


def _bucket(player: Player) -> str:
    """Filler bucket for a player: HITTER, or SP/RP for a pitcher."""
    if player.player_type == PlayerType.PITCHER:
        return pitcher_role(player)
    return _HITTER


def _filler_value(pid: str, player_lookup: Mapping[str, Any], player: Player) -> float:
    """Value proxy for ranking a team's fillers (lower = more marginal).

    Prefer the board's ``var`` (the value ordering the draft itself uses); fall
    back to ROS SGP, then 0.0.
    """
    row = player_lookup.get(pid)
    if row is not None and row.get("var") is not None:
        return float(row["var"])
    ros = player.rest_of_season
    sgp = getattr(ros, "sgp", None) if ros is not None else None
    if sgp is not None:
        return float(sgp)
    return 0.0


def marginal_fillers(
    *,
    current_roster: list[str],
    final_roster: list[str],
    board_by_id: Mapping[str, Player],
    player_lookup: Mapping[str, Any],
) -> dict[str, Player]:
    """Worst (most marginal) projected filler per bucket for the picking team.

    The fillers are the players ``forward_fill_rosters`` added to the team's
    roster (final minus current). A candidate displaces the lowest-value filler
    in its bucket -- the realistic alternative the team would otherwise roster
    at that slot. Returns ``{bucket: Player}`` for whichever of HITTER/SP/RP the
    team has fillers in.
    """
    current_set = set(current_roster)
    filler_pids = [pid for pid in final_roster if pid not in current_set]
    by_bucket: dict[str, tuple[float, Player]] = {}
    for pid in filler_pids:
        player = board_by_id.get(pid)
        if player is None:
            continue
        bucket = _bucket(player)
        val = _filler_value(pid, player_lookup, player)
        if bucket not in by_bucket or val < by_bucket[bucket][0]:
            by_bucket[bucket] = (val, player)
    return {bucket: player for bucket, (_, player) in by_bucket.items()}


def build_finalslate_field(
    *,
    team_rosters: Mapping[int, list[str]],
    team_name_by_num: Mapping[int, str],
    board_by_id: Mapping[str, Player],
    player_lookup: Mapping[str, Any],
    adp_table: ADPTable,
    roster_slots: Mapping[str, int],
    scarcity_order: list[str],
    num_teams: int,
    current_pick: int,
    total_picks: int,
    drafted: set[str],
) -> tuple[
    ProjectedStandings,
    dict[str, dict[Category, float]],
    dict[int, list[str]],
    dict[str, str],
]:
    """Forward-fill the field and project it.

    Returns ``(field_standings, field_team_sds, final_rosters_by_num, holders)``
    -- the realistic end-state standings (name-keyed), the per-team category SDs
    over those rosters, the raw final rosters (so the caller can derive the
    picking team's marginal fillers), and ``holders`` mapping each rostered
    player id to its projected team name (for the snipe-aware ranker).
    """
    final_by_num = forward_fill_rosters(
        team_rosters=team_rosters,
        board_by_id=board_by_id,
        player_lookup=player_lookup,
        adp_table=adp_table,
        roster_slots=roster_slots,
        scarcity_order=scarcity_order,
        num_teams=num_teams,
        current_pick=current_pick,
        total_picks=total_picks,
        drafted=drafted,
    )
    rosters_by_name: dict[str, list[Player]] = {}
    holders: dict[str, str] = {}
    for num, pids in final_by_num.items():
        name = team_name_by_num.get(num, f"Team {num}")
        rosters_by_name[name] = [p for pid in pids if (p := board_by_id.get(pid)) is not None]
        for pid in pids:
            holders[pid] = name
    field_standings = build_projected_standings(rosters_by_name)
    field_team_sds = build_team_sds(rosters_by_name, sd_scale=1.0)
    return field_standings, field_team_sds, final_by_num, holders


__all__ = [
    "build_finalslate_field",
    "forward_fill_rosters",
    "marginal_fillers",
]
