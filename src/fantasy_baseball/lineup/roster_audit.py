"""Roster audit — evaluate every roster slot against the best available FA."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball.lineup.delta_roto import compute_delta_roto
from fantasy_baseball.lineup.optimizer import (
    HitterAssignment,
    PitcherStarter,
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import IL_STATUSES, Category
from fantasy_baseball.utils.positions import can_cover_slots

logger = logging.getLogger(__name__)


POSITION_POOL_SIZES: dict[str, int] = {
    "C": 5,
    "1B": 5,
    "2B": 5,
    "3B": 5,
    "SS": 5,
    "OF": 15,
    "SP": 20,
    "RP": 10,
}

# Projected-SV threshold separating starters from relievers. Yahoo reports
# pitchers as positions=["P"] in leagues without SP/RP slots, so we can't
# rely on the position string to bucket them. Projected saves is a clean
# signal: sub-5 ≈ starter, 5+ ≈ closer/setup.
RP_SV_THRESHOLD = 5


def build_position_pools(
    free_agents: list[Player],
    denoms: dict[Category, float] | None = None,
) -> dict[str, list[Player]]:
    """Bucket FAs into per-position pools, each sorted by raw SGP desc
    and truncated to POSITION_POOL_SIZES[pos].

    - Hitter pools (C/1B/2B/3B/SS/OF) key on ``pos in fa.positions`` — a
      multi-eligible hitter lives in every pool that matches.
    - Pitcher pools (SP/RP) key on projected saves: sv < RP_SV_THRESHOLD
      goes to SP, sv >= RP_SV_THRESHOLD goes to RP. This works around
      Yahoo leagues that only surface a generic "P" slot, where
      fa.positions == ["P"] for every pitcher.
    """
    if denoms is None:
        denoms = get_sgp_denominators()
    pools: dict[str, list[Player]] = {}
    for pos, n in POSITION_POOL_SIZES.items():
        if pos == "SP":
            eligible = [
                fa
                for fa in free_agents
                if isinstance(fa.rest_of_season, PitcherStats)
                and fa.rest_of_season.sv < RP_SV_THRESHOLD
            ]
        elif pos == "RP":
            eligible = [
                fa
                for fa in free_agents
                if isinstance(fa.rest_of_season, PitcherStats)
                and fa.rest_of_season.sv >= RP_SV_THRESHOLD
            ]
        else:
            eligible = [fa for fa in free_agents if pos in fa.positions]
        eligible.sort(
            key=lambda p: (
                calculate_player_sgp(p.rest_of_season, denoms) if p.rest_of_season else 0.0
            ),
            reverse=True,
        )
        pools[pos] = eligible[:n]
    return pools


LINEUP_ONLY_SLOTS: set[str] = {"IF", "UTIL", "Util", "P", "BN", "IL"}


HITTER_SOURCE_POSITIONS: tuple[str, ...] = ("C", "1B", "2B", "3B", "SS", "OF")


def worst_roster_by_position(
    roster: list[Player],
    denoms: dict[Category, float] | None = None,
) -> dict[str, str]:
    """Return ``{pool_pos: worst_roster_player_name}`` — the lowest-SGP
    roster player eligible at each pool position.

    Buckets mirror :func:`build_position_pools`: hitter source positions
    (C/1B/2B/3B/SS/OF) pick the lowest-SGP roster hitter eligible there,
    pitchers split on ``RP_SV_THRESHOLD`` into SP/RP buckets. This is the
    "drop candidate" used when pricing an FA's impact on the browse page.
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    def _sgp(p: Player) -> float:
        if p.rest_of_season is None:
            return 0.0
        return calculate_player_sgp(p.rest_of_season, denoms)

    result: dict[str, str] = {}
    for pos in HITTER_SOURCE_POSITIONS:
        eligible = [p for p in roster if p.player_type == PlayerType.HITTER and pos in p.positions]
        if eligible:
            result[pos] = min(eligible, key=_sgp).name

    sps = [
        p
        for p in roster
        if isinstance(p.rest_of_season, PitcherStats) and p.rest_of_season.sv < RP_SV_THRESHOLD
    ]
    rps = [
        p
        for p in roster
        if isinstance(p.rest_of_season, PitcherStats) and p.rest_of_season.sv >= RP_SV_THRESHOLD
    ]
    if sps:
        result["SP"] = min(sps, key=_sgp).name
    if rps:
        result["RP"] = min(rps, key=_sgp).name
    return result


def fa_target_positions(
    player_type: str,
    positions: Sequence[str],
    sv: float,
) -> list[str]:
    """Return the pool positions an FA should be compared against.

    Hitters: their Yahoo positions intersected with hitter source positions.
    Pitchers: SP or RP based on projected saves (``RP_SV_THRESHOLD``).
    """
    if player_type == PlayerType.PITCHER.value or player_type == "pitcher":
        return ["SP" if sv < RP_SV_THRESHOLD else "RP"]
    return [p for p in positions if p in HITTER_SOURCE_POSITIONS]


def candidates_for_player(
    player: Player,
    pools: dict[str, list[Player]],
) -> list[Player]:
    """Return candidate FAs for this roster player as a deduped list.

    - Hitters: union of pools for each of player's Yahoo positions (lineup-only
      slots like ``UTIL``/``IF`` don't contribute pools because they're not
      Yahoo source positions).
    - Pitchers: SP pool + RP pool (all Yahoo roster pitchers come through as
      ``positions=["P"]``; we don't distinguish SP/RP on the user's roster).
    - Dedup key: ``yahoo_id`` when present, else ``name::player_type``.
    """
    if player.player_type == PlayerType.PITCHER:
        source_positions = ["SP", "RP"]
    else:
        source_positions = [p for p in player.positions if p not in LINEUP_ONLY_SLOTS]

    seen: set[str] = set()
    result: list[Player] = []
    for pos in source_positions:
        for fa in pools.get(pos, []):
            key = fa.yahoo_id or f"{fa.name}::{fa.player_type.value}"
            if key in seen:
                continue
            seen.add(key)
            result.append(fa)
    return result


@dataclass
class AuditEntry:
    """Result of evaluating one roster slot against the FA pool."""

    player: str
    player_type: str
    positions: list[str]
    slot: str
    player_sgp: float
    player_id: str | None = None
    best_fa: str | None = None
    best_fa_type: str | None = None
    best_fa_positions: list[str] | None = None
    best_fa_sgp: float | None = None
    best_fa_id: str | None = None
    gap: float = 0.0
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "player_type": self.player_type,
            "positions": self.positions,
            "slot": self.slot,
            "player_sgp": self.player_sgp,
            "player_id": self.player_id,
            "best_fa": self.best_fa,
            "best_fa_type": self.best_fa_type,
            "best_fa_positions": self.best_fa_positions,
            "best_fa_sgp": self.best_fa_sgp,
            "best_fa_id": self.best_fa_id,
            "gap": self.gap,
            "candidates": self.candidates,
        }


def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    roster_slots: dict[str, int],
    *,
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    optimal_hitters: list[HitterAssignment] | None = None,
    optimal_pitchers: list[PitcherStarter] | None = None,
) -> list[AuditEntry]:
    """Evaluate every roster slot against the best available FA.

    For each roster player, finds the FA that produces the largest team
    ΔRoto when swapped in.  Returns an entry for every roster player,
    sorted by top-candidate ΔRoto descending (biggest upgrades first).
    Entries with no upgrade available have gap=0.0 and best_fa=None.

    IL players are excluded from lineup optimization (they can't play)
    but still appear in the output with slot="IL".

    The ``optimal_hitters`` / ``optimal_pitchers`` inputs are the outputs
    of :func:`optimize_hitter_lineup` / :func:`optimize_pitcher_lineup`
    on the active roster. They drive the per-player "slot" column. When
    omitted they're computed here; callers that already solved the
    lineup (e.g. the refresh pipeline) should pass them in to avoid
    duplicate work.

    ``team_sds`` is threaded into ``compute_delta_roto`` so within-
    uncertainty swaps produce fractional deltas instead of full ±1.0
    rank flips. ``None`` preserves exact-rank semantics.
    """
    if not roster:
        return []

    def _is_il(player) -> bool:
        """A player is on IL if either the status string or the
        selected_position slot indicates IL.

        Covers three production shapes seen in Yahoo roster data:
          - Soto: selected_position='BN' + status='IL10' — the
            status check catches this (bench-slotted IL player).
          - Strider: selected_position='IL' + status='IL15' — both
            checks catch this (formally in the IL slot).
          - Hader: selected_position='IL' + status='' — only the
            slot check catches this (Yahoo sometimes omits status
            on freshly-slotted IL players).
        """
        if player.status in IL_STATUSES:
            return True
        slot = player.selected_position
        if slot is None:
            return False
        return slot in IL_SLOTS

    active_roster = [p for p in roster if not _is_il(p)]
    il_players = [p for p in roster if _is_il(p)]
    active_fas = [fa for fa in free_agents if not _is_il(fa)]

    denoms = get_sgp_denominators()

    # Slot assignments come from the already-solved ERoto lineup. Compute
    # here only if the caller didn't pass them.
    if optimal_hitters is None:
        active_hitters = [p for p in active_roster if p.player_type != PlayerType.PITCHER]
        optimal_hitters = optimize_hitter_lineup(
            hitters=active_hitters,
            full_roster=roster,
            projected_standings=projected_standings,
            team_name=team_name,
            roster_slots=roster_slots,
            team_sds=team_sds,
        )
    if optimal_pitchers is None:
        active_pitchers = [p for p in active_roster if p.player_type == PlayerType.PITCHER]
        optimal_pitchers, _ = optimize_pitcher_lineup(
            pitchers=active_pitchers,
            full_roster=roster,
            projected_standings=projected_standings,
            team_name=team_name,
            slots=roster_slots.get("P", 9),
            team_sds=team_sds,
        )

    slot_lookup: dict[str, str] = {a.name: a.slot.value for a in optimal_hitters}
    for s in optimal_pitchers:
        slot_lookup[s.name] = "P"

    # Pre-compute per-player raw SGP for roster + FAs (used for display).
    player_sgp: dict[str, float] = {
        p.name: calculate_player_sgp(p.rest_of_season, denoms)
        for p in active_roster
        if p.rest_of_season is not None
    }
    fa_sgp: dict[str, float] = {
        fa.name: calculate_player_sgp(fa.rest_of_season, denoms)
        for fa in active_fas
        if fa.rest_of_season is not None
    }

    # Build per-position SGP pools once for this audit
    pools = build_position_pools(active_fas, denoms=denoms)

    p_slots = roster_slots.get("P", 9)

    entries: list[AuditEntry] = []
    for player in active_roster:
        entry = AuditEntry(
            player=player.name,
            player_type=player.player_type.value,
            positions=list(player.positions),
            slot=slot_lookup.get(player.name, "BN"),
            player_sgp=round(player_sgp.get(player.name, 0.0), 2),
            player_id=player.yahoo_id,
        )

        candidates = candidates_for_player(player, pools)

        scored: list[dict[str, Any]] = []

        for fa in candidates:
            new_roster = [p for p in active_roster if p.name != player.name] + [fa]
            new_pitchers = [p for p in new_roster if p.player_type == PlayerType.PITCHER]

            # Cross-type feasibility: pool structure already blocks most cross-type
            # swaps, but defense-in-depth against bad pool logic.
            if player.player_type == PlayerType.HITTER or fa.player_type == PlayerType.HITTER:
                new_hitters = [p for p in new_roster if p.player_type != PlayerType.PITCHER]
                if not can_cover_slots([list(p.positions) for p in new_hitters], roster_slots):
                    continue
            # Pitcher count threshold: only pitcher→hitter swaps reduce the
            # count. Same-type pitcher swaps preserve it — gating on p_slots
            # would reject every upgrade when the user's active pitcher count
            # is already below p_slots (common with IL pitchers).
            if (
                player.player_type == PlayerType.PITCHER
                and fa.player_type == PlayerType.HITTER
                and len(new_pitchers) < p_slots
            ):
                continue

            try:
                dr = compute_delta_roto(
                    drop_name=player.name,
                    add_player=fa,
                    user_roster=roster,
                    projected_standings=projected_standings,
                    team_name=team_name,
                    team_sds=team_sds,
                )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "deltaRoto failed for %s → %s: %s",
                    player.name,
                    fa.name,
                    exc,
                )
                continue

            sgp_gap = round(fa_sgp.get(fa.name, 0.0) - player_sgp.get(player.name, 0.0), 2)

            scored.append(
                {
                    "name": fa.name,
                    "player_type": fa.player_type.value,
                    "positions": list(fa.positions),
                    "sgp": round(fa_sgp.get(fa.name, 0.0), 2),
                    "gap": sgp_gap,
                    "delta_roto": dr.to_dict(),
                    "player_id": fa.yahoo_id,
                }
            )

        scored.sort(key=lambda c: c["delta_roto"]["total"], reverse=True)
        entry.candidates = scored

        # Top-1 becomes the recommendation only if it's a real upgrade.
        if scored and scored[0]["delta_roto"]["total"] > 0:
            top = scored[0]
            entry.best_fa = top["name"]
            entry.best_fa_type = top["player_type"]
            entry.best_fa_positions = top["positions"]
            entry.best_fa_sgp = top["sgp"]
            entry.best_fa_id = top["player_id"]
            entry.gap = top["gap"]
        # else: best_fa stays None, gap stays 0.0 — "No upgrade available"

        entries.append(entry)

    # Sort entries by top-candidate deltaRoto desc (entries with best_fa=None
    # sort to the bottom).
    entries.sort(
        key=lambda e: (
            e.candidates[0]["delta_roto"]["total"] if e.best_fa is not None else float("-inf")
        ),
        reverse=True,
    )

    # Add IL players at the end — they can't be swapped
    for player in il_players:
        entries.append(
            AuditEntry(
                player=player.name,
                player_type=player.player_type.value,
                positions=list(player.positions),
                slot="IL",
                player_sgp=0.0,
                player_id=player.yahoo_id,
            )
        )

    return entries
