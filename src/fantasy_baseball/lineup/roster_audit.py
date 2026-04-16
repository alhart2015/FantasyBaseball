"""Roster audit — evaluate every roster slot against the best available FA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from fantasy_baseball.lineup.delta_roto import compute_delta_roto
from fantasy_baseball.lineup.team_optimizer import compute_team_wsgp, build_lineup_summary
from fantasy_baseball.lineup.waivers import evaluate_pickup
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import IL_STATUSES
from fantasy_baseball.utils.positions import can_cover_slots


logger = logging.getLogger(__name__)


POSITION_POOL_SIZES: dict[str, int] = {
    "C": 5, "1B": 5, "2B": 5, "3B": 5, "SS": 5,
    "OF": 15, "SP": 20, "RP": 10,
}

# Projected-SV threshold separating starters from relievers. Yahoo reports
# pitchers as positions=["P"] in leagues without SP/RP slots, so we can't
# rely on the position string to bucket them. Projected saves is a clean
# signal: sub-5 ≈ starter, 5+ ≈ closer/setup.
RP_SV_THRESHOLD = 5


def build_position_pools(
    free_agents: list[Player],
    denoms: dict[str, float] | None = None,
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
                fa for fa in free_agents
                if fa.player_type == PlayerType.PITCHER
                and fa.rest_of_season.sv < RP_SV_THRESHOLD
            ]
        elif pos == "RP":
            eligible = [
                fa for fa in free_agents
                if fa.player_type == PlayerType.PITCHER
                and fa.rest_of_season.sv >= RP_SV_THRESHOLD
            ]
        else:
            eligible = [fa for fa in free_agents if pos in fa.positions]
        eligible.sort(
            key=lambda p: calculate_player_sgp(p.rest_of_season, denoms),
            reverse=True,
        )
        pools[pos] = eligible[:n]
    return pools


LINEUP_ONLY_SLOTS: set[str] = {"IF", "UTIL", "Util", "P", "BN", "IL"}


def candidates_for_player(
    player: Player,
    pools: dict[str, list[Player]],
) -> list[Player]:
    """Return candidate FAs for this roster player as a deduped list.

    - Hitters: union of pools for each of player's Yahoo positions (lineup-only
      slots like ``UTIL``/``IF`` don't contribute pools because they're not
      Yahoo source positions).
    - Pitchers: SP pool ∪ RP pool (all Yahoo roster pitchers come through as
      ``positions=["P"]``; we don't distinguish SP/RP on the user's roster).
    - Dedup key: ``yahoo_id`` when present, else ``name::player_type``.
    """
    if player.player_type == PlayerType.PITCHER:
        source_positions = ["SP", "RP"]
    else:
        source_positions = [
            p for p in player.positions if p not in LINEUP_ONLY_SLOTS
        ]

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
    player_wsgp: float
    player_id: Optional[str] = None
    best_fa: Optional[str] = None
    best_fa_type: Optional[str] = None
    best_fa_positions: Optional[list[str]] = None
    best_fa_wsgp: Optional[float] = None
    best_fa_id: Optional[str] = None
    gap: float = 0.0
    categories: dict[str, float] = field(default_factory=dict)
    classification: str = ""
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "player_type": self.player_type,
            "positions": self.positions,
            "slot": self.slot,
            "player_wsgp": self.player_wsgp,
            "player_id": self.player_id,
            "best_fa": self.best_fa,
            "best_fa_type": self.best_fa_type,
            "best_fa_positions": self.best_fa_positions,
            "best_fa_wsgp": self.best_fa_wsgp,
            "best_fa_id": self.best_fa_id,
            "gap": self.gap,
            "categories": self.categories,
            "classification": self.classification,
            "candidates": self.candidates,
        }


def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
    *,
    projected_standings: list[dict],
    team_name: str,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> list[AuditEntry]:
    """Evaluate every roster slot against the best available FA.

    For each roster player, finds the FA that produces the largest team
    wSGP gain when swapped in.  Returns an entry for every roster player,
    sorted by gap descending (biggest problems first).  Entries with no
    upgrade available have gap=0.0 and best_fa=None.

    IL players are excluded from lineup optimization (they can't play)
    but still appear in the output with slot="IL".

    When ``team_sds`` is provided, deltaRoto for each candidate uses
    EV-based ``score_roto`` so within-uncertainty swaps produce
    fractional deltas instead of full ±1.0 rank flips. ``None``
    preserves exact-rank semantics.
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

    # Baseline optimal lineup (active players only)
    baseline = compute_team_wsgp(active_roster, leverage, roster_slots, denoms=denoms)
    baseline_wsgp = baseline["total_wsgp"]
    baseline_summary = build_lineup_summary(
        baseline["hitter_lineup"], baseline["pitcher_starters"],
        baseline["player_wsgp"], [p.name for p in roster],
    )

    # Map player name → assigned slot from baseline
    slot_lookup = {e["name"]: e["slot"] for e in baseline_summary}

    # Pre-compute FA wSGP (kept as informational column alongside deltaRoto)
    fa_wsgp: dict[str, float] = {}
    for fa in active_fas:
        fa_wsgp[fa.name] = calculate_weighted_sgp(fa.rest_of_season, leverage, denoms=denoms)

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
            player_wsgp=round(baseline["player_wsgp"].get(player.name, 0.0), 2),
            player_id=player.yahoo_id,
            classification=player.classification,
        )

        # Protected players: high league-wide value, skip FA comparison
        if player.classification in ("core", "trade_candidate"):
            entries.append(entry)
            continue

        # Gather candidates from the per-position pools
        candidates = candidates_for_player(player, pools)

        # Pre-build wSGP dict without this player for swap simulation
        base_wsgp = {k: v for k, v in baseline["player_wsgp"].items()
                     if k != player.name}

        # scored: list of (candidate_dict, Player) tuples — Player retained so
        # we can pass it to evaluate_pickup when selecting the top candidate.
        scored: list[tuple[dict, Player]] = []

        for fa in candidates:
            new_roster = [p for p in active_roster if p.name != player.name] + [fa]
            new_pitchers = [p for p in new_roster if p.player_type == PlayerType.PITCHER]

            # Cross-type feasibility: pool structure already blocks most cross-type
            # swaps, but defense-in-depth against bad pool logic.
            if player.player_type == PlayerType.HITTER or fa.player_type == PlayerType.HITTER:
                new_hitters = [p for p in new_roster if p.player_type != PlayerType.PITCHER]
                if not can_cover_slots([list(p.positions) for p in new_hitters], roster_slots):
                    continue
            if player.player_type == PlayerType.PITCHER or fa.player_type == PlayerType.PITCHER:
                if len(new_pitchers) < p_slots:
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
                    "deltaRoto failed for %s → %s: %s", player.name, fa.name, exc,
                )
                continue

            # wSGP gap (informational column)
            swap_wsgp = dict(base_wsgp)
            swap_wsgp[fa.name] = fa_wsgp[fa.name]
            new_result = compute_team_wsgp(
                new_roster, leverage, roster_slots,
                denoms=denoms, player_wsgp=swap_wsgp,
            )
            gap = round(new_result["total_wsgp"] - baseline_wsgp, 2)

            cand_dict = {
                "name": fa.name,
                "player_type": fa.player_type.value,
                "positions": list(fa.positions),
                "wsgp": round(fa_wsgp.get(fa.name, 0.0), 2),
                "gap": gap,
                "delta_roto": dr.to_dict(),
                "player_id": fa.yahoo_id,
            }
            scored.append((cand_dict, fa))

        scored.sort(key=lambda t: t[0]["delta_roto"]["total"], reverse=True)
        entry.candidates = [cand for cand, _ in scored]

        # Top-1 becomes the recommendation only if it's a real upgrade.
        if scored and scored[0][0]["delta_roto"]["total"] > 0:
            top_dict, top_player = scored[0]
            cat_result = evaluate_pickup(top_player, player, leverage)
            entry.best_fa = top_dict["name"]
            entry.best_fa_type = top_dict["player_type"]
            entry.best_fa_positions = top_dict["positions"]
            entry.best_fa_wsgp = top_dict["wsgp"]
            entry.best_fa_id = top_dict["player_id"]
            entry.gap = top_dict["gap"]
            entry.categories = cat_result["categories"]
        # else: best_fa stays None, gap stays 0.0 — "No upgrade available"

        entries.append(entry)

    # Sort entries by top-candidate deltaRoto desc (entries with best_fa=None
    # sort to the bottom).
    entries.sort(
        key=lambda e: (
            e.candidates[0]["delta_roto"]["total"]
            if e.best_fa is not None
            else float("-inf")
        ),
        reverse=True,
    )

    # Add IL players at the end — they can't be swapped
    for player in il_players:
        entries.append(AuditEntry(
            player=player.name,
            player_type=player.player_type.value,
            positions=list(player.positions),
            slot="IL",
            player_wsgp=0.0,
            player_id=player.yahoo_id,
            classification=player.classification,
        ))

    return entries
