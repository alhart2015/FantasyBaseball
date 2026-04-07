"""Roster audit — evaluate every roster slot against the best available FA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from fantasy_baseball.lineup.team_optimizer import compute_team_wsgp, build_lineup_summary
from fantasy_baseball.lineup.waivers import evaluate_pickup
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import IL_STATUSES
from fantasy_baseball.utils.positions import can_cover_slots


@dataclass
class AuditEntry:
    """Result of evaluating one roster slot against the FA pool."""

    player: str
    player_type: str
    positions: list[str]
    slot: str
    player_wsgp: float
    best_fa: Optional[str] = None
    best_fa_type: Optional[str] = None
    best_fa_positions: Optional[list[str]] = None
    best_fa_wsgp: Optional[float] = None
    gap: float = 0.0
    categories: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "player_type": self.player_type,
            "positions": self.positions,
            "slot": self.slot,
            "player_wsgp": self.player_wsgp,
            "best_fa": self.best_fa,
            "best_fa_type": self.best_fa_type,
            "best_fa_positions": self.best_fa_positions,
            "best_fa_wsgp": self.best_fa_wsgp,
            "gap": self.gap,
            "categories": self.categories,
        }


def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
) -> list[AuditEntry]:
    """Evaluate every roster slot against the best available FA.

    For each roster player, finds the FA that produces the largest team
    wSGP gain when swapped in.  Returns an entry for every roster player,
    sorted by gap descending (biggest problems first).  Entries with no
    upgrade available have gap=0.0 and best_fa=None.

    IL players are excluded from lineup optimization (they can't play)
    but still appear in the output with slot="IL".
    """
    if not roster:
        return []

    active_roster = [p for p in roster if p.status not in IL_STATUSES]
    il_players = [p for p in roster if p.status in IL_STATUSES]
    active_fas = [fa for fa in free_agents if fa.status not in IL_STATUSES]

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

    # Pre-compute FA wSGP
    fa_wsgp: dict[str, float] = {}
    for fa in active_fas:
        fa_wsgp[fa.name] = calculate_weighted_sgp(fa.ros, leverage, denoms=denoms)

    p_slots = roster_slots.get("P", 9)

    entries: list[AuditEntry] = []
    for player in active_roster:
        entry = AuditEntry(
            player=player.name,
            player_type=player.player_type.value,
            positions=list(player.positions),
            slot=slot_lookup.get(player.name, "BN"),
            player_wsgp=round(baseline["player_wsgp"].get(player.name, 0.0), 2),
        )

        best_gain = 0.0
        best_fa_player = None

        # Pre-build wSGP dict without this player for swap simulation
        base_wsgp = {k: v for k, v in baseline["player_wsgp"].items()
                     if k != player.name}

        for fa in active_fas:
            new_roster = [p for p in active_roster if p.name != player.name] + [fa]
            new_pitchers = [p for p in new_roster if p.player_type == PlayerType.PITCHER]

            # Cross-type feasibility: swapping across types can't leave
            # fewer hitters or pitchers than required slots.
            if player.player_type == PlayerType.HITTER or fa.player_type == PlayerType.HITTER:
                new_hitters = [p for p in new_roster if p.player_type != PlayerType.PITCHER]
                if not can_cover_slots([list(p.positions) for p in new_hitters], roster_slots):
                    continue
            if player.player_type == PlayerType.PITCHER or fa.player_type == PlayerType.PITCHER:
                if len(new_pitchers) < p_slots:
                    continue

            swap_wsgp = dict(base_wsgp)
            swap_wsgp[fa.name] = fa_wsgp[fa.name]

            new_result = compute_team_wsgp(
                new_roster, leverage, roster_slots,
                denoms=denoms, player_wsgp=swap_wsgp,
            )
            gain = round(new_result["total_wsgp"] - baseline_wsgp, 2)

            if gain > best_gain:
                best_gain = gain
                best_fa_player = fa

        if best_fa_player:
            cat_result = evaluate_pickup(best_fa_player, player, leverage)
            entry.best_fa = best_fa_player.name
            entry.best_fa_type = best_fa_player.player_type.value
            entry.best_fa_positions = list(best_fa_player.positions)
            entry.best_fa_wsgp = round(fa_wsgp.get(best_fa_player.name, 0.0), 2)
            entry.gap = best_gain
            entry.categories = cat_result["categories"]

        entries.append(entry)

    # Add IL players at the end — they can't be swapped
    for player in il_players:
        entries.append(AuditEntry(
            player=player.name,
            player_type=player.player_type.value,
            positions=list(player.positions),
            slot="IL",
            player_wsgp=0.0,
        ))

    entries.sort(key=lambda e: e.gap, reverse=True)
    return entries
