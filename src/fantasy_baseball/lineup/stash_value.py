"""Stash board -- rank injured players (owned IL + injured FAs) by their
leverage-aware marginal active value, and allocate the scarce IL slots.

Sibling of ``il_return_planner``: reuses the optimizer and the
double-count-safe deltaRoto band. A candidate's Gain is the band mean of
activating him into the optimized lineup (floored at ~0 when he can't crack
it -- "no harm, no foul"). Cost is the IL-slot allocation cost: 0 when a slot
is open, else the Gain of the weakest owned IL stash he displaces.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band
from fantasy_baseball.lineup.optimizer import (
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.utils.constants import Category

__all__ = ["StashResult", "StashScore", "score_stash_candidates"]


@dataclass
class StashScore:
    """One injured player's stash evaluation."""

    name: str
    player_type: str
    status: str  # IL10 / IL15 / IL60 / DTD / ...
    owned: bool  # already on the user's roster
    gain: float  # marginal active value (deltaRoto band mean), floored at ~0
    cost: float  # deltaRoto sacrificed to roster him (0 if open IL slot)
    stash_value: float  # gain - cost
    band: dict[str, Any]  # {mean, sd, p_positive, verdict}
    recommended_drop: str | None  # who to drop to make room (None if free slot)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StashResult:
    """Ranked stash board."""

    open_il_slots: int
    cutline_rank: int  # = IL capacity; top-N are "hold/grab"
    candidates: list[StashScore] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_il_slots": self.open_il_slots,
            "cutline_rank": self.cutline_rank,
            "candidates": [c.to_dict() for c in self.candidates],
            "warning": self.warning,
        }


def _activate(p: Player) -> Player:
    """Copy with IL signals cleared so the optimizer treats the player as
    active-eligible. Identical to il_return_planner._activate."""
    return dataclasses.replace(p, status="", selected_position=None)


def _solve_active(
    pool: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> list[Player]:
    """Optimized active lineup (hitters + pitcher starters) over ``pool``."""
    hitters = [p for p in pool if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in pool if p.player_type == PlayerType.PITCHER]
    h_assign = optimize_hitter_lineup(
        hitters=hitters,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
        fraction_remaining=None,
    )
    p_starters, _bench = optimize_pitcher_lineup(
        pitchers=pitchers,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
        fraction_remaining=None,
    )
    return [a.player for a in h_assign] + [s.player for s in p_starters]


def _counted_pool(roster: list[Player], exclude_name: str | None = None) -> list[Player]:
    """Active + bench bodies (everything not in a true IL slot), optionally
    excluding one player by name."""
    out: list[Player] = []
    for p in roster:
        if p.selected_position in IL_SLOTS:
            continue
        if exclude_name is not None and p.name == exclude_name:
            continue
        out.append(p)
    return out


def _marginal_band(
    candidate: Player,
    *,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> dict[str, Any]:
    """Return the deltaRoto band dict for activating ``candidate``."""
    pool_with = [*_counted_pool(roster, exclude_name=candidate.name), _activate(candidate)]
    after_active = _solve_active(pool_with, roster_slots, projected_standings, team_name, team_sds)
    band = compute_delta_roto_band(
        before_active,
        after_active,
        projected_standings.field_stats(team_name),
        team_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )
    return band.to_dict()


def _marginal_value(
    candidate: Player,
    *,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> float:
    """Gain = band mean of activating ``candidate``. Floored at ~0."""
    band = _marginal_band(
        candidate,
        before_active=before_active,
        roster=roster,
        roster_slots=roster_slots,
        projected_standings=projected_standings,
        team_name=team_name,
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
    )
    return band["mean"]


def _open_il_slots(roster: list[Player], roster_slots: dict[str, int]) -> int:
    """IL capacity minus players currently in true IL slots."""
    capacity = roster_slots.get("IL", 0)
    occupied = sum(1 for p in roster if p.selected_position in IL_SLOTS)
    return max(0, capacity - occupied)


def _owned_il_stashes(roster: list[Player]) -> list[Player]:
    """Owned players on the IL (slot or status)."""
    return [p for p in roster if p.is_on_il()]


def _cost_and_drop(
    candidate: Player,
    *,
    gain_by_name: dict[str, float],
    roster: list[Player],
    roster_slots: dict[str, int],
) -> tuple[float, str | None]:
    """Cost to roster ``candidate`` and the recommended drop.

    - IL-eligible + open IL slot -> (0, None).
    - IL-eligible + IL full -> displace the lowest-Gain owned IL stash
      (IL-for-IL, the user's rule). Cost = that stash's Gain.
    - Not IL-eligible (e.g. DTD) -> cannot use an IL slot; if an active/bench
      body is open, (0, None), else displace the lowest-Gain active/bench body.
    """
    il_eligible = candidate.is_on_il()
    if il_eligible and _open_il_slots(roster, roster_slots) > 0:
        return 0.0, None

    if il_eligible:
        # Displace the weakest owned IL stash (exclude the candidate itself).
        pool = [p for p in _owned_il_stashes(roster) if p.name != candidate.name]
    else:
        from fantasy_baseball.lineup.il_return_planner import roster_capacity

        counted = _counted_pool(roster, exclude_name=candidate.name)
        if len(counted) < roster_capacity(roster_slots):
            return 0.0, None
        pool = counted

    if not pool:
        return 0.0, None
    drop = min(pool, key=lambda p: gain_by_name.get(p.name, 0.0))
    return gain_by_name.get(drop.name, 0.0), drop.name


def score_stash_candidates(
    roster: list[Player],
    free_agents: list[Player],
    projected_standings: ProjectedStandings,
    roster_slots: dict[str, int],
    team_name: str,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
    max_candidates: int = 25,
) -> StashResult:
    """Rank injured players (owned IL + injured FAs) by stash value.

    Gain = marginal active value (band mean, floored at ~0). Cost = IL-slot
    allocation cost. stash_value = gain - cost, ranked descending. The top
    ``IL`` -capacity candidates are worth a slot.
    """
    il_capacity = roster_slots.get("IL", 0)
    owned_il = _owned_il_stashes(roster)
    injured_fas = [fa for fa in free_agents if fa.is_on_il()]

    # before_active is identical for every candidate: the optimized lineup over
    # the counted (non-IL-slot) bodies, with NO candidate activated.
    before_active = _solve_active(
        _counted_pool(roster), roster_slots, projected_standings, team_name, team_sds
    )

    # Pass 1: Gain + band for every candidate (owned + FA).
    bands: dict[str, dict[str, Any]] = {}
    candidates_in: list[tuple[Player, bool]] = [(p, True) for p in owned_il] + [
        (p, False) for p in injured_fas
    ]
    for player, _owned in candidates_in:
        bands[player.name] = _marginal_band(
            player,
            before_active=before_active,
            roster=roster,
            roster_slots=roster_slots,
            projected_standings=projected_standings,
            team_name=team_name,
            team_sds=team_sds,
            fraction_remaining=fraction_remaining,
        )
    gain_by_name = {name: b["mean"] for name, b in bands.items()}

    # Pass 2: Cost + stash value.
    scores: list[StashScore] = []
    for player, owned in candidates_in:
        band = bands[player.name]
        gain = band["mean"]
        cost, drop = _cost_and_drop(
            player,
            gain_by_name=gain_by_name,
            roster=roster,
            roster_slots=roster_slots,
        )
        scores.append(
            StashScore(
                name=player.name,
                player_type=player.player_type.value,
                status=player.status,
                owned=owned,
                gain=round(gain, 2),
                cost=round(cost, 2),
                stash_value=round(gain - cost, 2),
                band=band,
                recommended_drop=drop,
            )
        )

    scores.sort(key=lambda s: s.stash_value, reverse=True)
    return StashResult(
        open_il_slots=_open_il_slots(roster, roster_slots),
        cutline_rank=il_capacity,
        candidates=scores[:max_candidates],
    )
