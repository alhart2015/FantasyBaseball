"""ERoto-delta recommender for the live draft.

Wraps ``lineup.delta_roto.compute_delta_roto`` — same math powering
in-season trade evaluation. For a candidate player and the team on the
clock, we compute ``score_roto(team_with_player) - score_roto(team_with_replacement)``
across all 10 teams. The delta is context-dependent: a 40-HR candidate is
worth more to an HR-weak roster than to an HR-strong one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fantasy_baseball.draft.adp import ADPTable
from fantasy_baseball.lineup.delta_roto import DeltaRotoResult, compute_delta_roto
from fantasy_baseball.models.player import Player
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.utils.constants import Category


@dataclass
class DeltaBreakdown:
    """Per-category + total ERoto delta for a single candidate pick."""

    total: float
    per_category: dict[str, float]


def immediate_delta(
    *,
    candidate: Player,
    replacement: Player,
    team_name: str,
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> DeltaBreakdown:
    """ERoto delta from swapping ``replacement`` out for ``candidate``."""
    result: DeltaRotoResult = compute_delta_roto(
        drop_name=replacement.name,
        add_player=candidate,
        user_roster=[replacement],
        projected_standings=projected_standings,
        team_name=team_name,
        team_sds=team_sds,
    )
    return DeltaBreakdown(
        total=result.total,
        per_category={cat: cd.roto_delta for cat, cd in result.categories.items()},
    )


@dataclass
class RecRow:
    """One recommendation row for the dashboard."""

    player_id: str
    name: str
    positions: list[str]
    immediate_delta: float
    immediate_delta_sd: float
    value_of_picking_now: float
    per_category: dict[str, float] = field(default_factory=dict)


def rank_candidates(
    *,
    candidates: list[Player],
    replacements: Mapping[str, Player],
    team_name: str,
    projected_standings: ProjectedStandings,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    picks_until_next_turn: int = 0,
    adp_table: ADPTable | None = None,
) -> list[RecRow]:
    """Score every candidate's immediate ERoto delta + value-of-picking-now."""
    immediate_rows: list[tuple[Player, DeltaBreakdown]] = []
    for candidate in candidates:
        replacement = _pick_replacement(candidate, replacements)
        delta = immediate_delta(
            candidate=candidate,
            replacement=replacement,
            team_name=team_name,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )
        immediate_rows.append((candidate, delta))

    # Forward-model: opponents pick lowest-ADP first. After picks_until_next_turn
    # picks they take a fixed set of candidates (the "sniped" ones); the rest
    # are "surviving" — we'd still see them at our next turn.
    surviving_ids = {_candidate_id(c) for c in candidates}
    if picks_until_next_turn > 0 and adp_table is not None:
        by_adp = sorted(candidates, key=lambda c: adp_table.get(_candidate_id(c)))
        snipes_left = picks_until_next_turn
        for c in by_adp:
            if snipes_left <= 0:
                break
            surviving_ids.discard(_candidate_id(c))
            snipes_left -= 1

    # Best immediate_delta among the candidates that will survive — the
    # baseline alternative I'd be left with if I let p go.
    best_surviving_delta = max(
        (d.total for c, d in immediate_rows if _candidate_id(c) in surviving_ids),
        default=0.0,
    )

    rows = []
    for c, d in immediate_rows:
        cid = _candidate_id(c)
        if cid in surviving_ids:
            # I can wait — p will still be there at my next turn, so the
            # order I pick doesn't affect my two-pick total. No regret.
            vopn = 0.0
        else:
            # p will be sniped. Regret = gap between p and the best
            # alternative I could still grab at my next turn. Positive
            # means urgent (p > best_survivor); negative means even
            # waiting beats panic-grabbing p now.
            vopn = d.total - best_surviving_delta
        rows.append(
            RecRow(
                player_id=cid,
                name=c.name,
                positions=[str(p) for p in c.positions],
                immediate_delta=d.total,
                immediate_delta_sd=0.0,
                value_of_picking_now=vopn,
                per_category=d.per_category,
            )
        )
    rows.sort(key=lambda r: r.immediate_delta, reverse=True)
    return rows


def _pick_replacement(candidate: Player, replacements: Mapping[str, Player]) -> Player:
    """Choose the replacement-level player the candidate would displace.

    For v1, use the candidate's primary position. Phase 3 can be smarter
    (scarcity-based slot pick) — call out to roster_state helpers then.
    """
    primary = str(candidate.positions[0]) if candidate.positions else ""
    if primary in replacements:
        return replacements[primary]
    return next(iter(replacements.values()))


def _candidate_id(player: Player) -> str:
    """Stable ID for a candidate. Falls back to ``name::player_type`` when
    ``yahoo_id`` is missing — same convention used throughout the codebase.
    """
    if player.yahoo_id:
        return player.yahoo_id
    return f"{player.name}::{player.player_type.value}"
