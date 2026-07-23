"""Shared assembly of evaluate_multi_trade's inputs from cached blobs. Used by the
/api/evaluate-trade route (local cache) and the keeper-trade generator (Upstash).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.scoring import team_sds_from_json
from fantasy_baseball.trades.multi_trade import build_waiver_pool
from fantasy_baseball.utils.constants import Category


@dataclass(frozen=True)
class TradeEvalContext:
    hart_name: str
    hart_roster: list[Player]
    opp_rosters: dict[str, list[Player]]
    waiver_pool: dict[str, Player]
    projected_standings: ProjectedStandings
    team_sds: dict[str, dict[Category, float]] | None
    fraction_remaining: float
    roster_slots: dict[str, int]


def load_trade_eval_context(
    *,
    hart_name: str,
    roster_raw: list[dict[str, Any]],
    opp_rosters_raw: dict[str, list[dict[str, Any]]],
    proj_cache: dict[str, Any],
    ros_cache: dict[str, Any],
    roster_slots: dict[str, int],
) -> TradeEvalContext:
    """Assemble every input evaluate_multi_trade needs from already-read cache blobs.
    Callers read their blobs their own way (route: local cache; generator: Upstash).
    """
    hart_roster = [Player.from_dict(p) for p in roster_raw]
    opp_rosters = {n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()}
    waiver_pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)
    raw_ps = proj_cache.get("projected_standings")
    if not raw_ps:
        raise ValueError("proj_cache missing 'projected_standings'")
    sds_raw = proj_cache.get("team_sds")
    fr = proj_cache.get("fraction_remaining")
    return TradeEvalContext(
        hart_name=hart_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        waiver_pool=waiver_pool,
        projected_standings=ProjectedStandings.from_json(raw_ps),
        team_sds=team_sds_from_json(sds_raw) if sds_raw else None,
        fraction_remaining=1.0 if fr is None else float(fr),
        roster_slots=roster_slots,
    )
