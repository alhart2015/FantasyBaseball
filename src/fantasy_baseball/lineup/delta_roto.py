"""deltaRoto — roto-point impact metric for player swaps.

Uses EV-based score_roto, so deltaRoto.total is simply the change in
total expected roto points across all categories. No tuning knobs,
no tie bands, no defensive-comfort heuristic — the Gaussian pairwise
win-probabilities price projection uncertainty and vulnerability
directly into the score.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

if TYPE_CHECKING:
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.standings import ProjectedStandings


@dataclass
class CategoryDelta:
    roto_delta: float


@dataclass
class DeltaRotoResult:
    total: float
    categories: dict[str, CategoryDelta]
    before_total: float
    after_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 2),
            "before_total": round(self.before_total, 2),
            "after_total": round(self.after_total, 2),
            "categories": {
                cat: {"roto_delta": round(cd.roto_delta, 2)} for cat, cd in self.categories.items()
            },
        }


def score_swap(
    roto_before: dict[str, dict[str, float]],
    roto_after: dict[str, dict[str, float]],
    team_name: str,
) -> DeltaRotoResult:
    """Per-category deltaRoto from before/after ``score_roto`` outputs.

    Total is the change in team total EV roto points. Each category's
    ``roto_delta`` is the change in that category's EV points for the
    user's team. No discounts, no penalties — the EV already reflects
    projection uncertainty, defensive vulnerability, and boundary
    proximity via the sigmoid on pairwise win probabilities.
    """
    categories: dict[str, CategoryDelta] = {}
    for cat in ALL_CATEGORIES:
        rd = roto_after[team_name][f"{cat.value}_pts"] - roto_before[team_name][f"{cat.value}_pts"]
        categories[cat.value] = CategoryDelta(roto_delta=rd)

    return DeltaRotoResult(
        total=roto_after[team_name]["total"] - roto_before[team_name]["total"],
        categories=categories,
        before_total=roto_before[team_name]["total"],
        after_total=roto_after[team_name]["total"],
    )


def compute_delta_roto(
    drop_name: str,
    add_player: Player,
    user_roster: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> DeltaRotoResult:
    """Compute deltaRoto for dropping one player and adding another.

    When ``team_sds`` is provided, ``score_roto`` uses pairwise Gaussian
    win-probabilities so a swap's impact reflects projection
    uncertainty (ERoto). Pass ``None`` explicitly for exact-rank
    semantics — no default: callers must make the choice so we can't
    silently fall back to integer roto by forgetting the argument.

    Args:
        drop_name: roster player to drop.
        add_player: Player to add.
        user_roster: current roster (used to resolve the dropped player's ROS).
        projected_standings: end-of-season stats for all teams.
        team_name: user's team name.
        team_sds: ``{team: {Category: sd}}`` for EV scoring, or ``None``
            for rank-based. Required keyword — no default.

    Raises:
        ValueError: if drop_name is not found on the roster.
    """
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.trades.evaluate import build_swap_standings, find_player_by_name

    dropped = find_player_by_name(drop_name, user_roster)
    if dropped is None:
        raise ValueError(f"Player '{drop_name}' not found on roster")

    all_before, all_after = build_swap_standings(
        dropped, add_player, projected_standings, team_name
    )
    roto_before = score_roto_dict(all_before, team_sds=team_sds)
    roto_after = score_roto_dict(all_after, team_sds=team_sds)

    return score_swap(roto_before, roto_after, team_name)


@dataclass
class DeltaRotoBand:
    """Monte-Carlo confidence band for a before->after deltaRoto.

    ``mean`` approximately tracks the EV point estimate from
    :func:`compute_delta_roto`; ``sd`` and ``p_positive`` describe how
    often the swap actually helps once playing-time and performance
    variance are sampled. ``to_dict`` calls :func:`band_class` so the
    crosses-zero verdict is computed once, in Python, for every surface.
    """

    mean: float
    sd: float
    p_positive: float

    def to_dict(self) -> dict[str, float | str]:
        from fantasy_baseball.lineup.band_format import band_class

        return {
            "mean": round(self.mean, 2),
            "sd": round(self.sd, 2),
            "p_positive": round(self.p_positive, 3),
            "verdict": band_class(self.mean, self.sd),
        }


def _sum_realized(realized_rows: list[dict[str, Any]]) -> CategoryStats:
    """Sum one realization's per-player rows into a team CategoryStats.

    Mirrors ``simulation.simulate_season``: counting stats sum directly,
    rate stats recombine from component totals (H/AB, ER/IP, (BB+H)/IP).
    """
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

    h = [r for r in realized_rows if r["player_type"] == PlayerType.HITTER]
    p = [r for r in realized_rows if r["player_type"] == PlayerType.PITCHER]
    ab = sum(x["ab"] for x in h)
    hits = sum(x["h"] for x in h)
    ip = sum(x["ip"] for x in p)
    er = sum(x["er"] for x in p)
    bb = sum(x["bb"] for x in p)
    ha = sum(x["h_allowed"] for x in p)
    return CategoryStats(
        r=sum(x["r"] for x in h),
        hr=sum(x["hr"] for x in h),
        rbi=sum(x["rbi"] for x in h),
        sb=sum(x["sb"] for x in h),
        avg=calculate_avg(hits, ab),
        w=sum(x["w"] for x in p),
        k=sum(x["k"] for x in p),
        sv=sum(x.get("sv", 0) for x in p),
        era=calculate_era(er, ip),
        whip=calculate_whip(bb, ha, ip),
    )


def compute_delta_roto_band(
    before_players: list[Player],
    after_players: list[Player],
    field_stats: Mapping[str, CategoryStats],
    team_name: str,
    fraction_remaining: float,
    *,
    n_draws: int = 400,
    seed: int = 0,
) -> DeltaRotoBand:
    """Monte-Carlo the deltaRoto of a before->after roster change.

    Each draw samples realized stats for the union of both rosters ONCE
    via the calibrated variance model (``simulation._apply_variance``),
    so players shared between before and after get identical realized
    stats within a draw (common random numbers). The before and after
    subsets of that single realization are summed and scored against a
    FIXED field (``field_stats``); the per-draw delta is
    ``after_total - before_total``. Returns the mean, sd, and fraction
    of draws above zero.

    Args:
        before_players: roster before the change.
        after_players: roster after the change.
        field_stats: other teams' fixed point CategoryStats, keyed by name.
        team_name: user's team name (the key the swap is scored under).
        fraction_remaining: portion of the season left to simulate.
        n_draws: number of Monte-Carlo draws.
        seed: RNG seed for reproducibility.
    """
    import numpy as np

    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.simulation import _apply_variance, _flatten_full_season

    before_names = [p.name for p in before_players]
    after_names = [p.name for p in after_players]
    union = list({p.name: p for p in (after_players + before_players)}.values())
    union_h = [_flatten_full_season(p) for p in union if p.player_type == PlayerType.HITTER]
    union_p = [_flatten_full_season(p) for p in union if p.player_type == PlayerType.PITCHER]

    rng = np.random.default_rng(seed)
    field_table = dict(field_stats)
    deltas = np.empty(n_draws)
    for i in range(n_draws):
        inj: list[tuple[str, float]] = []
        rows = _apply_variance(union_h, PlayerType.HITTER, rng, inj, fraction_remaining)
        rows += _apply_variance(union_p, PlayerType.PITCHER, rng, inj, fraction_remaining)
        by_name = {r["name"]: r for r in rows}
        before_cs = _sum_realized([by_name[n] for n in before_names])
        after_cs = _sum_realized([by_name[n] for n in after_names])
        b = score_roto_dict({team_name: before_cs, **field_table}, team_sds=None)
        a = score_roto_dict({team_name: after_cs, **field_table}, team_sds=None)
        deltas[i] = a[team_name]["total"] - b[team_name]["total"]

    return DeltaRotoBand(
        mean=float(np.mean(deltas)),
        sd=float(np.std(deltas)),
        p_positive=float(np.mean(deltas > 0)),
    )


def compute_one_for_one_band(
    drop_name: str,
    add_player: Player,
    active_players: list[Player],
    field_stats: Mapping[str, CategoryStats],
    team_name: str,
    fraction_remaining: float,
    *,
    n_draws: int = 400,
    seed: int = 0,
) -> DeltaRotoBand:
    """Band for a one-for-one swap: drop ``drop_name``, add ``add_player``.

    Thin wrapper that builds the before/after rosters and delegates to
    :func:`compute_delta_roto_band`.
    """
    before = list(active_players)
    after = [p for p in active_players if p.name != drop_name] + [add_player]
    return compute_delta_roto_band(
        before,
        after,
        field_stats,
        team_name,
        fraction_remaining,
        n_draws=n_draws,
        seed=seed,
    )
