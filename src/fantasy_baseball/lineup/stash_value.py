"""Stash board -- rank injured players (owned IL + injured FAs) by their
leverage-aware marginal active value, and allocate the scarce IL slots.

Every candidate (owned IL or injured FA) is valued the SAME way: the best
rate-over-return-window swap of his ROS rate for the W-window of the weakest
eligible active player, scored against the user's HEALTHY ACTIVE LINEUP (every
IL player excluded -- see ``_active_lineup_standings``). Because the baseline
excludes ALL of the user's injured players, a candidate is a clean addition
whether or not you already roster him -- no ownership-based double-count, and
owned and FA candidates are directly comparable on one ranking.

A 100-AB injured bat is judged on the 100 ABs he'll actually play once healthy
(rate over his return window), not his season total, so low-volume-but-elite
players surface instead of being buried by playing-time-weighted totals.

The board is ranked by P(helps) -- the probability the candidate's best swap
improves the user's roto total -- with the expected Value shown alongside. There
is no slot "cost": scarcity is priced by the cutline (the top IL-capacity
candidates earn a slot), and each above-cutline free agent's ``recommended_drop``
names the distinct below-cutline owned stash it would bump when the IL is full.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.lineup.band_format import band_class
from fantasy_baseball.lineup.delta_roto import DeltaRotoBand, compute_delta_roto_band
from fantasy_baseball.lineup.pitcher_swap import discount_factor, swap_window_ip
from fantasy_baseball.lineup.optimizer import (
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

__all__ = ["StashResult", "StashScore", "score_stash_candidates"]


@dataclass
class StashScore:
    """One injured player's stash evaluation.

    ``stash_value`` is the expected roto-point gain (band mean, floored at ~0);
    P(helps) is ``band["p_positive"]``. The board is ranked by P(helps) (see
    :func:`_rank_key`). There is no slot cost -- scarcity is priced by the
    cutline, and an above-cutline free agent's ``recommended_drop`` names the
    distinct below-cutline owned stash it bumps when the IL is full (see
    :func:`_assign_recommended_drops`)."""

    name: str
    player_type: str
    status: str  # IL10 / IL15 / IL60 / ...
    owned: bool  # already on the user's roster
    stash_value: float  # expected roto-point gain (band mean, floored at ~0)
    band: dict[str, Any]  # {mean, sd, p_positive, verdict}; P(helps) = p_positive
    recommended_drop: str | None  # below-cutline owned stash this FA bumps; else None

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
    """Active + bench bodies with NO IL players, optionally excluding one
    player by name.

    Excludes by ``is_on_il()`` (status OR slot), not just slot: a
    BN+IL-status player is on the IL even though he's not in a true IL slot,
    so he must not leak into the active baseline. This matches candidate
    selection (``_owned_il_stashes``) and the standings classifier."""
    out: list[Player] = []
    for p in roster:
        if p.is_on_il():
            continue
        if exclude_name is not None and p.name == exclude_name:
            continue
        out.append(p)
    return out


def _ros_volume(p: Player) -> float:
    """Rest-of-season playing-time volume: AB for hitters, IP for pitchers.

    This is the window the player will actually be back for -- the basis for
    the rate-over-return-window comparison."""
    ros = p.rest_of_season
    if isinstance(ros, PitcherStats):
        return float(ros.ip or 0.0)
    if isinstance(ros, HitterStats):
        return float(ros.ab or 0.0)
    return 0.0


def _zero_band() -> dict[str, Any]:
    """Neutral band: the candidate upgrades no eligible active player."""
    return {"mean": 0.0, "sd": 0.0, "p_positive": 0.5, "verdict": band_class(0.5)}


def _synthetic_swap_line(incumbent: Player, candidate: Player, w: float) -> Player:
    """Synthetic replacement line for swapping ``candidate`` in for the W-window
    of ``incumbent``'s playing time.

    ``synth = scale * incumbent.ros + candidate.ros`` (component-wise), where
    ``scale = max(0, vol_I - W) / vol_I`` keeps the share of the incumbent's
    season the candidate does NOT take. This makes the team delta exactly
    ``candidate.ros - rate_incumbent * W`` -- i.e. trading W of the incumbent's
    playing time for W of the candidate's, at ROS rates. ``W >= vol_I`` clamps
    ``scale`` to 0 (full replacement). Rate fields are recomputed from the
    summed components so the line is self-consistent for downstream scoring."""
    inc = incumbent.rest_of_season
    cand = candidate.rest_of_season
    if isinstance(inc, PitcherStats) and isinstance(cand, PitcherStats):
        # Use the shared pitcher_swap helpers so the stash board and the
        # displacement model in scoring.py agree on cross-role math. The ``w``
        # argument is kept in the function signature for backwards
        # compatibility but only used as a fallback when ``swap_window_ip``
        # returns 0 (candidate has no ROS IP -- the legacy path).
        window = swap_window_ip(candidate, incumbent)
        if window <= 0.0:
            window = float(w)
        scale = discount_factor(float(inc.ip), window)
        ip = scale * inc.ip + cand.ip
        er = scale * inc.er + cand.er
        bb = scale * inc.bb + cand.bb
        ha = scale * inc.h_allowed + cand.h_allowed
        pitch = PitcherStats(
            ip=ip,
            w=scale * inc.w + cand.w,
            k=scale * inc.k + cand.k,
            sv=scale * inc.sv + cand.sv,
            er=er,
            bb=bb,
            h_allowed=ha,
            era=calculate_era(er, ip),
            whip=calculate_whip(bb, ha, ip),
        )
        return dataclasses.replace(incumbent, name=f"{candidate.name}__swap", rest_of_season=pitch)

    # Same player type guaranteed by the caller, so both are HitterStats here.
    assert isinstance(inc, HitterStats) and isinstance(cand, HitterStats)
    vol = float(inc.ab)
    scale = max(0.0, vol - w) / vol if vol > 0 else 0.0
    ab = scale * inc.ab + cand.ab
    h = scale * inc.h + cand.h
    hit = HitterStats(
        pa=scale * inc.pa + cand.pa,
        ab=ab,
        h=h,
        r=scale * inc.r + cand.r,
        hr=scale * inc.hr + cand.hr,
        rbi=scale * inc.rbi + cand.rbi,
        sb=scale * inc.sb + cand.sb,
        avg=calculate_avg(h, ab),
    )
    return dataclasses.replace(incumbent, name=f"{candidate.name}__swap", rest_of_season=hit)


def _best_swap_band(
    candidate: Player,
    *,
    before_active: list[Player],
    field_stats: Mapping[str, CategoryStats],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> dict[str, Any]:
    """FA Gain: the best rate-normalized swap against the weakest eligible
    active player, over the candidate's return window.

    For each position-eligible active incumbent (same player type -- the
    UTIL/generic-P slots make same-type swaps feasible), build a synthetic
    replacement line so the team delta is ``candidate.ros - rate_I * W`` and
    score it through the leverage-aware deltaRoto band. Gain is the best such
    swap, floored at 0 ("no harm, no foul" if he upgrades no one).

    Used for owned IL players and FAs alike. The caller passes the
    healthy-active-lineup baseline (``_marginal_band``), which excludes every IL
    player, so the candidate is a clean addition regardless of ownership.
    """
    w = _ros_volume(candidate)
    if w <= 0.0:
        return _zero_band()

    best: DeltaRotoBand | None = None
    for incumbent in before_active:
        if incumbent.player_type != candidate.player_type:
            continue
        if _ros_volume(incumbent) <= 0.0:
            continue
        synth = _synthetic_swap_line(incumbent, candidate, w)
        after = [p for p in before_active if p is not incumbent]
        after.append(synth)
        band = compute_delta_roto_band(
            before_active,
            after,
            field_stats,
            team_name,
            fraction_remaining,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )
        if best is None or band.mean > best.mean:
            best = band

    if best is None or best.mean <= 0.0:
        return _zero_band()
    return best.to_dict()


def _active_lineup_standings(
    before_active: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
) -> ProjectedStandings:
    """Standings whose USER row is the healthy active lineup only (ALL of the
    user's injured players excluded), opponents unchanged.

    This is the shared baseline every candidate is valued against. Because it
    excludes every IL player, an owned IL candidate is a clean addition just
    like a free agent -- no ownership-based double-count -- so owned and FA
    candidates land on one comparable scale. Full-season source matches the
    opponents' ``from_rosters`` rows so leverage (who's contesting which
    category) is computed on the same basis.
    """
    from fantasy_baseball.scoring import project_team_stats

    user_row = project_team_stats(before_active, projection_source="full_season_projection")
    entries = [
        ProjectedStandingsEntry(team_name=e.team_name, stats=user_row)
        if e.team_name == team_name
        else e
        for e in projected_standings.entries
    ]
    return ProjectedStandings(effective_date=projected_standings.effective_date, entries=entries)


def _marginal_band(
    candidate: Player,
    *,
    before_active: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> dict[str, Any]:
    """deltaRoto band dict for ``candidate``'s stash Gain.

    Owned IL players and injured FAs are valued identically: the best
    rate-upgrade-over-return-window swap of the candidate's ROS rate for the
    W-window of the weakest eligible active player (``_best_swap_band``),
    scored against the user's HEALTHY ACTIVE LINEUP
    (``_active_lineup_standings``). That baseline excludes every IL player, so a
    candidate is a clean addition whether or not you already roster him -- no
    double-count, and the whole board is one comparable ranking. An injured
    player with little remaining VOLUME but a strong RATE still scores: a 100-AB
    bat is judged on the 100 ABs he'll play, not his season total.
    """
    baseline = _active_lineup_standings(before_active, projected_standings, team_name)
    return _best_swap_band(
        candidate,
        before_active=before_active,
        field_stats=baseline.field_stats(team_name),
        projected_standings=baseline,
        team_name=team_name,
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
    )


def _marginal_value(
    candidate: Player,
    *,
    before_active: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> float:
    """Gain = band mean of the candidate's best rate-swap. Floored at ~0.

    Thin wrapper over ``_marginal_band`` for the test-only callers."""
    band = _marginal_band(
        candidate,
        before_active=before_active,
        projected_standings=projected_standings,
        team_name=team_name,
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
    )
    return float(band["mean"])


def _rank_key(score: StashScore) -> tuple[float, float]:
    """Board sort key: P(helps) first, then expected Value as a deterministic
    tie-break. Higher is better -- callers sort with ``reverse=True``.

    P(helps) (``band["p_positive"]``) is the probability the candidate's best
    swap improves the user's roto total. Ranking by it is risk-averse: a
    smaller-but-likelier upgrade outranks a larger-but-shakier one, and it is
    NOT the same order as ranking by Value (see the v3 design doc). Value breaks
    ties so equal-probability rows order deterministically."""
    return (score.band["p_positive"], score.stash_value)


def _open_il_slots(roster: list[Player], roster_slots: dict[str, int]) -> int:
    """IL capacity minus players currently in true IL slots."""
    capacity = roster_slots.get("IL", 0)
    occupied = sum(1 for p in roster if p.selected_position in IL_SLOTS)
    return max(0, capacity - occupied)


def _owned_il_stashes(roster: list[Player]) -> list[Player]:
    """Owned players on the IL (slot or status)."""
    return [p for p in roster if p.is_on_il()]


def _assign_recommended_drops(
    scores: list[StashScore], *, cutline_rank: int, open_il_slots: int
) -> None:
    """Set ``recommended_drop`` on each above-cutline free agent, in place.

    ``scores`` must already be sorted best-first by :func:`_rank_key`. The top
    ``cutline_rank`` rows earn the IL slots. Walking those above-cutline rows
    best-first: an owned player keeps the slot he already holds (no drop); a
    free agent fills an open IL slot if one remains, otherwise he bumps a
    BELOW-cutline owned stash -- one of the players who lost his slot to a better
    candidate. Drops are paired WORST-first: the best free agent bumps the
    weakest below-cutline owned stash, the next free agent the next-weakest, and
    so on. So if you act on only the top recommendation you drop the stash you'd
    miss least, and each free agent gets a DISTINCT owned stash -- the board
    never frees the same slot twice or names an above-cutline keeper.
    Below-cutline rows and owned rows keep ``None``.

    The below-cutline owned stashes exactly cover the above-cutline free agents
    that still need a slot once the open ones are used (the cutline balances by
    construction); the ``next_drop`` bound is a guard against messy real-world
    states where IL-by-status and IL-by-slot counts disagree."""
    above = scores[:cutline_rank]
    # Worst-first (reverse the best-first board tail): the top FA bumps the
    # weakest owned stash, so a single add costs the user the least.
    droppable = [s.name for s in reversed(scores[cutline_rank:]) if s.owned]
    open_slots = open_il_slots
    next_drop = 0
    for score in above:
        if score.owned:
            continue  # already holds a slot
        if open_slots > 0:
            open_slots -= 1  # fills an open IL slot -- nothing to drop
            continue
        if next_drop < len(droppable):
            score.recommended_drop = droppable[next_drop]
            next_drop += 1


def _cap_candidates(scores: list[StashScore], max_candidates: int) -> list[StashScore]:
    """Cap the board at ``max_candidates`` rows without ever hiding an owned
    stash. An owned stash can be named as another row's ``recommended_drop``, so
    it must stay on the rendered board; only free agents are truncated. Owned
    stashes are bounded by IL capacity, so the total never exceeds the cap.
    Sorted order is preserved (callers pass an already-sorted list)."""
    fa_budget = max(0, max_candidates - sum(1 for s in scores if s.owned))
    capped: list[StashScore] = []
    for score in scores:
        if score.owned:
            capped.append(score)
        elif fa_budget > 0:
            capped.append(score)
            fa_budget -= 1
    return capped


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
    """Rank injured players (owned IL + injured FAs) by P(helps).

    Each candidate's Value is its marginal active value (band mean, floored at
    ~0); P(helps) is ``band["p_positive"]``. The board sorts by P(helps) with
    Value breaking ties (:func:`_rank_key`); the top ``IL``-capacity candidates
    earn a slot. Each above-cutline free agent's ``recommended_drop`` names the
    distinct below-cutline owned stash it bumps when the IL is full
    (:func:`_assign_recommended_drops`). Slot scarcity is priced by the cutline,
    not by a per-row cost.
    """
    il_capacity = roster_slots.get("IL", 0)
    owned_il = _owned_il_stashes(roster)
    injured_fas = [fa for fa in free_agents if fa.is_on_il()]

    # No injured players -> nothing to rank; skip the optimizer baseline entirely.
    if not owned_il and not injured_fas:
        return StashResult(
            open_il_slots=_open_il_slots(roster, roster_slots),
            cutline_rank=il_capacity,
        )

    # before_active is identical for every candidate: the optimized lineup over
    # the counted (non-IL-slot) bodies, with NO candidate activated.
    before_active = _solve_active(
        _counted_pool(roster), roster_slots, projected_standings, team_name, team_sds
    )

    # Value + band for every candidate (owned + FA).
    candidates_in: list[tuple[Player, bool]] = [(p, True) for p in owned_il] + [
        (p, False) for p in injured_fas
    ]
    bands: dict[str, dict[str, Any]] = {}
    for player, _owned in candidates_in:
        bands[player.name] = _marginal_band(
            player,
            before_active=before_active,
            projected_standings=projected_standings,
            team_name=team_name,
            team_sds=team_sds,
            fraction_remaining=fraction_remaining,
        )

    scores: list[StashScore] = []
    for player, owned in candidates_in:
        band = bands[player.name]
        scores.append(
            StashScore(
                name=player.name,
                player_type=player.player_type.value,
                status=player.status,
                owned=owned,
                stash_value=round(band["mean"], 2),
                band=band,
                recommended_drop=None,
            )
        )

    scores.sort(key=_rank_key, reverse=True)
    # Drops are assigned by cutline position, so the board must be sorted first:
    # each above-cutline FA bumps a distinct below-cutline owned stash.
    open_slots = _open_il_slots(roster, roster_slots)
    _assign_recommended_drops(scores, cutline_rank=il_capacity, open_il_slots=open_slots)
    return StashResult(
        open_il_slots=open_slots,
        cutline_rank=il_capacity,
        candidates=_cap_candidates(scores, max_candidates),
    )
