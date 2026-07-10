import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.positions import HITTER_ELIGIBLE, PITCHER_ELIGIBLE, Position
from fantasy_baseball.models.standings import ProjectedStandings, Standings, TeamYtdComponents
from fantasy_baseball.scoring import (
    LeagueContext,
    project_ros_components,
    score_roto_dict,
    team_end_of_season,
)
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS, Category
from fantasy_baseball.utils.positions import can_fill_slot


@dataclass
class HitterAssignment:
    slot: Position
    name: str
    player: Player
    roto_delta: float
    band: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot.value,
            "name": self.name,
            "roto_delta": round(self.roto_delta, 2),
            "band": self.band,
        }


@dataclass
class PitcherStarter:
    name: str
    player: Player
    roto_delta: float
    band: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "roto_delta": round(self.roto_delta, 2),
            "band": self.band,
        }


def _build_hitter_slot_positions(roster_slots: dict[str, int]) -> list[Position]:
    """Return the ordered list of active hitter slot Position values from config."""
    slots: list[Position] = []
    for pos_key, count in roster_slots.items():
        pos = pos_key if isinstance(pos_key, Position) else Position.parse(pos_key)
        if pos not in HITTER_ELIGIBLE:
            continue
        for _ in range(count):
            slots.append(pos)
    return slots


def _feasible_assignment(
    subset: list[Player],
    slot_positions: list[Position],
) -> list[Position] | None:
    """Return a list parallel to ``subset`` giving each player's assigned slot,
    or None if no valid assignment exists.

    Uses Hungarian on a 0/1 feasibility matrix (cost 0 for eligible, large cost
    for ineligible). A valid matching has zero total cost.
    """
    n_players = len(subset)
    n_slots = len(slot_positions)
    if n_players != n_slots:
        return None
    size = n_players
    cost = np.full((size, size), 1e9)
    for i, p in enumerate(subset):
        for j, slot in enumerate(slot_positions):
            if can_fill_slot(p.positions, slot.value):
                cost[i][j] = 0.0
    row_idx, col_idx = linear_sum_assignment(cost)
    matched: dict[int, Position] = {}
    for r, c in zip(row_idx, col_idx, strict=False):
        if cost[r][c] > 0.5:
            return None
        matched[r] = slot_positions[c]
    # linear_sum_assignment returns one column per row, so every row index is
    # present in ``matched``; rebuild the subset-parallel list in order.
    return [matched[i] for i in range(n_players)]


@dataclass
class _TeamContext:
    """Scoring-side inputs passed through every ERoto evaluation.

    ``user_ytd_components`` (Team-YTD ingredients pulled from the live Yahoo
    standings snapshot) anchors the user's row at the same end-of-season
    scale as :func:`ProjectedStandings.from_rosters` uses for opponents:
    ``team_end_of_season(user_ytd, project_ros_components(hypothetical))``.

    Without it the user row collapses to ROS-only while opponents are
    team_YTD + ROS, so the user lives in a low-mu region of the
    ``score_roto`` S-curve and counting-cat deltas saturate. The default
    is zero components, matching pre-season behavior and every legacy
    caller that doesn't have a live standings snapshot.
    """

    full_roster: list[Player]
    projected_standings: ProjectedStandings
    team_name: str
    team_sds: Mapping[str, Mapping[Category, float]] | None = None
    user_ytd_components: TeamYtdComponents = field(default_factory=TeamYtdComponents)
    league_context: LeagueContext | None = None
    """ROS displacement context for the user row, threaded into
    ``project_ros_components``. When provided, pitcher displacement uses the
    ROTO-optimal pair-swap pool model -- the SAME model
    :func:`ProjectedStandings.from_rosters` uses to build the opponent rows and
    the standings the displayed deltaRoto band reads. Without it, displacement
    falls back to the legacy SGP picker, which systematically zeroes elite
    low-volume closers' saves when an IL pitcher is rostered (see
    :class:`fantasy_baseball.scoring.LeagueContext`), so the optimizer's
    selection silently disagrees with the band it displays. ``None`` preserves
    the legacy SGP path for callers that build a ``_TeamContext`` directly."""


def apply_lineup_to_roster(
    full_roster: list[Player],
    active_slots: dict[str, Position],
    bench_names: set[str],
) -> list[Player]:
    """Return a copy of ``full_roster`` with selected_position overridden.

    - Names in ``active_slots`` -> selected_position set to the mapped Position.
    - Names in ``bench_names`` -> selected_position set to Position.BN.
    - All other players (IL, opposite type) unchanged.
    """
    result: list[Player] = []
    for p in full_roster:
        if p.name in bench_names:
            result.append(dataclasses.replace(p, selected_position=Position.BN))
        elif p.name in active_slots:
            result.append(dataclasses.replace(p, selected_position=active_slots[p.name]))
        else:
            result.append(p)
    return result


def team_roto_total(hypothetical: list[Player], ctx: _TeamContext) -> float:
    """Score ``hypothetical`` as the user's row against ``ctx.projected_standings``.

    The user row is built as
    ``team_end_of_season(ctx.user_ytd_components, project_ros_components(hypothetical))``
    -- the same team_YTD + ROS math :func:`ProjectedStandings.from_rosters`
    uses for the opponent rows in ``ctx.projected_standings``. Without
    matching scales the user lives in a low-mu region of the ``score_roto``
    S-curve and marginal counting-cat deltas saturate (the bug PR #110
    fixed for the stash board via ``_active_lineup_standings`` and missed
    here; see the team-YTD refactor docstring on
    :func:`fantasy_baseball.scoring.project_team_stats`).

    ``displacement=True`` matches the pre-fix call (the original
    ``project_team_stats(hypothetical, displacement=True)``):
    ``hypothetical`` is the user's full roster with active/bench/IL slots
    assigned, so the standard IL-displaces-worst-active scaling still
    applies. (Contrast with the stash board's ``_active_lineup_standings``,
    which receives a pre-filtered active pool and so uses
    ``displacement=False``.)
    """
    user_ros = project_ros_components(
        hypothetical, displacement=True, league_context=ctx.league_context
    )
    my_stats = team_end_of_season(ctx.user_ytd_components, user_ros).to_dict()
    all_stats = {e.team_name: e.stats.to_dict() for e in ctx.projected_standings.entries}
    all_stats[ctx.team_name] = my_stats
    return score_roto_dict(all_stats, team_sds=ctx.team_sds)[ctx.team_name]["total"]


def _score_hitter_subset(
    ctx: _TeamContext,
    subset: list[Player],
    assignment: list[Position],
    bench: list[Player],
) -> float:
    hypothetical = apply_lineup_to_roster(
        ctx.full_roster,
        active_slots={p.name: slot for p, slot in zip(subset, assignment, strict=False)},
        bench_names={h.name for h in bench},
    )
    return team_roto_total(hypothetical, ctx)


def _pitcher_active_slots(subset: list[Player]) -> dict[str, Position]:
    return {
        p.name: next((pos for pos in p.positions if pos in PITCHER_ELIGIBLE), Position.P)
        for p in subset
    }


def _score_pitcher_subset(
    ctx: _TeamContext,
    subset: list[Player],
    bench: list[Player],
    memo: dict[frozenset[str], float] | None = None,
) -> float:
    """Score the team roto total with ``subset`` as the active pitchers.

    Within a single :func:`optimize_pitcher_lineup` call ``ctx`` is fixed
    (same hitters, IL pitchers, ``league_context``), so the total is a pure
    function of *which* pitchers are active: benched pitchers contribute
    nothing and the IL/hitter halves are constant. The optional ``memo``
    caches results keyed on the active-pitcher name set, so the per-starter
    ``roto_delta`` loop -- which re-scores subsets the selection loop already
    scored, with heavy overlap across starters -- does not re-run the
    expensive IL pair-swap pool model
    (:func:`scoring._compute_pitcher_pool_factors`) for an active set it has
    already priced.
    """
    key: frozenset[str] | None = None
    if memo is not None:
        key = frozenset(p.name for p in subset)
        if key in memo:
            return memo[key]
    hypothetical = apply_lineup_to_roster(
        ctx.full_roster,
        _pitcher_active_slots(subset),
        {p.name for p in bench},
    )
    total = team_roto_total(hypothetical, ctx)
    if memo is not None and key is not None:
        memo[key] = total
    return total


def _resolve_user_ytd_components(
    actual_standings: Standings | None, team_name: str
) -> TeamYtdComponents:
    """Extract the user's YTD components from the live standings snapshot.

    Returns zero components when ``actual_standings`` is None (pre-season,
    legacy callers) or when the user's team is missing from the snapshot --
    ``team_end_of_season(zero, ros)`` collapses to ROS-only, matching the
    pre-team-YTD-refactor behavior. Called ONCE per public-optimizer entry
    so the constant doesn't get recomputed inside the combinatorial loops.
    """
    if actual_standings is None:
        return TeamYtdComponents()
    for entry in actual_standings.entries:
        if entry.team_name == team_name:
            return entry.ytd_components()
    return TeamYtdComponents()


def _build_league_context(
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float | None,
) -> LeagueContext:
    """Build the ROS-displacement context for the user row.

    Mirrors the ``LeagueContext`` :func:`ProjectedStandings.from_rosters`
    constructs per team, so the optimizer's ``team_roto_total`` displaces the
    user's pitchers with the SAME ROTO-optimal pool model that built the
    standings -- making the selector consistent with the displayed deltaRoto
    band and fixing the elite-closer-saves-zeroing bug.

    ``baseline_other_team_stats`` is the frozen field (opponent end-of-season
    rows) from ``projected_standings``; the picker scores candidate
    displacements against it. ``team_sds`` falls back to an empty mapping
    (rank-based scoring, matching ``team_roto_total`` when no SDs are given).
    ``fraction_remaining`` falls back to 1.0 (whole season) for callers that
    omit it.
    """
    return LeagueContext(
        baseline_other_team_stats=projected_standings.field_stats(team_name),
        team_sds=team_sds or {},
        team_name=team_name,
        fraction_remaining=fraction_remaining if fraction_remaining is not None else 1.0,
    )


def optimize_hitter_lineup(
    hitters: list[Player],
    full_roster: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    roster_slots: dict[str, int] | None = None,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    fraction_remaining: float | None = None,
    actual_standings: Standings | None = None,
    compute_bands: bool = True,
) -> list[HitterAssignment]:
    """Return the ERoto-maximizing active hitter lineup.

    ``actual_standings`` is the live Yahoo standings snapshot at the same
    effective_date as ``projected_standings``. When provided, the user's
    row inside ``team_roto_total`` is built as
    ``team_end_of_season(team_YTD, project_ros_components(hypothetical))``,
    matching the opponent rows produced by
    :func:`ProjectedStandings.from_rosters`. ``None`` collapses to ROS-only
    -- preserves pre-team-YTD-refactor behavior for pre-season callers and
    legacy tests.

    ``compute_bands`` gates per-starter band computation independently of
    ``fraction_remaining`` (a band still requires a non-None
    ``fraction_remaining`` for its variance scale). Callers that want the
    lineup but not the bands pass ``compute_bands=False``.

    No ``league_context`` is threaded here: the hitter selection's pitcher
    half is identical across every hitter subset, so the pitcher-displacement
    model cancels in the marginal and cannot change the hitter pick. Only
    :func:`optimize_pitcher_lineup` needs the ROTO-optimal pool model (where
    the elite-closer-saves-zeroing bug lives); keeping it out of the hitter
    combinatorial loop avoids running the per-subset pool picker thousands of
    times for no effect.
    """
    if not hitters:
        return []
    slot_positions = _build_hitter_slot_positions(
        roster_slots if roster_slots is not None else DEFAULT_ROSTER_SLOTS
    )
    n_slots = len(slot_positions)
    user_ytd = _resolve_user_ytd_components(actual_standings, team_name)
    ctx = _TeamContext(
        full_roster,
        projected_standings,
        team_name,
        team_sds,
        user_ytd_components=user_ytd,
    )

    field_stats = projected_standings.field_stats(team_name)

    if n_slots == 0 or len(hitters) < n_slots:
        # Fewer hitters than slots -- fall back to the best feasible partial lineup.
        partial_best: tuple[float, list[Player], list[Position]] | None = None
        for size in range(min(len(hitters), n_slots), 0, -1):
            for subset in combinations(hitters, size):
                assn = _feasible_assignment(list(subset), slot_positions[:size])
                if assn is None:
                    continue
                bench = [h for h in hitters if h not in subset]
                total = _score_hitter_subset(ctx, list(subset), assn, bench)
                if partial_best is None or total > partial_best[0]:
                    partial_best = (total, list(subset), assn)
            if partial_best is not None:
                break
        if partial_best is None:
            return []
        return [
            HitterAssignment(slot=slot, name=p.name, player=p, roto_delta=0.0)
            for p, slot in zip(partial_best[1], partial_best[2], strict=False)
        ]

    best: tuple[float, list[Player], list[Position], list[Player]] | None = None
    for subset in combinations(hitters, n_slots):
        assn = _feasible_assignment(list(subset), slot_positions)
        if assn is None:
            continue
        bench = [h for h in hitters if h not in subset]
        total = _score_hitter_subset(ctx, list(subset), assn, bench)
        if best is None or total > best[0]:
            best = (total, list(subset), assn, bench)

    if best is None:
        return []

    best_total, active_subset, assignment, bench = best

    # Active pitchers on this roster -- identical in before/after, so they
    # cancel in the marginal but anchor the band at the correct full-team
    # operating point on the win-probability S-curve.
    pitcher_half = [
        p for p in full_roster if set(p.positions) & PITCHER_ELIGIBLE and not p.is_on_il()
    ]

    band_reference: list[Player] | None = None
    if compute_bands and fraction_remaining is not None:
        from fantasy_baseball.lineup.delta_roto import band_reference_lineup

        band_reference = band_reference_lineup(hitters, pitcher_half)

    roto_deltas: dict[str, float] = {}
    bands: dict[str, dict[str, Any]] = {}
    for starter in active_subset:
        remaining_hitters = [h for h in hitters if h is not starter]
        alt_best: float | None = None
        alt_best_subset: list[Player] = []
        for sub in combinations(remaining_hitters, n_slots):
            assn = _feasible_assignment(list(sub), slot_positions)
            if assn is None:
                continue
            sub_bench = [h for h in remaining_hitters if h not in sub] + [starter]
            t = _score_hitter_subset(ctx, list(sub), assn, sub_bench)
            if alt_best is None or t > alt_best:
                alt_best = t
                alt_best_subset = list(sub)
        if alt_best is None:
            # No feasible full-size replacement lineup (the roster is too
            # thin to cover every slot without this starter). Counterfactual
            # is "starter benched, their slot left empty" -- score the rest
            # of the optimal lineup without them.
            no_rep_subset = [p for p in active_subset if p is not starter]
            no_rep_assn = [
                a for p, a in zip(active_subset, assignment, strict=False) if p is not starter
            ]
            alt_best = _score_hitter_subset(ctx, no_rep_subset, no_rep_assn, [*bench, starter])
            alt_best_subset = no_rep_subset
        roto_deltas[starter.name] = best_total - alt_best

        if compute_bands and fraction_remaining is not None:
            from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band

            band_result = compute_delta_roto_band(
                [*alt_best_subset, *pitcher_half],
                [*active_subset, *pitcher_half],
                field_stats,
                team_name,
                fraction_remaining,
                projected_standings=projected_standings,
                team_sds=team_sds,
                reference_players=band_reference,
            )
            bands[starter.name] = band_result.to_dict()

    return [
        HitterAssignment(
            slot=slot,
            name=p.name,
            player=p,
            roto_delta=roto_deltas.get(p.name, 0.0),
            band=bands.get(p.name),
        )
        for p, slot in zip(active_subset, assignment, strict=False)
    ]


def optimize_pitcher_lineup(
    pitchers: list[Player],
    full_roster: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    slots: int = 9,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    fraction_remaining: float | None = None,
    actual_standings: Standings | None = None,
    compute_bands: bool = True,
) -> tuple[list[PitcherStarter], list[Player]]:
    """Return (starters with roto_delta, bench) maximizing ERoto.

    ``actual_standings`` is threaded through to ``team_roto_total`` so the
    user's row is built as team_YTD + ROS, matching the opponent rows in
    ``projected_standings``. See :func:`optimize_hitter_lineup`.

    A ``league_context`` IS built here (unlike the hitter optimizer) so
    pitcher displacement uses the ROTO-optimal pair-swap pool model that the
    standings/band use -- without it the legacy SGP picker zeroes an elite
    low-volume closer's saves whenever an IL pitcher is rostered. The pool
    model's slot-share sizing depends on ``fraction_remaining``: pass the REAL
    remaining-season fraction (NOT None) so a returning IL arm is measured
    against the remaining season, not the whole year. ``fraction_remaining``
    falls back to 1.0 inside the context only for pre-season/unknown callers.

    ``compute_bands`` gates per-starter band computation independently of
    ``fraction_remaining``, so callers needing correct in-season displacement
    but not the (expensive) per-starter bands -- the stash board, IL-return
    planner, roster audit -- pass the real ``fraction_remaining`` with
    ``compute_bands=False``.
    """
    if not pitchers or slots <= 0:
        return [], list(pitchers)
    k = min(slots, len(pitchers))
    user_ytd = _resolve_user_ytd_components(actual_standings, team_name)
    ctx = _TeamContext(
        full_roster,
        projected_standings,
        team_name,
        team_sds,
        user_ytd_components=user_ytd,
        league_context=_build_league_context(
            projected_standings, team_name, team_sds, fraction_remaining
        ),
    )

    field_stats = projected_standings.field_stats(team_name)

    # Cache subset scores across the selection loop and the per-starter
    # roto_delta loop. team_roto_total is a pure function of the active-pitcher
    # set for this fixed ctx, and the delta loop re-scores subsets the
    # selection loop already priced (with overlap across starters); without the
    # memo each rescore re-runs the IL pair-swap pool model from scratch.
    score_memo: dict[frozenset[str], float] = {}

    best = None
    for subset in combinations(pitchers, k):
        bench = [p for p in pitchers if p not in subset]
        total = _score_pitcher_subset(ctx, list(subset), bench, score_memo)
        if best is None or total > best[0]:
            best = (total, list(subset), bench)

    # ``pitchers`` is non-empty and ``k >= 1`` (guarded at function top), so
    # ``combinations`` yields at least one subset and ``best`` is always set.
    assert best is not None
    best_total, active_subset, bench = best

    # Active hitters on this roster -- identical in before/after, so they
    # cancel in the marginal but anchor the band at the correct full-team
    # operating point on the win-probability S-curve.
    hitter_half = [
        p for p in full_roster if not (set(p.positions) & PITCHER_ELIGIBLE) and not p.is_on_il()
    ]

    band_reference: list[Player] | None = None
    if compute_bands and fraction_remaining is not None:
        from fantasy_baseball.lineup.delta_roto import band_reference_lineup

        band_reference = band_reference_lineup(pitchers, hitter_half)

    roto_deltas: dict[str, float] = {}
    bands: dict[str, dict[str, Any]] = {}
    for starter in active_subset:
        remaining = [p for p in pitchers if p is not starter]
        alt_best: float | None = None
        alt_best_subset: list[Player] = []
        if len(remaining) >= k:
            for sub in combinations(remaining, k):
                sub_bench = [p for p in remaining if p not in sub] + [starter]
                t = _score_pitcher_subset(ctx, list(sub), sub_bench, score_memo)
                if alt_best is None or t > alt_best:
                    alt_best = t
                    alt_best_subset = list(sub)
        if alt_best is None:
            # No feasible full-size replacement (roster has exactly k
            # pitchers, no bench). Counterfactual is "starter benched,
            # their slot left empty" -- score the other k-1 starters alone.
            no_rep_subset = [p for p in active_subset if p is not starter]
            alt_best = _score_pitcher_subset(ctx, no_rep_subset, [*bench, starter], score_memo)
            alt_best_subset = no_rep_subset
        roto_deltas[starter.name] = best_total - alt_best

        if compute_bands and fraction_remaining is not None:
            from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band

            band_result = compute_delta_roto_band(
                [*alt_best_subset, *hitter_half],
                [*active_subset, *hitter_half],
                field_stats,
                team_name,
                fraction_remaining,
                projected_standings=projected_standings,
                team_sds=team_sds,
                reference_players=band_reference,
            )
            bands[starter.name] = band_result.to_dict()

    starters = [
        PitcherStarter(
            name=p.name,
            player=p,
            roto_delta=roto_deltas[p.name],
            band=bands.get(p.name),
        )
        for p in active_subset
    ]
    return starters, bench


def combined_team_roto(
    roster: list[Player],
    hitters: list[Player],
    hitter_lineup: list[HitterAssignment],
    pitcher_starters: list[PitcherStarter],
    pitcher_bench: list[Player],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    actual_standings: Standings | None = None,
) -> float:
    """Score a combined hitter + pitcher lineup as a single ERoto total.

    The per-side optimizers score independently; this recomputes once on the
    combined lineup so the reported total reflects both sides together.

    ``actual_standings`` is threaded through so the user's row matches the
    team_YTD + ROS scale of the opponent rows. See
    :func:`optimize_hitter_lineup`.
    """
    active_slots: dict[str, Position] = {a.name: a.slot for a in hitter_lineup}
    active_slots.update(_pitcher_active_slots([s.player for s in pitcher_starters]))
    bench_names = {h.name for h in hitters} - {a.name for a in hitter_lineup} | {
        p.name for p in pitcher_bench
    }
    hypothetical = apply_lineup_to_roster(roster, active_slots, bench_names)
    user_ytd = _resolve_user_ytd_components(actual_standings, team_name)
    ctx = _TeamContext(
        roster,
        projected_standings,
        team_name,
        team_sds,
        user_ytd_components=user_ytd,
    )
    return team_roto_total(hypothetical, ctx)
