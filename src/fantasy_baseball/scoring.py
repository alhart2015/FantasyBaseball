"""Roto scoring and team stat projection — shared across all modules.

Provides core functions:
- project_team_stats: sum projected stats for a roster into a
  CategoryStats. Accepts Player dataclass objects OR flat dicts for
  backwards compatibility with draft/script callers that still build
  rosters as plain dicts.
- project_team_sds: analytical team-level standard deviations for
  each category under player-independence, used to price projection
  uncertainty into score_roto.
- score_roto: assign expected roto points via pairwise Gaussian
  win-probabilities. With team_sds=None, collapses to rank-based
  scoring with averaged ties (backwards-compatible default).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import erf, sqrt
from typing import Literal, Protocol

from fantasy_baseball.lineup.pitcher_swap import discount_factor, swap_window_ip
from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import (
    CategoryPoints,
    CategoryStats,
    ProjectedStandingsEntry,
    TeamYtdComponents,
)
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import (
    AB_PER_PA,
    HITTING_COUNTING,
    PITCHING_COUNTING,
    Category,
    role_from_ip,
)
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.constants import (
    INVERSE_STATS as INVERSE_CATS,
)
from fantasy_baseball.utils.constants import (
    safe_float as _safe,
)
from fantasy_baseball.utils.dispersion import negbin_perf_variance
from fantasy_baseball.utils.playing_time import playing_time_params
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

ProjectionSource = Literal["rest_of_season", "full_season_projection"]


class TeamStatsRow(Protocol):
    """Minimal shape for a standings row: ``team_name`` plus ``CategoryStats``."""

    team_name: str
    stats: CategoryStats


class TeamStatsTable(Protocol):
    """Any object with a sequence of ``TeamStatsRow`` entries.

    Concrete implementations: :class:`Standings`, :class:`ProjectedStandings`.
    """

    entries: Sequence[TeamStatsRow]


def _get(p, key, default=0):
    """Read a field from a Player dataclass or a plain dict."""
    if hasattr(p, key):
        return getattr(p, key)
    if isinstance(p, dict):
        return p.get(key, default)
    return default


def _stat(p, key, source: ProjectionSource = "rest_of_season"):
    """Read a stat from a Player's projection stats or from a flat dict.

    ``source`` selects which projection field on a :class:`Player` to read
    from: ``"rest_of_season"`` (the default — forward-looking decision math
    used by the optimizer, recs, and trade evaluation) or
    ``"full_season_projection"`` (= ROS + YTD; used by
    :meth:`ProjectedStandings.from_rosters` to preserve the
    end-of-season-totals projection until a proper standings + ROS
    combination ships).

    When the requested source is missing on the Player (e.g. preseason
    rosters are matched against a single projection frame and end up with
    ``full_season_projection=None`` — the preseason values live in
    ``rest_of_season``), falls back to ``rest_of_season``. Without this
    fallback, ``ProjectedStandings.from_rosters`` would read zeros for
    every preseason team and the preseason-standings widget would show
    a default-only (R=0, ERA=99) board.

    Flat-dict input (draft scripts) is unaffected by ``source`` — those
    rosters carry a single set of stat keys with no ROS/full distinction.
    """
    # Player dataclass: stats live on the .{source} attribute.
    stats = getattr(p, source, None)
    if stats is None and source != "rest_of_season":
        stats = getattr(p, "rest_of_season", None)
    if stats is not None and hasattr(stats, key):
        return _safe(getattr(stats, key, 0))
    # Flat dict or pandas Series (legacy callers, tests, draft scripts).
    # Series is not a dict but exposes the same ``.get`` -- gating on
    # ``isinstance(p, dict)`` silently zeroed Series rosters (the draft
    # simulator feeds Series rows), tying every team's roto categories.
    get = getattr(p, "get", None)
    if callable(get):
        return _safe(get(key, 0))
    return 0.0


def _prob_beats(
    mu_a: float,
    mu_b: float,
    sd_a: float,
    sd_b: float,
    *,
    higher_is_better: bool,
) -> float:
    """P(team A's category total exceeds team B's) under Gaussian independence.

    When combined SD is zero, this is a step function: 1.0 if A is ahead,
    0.0 if behind, 0.5 on exact equality. Positive combined SD smooths the
    step into a continuous sigmoid. The ``higher_is_better`` flag flips
    the direction for inverse categories (ERA, WHIP).

    This is the pairwise primitive the EV-based ``score_roto`` sums over.
    """
    diff = (mu_a - mu_b) if higher_is_better else (mu_b - mu_a)
    combined = sqrt(sd_a * sd_a + sd_b * sd_b)
    if combined == 0.0:
        if diff > 0:
            return 1.0
        if diff < 0:
            return 0.0
        return 0.5
    return 0.5 * (1.0 + erf(diff / (combined * sqrt(2.0))))


# ── Displacement helpers ────────────────────────────────────────────

# Generic slots that are ignored when matching positions for displacement.
_GENERIC_SLOTS: frozenset[Position] = frozenset(
    {
        Position.P,
        Position.UTIL,
        Position.IF,
        Position.DH,
        Position.BN,
        Position.IL,
        Position.IL_PLUS,
        Position.DL,
        Position.DL_PLUS,
    }
)


def _is_bench(p: Player) -> bool:
    """True if the player is benched (BN slot) and NOT on the IL."""
    return p.selected_position == Position.BN and not p.is_on_il()


def _classify_roster(
    roster: list[Player],
) -> tuple[list[Player], list[Player], list[Player]]:
    """Slot-first partition of a roster into (active, il, bench).

    - Active slot → active (counted at face value; may be displaced).
    - IL / IL+ / DL / DL+ slot → il.
    - BN + IL status → il (same displacement path as IL-slotted).
    - BN + healthy → bench (excluded).

    Non-Player entries (dict inputs from draft scripts) are skipped
    entirely — callers that support them must handle dicts separately.
    """
    active: list[Player] = []
    il_players: list[Player] = []
    bench: list[Player] = []
    for p in roster:
        if not isinstance(p, Player):
            continue
        slot = p.selected_position
        if slot == Position.BN:
            if p.is_on_il():
                il_players.append(p)
            else:
                bench.append(p)
            continue
        if slot in IL_SLOTS:
            il_players.append(p)
            continue
        active.append(p)
    return active, il_players, bench


def _playing_time(p: Player) -> float:
    """Return the playing-time measure: IP for pitchers, PA (or AB) for hitters."""
    if p.rest_of_season is None:
        return 0.0
    if isinstance(p.rest_of_season, PitcherStats):
        return _safe(p.rest_of_season.ip)
    # Hitters: prefer PA, fall back to AB
    pa = _safe(getattr(p.rest_of_season, "pa", 0))
    if pa > 0:
        return pa
    return _safe(getattr(p.rest_of_season, "ab", 0))


def _pitcher_role(p: Player) -> str:
    """Classify a pitcher as 'SP' or 'RP' based on projected IP."""
    ip = _safe(p.rest_of_season.ip) if isinstance(p.rest_of_season, PitcherStats) else 0.0
    return role_from_ip(ip)


def _real_positions(p: Player) -> frozenset[Position]:
    """Return the player's eligible positions minus generic/bench/IL slots."""
    return frozenset(p.positions) - _GENERIC_SLOTS


@dataclass(frozen=True)
class LeagueContext:
    """League-wide context that lets displacement target selection use
    ΔRoto (team roto pts impact) instead of raw SGP.

    SGP-based selection is volume-weighted and systematically picks
    elite-but-low-volume players (closers) for displacement, since
    their total SGP is small next to a 150-IP starter. ΔRoto sees that
    losing 27 SV costs a roto point in SV but losing one SP's IP
    barely moves K/W standings — and picks the actually-droppable arm.

    Used at the standings-build layer only (``ProjectedStandings`` and
    ``build_standings_breakdown_payload``); other callers pass ``None``
    and keep the SGP picker.

    Fields:
        baseline_other_team_stats: ``{team_name: CategoryStats}`` for
            every OTHER team in the league, frozen from a pass-1 SGP
            standings build. Excludes the team being optimized.
        team_sds: per-team category SDs (from ``build_team_sds``), used
            by the Gaussian-pairwise scorer for variance pricing.
        team_name: the team whose roster the picker is currently
            evaluating displacement candidates for.
        fraction_remaining: fraction of the season still to play. Passed
            to ``pitcher_swap.swap_window_ip`` so a returning IL pitcher's
            slot-share is measured against a healthy *remaining-season*
            workload (``preseason.ip * fraction_remaining``), not the full
            season -- otherwise an arm back for the whole rest of the year
            reads as a part-timer. Defaults to 1.0 (whole season remaining)
            for callers that don't supply it.
    """

    baseline_other_team_stats: Mapping[str, CategoryStats]
    team_sds: Mapping[str, Mapping[Category, float]]
    team_name: str
    fraction_remaining: float = 1.0


def _find_worst_match(
    il_player: Player,
    active_players: list[Player],
    already_displaced: set[str],
    *,
    league_context: LeagueContext | None = None,
    current_roster: list[Player | dict] | None = None,
) -> Player | None:
    """Find the displacement target for ``il_player``.

    For pitchers: any active pitcher is eligible (SP/RP roles are not
    fungible in real-life production but ARE fungible in fantasy P
    slots — Hader returning bumps the worst arm overall, not the only
    same-role match). For hitters: prefer overlapping real positions
    (1B-only can't fill SS slot), fall back to any active hitter.

    Selection rule: when ``league_context`` is provided, picks the
    candidate that maximizes the team's expected roto pts after the
    displacement (ΔRoto-optimal). Otherwise picks the lowest-SGP
    candidate (the historical behavior).

    Returns None if no eligible active player exists.
    """
    candidates: list[Player] = []

    if il_player.player_type == PlayerType.PITCHER:
        for a in active_players:
            if a.name in already_displaced:
                continue
            if a.player_type != PlayerType.PITCHER:
                continue
            candidates.append(a)
    else:
        il_positions = _real_positions(il_player)
        # First pass: overlapping real positions
        for a in active_players:
            if a.name in already_displaced:
                continue
            if a.player_type != PlayerType.HITTER:
                continue
            if il_positions & _real_positions(a):
                candidates.append(a)
        # Fallback: any active hitter
        if not candidates:
            for a in active_players:
                if a.name in already_displaced:
                    continue
                if a.player_type != PlayerType.HITTER:
                    continue
                candidates.append(a)

    if not candidates:
        return None

    if league_context is not None and current_roster is not None:
        return _find_delta_roto_optimal(
            il_player,
            candidates,
            current_roster,
            league_context,
        )

    # Fallback: SGP-based (legacy behavior, used when no league context).
    # Hitters only: pitchers with league_context use the pair-swap model
    # (_compute_pitcher_pool_factors), which avoids the
    # elite-low-volume-closer pathology this path was previously vulnerable to.
    # Standings/breakdown call sites should pass league_context to enable
    # the delta-Roto-optimal picker.
    return min(candidates, key=lambda a: _player_sgp(a))


def _find_delta_roto_optimal(
    il_player: Player,
    candidates: list[Player],
    current_roster: list[Player | dict],
    ctx: LeagueContext,
) -> Player | None:
    """Pick the candidate whose displacement maximizes team roto pts.

    For each candidate, builds a hypothetical roster with that
    candidate scaled by the displacement factor, sums the team's
    projected stats (with ``displacement=False`` to avoid recursion;
    upstream displacement state is already baked into
    ``current_roster``), then scores via :func:`score_roto_dict`
    against the frozen baseline of other teams' stats.
    """
    il_pt = _playing_time(il_player)
    if il_pt <= 0:
        # IL player has no playing-time projection — fall back to SGP picker
        # so we at least pick *some* target deterministically.
        return min(candidates, key=lambda a: _player_sgp(a))

    best_target: Player | None = None
    best_pts = -float("inf")
    for cand in candidates:
        active_pt = _playing_time(cand)
        if active_pt <= 0:
            continue
        factor = max(0.0, active_pt - il_pt) / active_pt
        hyp_roster: list[Player | dict] = [
            _scale_stats(cand, factor) if isinstance(p, Player) and p.name == cand.name else p
            for p in current_roster
        ]
        team_stats = project_team_stats(hyp_roster, displacement=False)
        all_team_stats: dict[str, CategoryStats] = dict(ctx.baseline_other_team_stats)
        all_team_stats[ctx.team_name] = team_stats
        roto = score_roto(_dict_table(all_team_stats), team_sds=ctx.team_sds)
        pts = roto[ctx.team_name].total
        if pts > best_pts:
            best_pts = pts
            best_target = cand
    return best_target


def _player_sgp(p: Player) -> float:
    """Calculate total SGP for a player, returning 0 if no ROS stats."""
    if p.rest_of_season is None:
        return 0.0
    # Default SGP denominators (not league sgp_overrides): no league config
    # reaches the project_team_stats displacement layer (LeagueContext carries
    # standings/SDs only), and this value is used purely ordinally to pick the
    # min-SGP drop candidate in rare fallback paths.
    return calculate_player_sgp(p.rest_of_season)


def _compute_displacement_factors(
    active: list[Player],
    il_players: list[Player],
    *,
    league_context: LeagueContext | None = None,
) -> dict[str, float]:
    """Map player-name -> scale factor for IL-induced displacement.

    Hitters (always) and pitchers (when ``league_context`` is None) use
    the legacy substitution model: process IL players in descending
    playing-time order; each picks the displacement target via
    :func:`_find_worst_match` and scales them by
    ``max(0, active_pt - il_pt) / active_pt``.

    Pitchers WITH ``league_context`` use the pair-swap model: each IL
    pitcher (sorted by descending preseason IP) is activated at full ROS
    and one active target absorbs a rate-aware discount via
    ``pitcher_swap.swap_window_ip`` + ``pitcher_swap.discount_factor``.
    Same-stat invariant: the chosen target is one pitcher, scaled
    identically across every counting category. See
    :func:`_compute_pitcher_pool_factors` for the full algorithm.

    Each player appears in factors at most once. Players not in the
    returned dict contribute at full scale (sf=1.0 implicit).
    """
    factors: dict[str, float] = {}

    # Hitters: always legacy substitution (position constraints make a
    # pool-slot model more complex; out of scope for the current change).
    active_hitters = [p for p in active if p.player_type == PlayerType.HITTER]
    il_hitters = [p for p in il_players if p.player_type == PlayerType.HITTER]
    factors.update(
        _compute_substitution_factors(
            active_hitters,
            il_hitters,
            league_context=league_context,
            all_active=active,
            all_il=il_players,
        )
    )

    # Pitchers: pair-swap model with league_context, substitution otherwise.
    active_pitchers = [p for p in active if p.player_type == PlayerType.PITCHER]
    il_pitchers = [p for p in il_players if p.player_type == PlayerType.PITCHER]
    if league_context is not None:
        factors.update(
            _compute_pitcher_pool_factors(
                active_pitchers,
                il_pitchers,
                all_active=active,
                all_il=il_players,
                league_context=league_context,
            )
        )
    else:
        factors.update(
            _compute_substitution_factors(
                active_pitchers,
                il_pitchers,
                league_context=None,
                all_active=active,
                all_il=il_players,
            )
        )

    return factors


def _compute_substitution_factors(
    active_subset: list[Player],
    il_subset: list[Player],
    *,
    league_context: LeagueContext | None,
    all_active: list[Player],
    all_il: list[Player],
) -> dict[str, float]:
    """Legacy per-IL-player substitution displacement, restricted to a
    player-type subset. ``all_active``/``all_il`` (the full team) are
    used to build the ΔRoto picker's running-state roster so cross-type
    interactions are scored correctly.
    """
    il_sorted = sorted(il_subset, key=_playing_time, reverse=True)
    already_displaced: set[str] = set()
    factors: dict[str, float] = {}

    running_roster: list[Player | dict] = []
    if league_context is not None:
        running_roster = [*all_il, *all_active]

    for il_p in il_sorted:
        il_pt = _playing_time(il_p)
        if il_pt <= 0:
            continue
        target = _find_worst_match(
            il_p,
            active_subset,
            already_displaced,
            league_context=league_context,
            current_roster=running_roster if league_context is not None else None,
        )
        if target is None:
            continue
        active_pt = _playing_time(target)
        if active_pt <= 0:
            continue
        factor = max(0.0, active_pt - il_pt) / active_pt
        already_displaced.add(target.name)
        factors[target.name] = factor
        if league_context is not None:
            running_roster = [
                _scale_stats(p, factor) if isinstance(p, Player) and p.name == target.name else p
                for p in running_roster
            ]
    return factors


def _compute_pitcher_pool_factors(
    active_pitchers: list[Player],
    il_pitchers: list[Player],
    *,
    all_active: list[Player],
    all_il: list[Player],
    league_context: LeagueContext,
) -> dict[str, float]:
    """Pair-swap pitcher displacement: each IL pitcher who improves team
    DeltaRoto is activated at full ROS; one active target absorbs the discount.

    For each IL pitcher (processed in descending preseason IP -- biggest
    expected workload first), the picker evaluates every still-undiscounted
    active pitcher as the swap target, scoring the resulting team via
    :func:`project_team_stats` against the frozen baseline of other teams'
    stats. The pair maximizing team roto pts wins; the IL pitcher stays at
    implicit sf=1.0 and the target gets ``discount_factor(target.ros.ip,
    swap_window_ip(il, target))`` applied.

    The IL pitcher is NEVER the displaced one. If no positive-DeltaRoto swap
    exists (every candidate target would hurt the team), the IL pitcher is
    set to sf=0 -- the legacy zero-out for cases where the returning arm
    truly cannot find a home in the lineup.

    Same-stat invariant: the chosen target is one pitcher, scaled
    identically across every counting category. Per-stat target selection
    is explicitly prohibited (it would let strikeouts use one displacement
    target and WHIP use another, producing inconsistent reasoning).

    Cross-role swaps use ``pitcher_swap.swap_window_ip``, which prorates
    by preseason IP -- a 60 ROS-IP starter consumes ~30% of a reliever's
    preseason IP (not 60 IP, which would zero a reliever).

    No-op when there are no IL pitchers to evaluate. Skips pitchers
    without ROS projections.
    """
    il_candidates = [
        p for p in il_pitchers if p.rest_of_season is not None and _playing_time(p) > 0
    ]
    if not il_candidates:
        return {}

    active_pool = [
        p for p in active_pitchers if p.rest_of_season is not None and _playing_time(p) > 0
    ]
    if not active_pool:
        # No one to discount; can't realize the swap. Conservative: skip.
        return {}

    full_pool_roster: list[Player | dict] = [*all_il, *all_active]

    def team_pts(stats: CategoryStats) -> float:
        all_team_stats: dict[str, CategoryStats] = dict(league_context.baseline_other_team_stats)
        all_team_stats[league_context.team_name] = stats
        return score_roto(_dict_table(all_team_stats), team_sds=league_context.team_sds)[
            league_context.team_name
        ].total

    def state_with(overrides: dict[str, float]) -> list[Player | dict]:
        """Roster with every name in `overrides` replaced by a `_scale_stats`
        dict at the given factor; all other players pass through."""
        out: list[Player | dict] = []
        for p in full_pool_roster:
            if isinstance(p, Player) and p.name in overrides:
                out.append(_scale_stats(p, overrides[p.name]))
            else:
                out.append(p)
        return out

    factors: dict[str, float] = {}
    already_discounted: set[str] = set()

    # Larger preseason workloads first -- same volume-priority intent as the
    # legacy substitution model, but for the pair-swap selector.
    def _preseason_or_ros(p: Player) -> float:
        if p.preseason is not None:
            return _safe(getattr(p.preseason, "ip", 0))
        return _playing_time(p)

    il_sorted = sorted(il_candidates, key=_preseason_or_ros, reverse=True)

    for il_p in il_sorted:
        # "No swap" baseline for this IL pitcher: bench them (sf=0) and keep
        # all currently-committed discounts. This is the cost of NOT activating
        # the IL pitcher -- compare every candidate target against this.
        bench_il_overrides = {**factors, il_p.name: 0.0}
        bench_il_stats = project_team_stats(
            state_with(bench_il_overrides),
            displacement=False,
        )
        bench_il_pts = team_pts(bench_il_stats)

        # Find the (target, factor) pair that maximizes team pts when the IL
        # pitcher is activated at full ROS and the target absorbs the swap.
        best_target: Player | None = None
        best_factor: float = 1.0
        best_pts: float = -float("inf")
        for target in active_pool:
            if target.name in already_discounted:
                continue
            window = swap_window_ip(
                il_p, target, fraction_remaining=league_context.fraction_remaining
            )
            tgt_ros_ip = _safe(getattr(target.rest_of_season, "ip", 0))
            f = discount_factor(tgt_ros_ip, window)
            # Swap state: IL pitcher at full ROS (not in overrides = sf=1.0),
            # target at discount_factor, all previously committed discounts applied.
            overrides = {**factors, target.name: f}
            stats = project_team_stats(
                state_with(overrides),
                displacement=False,
            )
            pts = team_pts(stats)
            if pts > best_pts:
                best_pts = pts
                best_target = target
                best_factor = f

        if best_target is None or best_pts <= bench_il_pts:
            # No target makes the swap worth it relative to benching the IL
            # pitcher. Bench the IL pitcher (legacy zero-out for "this returning
            # arm wouldn't actually displace anyone").
            factors[il_p.name] = 0.0
            continue

        factors[best_target.name] = best_factor
        already_discounted.add(best_target.name)

    return factors


def _scale_stats(
    p: Player,
    factor: float,
) -> dict[str, float | PlayerType]:
    """Return per-player scaled counting stats: ``ROS * factor``.

    The YTD floor added by PR #108 is removed -- per-player YTD attribution
    incorrectly credits whoever currently owns the player with stats accrued
    before the acquisition (e.g., a mid-season pickup's pre-pickup K's would
    bleed into the new owner's projected total).

    Team YTD is now captured at the team level via :func:`team_end_of_season`
    and :class:`TeamYtdComponents` from Yahoo standings -- i.e., only stats
    accrued while the player was on the team count.

    ``factor=0.0`` zeros the contribution entirely; ``factor=1.0`` returns
    the full ROS projection.
    """
    result: dict[str, float | PlayerType] = {}
    if p.rest_of_season is None:
        return result
    keys = HITTING_COUNTING if p.player_type == PlayerType.HITTER else PITCHING_COUNTING
    for key in keys:
        ros_val = _safe(getattr(p.rest_of_season, key, 0))
        result[key] = ros_val * factor
    result["player_type"] = p.player_type
    return result


# ── Breakdown types (per-team, per-player contribution view) ────────


class ContributionStatus(StrEnum):
    """Why a player contributes at the level they do.

    Derived from the same classification ``_apply_displacement`` uses;
    exposed on per-player breakdowns for UI tooling.
    """

    ACTIVE = "active"
    IL_FULL = "il_full"
    DISPLACED = "displaced"
    BENCH = "bench"
    NO_PROJECTION = "no_projection"


@dataclass(frozen=True)
class PlayerContribution:
    """One player's contribution to a team's projected totals."""

    name: str
    player_type: PlayerType
    status: ContributionStatus
    scale_factor: float  # 0.0 to 1.0
    raw_stats: dict[str, float]  # pre-scale projection stats (for display)
    contribution_stats: dict[str, float] = field(default_factory=dict)
    """Actual ROS-scaled contribution per counting stat -- what flows into
    the team total: ``ROS * scale_factor``. The modal renders these values
    directly so the per-row sum matches the ROS portion of the standings
    widget. Team YTD lives at the team level via :func:`team_end_of_season`,
    not on these per-player rows.
    """

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "player_type": self.player_type.value,
            "status": self.status.value,
            "scale_factor": self.scale_factor,
            "raw_stats": dict(self.raw_stats),
            "contribution_stats": dict(self.contribution_stats),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlayerContribution:
        # `or {}` guards against explicit null values in persisted JSON.
        # d.get(k, {}) returns the stored None when the key exists but is
        # null, so the subsequent .items() would raise AttributeError.
        raw_stats = {k: float(v) for k, v in (d.get("raw_stats") or {}).items()}
        # No back-compat fabrication: contribution_stats is taken exactly as
        # stored. raw_stats is FULL-SEASON (``_raw_stats_for`` reads
        # ``full_season_projection``) while contribution_stats is ``ROS *
        # factor``, so the old ``raw_stats * scale_factor`` fallback rendered
        # ``full_season * factor`` -- the pre-#110 YTD double-count (team YTD
        # is added separately at the team level). Every refresh writes
        # contribution_stats, so an absent value now means a corrupt/stale
        # payload; per the repo's "no plausible-wrong answer" rule we leave it
        # empty (renders as visible zero) rather than fabricate wrong math.
        scale_factor = float(d["scale_factor"])
        contribution_stats = {k: float(v) for k, v in (d.get("contribution_stats") or {}).items()}
        return cls(
            name=d["name"],
            player_type=PlayerType(d["player_type"]),
            status=ContributionStatus(d["status"]),
            scale_factor=scale_factor,
            raw_stats=raw_stats,
            contribution_stats=contribution_stats,
        )


@dataclass(frozen=True)
class RosterBreakdown:
    """Per-player contributions for one team, partitioned by player type.

    ``team_ytd`` is the team's YTD totals block (uppercase-string-keyed,
    matching :class:`CategoryStats`'s ``to_dict`` for counting stats plus
    rate-stat ingredients H, AB, IP, ER, BB_plus_H_allowed). It is a
    first-class field so it survives the
    ``RosterBreakdown.from_dict(...).to_dict()`` round-trip that
    ``season_routes`` does to backfill stale KV blobs. Defaults to an
    empty dict for callers (and legacy persisted payloads) that don't
    carry team-level YTD.
    """

    team_name: str
    hitters: list[PlayerContribution]
    pitchers: list[PlayerContribution]
    team_ytd: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "hitters": [c.to_dict() for c in self.hitters],
            "pitchers": [c.to_dict() for c in self.pitchers],
            "team_ytd": dict(self.team_ytd),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RosterBreakdown:
        # `or {}` guards against explicit null values in persisted JSON --
        # d.get("team_ytd", {}) would return None when the key exists but
        # is null, breaking dict() construction below.
        return cls(
            team_name=d["team_name"],
            hitters=[PlayerContribution.from_dict(x) for x in d.get("hitters", [])],
            pitchers=[PlayerContribution.from_dict(x) for x in d.get("pitchers", [])],
            team_ytd=dict(d.get("team_ytd") or {}),
        )


def _apply_displacement(
    roster: list[Player],
    *,
    league_context: LeagueContext | None = None,
) -> list[Player | dict]:
    """Partition roster into active/bench/IL and apply displacement scaling.

    Classification is slot-first:

    - Any slot that is neither ``BN`` nor in ``IL_SLOTS`` → active.
      Counted at face value; may be a displacement target. Yahoo IL
      status on an active-slotted player is ignored — the manager's
      slot choice wins.
    - Slot in ``IL_SLOTS`` (IL, IL+, DL, DL+) → IL. Counted at full
      ROS and displaces an active player (per ``_find_worst_match``).
    - BN slot + IL status → IL (same displacement path).
    - BN slot + healthy → excluded.

    When ``league_context`` is provided, pitcher displacement uses the
    pair-swap model (each IL pitcher activates at full ROS; one active
    target absorbs a rate-aware discount; see
    :func:`_compute_pitcher_pool_factors`) and hitter displacement uses
    DeltaRoto-optimal substitution. Otherwise both use the legacy SGP
    substitution model.

    Note: the pair-swap model only acts when there are IL pitchers to
    activate. When the active pool exceeds the slot count with NO IL
    pitchers present, no displacement is applied -- this is a known gap
    relative to the legacy zero-out behavior and is acceptable for
    typical rosters (9 P slots, rarely >9 active without IL).

    Returns a list where each entry is either an unmodified Player
    (active, unaffected; or IL, full-scale) or a dict of scaled stats
    (any player with a sub-1.0 factor). Non-Player entries in the input
    (dicts from draft scripts) pass through untouched.
    """
    # Non-Player entries pass through untouched (draft scripts use dicts).
    pass_through: list[Player | dict] = [p for p in roster if not isinstance(p, Player)]
    typed = [p for p in roster if isinstance(p, Player)]

    active, il_players, _bench = _classify_roster(typed)
    displacement_factors = _compute_displacement_factors(
        active,
        il_players,
        league_context=league_context,
    )

    # Build output: each player in active or IL is either passed through
    # at full scale or scaled per the factors dict. Pool model can put an
    # IL pitcher in factors with sf=0; substitution model only ever puts
    # active players in factors. Bench players are excluded entirely.
    result: list[Player | dict] = list(pass_through)
    for p in [*il_players, *active]:
        if p.name in displacement_factors:
            result.append(_scale_stats(p, displacement_factors[p.name]))
        else:
            result.append(p)

    return result


def _contribution_stats_for(p: Player, factor: float) -> dict[str, float]:
    """Per-player counting-stat contribution to the team total at this factor.

    Mirrors the math in :func:`_scale_stats` (which produces the dicts
    summed by :func:`project_team_stats`), but returns only the counting
    stats (no ``player_type`` key). Used by :func:`compute_roster_breakdown`
    so the modal can render the actual contribution directly instead of
    multiplying raw_stats * scale_factor client-side -- the latter would
    lose any factor-aware scaling.

    Returns ``ROS * factor`` per stat. Team YTD is now captured at the
    team level via :func:`team_end_of_season` (not per-player), so this
    helper is ROS-only regardless of caller mode.

    Returns ``{}`` when the player has no ROS projection.
    """
    if p.rest_of_season is None:
        return {}
    scaled = _scale_stats(p, factor)
    # _scale_stats includes a "player_type" key for routing; strip it.
    return {k: float(v) for k, v in scaled.items() if k != "player_type"}


def _raw_stats_for(p: Player) -> dict[str, float]:
    """Extract the per-player raw stats for the standings breakdown drilldown.

    Reads ``full_season_projection`` (so per-player rows in the
    breakdown modal sum to the end-of-season totals shown in the
    standings widget), falling back to ``rest_of_season`` when the
    full-season field is unset (preseason rosters store preseason
    values there). Returns ``{}`` when neither projection is set.
    """
    stats = p.full_season_projection or p.rest_of_season
    if stats is None:
        return {}
    keys = PITCHING_COUNTING if p.player_type == PlayerType.PITCHER else HITTING_COUNTING
    return {k: _safe(getattr(stats, k, 0)) for k in keys}


def compute_roster_breakdown(
    team_name: str,
    roster: list[Player],
    *,
    league_context: LeagueContext | None = None,
    team_ytd: dict[str, float] | None = None,
) -> RosterBreakdown:
    """Return per-player contributions for ``roster``, tagged with status.

    Uses the same slot-first classification as :func:`_apply_displacement`,
    and the same displacement-factor math. The aggregate over
    ``contribution_stats[cat]`` per category equals
    :func:`project_team_stats` with ``displacement=True``.

    ``raw_stats`` is the unscaled projection (``full_season_projection``
    when available, ROS otherwise) -- shown in the modal for context.
    ``contribution_stats`` is the actual ROS-scaled contribution per
    category -- what flows into the team total. The modal must render
    ``contribution_stats[cat]`` directly to match the standings widget;
    multiplying ``raw_stats * scale_factor`` would not match because
    ``raw_stats`` may be full-season while contributions are ROS-only.

    Per-player YTD is intentionally NOT included in ``contribution_stats``
    -- team YTD lives at the team level via :func:`team_end_of_season`
    (from :class:`TeamYtdComponents`). Callers (the refresh pipeline) can
    pass that block via ``team_ytd`` so it becomes a first-class field on
    the returned :class:`RosterBreakdown` and survives the
    season_routes ``from_dict().to_dict()`` round-trip. ``None`` defaults
    to an empty dict (preseason / no actual standings).

    When ``league_context`` is provided, displacement targets are chosen
    via DeltaRoto-optimal selection (matches the standings layer); without
    it, the legacy SGP picker is used.
    """
    active, il_players, bench = _classify_roster(roster)
    displacement_factors = _compute_displacement_factors(
        active,
        il_players,
        league_context=league_context,
    )

    contributions: list[PlayerContribution] = []

    for p in active:
        if p.rest_of_season is None:
            status = ContributionStatus.NO_PROJECTION
            factor = 0.0
        elif p.name in displacement_factors:
            status = ContributionStatus.DISPLACED
            factor = displacement_factors[p.name]
        else:
            status = ContributionStatus.ACTIVE
            factor = 1.0
        contributions.append(
            PlayerContribution(
                name=p.name,
                player_type=p.player_type,
                status=status,
                scale_factor=factor,
                raw_stats=_raw_stats_for(p),
                contribution_stats=_contribution_stats_for(p, factor),
            )
        )

    for p in il_players:
        if p.rest_of_season is None:
            status = ContributionStatus.NO_PROJECTION
            factor = 0.0
        elif p.name in displacement_factors:
            # Pool model can put an IL pitcher in the bench tier (sf=0)
            # when the team's other pitchers are projected to outproduce
            # the returning IL guy. Tag as DISPLACED to surface this in
            # the breakdown UI rather than IL_FULL (which would
            # mis-imply they're contributing).
            status = ContributionStatus.DISPLACED
            factor = displacement_factors[p.name]
        else:
            status = ContributionStatus.IL_FULL
            factor = 1.0
        contributions.append(
            PlayerContribution(
                name=p.name,
                player_type=p.player_type,
                status=status,
                scale_factor=factor,
                raw_stats=_raw_stats_for(p),
                contribution_stats=_contribution_stats_for(p, factor),
            )
        )

    for p in bench:
        contributions.append(
            PlayerContribution(
                name=p.name,
                player_type=p.player_type,
                status=ContributionStatus.BENCH,
                scale_factor=0.0,
                raw_stats=_raw_stats_for(p),
                contribution_stats={},  # bench players are excluded from team totals
            )
        )

    hitters = [c for c in contributions if c.player_type == PlayerType.HITTER]
    pitchers = [c for c in contributions if c.player_type == PlayerType.PITCHER]
    return RosterBreakdown(
        team_name=team_name,
        hitters=hitters,
        pitchers=pitchers,
        team_ytd=dict(team_ytd) if team_ytd else {},
    )


@dataclass(frozen=True)
class TeamRosComponents:
    """Rest-of-season counting + rate-stat ingredients for a roster,
    matching the :class:`TeamYtdComponents` shape so the two can be summed
    by ``team_end_of_season`` (added in a follow-up task).

    Same field set as ``TeamYtdComponents``: counting stats (R, HR, RBI,
    SB, W, K, SV) plus rate-stat ingredients (H, AB, IP, ER, BB+H_allowed).
    """

    r: float = 0.0
    hr: float = 0.0
    rbi: float = 0.0
    sb: float = 0.0
    w: float = 0.0
    k: float = 0.0
    sv: float = 0.0
    h: float = 0.0
    ab: float = 0.0
    ip: float = 0.0
    er: float = 0.0
    bb_plus_h_allowed: float = 0.0


def project_ros_components(
    roster: list[Player],
    *,
    displacement: bool = True,
    league_context: LeagueContext | None = None,
) -> TeamRosComponents:
    """Sum rest-of-season counting + rate-stat ingredients across a roster.

    Mirrors :func:`project_team_stats` but returns a :class:`TeamRosComponents`
    instead of :class:`CategoryStats`. Designed to be combined with
    :class:`TeamYtdComponents` from Yahoo standings to produce projected
    end-of-season totals via a ``team_end_of_season`` helper (added in a
    follow-up task) that recomputes AVG / ERA / WHIP from the summed
    ingredients rather than averaging the precomputed rates.

    Displacement scaling (active pool overlap, IL pitcher pair-swap) is
    applied via the standard :func:`_apply_displacement` path when
    ``displacement=True``. Each displaced player's components are scaled
    by their displacement factor.
    """
    players: list[Player | dict]
    if displacement:
        players = _apply_displacement(
            roster,
            league_context=league_context,
        )
    else:
        players = list(roster)

    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in players:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            r += _stat(p, "r", "rest_of_season")
            hr += _stat(p, "hr", "rest_of_season")
            rbi += _stat(p, "rbi", "rest_of_season")
            sb += _stat(p, "sb", "rest_of_season")
            h_total += _stat(p, "h", "rest_of_season")
            ab_total += _stat(p, "ab", "rest_of_season")
        elif ptype == PlayerType.PITCHER:
            w += _stat(p, "w", "rest_of_season")
            k += _stat(p, "k", "rest_of_season")
            sv += _stat(p, "sv", "rest_of_season")
            ip_total += _stat(p, "ip", "rest_of_season")
            er_total += _stat(p, "er", "rest_of_season")
            bb_total += _stat(p, "bb", "rest_of_season")
            ha_total += _stat(p, "h_allowed", "rest_of_season")

    return TeamRosComponents(
        r=r,
        hr=hr,
        rbi=rbi,
        sb=sb,
        w=w,
        k=k,
        sv=sv,
        h=h_total,
        ab=ab_total,
        ip=ip_total,
        er=er_total,
        bb_plus_h_allowed=bb_total + ha_total,
    )


def team_end_of_season(
    ytd: TeamYtdComponents,
    ros: TeamRosComponents,
) -> CategoryStats:
    """Combine team YTD components and ROS components into end-of-season totals.

    Counting stats: simple sum (YTD + ROS).
    Rate stats: recomputed from summed components --
      - AVG = (YTD.h + ROS.h) / (YTD.ab + ROS.ab)
      - ERA = 9 * (YTD.er + ROS.er) / (YTD.ip + ROS.ip)
      - WHIP = (YTD.bb_plus_h_allowed + ROS.bb_plus_h_allowed) / (YTD.ip + ROS.ip)

    Zero-AB defaults AVG to 0.0 (non-inverse: 0.0 = batting nothing = worst).
    Zero-IP defaults ERA/WHIP to 99.0 -- matching :class:`CategoryStats`'s
    inverse-stat defaults and :func:`utils.rate_stats.calculate_era` /
    :func:`calculate_whip`. Returning 0.0 here would silently make a zero-IP
    team WIN ERA/WHIP (lower is better), and Pass-1 (:func:`project_team_stats`,
    which uses ``calculate_era``/``calculate_whip``) would disagree with
    Pass-2 (this function) by ~99 on the same zero input.

    See :class:`TeamYtdComponents` for the YTD side (from
    :meth:`StandingsEntry.ytd_components`) and :func:`project_ros_components`
    for the ROS side.
    """
    total_ab = ytd.ab + ros.ab
    total_ip = ytd.ip + ros.ip
    avg = (ytd.h + ros.h) / total_ab if total_ab > 0 else 0.0
    era = 9.0 * (ytd.er + ros.er) / total_ip if total_ip > 0 else 99.0
    whip = (ytd.bb_plus_h_allowed + ros.bb_plus_h_allowed) / total_ip if total_ip > 0 else 99.0
    return CategoryStats(
        r=ytd.r + ros.r,
        hr=ytd.hr + ros.hr,
        rbi=ytd.rbi + ros.rbi,
        sb=ytd.sb + ros.sb,
        avg=avg,
        w=ytd.w + ros.w,
        k=ytd.k + ros.k,
        sv=ytd.sv + ros.sv,
        era=era,
        whip=whip,
    )


def project_team_stats(
    roster,
    *,
    displacement: bool = False,
    league_context: LeagueContext | None = None,
) -> CategoryStats:
    """Sum projected ROS counting stats for a roster into a CategoryStats.

    Accepts Player dataclass objects OR plain dicts with flat stat
    keys. Rate stats (AVG, ERA, WHIP) are computed from component
    totals rather than simple sums, so the result is mathematically
    correct rather than just a naive average.

    Sums ``Player.rest_of_season`` only -- forward-looking math used by
    the optimizer, recs, and trade evaluation. A hot-YTD player and a
    cold-YTD player with the same ROS-remaining contribute identically,
    so start/sit decisions are not biased by locked YTD totals. Team
    YTD is captured at the team level via :func:`team_end_of_season`
    (combining :class:`TeamYtdComponents` from Yahoo standings with
    :func:`project_ros_components`), not via a per-player projection
    field. The standings layer is responsible for adding team YTD to
    these ROS totals when end-of-season projections are needed.

    When ``displacement=True``, bench players are excluded and IL
    players displace the worst positional match among active players,
    scaling down the displaced player's stats proportionally based on
    playing time. Only activates for Player dataclass objects -- dict
    input callers are unaffected. Displaced players contribute
    ``ROS * factor``.

    The dict-input path exists for backwards compatibility with
    draft-side scripts (``scripts/simulate_draft.py``,
    ``scripts/summary.py``) that build rosters as plain dicts. Those
    scripts are explicitly out of scope for the League data model
    refactor and would need significant rework to use Player objects.
    Step 9 cleanup can revisit.
    """
    if displacement:
        roster = _apply_displacement(
            roster,
            league_context=league_context,
        )

    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            r += _stat(p, "r", "rest_of_season")
            hr += _stat(p, "hr", "rest_of_season")
            rbi += _stat(p, "rbi", "rest_of_season")
            sb += _stat(p, "sb", "rest_of_season")
            h_total += _stat(p, "h", "rest_of_season")
            ab_total += _stat(p, "ab", "rest_of_season")
        elif ptype == PlayerType.PITCHER:
            w += _stat(p, "w", "rest_of_season")
            k += _stat(p, "k", "rest_of_season")
            sv += _stat(p, "sv", "rest_of_season")
            ip_total += _stat(p, "ip", "rest_of_season")
            er_total += _stat(p, "er", "rest_of_season")
            bb_total += _stat(p, "bb", "rest_of_season")
            ha_total += _stat(p, "h_allowed", "rest_of_season")

    return CategoryStats(
        r=r,
        hr=hr,
        rbi=rbi,
        sb=sb,
        avg=calculate_avg(h_total, ab_total),
        w=w,
        k=k,
        sv=sv,
        era=calculate_era(er_total, ip_total),
        whip=calculate_whip(bb_total, ha_total, ip_total),
    )


def _full_season_volume(p, is_hitter: bool) -> float:
    """Full-season projected playing time for the playing-time curve lookup.

    Reads PA (hitters) / IP (pitchers) from the full-season projection so the
    cv_pt band matches the calibration and the MC. Falls back to ROS at
    preseason (``_stat`` handles that) and to AB/0.90 when a dict caller has
    ``ab`` but not ``pa``.
    """
    if is_hitter:
        pa = _stat(p, "pa", "full_season_projection")
        if pa > 0:
            return float(pa)
        return float(_stat(p, "ab", "full_season_projection")) / AB_PER_PA
    return float(_stat(p, "ip", "full_season_projection"))


def player_category_variance(player) -> dict[Category | str, float]:
    """Per-player variance contribution for each category.

    Returns a dict keyed by :class:`Category` for counting categories and by
    raw string keys for the rate-assembly components that ``project_team_sds``
    needs to fold into team-level rate SDs.

    **Counting categories** (R, HR, RBI, SB for hitters; W, K, SV for pitchers):
    Combines performance (NegBin dispersion via ``negbin_perf_variance``) and
    playing-time (``cv_pt``) variance for a single player::

        var_cat = negbin_perf_variance(stat, mu) + mu^2 * cv_pt^2

    where ``mu`` is the projected stat. The performance term is the same
    ``mu + mu^2/r`` quantity the MC uses (single source of truth for per-stat
    dispersion); it is conditional on realized playing time, so the PT variance
    is added separately.

    **Rate categories** (AVG, ERA, WHIP) are NOT representable as simple
    per-player variances because the rate denominator (total AB / total IP) is a
    team-level quantity.  Instead, this function returns the per-component
    PERFORMANCE variance (NegBin, playing-time-invariant) so the caller can
    assemble the team-level rate SD:

    Hitters expose:
      - ``"h_var"`` (float): NegBin performance variance of projected hits
      - ``"ab"``    (float): projected at-bats

    Pitchers expose:
      - ``"er_var"`` (float): NegBin performance variance of projected ER
      - ``"bb_var"`` (float): NegBin performance variance of projected BB
      - ``"ha_var"`` (float): NegBin performance variance of projected hits allowed
      - ``"ip"``     (float): projected innings pitched

    Playing time does NOT contribute to rate-component sums (missed time cancels
    in numerator/denominator; see ``project_team_sds`` docstring).

    Unknown player types return an empty dict.
    """
    ptype = _get(player, "player_type")
    result: dict[Category | str, float] = {}

    if ptype == PlayerType.HITTER:
        cv_pt_sq = playing_time_params(PlayerType.HITTER, _full_season_volume(player, True))[1] ** 2
        for stat_key, cat in [
            ("r", Category.R),
            ("hr", Category.HR),
            ("rbi", Category.RBI),
            ("sb", Category.SB),
        ]:
            v = _stat(player, stat_key)
            result[cat] = float(negbin_perf_variance(stat_key, v)) + v * v * cv_pt_sq
        # Rate-assembly: per-component PERFORMANCE variance (cv_pt cancels in a rate).
        result["h_var"] = float(negbin_perf_variance("h", _stat(player, "h")))
        result["ab"] = _stat(player, "ab")

    elif ptype == PlayerType.PITCHER:
        cv_pt_sq = (
            playing_time_params(PlayerType.PITCHER, _full_season_volume(player, False))[1] ** 2
        )
        for stat_key, cat in [
            ("w", Category.W),
            ("k", Category.K),
            ("sv", Category.SV),
        ]:
            v = _stat(player, stat_key)
            result[cat] = float(negbin_perf_variance(stat_key, v)) + v * v * cv_pt_sq
        result["er_var"] = float(negbin_perf_variance("er", _stat(player, "er")))
        result["bb_var"] = float(negbin_perf_variance("bb", _stat(player, "bb")))
        result["ha_var"] = float(negbin_perf_variance("h_allowed", _stat(player, "h_allowed")))
        result["ip"] = _stat(player, "ip")

    return result


def project_team_sds(
    roster,
    *,
    displacement: bool = True,
) -> dict[Category, float]:
    """Aggregate per-player projection variance into team-level SDs.

    Combines two independent per-player variance sources for the counting
    categories: performance (NegBin dispersion ``mu + mu^2/r`` via
    ``negbin_perf_variance`` -- the single source of truth shared with the MC)
    and playing time (``cv_pt`` from the calibrated playing-time model,
    ``utils.playing_time``):

        SD_cat_team = sqrt(sum_i [negbin_perf_variance(stat, mu_i) + mu_i^2 * cv_pt_i^2])

    Under a player-independence assumption. Playing-time variance is NOT added
    to the rate categories (AVG/ERA/WHIP): a player's missed time scales his
    numerator and denominator together, so it cancels out of a rate. Those keep
    the performance-only propagation, with the per-component NegBin variance
    summed across players:

        SD_AVG  = sqrt(sum negbin_perf_variance(h, h_i)) / sum_AB
        SD_ERA  = 9 * sqrt(sum negbin_perf_variance(er, er_i)) / sum_IP
        SD_WHIP = sqrt(sum negbin_perf_variance(bb, bb_i)
                       + sum negbin_perf_variance(h_allowed, ha_i)) / sum_IP

    ``displacement`` matches :func:`project_team_stats` — bench excluded,
    IL players displace their worst active positional match.

    Returns ``{Category: sd}`` keyed by :class:`Category` enum for every
    category in ``ALL_CATS``. Use :func:`team_sds_to_json` at the cache
    boundary for JSON serialization. Empty roster returns zeros.
    """
    if displacement:
        roster = _apply_displacement(roster)

    # Counting-category variance sums (from player_category_variance).
    h_var: dict[Category, float] = {
        c: 0.0 for c in (Category.R, Category.HR, Category.RBI, Category.SB)
    }
    p_var: dict[Category, float] = {c: 0.0 for c in (Category.W, Category.K, Category.SV)}
    # Rate-assembly PERFORMANCE-variance sums (playing-time-invariant).
    h_sum_var = 0.0
    p_sum_var: dict[str, float] = {k: 0.0 for k in ("er", "bb", "h_allowed")}
    total_ab = 0.0
    total_ip = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            contrib = player_category_variance(p)
            for cat in (Category.R, Category.HR, Category.RBI, Category.SB):
                h_var[cat] += contrib.get(cat, 0.0)
            # Rate components
            h_sum_var += contrib.get("h_var", 0.0)
            total_ab += contrib.get("ab", 0.0)
        elif ptype == PlayerType.PITCHER:
            contrib = player_category_variance(p)
            for cat in (Category.W, Category.K, Category.SV):
                p_var[cat] += contrib.get(cat, 0.0)
            # Rate components
            p_sum_var["er"] += contrib.get("er_var", 0.0)
            p_sum_var["bb"] += contrib.get("bb_var", 0.0)
            p_sum_var["h_allowed"] += contrib.get("ha_var", 0.0)
            total_ip += contrib.get("ip", 0.0)

    sds: dict[Category, float] = dict.fromkeys(ALL_CATS, 0.0)
    for cat in (Category.R, Category.HR, Category.RBI, Category.SB):
        sds[cat] = sqrt(h_var[cat])
    for cat in (Category.W, Category.K, Category.SV):
        sds[cat] = sqrt(p_var[cat])
    if total_ab > 0:
        sds[Category.AVG] = sqrt(h_sum_var) / total_ab
    if total_ip > 0:
        sds[Category.ERA] = 9.0 * sqrt(p_sum_var["er"]) / total_ip
        whip_var = p_sum_var["bb"] + p_sum_var["h_allowed"]
        sds[Category.WHIP] = sqrt(whip_var) / total_ip
    return sds


def build_team_sds(
    team_rosters: dict[str, list],
    sd_scale: float,
) -> dict[str, dict[Category, float]]:
    """Build the typed ``team_sds`` table for a set of team rosters.

    Each team's per-category SDs from :func:`project_team_sds` are
    scaled by ``sd_scale`` — typically ``sqrt(fraction_remaining)`` so
    variance damps as the season progresses and less of the roto total
    is still up for grabs.

    Returns ``{team_name: {Category: sd}}``. Use
    :func:`team_sds_to_json` / :func:`team_sds_from_json` at the cache
    I/O boundary; in-memory consumers index by :class:`Category` enum.
    """
    return {
        tname: {
            cat: sd * sd_scale for cat, sd in project_team_sds(roster, displacement=True).items()
        }
        for tname, roster in team_rosters.items()
    }


def team_sds_to_json(
    team_sds: Mapping[str, Mapping[Category, float]],
) -> dict[str, dict[str, float]]:
    """Serialize a typed ``team_sds`` table for JSON-backed caches.

    Inverse of :func:`team_sds_from_json`. Inner keys become the
    uppercase string form of :class:`Category` (``"R"``, ``"HR"``, …).
    """
    return {
        team: {cat.value: float(sd) for cat, sd in sds.items()} for team, sds in team_sds.items()
    }


def team_sds_from_json(
    raw: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[Category, float]]:
    """Deserialize a JSON-shaped ``team_sds`` payload into the typed form.

    Inverse of :func:`team_sds_to_json`. Missing category keys default
    to 0.0; unknown keys are ignored.
    """
    return {
        team: {cat: float(sds.get(cat.value, 0.0)) for cat in ALL_CATS} for team, sds in raw.items()
    }


def score_roto(
    standings: TeamStatsTable,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict[str, CategoryPoints]:
    """Assign expected-value roto points per team per category.

    For each team in each category, points equal

        pts = 1 + Σ_{j≠me} P(me > j)

    where ``P(A > B) = Phi((mu_A - mu_B) / sqrt(sd_A^2 + sd_B^2))`` under
    Gaussian independence of team totals (Phi is the standard-normal CDF).
    When ``team_sds`` is ``None`` or every sd is zero, this reduces to the
    step function that recovers the standard rank-based scoring,
    including the averaged-ranks convention on exact ties.

    Args:
        standings: any :class:`TeamStatsTable` — concretely
            :class:`Standings` or :class:`ProjectedStandings`. Each
            entry supplies ``team_name`` and a :class:`CategoryStats`
            under ``.stats``.
        team_sds: optional ``{team: {Category: sd}}``. ``None`` disables
            uncertainty (exact-rank behavior).

    Returns:
        ``{team_name: CategoryPoints}``. ``CategoryPoints.values`` is
        the per-category map (keyed by :class:`Category` enum); ``total``
        is the sum across all categories.
    """
    teams = [e.team_name for e in standings.entries]
    stats_by_team: dict[str, CategoryStats] = {e.team_name: e.stats for e in standings.entries}

    per_team_cat: dict[str, dict[Category, float]] = {t: {} for t in teams}

    for cat in ALL_CATS:
        higher_is_better = cat not in INVERSE_CATS
        for me in teams:
            mu_me = stats_by_team[me][cat]
            sd_me = team_sds.get(me, {}).get(cat, 0.0) if team_sds else 0.0
            pts = 1.0
            for other in teams:
                if other == me:
                    continue
                mu_o = stats_by_team[other][cat]
                sd_o = team_sds.get(other, {}).get(cat, 0.0) if team_sds else 0.0
                pts += _prob_beats(
                    mu_me,
                    mu_o,
                    sd_me,
                    sd_o,
                    higher_is_better=higher_is_better,
                )
            per_team_cat[me][cat] = pts

    return {
        t: CategoryPoints(
            values=per_team_cat[t],
            total=sum(per_team_cat[t].values()),
        )
        for t in teams
    }


@dataclass
class _AdHocStatsTable:
    """Private adapter for callers that still hold a ``{team: stats}`` dict.

    Implements the :class:`TeamStatsTable` protocol so those sites can
    call ``score_roto`` without building a full ``ProjectedStandings``.
    Phase 3.2+ migrates them to typed standings; until then, this keeps
    the legacy path minimal and localized.
    """

    entries: Sequence[TeamStatsRow]


def _dict_table(
    stats_by_team: Mapping[str, Mapping[str, float] | CategoryStats],
) -> _AdHocStatsTable:
    """Wrap a ``{team: stats}`` dict in a :class:`TeamStatsTable`.

    Accepts either ``CategoryStats`` values or uppercase-string-keyed
    dicts. Used internally by :func:`score_roto_dict` and by legacy
    callers that still operate on dict-shaped stats.
    """
    entries: list[TeamStatsRow] = []
    for name, stats in stats_by_team.items():
        cs = stats if isinstance(stats, CategoryStats) else CategoryStats.from_dict(stats)
        entries.append(ProjectedStandingsEntry(team_name=name, stats=cs))
    return _AdHocStatsTable(entries=entries)


def score_roto_dict(
    all_team_stats: Mapping[str, Mapping[str, float] | CategoryStats],
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Dict-shaped wrapper around :func:`score_roto`.

    Takes stats as ``{team: {stat_str: value}}`` (used by
    swap-simulation sites that mutate one team's row in-place) and
    returns ``{team: {"R_pts": ..., "HR_pts": ..., ..., "total": ...}}``
    for :mod:`delta_roto` and friends that still compare via
    string-keyed per-category deltas. ``team_sds`` is the typed
    :class:`Category`-keyed form — no string-keyed fallback.
    """
    roto = score_roto(_dict_table(all_team_stats), team_sds=team_sds)
    return {
        team: {
            **{f"{cat.value}_pts": cp.values[cat] for cat in ALL_CATS},
            "total": cp.total,
        }
        for team, cp in roto.items()
    }
