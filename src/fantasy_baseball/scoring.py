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
from dataclasses import dataclass
from enum import StrEnum
from math import erf, sqrt
from typing import Literal, Protocol

from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import (
    CategoryPoints,
    CategoryStats,
    ProjectedStandingsEntry,
)
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.constants import (
    HITTING_COUNTING,
    PITCHING_COUNTING,
    STARTER_IP_THRESHOLD,
    STAT_VARIANCE,
    Category,
)
from fantasy_baseball.utils.constants import (
    INVERSE_STATS as INVERSE_CATS,
)
from fantasy_baseball.utils.constants import (
    safe_float as _safe,
)
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
    # Flat dict (legacy callers, tests, draft scripts)
    if isinstance(p, dict):
        return _safe(p.get(key, 0))
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
    return "SP" if ip > STARTER_IP_THRESHOLD else "RP"


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
    """

    baseline_other_team_stats: Mapping[str, CategoryStats]
    team_sds: Mapping[str, Mapping[Category, float]]
    team_name: str


def _find_worst_match(
    il_player: Player,
    active_players: list[Player],
    already_displaced: set[str],
    *,
    league_context: LeagueContext | None = None,
    current_roster: list[Player | dict] | None = None,
    projection_source: ProjectionSource = "rest_of_season",
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
            projection_source,
        )

    # Fallback: SGP-based (legacy behavior, used when no league context).
    # Note for pitchers: this can incorrectly pick an elite low-volume
    # closer (Mason Miller) over a struggling starter, because SGP is
    # volume-weighted. Standings/breakdown call sites should pass
    # league_context to enable the ΔRoto-optimal picker.
    return min(candidates, key=lambda a: _player_sgp(a))


def _find_delta_roto_optimal(
    il_player: Player,
    candidates: list[Player],
    current_roster: list[Player | dict],
    ctx: LeagueContext,
    projection_source: ProjectionSource,
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
        team_stats = project_team_stats(
            hyp_roster, displacement=False, projection_source=projection_source
        )
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
    return calculate_player_sgp(p.rest_of_season)


def _compute_displacement_factors(
    active: list[Player],
    il_players: list[Player],
    *,
    league_context: LeagueContext | None = None,
    projection_source: ProjectionSource = "rest_of_season",
) -> dict[str, float]:
    """Map player-name → scale factor for IL-induced displacement.

    Hitters (always) and pitchers (when ``league_context`` is None) use
    the legacy substitution model: process IL players in descending
    playing-time order; each picks the displacement target via
    :func:`_find_worst_match` and scales them by
    ``max(0, active_pt - il_pt) / active_pt``.

    Pitchers WITH ``league_context`` use the pool-slot model: pool all
    pitchers (active + IL), rank by leave-one-out team-ΔRoto, the
    bottom (pool_size - active_p_slots) get sf=0 — even when those are
    IL pitchers themselves. This matches real-world fantasy decisions
    (when an IL guy returns, the manager benches the worst remaining
    pitcher overall, not necessarily the role-matched one), and avoids
    the substitution model's tendency to zero out elite low-volume
    closers when high-volume IL starters are returning.

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
            projection_source=projection_source,
            all_active=active,
            all_il=il_players,
        )
    )

    # Pitchers: pool-slot model with league_context, substitution otherwise.
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
                projection_source=projection_source,
            )
        )
    else:
        factors.update(
            _compute_substitution_factors(
                active_pitchers,
                il_pitchers,
                league_context=None,
                projection_source=projection_source,
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
    projection_source: ProjectionSource,
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
            projection_source=projection_source,
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
    projection_source: ProjectionSource,
) -> dict[str, float]:
    """Pool-slot pitcher displacement: keep top-N (by team-ΔRoto), where
    N is the count of active P-slot pitchers.

    Excess pitchers get sf=0. Excess can include IL pitchers — the
    model treats IL guys as competing for the same playing-time pool
    as active pitchers, so a weak IL pitcher with poor projection can
    be "benched" while an elite active closer is preserved.

    Iterative-greedy selection: each round, evaluate every remaining
    candidate against the current post-cut roster state and bench the
    one whose removal preserves the highest team roto pts. After each
    cut, the next round's marginals are recomputed from the new state,
    so cumulative damage in scarce categories (e.g. cutting a second
    closer after losing the first) shows up as a much larger marginal
    than independent leave-one-out would suggest. This avoids the
    pathology where two closers each look "affordable" in isolation
    but together crash a category by several roto pts.

    No-op when the pool already fits in the available slots. Skips
    pitchers without ROS projections.
    """
    pool = [
        p
        for p in [*active_pitchers, *il_pitchers]
        if p.rest_of_season is not None and _playing_time(p) > 0
    ]
    slot_count = len(active_pitchers)
    excess = len(pool) - slot_count
    if excess <= 0:
        return {}

    full_pool_roster: list[Player | dict] = [*all_il, *all_active]

    def team_pts(stats: CategoryStats) -> float:
        all_team_stats: dict[str, CategoryStats] = dict(league_context.baseline_other_team_stats)
        all_team_stats[league_context.team_name] = stats
        return score_roto(_dict_table(all_team_stats), team_sds=league_context.team_sds)[
            league_context.team_name
        ].total

    def state_with(zero_names: set[str]) -> list[Player | dict]:
        return [
            _scale_stats(p, 0.0) if isinstance(p, Player) and p.name in zero_names else p
            for p in full_pool_roster
        ]

    benched: set[str] = set()
    remaining = list(pool)
    for _ in range(excess):
        best_cand: Player | None = None
        best_pts = -float("inf")
        for cand in remaining:
            hyp_roster = state_with(benched | {cand.name})
            stats = project_team_stats(
                hyp_roster,
                displacement=False,
                projection_source=projection_source,
            )
            pts = team_pts(stats)
            if pts > best_pts:
                best_pts = pts
                best_cand = cand
        if best_cand is None:
            break
        benched.add(best_cand.name)
        remaining = [p for p in remaining if p.name != best_cand.name]

    return {name: 0.0 for name in benched}


def _scale_stats(p: Player, factor: float) -> dict[str, float | PlayerType]:
    """Return a dict of scaled counting stats for the player.

    factor=1.0 means full stats; factor=0.0 means zeroed out. The
    ``player_type`` key is included so callers can route the result the
    same way they would a full Player.
    """
    result: dict[str, float | PlayerType] = {}
    if p.rest_of_season is None:
        return result
    if p.player_type == PlayerType.HITTER:
        for key in HITTING_COUNTING:
            result[key] = _safe(getattr(p.rest_of_season, key, 0)) * factor
    elif p.player_type == PlayerType.PITCHER:
        for key in PITCHING_COUNTING:
            result[key] = _safe(getattr(p.rest_of_season, key, 0)) * factor
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
    raw_stats: dict[str, float]  # pre-scale ROS stats; empty when NO_PROJECTION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "player_type": self.player_type.value,
            "status": self.status.value,
            "scale_factor": self.scale_factor,
            "raw_stats": dict(self.raw_stats),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlayerContribution:
        return cls(
            name=d["name"],
            player_type=PlayerType(d["player_type"]),
            status=ContributionStatus(d["status"]),
            scale_factor=float(d["scale_factor"]),
            raw_stats={k: float(v) for k, v in d.get("raw_stats", {}).items()},
        )


@dataclass(frozen=True)
class RosterBreakdown:
    """Per-player contributions for one team, partitioned by player type."""

    team_name: str
    hitters: list[PlayerContribution]
    pitchers: list[PlayerContribution]

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "hitters": [c.to_dict() for c in self.hitters],
            "pitchers": [c.to_dict() for c in self.pitchers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RosterBreakdown:
        return cls(
            team_name=d["team_name"],
            hitters=[PlayerContribution.from_dict(x) for x in d.get("hitters", [])],
            pitchers=[PlayerContribution.from_dict(x) for x in d.get("pitchers", [])],
        )


def _apply_displacement(
    roster: list[Player],
    *,
    league_context: LeagueContext | None = None,
    projection_source: ProjectionSource = "rest_of_season",
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
    pool-slot model (top-N by team-ΔRoto play; bottom of pool gets sf=0,
    even when those are IL pitchers themselves) and hitter displacement
    uses ΔRoto-optimal substitution. Otherwise both use the legacy SGP
    substitution model.

    Returns a list where each entry is either an unmodified Player
    (active, unaffected; or IL, full-scale) or a dict of scaled stats
    (any player with a sub-1.0 factor — including IL pitchers in the
    pool model). Non-Player entries in the input (dicts from draft
    scripts) pass through untouched.
    """
    # Non-Player entries pass through untouched (draft scripts use dicts).
    pass_through: list[Player | dict] = [p for p in roster if not isinstance(p, Player)]
    typed = [p for p in roster if isinstance(p, Player)]

    active, il_players, _bench = _classify_roster(typed)
    displacement_factors = _compute_displacement_factors(
        active,
        il_players,
        league_context=league_context,
        projection_source=projection_source,
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
    projection_source: ProjectionSource = "rest_of_season",
) -> RosterBreakdown:
    """Return per-player contributions for ``roster``, tagged with status.

    Uses the same slot-first classification as :func:`_apply_displacement`,
    and the same displacement-factor math. The aggregate over
    ``raw_stats[cat] * scale_factor`` per category equals
    :func:`project_team_stats` with ``displacement=True``.

    When ``league_context`` is provided, displacement targets are chosen
    via ΔRoto-optimal selection (matches the standings layer); without
    it, the legacy SGP picker is used.
    """
    active, il_players, bench = _classify_roster(roster)
    displacement_factors = _compute_displacement_factors(
        active,
        il_players,
        league_context=league_context,
        projection_source=projection_source,
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
            )
        )

    hitters = [c for c in contributions if c.player_type == PlayerType.HITTER]
    pitchers = [c for c in contributions if c.player_type == PlayerType.PITCHER]
    return RosterBreakdown(team_name=team_name, hitters=hitters, pitchers=pitchers)


def project_team_stats(
    roster,
    *,
    displacement: bool = False,
    projection_source: ProjectionSource = "rest_of_season",
    league_context: LeagueContext | None = None,
) -> CategoryStats:
    """Sum projected stats for a roster into a CategoryStats.

    Accepts Player dataclass objects OR plain dicts with flat stat
    keys. Rate stats (AVG, ERA, WHIP) are computed from component
    totals rather than simple sums, so the result is mathematically
    correct rather than just a naive average.

    ``projection_source`` selects which projection field on each
    :class:`Player` to sum. The default ``"rest_of_season"`` is the
    forward-looking decision math used by the optimizer, recs, and
    trade evaluation: a hot-YTD player and a cold-YTD player with the
    same ROS-remaining contribute identically, so start/sit decisions
    are not biased by locked YTD totals. ``"full_season_projection"``
    sums ROS + YTD and is used only by
    :meth:`ProjectedStandings.from_rosters` to preserve the
    end-of-season-totals projection until proper standings + ROS
    combination ships (Yahoo standings only surface AVG, not the H/AB
    components needed to recombine rate stats correctly).

    Note: ``projection_source`` controls only the per-player reads in
    ``_stat``. Displacement scaling (``_apply_displacement`` →
    ``_scale_stats``) continues to operate on ROS playing time and ROS
    counting stats, so a displaced active player's contribution is
    ROS-times-factor regardless of source. With
    ``projection_source="full_season_projection"`` this is a documented
    approximation: undisplaced active players contribute full-season
    totals while displaced players contribute scaled ROS, mixing units.
    The proper combination (YTD + scaled ROS for displaced; full-season
    for undisplaced) requires the team-level YTD ingest deferred to a
    follow-up phase.

    When ``displacement=True``, bench players are excluded and IL
    players displace the worst positional match among active players,
    scaling down the displaced player's stats proportionally based on
    playing time. Only activates for Player dataclass objects — dict
    input callers are unaffected.

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
            projection_source=projection_source,
        )

    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            r += _stat(p, "r", projection_source)
            hr += _stat(p, "hr", projection_source)
            rbi += _stat(p, "rbi", projection_source)
            sb += _stat(p, "sb", projection_source)
            h_total += _stat(p, "h", projection_source)
            ab_total += _stat(p, "ab", projection_source)
        elif ptype == PlayerType.PITCHER:
            w += _stat(p, "w", projection_source)
            k += _stat(p, "k", projection_source)
            sv += _stat(p, "sv", projection_source)
            ip_total += _stat(p, "ip", projection_source)
            er_total += _stat(p, "er", projection_source)
            bb_total += _stat(p, "bb", projection_source)
            ha_total += _stat(p, "h_allowed", projection_source)

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


def project_team_sds(
    roster,
    *,
    displacement: bool = True,
) -> dict[Category, float]:
    """Aggregate per-player projection variance into team-level SDs.

    Uses ``STAT_VARIANCE`` (per-stat CV calibrated from 2022-2024
    Steamer+ZiPS vs actuals) under a player-independence assumption:

        SD_cat_team = CV_cat * sqrt(sum_over_players(stat_i^2))

    Rate stats propagate through their component totals:

        SD_AVG  = CV_h * sqrt(sum h_i^2) / sum_AB
        SD_ERA  = 9 * CV_er * sqrt(sum er_i^2) / sum_IP
        SD_WHIP = sqrt(CV_bb^2 * sum bb_i^2 + CV_ha^2 * sum ha_i^2) / sum_IP

    ``displacement`` matches :func:`project_team_stats` — bench excluded,
    IL players displace their worst active positional match.

    Returns ``{Category: sd}`` keyed by :class:`Category` enum for every
    category in ``ALL_CATS``. Use :func:`team_sds_to_json` at the cache
    boundary for JSON serialization. Empty roster returns zeros.
    """
    if displacement:
        roster = _apply_displacement(roster)

    h_sum_sq: dict[str, float] = {k: 0.0 for k in HITTING_COUNTING}
    p_sum_sq: dict[str, float] = {k: 0.0 for k in PITCHING_COUNTING}
    total_ab = 0.0
    total_ip = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            for k in HITTING_COUNTING:
                v = _stat(p, k)
                h_sum_sq[k] += v * v
            total_ab += _stat(p, "ab")
        elif ptype == PlayerType.PITCHER:
            for k in PITCHING_COUNTING:
                v = _stat(p, k)
                p_sum_sq[k] += v * v
            total_ip += _stat(p, "ip")

    sds: dict[Category, float] = dict.fromkeys(ALL_CATS, 0.0)
    for stat_key, cat in [
        ("r", Category.R),
        ("hr", Category.HR),
        ("rbi", Category.RBI),
        ("sb", Category.SB),
    ]:
        sds[cat] = STAT_VARIANCE[stat_key] * sqrt(h_sum_sq[stat_key])
    for stat_key, cat in [("w", Category.W), ("k", Category.K), ("sv", Category.SV)]:
        sds[cat] = STAT_VARIANCE[stat_key] * sqrt(p_sum_sq[stat_key])
    if total_ab > 0:
        sds[Category.AVG] = STAT_VARIANCE["h"] * sqrt(h_sum_sq["h"]) / total_ab
    if total_ip > 0:
        sds[Category.ERA] = 9.0 * STAT_VARIANCE["er"] * sqrt(p_sum_sq["er"]) / total_ip
        whip_var = (STAT_VARIANCE["bb"] ** 2) * p_sum_sq["bb"] + (
            STAT_VARIANCE["h_allowed"] ** 2
        ) * p_sum_sq["h_allowed"]
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
