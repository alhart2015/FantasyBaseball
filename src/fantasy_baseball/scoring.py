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
from typing import Protocol

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
    IL_STATUSES,
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


def _stat(p, key):
    """Read a stat from a Player's ROS stats or from a flat dict."""
    # Player dataclass: stats live on the .rest_of_season attribute
    ros = getattr(p, "rest_of_season", None)
    if ros is not None and hasattr(ros, key):
        return _safe(getattr(ros, key, 0))
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


def _is_il(p: Player) -> bool:
    """True if the player is on the IL (by slot or by Yahoo status)."""
    if p.selected_position in IL_SLOTS:
        return True
    return p.status in IL_STATUSES


def _is_bench(p: Player) -> bool:
    """True if the player is benched (BN slot) and NOT on the IL."""
    return p.selected_position == Position.BN and not _is_il(p)


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


def _find_worst_match(
    il_player: Player,
    active_players: list[Player],
    already_displaced: set[str],
) -> Player | None:
    """Find the worst active player (by SGP) sharing a positional role.

    For pitchers: match SP vs RP role.
    For hitters: match on overlapping real positions; fallback to worst
    hitter overall if no position match.

    Returns None if no eligible active player exists.
    """
    candidates: list[Player] = []

    if il_player.player_type == PlayerType.PITCHER:
        role = _pitcher_role(il_player)
        for a in active_players:
            if a.name in already_displaced:
                continue
            if a.player_type != PlayerType.PITCHER:
                continue
            if _pitcher_role(a) == role:
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

    # Worst = lowest total SGP
    return min(candidates, key=lambda a: _player_sgp(a))


def _player_sgp(p: Player) -> float:
    """Calculate total SGP for a player, returning 0 if no ROS stats."""
    if p.rest_of_season is None:
        return 0.0
    return calculate_player_sgp(p.rest_of_season)


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


def _apply_displacement(roster: list[Player]) -> list[Player | dict]:
    """Partition roster into active/bench/IL and apply displacement scaling.

    Classification is slot-first:

    - Any slot that is neither ``BN`` nor in ``IL_SLOTS`` → active.
      Counted at face value; may be a displacement target. Yahoo IL
      status on an active-slotted player is ignored — the manager's
      slot choice wins.
    - Slot in ``IL_SLOTS`` (IL, IL+, DL, DL+) → IL. Counted at full
      ROS and displaces the worst SGP-matched active player.
    - BN slot + IL status → IL (same displacement path).
    - BN slot + healthy → excluded.

    Returns a list where each entry is either an unmodified Player
    (active, unaffected; or IL, full-scale) or a dict of scaled stats
    (active, displaced).
    """
    # Separate players into categories
    active: list[Player] = []
    il_players: list[Player] = []

    for p in roster:
        if not isinstance(p, Player):
            # Dict-input callers: pass through unmodified
            active.append(p)
            continue
        slot = p.selected_position
        if slot == Position.BN:
            # BN + IL status still displaces; healthy bench is excluded.
            if _is_il(p):
                il_players.append(p)
            continue
        if slot in IL_SLOTS:
            il_players.append(p)
            continue
        # Active slot: treat at face value, regardless of Yahoo status.
        active.append(p)

    # Sort IL players by descending playing time (highest PT gets first pick)
    il_players.sort(key=_playing_time, reverse=True)

    # Track which active players have already been displaced
    already_displaced: set[str] = set()
    # Map from player name to scale factor for displaced players
    displacement_factors: dict[str, float] = {}

    for il_p in il_players:
        il_pt = _playing_time(il_p)
        if il_pt <= 0:
            continue  # No playing time -> no displacement

        target = _find_worst_match(il_p, active, already_displaced)
        if target is None:
            continue

        active_pt = _playing_time(target)
        if active_pt <= 0:
            continue

        factor = max(0.0, active_pt - il_pt) / active_pt
        already_displaced.add(target.name)
        displacement_factors[target.name] = factor

    # Build output: IL players at full scale + active with displacement
    result: list = []
    # IL players contribute their full projected stats
    for p in il_players:
        result.append(p)
    # Active players: apply displacement factors to affected ones
    for p in active:
        if not isinstance(p, Player):
            result.append(p)
            continue
        if p.name in displacement_factors:
            factor = displacement_factors[p.name]
            scaled = _scale_stats(p, factor)
            result.append(scaled)
        else:
            result.append(p)

    return result


_HITTER_RAW_KEYS: tuple[str, ...] = ("r", "hr", "rbi", "sb", "h", "ab")
_PITCHER_RAW_KEYS: tuple[str, ...] = ("w", "k", "sv", "ip", "er", "bb", "h_allowed")


def _raw_stats_for(p: Player) -> dict[str, float]:
    """Extract the ROS raw stats the breakdown needs, or ``{}`` if absent."""
    if p.rest_of_season is None:
        return {}
    keys = _PITCHER_RAW_KEYS if p.player_type == PlayerType.PITCHER else _HITTER_RAW_KEYS
    return {k: _safe(getattr(p.rest_of_season, k, 0)) for k in keys}


def compute_roster_breakdown(team_name: str, roster: list[Player]) -> RosterBreakdown:
    """Return per-player contributions for ``roster``, tagged with status.

    Uses the same slot-first classification as :func:`_apply_displacement`,
    and the same displacement-factor math. The aggregate over
    ``raw_stats[cat] * scale_factor`` per category equals
    :func:`project_team_stats` with ``displacement=True``.
    """
    active: list[Player] = []
    il_players: list[Player] = []
    bench: list[Player] = []
    no_projection: list[Player] = []

    for p in roster:
        if not isinstance(p, Player):
            continue  # dict-input callers aren't supported here
        slot = p.selected_position
        if slot == Position.BN:
            if _is_il(p):
                il_players.append(p)
            else:
                bench.append(p)
            continue
        if slot in IL_SLOTS:
            il_players.append(p)
            continue
        active.append(p)

    # Run the same displacement math _apply_displacement runs, without
    # allocating its output list — we only need the factors.
    il_players_sorted = sorted(il_players, key=_playing_time, reverse=True)
    already_displaced: set[str] = set()
    displacement_factors: dict[str, float] = {}
    for il_p in il_players_sorted:
        il_pt = _playing_time(il_p)
        if il_pt <= 0:
            continue
        target = _find_worst_match(il_p, active, already_displaced)
        if target is None:
            continue
        active_pt = _playing_time(target)
        if active_pt <= 0:
            continue
        factor = max(0.0, active_pt - il_pt) / active_pt
        already_displaced.add(target.name)
        displacement_factors[target.name] = factor

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
        status = (
            ContributionStatus.NO_PROJECTION
            if p.rest_of_season is None
            else ContributionStatus.IL_FULL
        )
        factor = 0.0 if status == ContributionStatus.NO_PROJECTION else 1.0
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

    for p in no_projection:
        contributions.append(
            PlayerContribution(
                name=p.name,
                player_type=p.player_type,
                status=ContributionStatus.NO_PROJECTION,
                scale_factor=0.0,
                raw_stats={},
            )
        )

    hitters = [c for c in contributions if c.player_type == PlayerType.HITTER]
    pitchers = [c for c in contributions if c.player_type == PlayerType.PITCHER]
    return RosterBreakdown(team_name=team_name, hitters=hitters, pitchers=pitchers)


def project_team_stats(roster, *, displacement: bool = False) -> CategoryStats:
    """Sum projected stats for a roster into a CategoryStats.

    Accepts Player dataclass objects OR plain dicts with flat stat
    keys. Rate stats (AVG, ERA, WHIP) are computed from component
    totals rather than simple sums, so the result is mathematically
    correct rather than just a naive average.

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
        roster = _apply_displacement(roster)

    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            r += _stat(p, "r")
            hr += _stat(p, "hr")
            rbi += _stat(p, "rbi")
            sb += _stat(p, "sb")
            h_total += _stat(p, "h")
            ab_total += _stat(p, "ab")
        elif ptype == PlayerType.PITCHER:
            w += _stat(p, "w")
            k += _stat(p, "k")
            sv += _stat(p, "sv")
            ip_total += _stat(p, "ip")
            er_total += _stat(p, "er")
            bb_total += _stat(p, "bb")
            ha_total += _stat(p, "h_allowed")

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
