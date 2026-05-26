"""Stash board -- rank injured players (owned IL + injured FAs) by their
leverage-aware marginal active value, and allocate the scarce IL slots.

Sibling of ``il_return_planner``: reuses the optimizer and the
double-count-safe deltaRoto band. A candidate's Gain depends on ownership,
because the standings anchor (``ProjectedStandings.from_rosters``) is built
with displacement over the FULL roster INCLUDING owned IL players:

  * Injured FA -- the anchor excludes him, so Gain is the ADD-GAIN of
    slotting him into the optimized lineup (band mean, ~0 when he can't crack
    it -- "no harm, no foul").
  * Owned IL stash -- the anchor already prices his ROS, so re-adding him
    would double-count. Gain is instead the DROP-COST (how much you'd lose by
    dropping him), mirroring ``il_return_planner``.

Cost is the IL-slot allocation cost: 0 when a slot is open, else the Gain of
the weakest owned IL stash he displaces.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.lineup.band_format import band_class
from fantasy_baseball.lineup.delta_roto import DeltaRotoBand, compute_delta_roto_band
from fantasy_baseball.lineup.il_return_planner import _activate
from fantasy_baseball.lineup.optimizer import (
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.models.standings import CategoryStats, ProjectedStandings
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

__all__ = ["StashResult", "StashScore", "score_stash_candidates"]


@dataclass
class StashScore:
    """One injured player's stash evaluation."""

    name: str
    player_type: str
    status: str  # IL10 / IL15 / IL60 / ...
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
        vol = float(inc.ip)
        scale = max(0.0, vol - w) / vol if vol > 0 else 0.0
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

    FA-only: owned IL players are already priced into the ``from_rosters``
    anchor via displacement and use the drop-cost path instead (PR #101).
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


def _marginal_band(
    candidate: Player,
    *,
    owned: bool,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> dict[str, Any]:
    """Return the deltaRoto band dict for ``candidate``'s stash Gain.

    - FA candidate (``owned=False``): Gain is the rate-upgrade-over-return-window
      value -- the best swap of his ROS rate for the W-window of the weakest
      eligible active player (see ``_best_swap_band``). This is why an injured
      player with little remaining VOLUME but a strong RATE still scores: a
      100-AB bat is judged on the 100 ABs he'll play, not his season total.
    - Owned IL candidate (``owned=True``): the ``ProjectedStandings.from_rosters``
      anchor already prices his ROS via displacement, so re-adding him would
      double-count (PR #101). Measure the DROP-COST instead -- the lineup WITH
      him active vs the baseline WITHOUT -- and negate it for the hold value.
    """
    field_stats = projected_standings.field_stats(team_name)

    if not owned:
        return _best_swap_band(
            candidate,
            before_active=before_active,
            field_stats=field_stats,
            projected_standings=projected_standings,
            team_name=team_name,
            team_sds=team_sds,
            fraction_remaining=fraction_remaining,
        )

    activated = _activate(candidate)
    lineup_with = _solve_active(
        [*_counted_pool(roster, exclude_name=candidate.name), activated],
        roster_slots,
        projected_standings,
        team_name,
        team_sds,
    )
    drop_band = compute_delta_roto_band(
        lineup_with,
        before_active,
        field_stats,
        team_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )
    p_hold = 1.0 - drop_band.p_positive
    return {
        "mean": round(-drop_band.mean, 2) + 0.0,  # + 0.0 flattens -0.0 -> 0.0 for JSON
        "sd": round(drop_band.sd, 2),
        "p_positive": round(p_hold, 3),
        "verdict": band_class(p_hold),
    }


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
    owned: bool = False,
) -> float:
    """Gain = band mean of rostering ``candidate`` active. Floored at ~0.

    ``owned`` defaults to ``False`` (FA add-gain) for the test-only callers
    that exercise the add-gain path; ``score_stash_candidates`` passes the
    real flag through."""
    band = _marginal_band(
        candidate,
        owned=owned,
        before_active=before_active,
        roster=roster,
        roster_slots=roster_slots,
        projected_standings=projected_standings,
        team_name=team_name,
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
    )
    return float(band["mean"])


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

    Every candidate is selected via ``is_on_il()`` (owned IL stashes + injured
    FAs), so each one uses an IL slot:

    - Open IL slot -> (0, None).
    - IL full -> displace the lowest-Gain owned IL stash (IL-for-IL, the
      user's rule). Cost = max(0, that stash's Gain): displacing a
      net-negative stash never CREDITS the candidate (no -cost -> inflated
      stash_value artifact).
    """
    if _open_il_slots(roster, roster_slots) > 0:
        return 0.0, None

    # Displace the weakest owned IL stash (exclude the candidate itself).
    pool = [p for p in _owned_il_stashes(roster) if p.name != candidate.name]
    if not pool:
        return 0.0, None
    drop = min(pool, key=lambda p: gain_by_name.get(p.name, 0.0))
    # every owned stash was scored in pass 1; floor so a negative-value stash
    # is a free drop, not a credit.
    return max(0.0, gain_by_name[drop.name]), drop.name


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

    # Pass 1: Gain + band for every candidate (owned + FA).
    bands: dict[str, dict[str, Any]] = {}
    candidates_in: list[tuple[Player, bool]] = [(p, True) for p in owned_il] + [
        (p, False) for p in injured_fas
    ]
    for player, owned in candidates_in:
        bands[player.name] = _marginal_band(
            player,
            owned=owned,
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
        # An owned player already holds his IL slot -- there is no acquisition
        # cost, so his stash value is just his (drop-cost) gain. _cost_and_drop
        # is only meaningful for a FA who must be slotted into a possibly-full IL.
        if owned:
            cost, drop = 0.0, None
        else:
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
