"""Trade-for-future-pick calculator.

Two separate views of trading a current player for a next-year draft pick:
the this-year ROS Monte Carlo impact (win% / top-3% / per-category) and the
next-year marginal value of the extra pick (VAR at the post-keeper-round draft
ordinal). See docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from fantasy_baseball.analysis.draft_value import ParCurve, projected_par_curve
from fantasy_baseball.analysis.injury_stress import McInputs, _replacement_ros
from fantasy_baseball.config import LeagueConfig
from fantasy_baseball.models.player import (
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
    RankInfo,
)
from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.simulation import _replacement_line, run_ros_monte_carlo
from fantasy_baseball.utils.constants import ALL_CATEGORIES
from fantasy_baseball.utils.name_utils import normalize_name


def keeper_rounds_for(config: LeagueConfig) -> int:
    """Number of keeper rounds = keepers per team = len(keepers) // num_teams.

    Requires an even split (every team keeps the same number, which is what
    "the first K rounds are keeper rounds" means). A non-divisible count breaks
    the nominal-to-drafted-round mapping, so fail loud rather than truncate.
    """
    nk = len(config.keepers)
    nt = config.num_teams
    if nt <= 0 or nk == 0 or nk % nt != 0:
        raise ValueError(
            f"Cannot derive keeper rounds: {nk} keepers / {nt} teams is not an "
            "even split. The nominal-to-drafted-round mapping needs a fixed "
            "keeper-rounds count."
        )
    return nk // nt


def pick_ordinal_range(
    nominal_round: int,
    keeper_rounds: int,
    num_teams: int,
    curve_len: int,
    pick_slot: str = "round",
) -> tuple[int, int]:
    """1-based inclusive par-curve ordinal range for a nominal pick round.

    drafted_round = nominal_round - keeper_rounds; the drafted round spans
    ordinals [(drafted_round-1)*num_teams + 1, drafted_round*num_teams] in the
    VAR-sorted par curve. ``pick_slot`` narrows to the top ("early"), middle
    ("mid"), or bottom ("late") third of that span. The upper bound is clamped
    to ``curve_len``; a range entirely beyond the curve is an error.
    """
    drafted_round = nominal_round - keeper_rounds
    if drafted_round < 1:
        raise ValueError(
            f"Round {nominal_round} is a keeper round; drafted picks start at "
            f"round {keeper_rounds + 1}."
        )
    lo = (drafted_round - 1) * num_teams + 1
    hi = drafted_round * num_teams
    if lo > curve_len:
        raise ValueError(
            f"drafted round {drafted_round} (ordinals {lo}-{hi}) is beyond the "
            f"par curve ({curve_len} picks)."
        )
    hi = min(hi, curve_len)
    if pick_slot == "round":
        return lo, hi
    if pick_slot not in ("early", "mid", "late"):
        raise ValueError("pick_slot must be one of: round, early, mid, late")
    span = hi - lo + 1
    third = max(1, span // 3)
    if pick_slot == "early":
        return lo, lo + third - 1
    if pick_slot == "late":
        return hi - third + 1, hi
    # mid: the middle third, clamped to stay non-empty inside [lo, hi]
    mlo = min(lo + third, hi)
    mhi = max(hi - third, mlo)
    return mlo, mhi


@dataclass(frozen=True)
class NextYearValue:
    nominal_round: int
    keeper_rounds: int
    drafted_round: int
    pick_slot: str
    expected_var: float
    early_var: float
    keeper_par: float
    ordinal_lo: int
    ordinal_hi: int


def _mean_over(par: ParCurve, lo: int, hi: int) -> float:
    vals = [par.par_for_slot(k) for k in range(lo, hi + 1)]
    return sum(vals) / len(vals)


def pick_value(
    par: ParCurve,
    nominal_round: int,
    keeper_rounds: int,
    num_teams: int,
    pick_slot: str = "round",
) -> NextYearValue:
    """Expected pick VAR = mean par over the (narrowed) drafted-round ordinals.

    Also reports an "early in the round" VAR (top third) for context and the
    keeper-average par (``par.keeper_par``, may be NaN).
    """
    curve_len = len(par.drafted_pars)
    lo, hi = pick_ordinal_range(nominal_round, keeper_rounds, num_teams, curve_len, pick_slot)
    elo, ehi = pick_ordinal_range(nominal_round, keeper_rounds, num_teams, curve_len, "early")
    return NextYearValue(
        nominal_round=nominal_round,
        keeper_rounds=keeper_rounds,
        drafted_round=nominal_round - keeper_rounds,
        pick_slot=pick_slot,
        expected_var=_mean_over(par, lo, hi),
        early_var=_mean_over(par, elo, ehi),
        keeper_par=par.keeper_par,
        ordinal_lo=lo,
        ordinal_hi=hi,
    )


def next_year_value(
    config: LeagueConfig, nominal_round: int, pick_slot: str = "round"
) -> NextYearValue:
    """Build the projected par curve and value a nominal-round pick on it."""
    par = projected_par_curve(config)
    return pick_value(par, nominal_round, keeper_rounds_for(config), config.num_teams, pick_slot)


def find_sent_player(
    roster: list[Player], name: str, player_type: str | None = None
) -> Player:
    """Locate the player being traded away, by normalized (accent-safe) name.

    When two roster players normalize to the same name and differ in type (a
    hitter and a pitcher), ``player_type`` disambiguates; without it, an
    ambiguous match is an error rather than a silent pick.
    """
    target = normalize_name(name)
    matches = [p for p in roster if normalize_name(p.name) == target]
    if player_type is not None:
        want = PlayerType(player_type)
        matches = [p for p in matches if p.player_type == want]
    if not matches:
        raise ValueError(f"{name!r} is not on your roster.")
    if len(matches) > 1:
        raise ValueError(
            f"{name!r} is ambiguous on your roster; pass --player-type hitter|pitcher."
        )
    return matches[0]


def build_replacement_filler(sent: Player) -> Player:
    """A replacement-level filler for the slot the sent player vacates.

    Distinct name (never aliases the real player now on the partner), the sent
    player's positions (so it is eligible for the vacated slot), and BOTH its
    ROS line and full-season line neutralized to replacement level -- the MC
    reads production off the ROS line (ROS-direct engine) but the full-season
    line still drives the playing-time-curve shape and the top-k fallback, so
    neutralize both. The active-lineup selection starts this filler only when no
    better bench player is available; otherwise it benches.
    """
    is_hitter = sent.player_type == PlayerType.HITTER
    ros_repl = _replacement_ros(sent)  # scaled to the sent player's ROS volume
    repl_line = _replacement_line(sent.to_flat_dict_full_season(), is_hitter)
    stats_cls = HitterStats if is_hitter else PitcherStats
    fs_repl = stats_cls.from_dict(repl_line)  # replacement full-season line
    pos_label = str(sent.positions[0]) if sent.positions else str(sent.player_type)
    return dataclasses.replace(
        sent,
        name=f"Replacement ({pos_label})",
        rest_of_season=ros_repl,
        full_season_projection=fs_repl,
        preseason=None,
        current=None,
        rank=RankInfo(),
        selected_position=None,
        fg_id=None,
        mlbam_id=None,
        yahoo_id=None,
    )


def worst_of_type(roster: list[Player], ptype: PlayerType, denoms) -> Player | None:
    """The lowest full-season-projected player of ``ptype``, or None if none exist.

    Ranked by full-season SGP so the partner drops a benched scrub (second-order)
    to fit the acquired star, keeping the partner's roster size constant. Players
    without a full-season projection are skipped (they cannot be scored).
    """
    cands = [
        p for p in roster if p.player_type == ptype and p.full_season_projection is not None
    ]
    if not cands:
        return None
    return min(
        cands, key=lambda p: calculate_player_sgp(p.full_season_projection, denoms=denoms)
    )


def build_trade_scenario(
    inputs: McInputs, sent: Player, partner: str
) -> dict[str, list[Player]]:
    """Team rosters after the trade: user loses ``sent`` and gains a replacement
    filler (size constant); the partner gains the intact ``sent`` and drops its
    worst-of-type player (size constant). ``inputs.team_rosters`` is not mutated.

    If the partner has no droppable player of the sent player's type (none on
    their roster), nothing is dropped and the partner grows by one -- you cannot
    drop what does not exist. Accepted (a benched extra body is second-order).
    """
    user = inputs.user_team_name
    filler = build_replacement_filler(sent)
    new_user = [p for p in inputs.team_rosters[user] if p is not sent] + [filler]

    partner_roster = inputs.team_rosters[partner]
    drop = worst_of_type(partner_roster, sent.player_type, inputs.denoms)
    new_partner = [p for p in partner_roster if p is not drop] + [sent]

    scenario = dict(inputs.team_rosters)
    scenario[user] = new_user
    scenario[partner] = new_partner
    return scenario


@dataclass(frozen=True)
class CategoryDelta:
    category: str
    base_first: float
    new_first: float
    base_top3: float
    new_top3: float


@dataclass(frozen=True)
class ThisYearImpact:
    base_win: float
    new_win: float
    base_top3: float
    new_top3: float
    categories: list[CategoryDelta]
    n_iter: int
    seed: int


def run_scenario(
    inputs: McInputs, team_rosters: dict[str, list[Player]], n_iter: int, seed: int
) -> dict:
    """Rebuild effective rosters for ``team_rosters`` and run the ROS Monte Carlo.

    eos_baseline / team_sds are held fixed (reused from ``inputs``), mirroring the
    injury stress-test: the first-order roster change flows through the rebuilt
    effective rosters and the MC scoring, while the league-context scaffolding
    stays constant so the baseline-vs-scenario delta is controlled.
    """
    eff = build_effective_rosters(
        team_rosters,
        inputs.eos_baseline,
        inputs.team_sds,
        inputs.fraction_remaining,
        denoms=inputs.denoms,
    )
    return run_ros_monte_carlo(
        team_rosters=team_rosters,
        actual_standings=inputs.actual_standings,
        fraction_remaining=inputs.fraction_remaining,
        h_slots=inputs.h_slots,
        p_slots=inputs.p_slots,
        user_team_name=inputs.user_team_name,
        n_iterations=n_iter,
        seed=seed,
        effective_rosters=eff,
    )


def this_year_impact(
    inputs: McInputs,
    sent: Player,
    partner: str,
    *,
    n_iter: int = 2000,
    seed: int = 42,
) -> ThisYearImpact:
    """Baseline vs post-trade ROS MC for the user, with common random numbers.

    Full swing: the sent player leaves the user (replaced by a bench-or-
    replacement filler) and joins the partner. Reports the user's overall win%
    and top-3% and per-category first%/top-3%.
    """
    user = inputs.user_team_name
    base = run_scenario(inputs, inputs.team_rosters, n_iter, seed)
    scen_rosters = build_trade_scenario(inputs, sent, partner)
    scen = run_scenario(inputs, scen_rosters, n_iter, seed)

    br = base["team_results"][user]
    sr = scen["team_results"][user]
    bcat = base["category_risk"]
    scat = scen["category_risk"]
    categories = [
        CategoryDelta(
            category=c.value,
            base_first=bcat[c.value]["first_pct"],
            new_first=scat[c.value]["first_pct"],
            base_top3=bcat[c.value]["top3_pct"],
            new_top3=scat[c.value]["top3_pct"],
        )
        for c in ALL_CATEGORIES
    ]
    return ThisYearImpact(
        base_win=br["first_pct"],
        new_win=sr["first_pct"],
        base_top3=br["top3_pct"],
        new_top3=sr["top3_pct"],
        categories=categories,
        n_iter=n_iter,
        seed=seed,
    )
