"""Trade-for-future-pick calculator.

Two separate views of trading a current player for a next-year draft pick:
the this-year ROS Monte Carlo impact (win% / top-3% / per-category) and the
next-year marginal value of the extra pick (VAR at the post-keeper-round draft
ordinal). See docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fantasy_baseball.analysis.draft_value import ParCurve, projected_par_curve
from fantasy_baseball.analysis.injury_stress import (
    McInputs,
    _replacement_ros,
    load_mc_inputs_from_upstash,
)
from fantasy_baseball.config import LeagueConfig, load_config
from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.models.player import (
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
    RankInfo,
)
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.simulation import _replacement_line, run_ros_monte_carlo
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category
from fantasy_baseball.utils.name_utils import normalize_name

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "league.yaml"


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


def find_sent_player(roster: list[Player], name: str, player_type: str | None = None) -> Player:
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


def place_in_slot(incoming: Player, slot: Position | None) -> Player:
    """Copy of ``incoming`` sitting in the roster spot ``slot``.

    A trade moves ``Player`` objects between rosters, and ``selected_position``
    is a property of the roster SPOT, not of the player -- but
    ``scoring._classify_roster`` reads each body's bucket (active / il / bench)
    straight off that field, and only active + IL bodies are simulated (healthy
    bench pitchers are dropped from the MC outright). So a body that keeps the
    slot it held on its old team silently changes bucket on its new one, which
    invents or deletes lineup production the trade never touched. Every body
    that changes rosters goes through here.

    Health, unlike the slot, belongs to the player, so the transplant never
    changes it: an injured body always lands on the IL and a healthy body never
    does. That guard has to run in BOTH directions, because ``is_on_il()`` reads
    status OR slot -- a player whose only IL signal is his old IL slot would be
    silently healed by a move to an active spot, and a healthy player dropped
    into a vacated IL slot would be silently injured.
    """
    if incoming.is_on_il():
        slot = slot if slot in IL_SLOTS else Position.IL
    elif slot in IL_SLOTS:
        slot = Position.BN
    return dataclasses.replace(incoming, selected_position=slot)


def build_replacement_filler(sent: Player) -> Player:
    """A replacement-level filler for the slot the sent player vacates.

    Distinct name (never aliases the real player now on the partner), the sent
    player's positions (so it is eligible for the vacated slot), and BOTH its
    ROS line and full-season line neutralized to replacement level -- the MC
    reads production off the ROS line (ROS-direct engine) but the full-season
    line still drives the playing-time-curve shape and the top-k fallback, so
    neutralize both.

    The filler inherits the vacated SLOT (via :func:`place_in_slot`) but not the
    sent player's ``status``: a body plucked off waivers is healthy even when the
    player it replaces was hurt, and inheriting an IL status would have the
    filler simulated as injured -- production the user would actually have.
    """
    is_hitter = sent.player_type == PlayerType.HITTER
    ros_repl = _replacement_ros(sent)  # scaled to the sent player's ROS volume
    repl_line = _replacement_line(sent.to_flat_dict_full_season(), is_hitter)
    stats_cls = HitterStats if is_hitter else PitcherStats
    fs_repl = stats_cls.from_dict(repl_line)  # replacement full-season line
    pos_label = str(sent.positions[0]) if sent.positions else str(sent.player_type)
    filler = dataclasses.replace(
        sent,
        name=f"Replacement ({pos_label})",
        rest_of_season=ros_repl,
        full_season_projection=fs_repl,
        preseason=None,
        current=None,
        rank=RankInfo(),
        status="",
        selected_position=None,
        fg_id=None,
        mlbam_id=None,
        yahoo_id=None,
    )
    return place_in_slot(filler, sent.selected_position)


def worst_of_type(
    roster: list[Player], ptype: PlayerType, denoms: dict[Category, float]
) -> Player | None:
    """The lowest full-season-projected player of ``ptype``, or None if none exist.

    Ranked by full-season SGP so the partner drops a benched scrub (second-order)
    to fit the acquired star, keeping the partner's roster size constant. Players
    without a full-season projection are skipped (they cannot be scored).
    """
    cands = [p for p in roster if p.player_type == ptype and p.full_season_projection is not None]
    if not cands:
        return None
    return min(cands, key=lambda p: calculate_player_sgp(p.full_season_projection, denoms=denoms))


def build_trade_scenario(
    inputs: McInputs, sent: Player, partner: str, received: Player | None = None
) -> dict[str, list[Player]]:
    """Team rosters after the trade. ``inputs.team_rosters`` is not mutated.

    Two shapes, and the difference matters:

    * **Player for player** (``received`` supplied). Each side gives one and gets
      one, so sizes hold on their own -- no filler is invented and nothing is
      dropped. This is the honest model of a real swap: the incoming player IS
      the replacement for the vacated slot, and modelling him as a generic filler
      would understate a good return and overstate a bad one.
    * **Player for a pick** (``received`` omitted). The user has a hole, so a
      replacement-level filler keeps the roster size constant, and the partner
      drops its worst-of-type to absorb the incoming player.

    In the pick case, if the partner has no droppable player of the sent player's
    type, nothing is dropped and the partner grows by one -- you cannot drop what
    does not exist. The extra body is BENCHED (accepted as second-order) rather
    than left in the active slot it held on the user's roster, which would hand
    the partner a lineup spot the trade did not create.

    Every body that changes teams is re-slotted into the spot it fills, via
    :func:`place_in_slot` -- see there for why carrying a slot across a trade is
    not safe.

    Both rosters are rebuilt by substituting the new body AT THE OUTGOING BODY'S
    INDEX, never by filtering and appending. The MC draws each team's per-player
    randomness as one block indexed by list position, so removing a player from
    the middle of a roster and appending his replacement at the end re-rolls
    every player after him -- which silently destroys the common random numbers
    :func:`this_year_impact` relies on to make its paired delta meaningful.
    """
    user = inputs.user_team_name
    user_roster = inputs.team_rosters[user]
    partner_roster = inputs.team_rosters[partner]

    if received is not None:
        if not any(p is received for p in partner_roster):
            # Otherwise the partner silently grows by one and the two sides of the
            # comparison are no longer the same size, which quietly biases the delta.
            raise ValueError(
                f"{received.name!r} is not on {partner!r}'s roster; a two-way trade has "
                "to move a player who is actually there"
            )
        into_user = place_in_slot(received, sent.selected_position)
        into_partner = place_in_slot(sent, received.selected_position)
        out_of_partner: Player | None = received
    else:
        into_user = build_replacement_filler(sent)
        drop = worst_of_type(partner_roster, sent.player_type, inputs.denoms)
        into_partner = place_in_slot(sent, drop.selected_position if drop else Position.BN)
        out_of_partner = drop

    new_user = [into_user if p is sent else p for p in user_roster]
    if out_of_partner is None:
        # Nothing to drop: the partner genuinely grows by one. Appending keeps
        # every existing body at its own index, so only the new tail differs.
        new_partner = [*partner_roster, into_partner]
    else:
        new_partner = [into_partner if p is out_of_partner else p for p in partner_roster]

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
) -> dict[str, Any]:
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
    received: Player | None = None,
    n_iter: int = 2000,
    seed: int = 42,
) -> ThisYearImpact:
    """Baseline vs post-trade ROS MC for the user, with common random numbers.

    Full swing: the sent player leaves the user and joins the partner. With
    ``received`` the real incoming player fills the slot; without it, a
    replacement-level filler does. Reports the user's overall win% and top-3%
    and per-category first%/top-3%.
    """
    user = inputs.user_team_name
    base = run_scenario(inputs, inputs.team_rosters, n_iter, seed)
    scen_rosters = build_trade_scenario(inputs, sent, partner, received=received)
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


@dataclass(frozen=True)
class TradePickResult:
    sent_name: str
    partner: str
    this_year: ThisYearImpact
    # Both picks are optional and independent: a trade can buy a pick, sell one,
    # swap two, or involve no draft capital at all (a straight player-for-player
    # swap). `None` means that leg is not part of the deal -- the report must then
    # say nothing about it rather than invent one.
    next_year: NextYearValue | None = None
    received_name: str | None = None
    sent_pick: NextYearValue | None = None

    @property
    def net_pick_var(self) -> float:
        """Incoming pick value minus outgoing. 0.0 when no picks change hands."""
        got = self.next_year.expected_var if self.next_year is not None else 0.0
        out = self.sent_pick.expected_var if self.sent_pick is not None else 0.0
        return got - out


def resolve_partner(team_rosters: dict[str, list[Player]], to: str, user: str) -> str:
    """Resolve the trade partner's team name (normalized match). Rejects the
    user's own team and an unknown name (listing valid teams)."""
    target = normalize_name(to)
    if normalize_name(user) == target:
        raise ValueError("You cannot trade to yourself; pick a different --to team.")
    for name in team_rosters:
        if normalize_name(name) == target:
            return name
    valid = ", ".join(sorted(t for t in team_rosters if t != user))
    raise ValueError(f"{to!r} is not a team in this league. Valid partners: {valid}")


def compute_trade_pick(
    send: str,
    to: str,
    pick_round: int | None = None,
    *,
    receive: str | None = None,
    send_pick_round: int | None = None,
    player_type: str | None = None,
    receive_player_type: str | None = None,
    pick_slot: str = "round",
    n_iter: int = 2000,
    seed: int = 42,
    config_path: Path | None = None,
) -> TradePickResult:
    """Load stored state, compute both halves, and return the combined result.

    `receive` names a player coming back from the partner, making this a real
    two-way swap rather than a player-for-pick sale. `send_pick_round` names a
    pick going the other way, so the reported next-year value is the NET of the
    two picks instead of the gross value of the one received.

    `pick_round` is optional: a straight player-for-player swap involves no draft
    capital, and forcing a round on it would have the report credit the deal with
    VAR that does not exist. Something has to come back, though -- a player, a
    pick, or both.
    """
    if receive is None and pick_round is None:
        raise ValueError(
            "This trade has nothing coming back. Pass --receive (a player), "
            "--pick-round (a pick), or both."
        )
    cfg_path = config_path or _CONFIG_PATH
    inputs = load_mc_inputs_from_upstash(cfg_path)
    config = load_config(cfg_path)
    partner = resolve_partner(inputs.team_rosters, to, inputs.user_team_name)
    sent = find_sent_player(inputs.team_rosters[inputs.user_team_name], send, player_type)
    got = (
        find_sent_player(inputs.team_rosters[partner], receive, receive_player_type)
        if receive
        else None
    )
    this_year = this_year_impact(inputs, sent, partner, received=got, n_iter=n_iter, seed=seed)
    # `is not None`, not truthiness: round 0 is falsy but is not "no pick", and
    # silently dropping that leg would mis-state the net. It is not a legal round
    # either, so let pick_ordinal_range reject it loudly.
    nxt = next_year_value(config, pick_round, pick_slot) if pick_round is not None else None
    out_pick = (
        next_year_value(config, send_pick_round, pick_slot) if send_pick_round is not None else None
    )
    return TradePickResult(
        sent.name,
        partner,
        this_year,
        nxt,
        received_name=got.name if got else None,
        sent_pick=out_pick,
    )


def _kp(x: float) -> str:
    """Keeper-par render: 'n/a' for NaN, else one-decimal VAR."""
    return "n/a" if x != x else f"{x:.1f}"


def render_report(result: TradePickResult) -> str:
    ty = result.this_year
    ny = result.next_year
    sp = result.sent_pick
    dwin = ty.new_win - ty.base_win
    dtop3 = ty.new_top3 - ty.base_top3
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("TRADE CALCULATOR")
    send_parts = [result.sent_name]
    if sp:
        send_parts.append(f"2027 R{sp.nominal_round} pick")
    lines.append(f"  Send:    {' + '.join(send_parts)}  ->  {result.partner}")
    recv_parts = []
    if result.received_name:
        recv_parts.append(result.received_name)
    if ny:
        recv_parts.append(f"2027 Round {ny.nominal_round} pick")
    keeper_note = (
        f"  (keeper rounds: {ny.keeper_rounds}  ->  drafted round {ny.drafted_round})" if ny else ""
    )
    lines.append(f"  Receive: {' + '.join(recv_parts)}{keeper_note}")
    lines.append("=" * 72)

    lines.append("")
    swap = (
        f"1. THIS YEAR -- {result.sent_name} out, {result.received_name} in "
        f"(swap with {result.partner})"
        if result.received_name
        else f"1. THIS YEAR WITHOUT {result.sent_name}  (full swing: joins {result.partner})"
    )
    lines.append(swap)
    lines.append("-" * 72)
    lines.append(f"  Win%   : {ty.base_win:5.1f}%  ->  {ty.new_win:5.1f}%   ({dwin:+.1f})")
    lines.append(f"  Top-3% : {ty.base_top3:5.1f}%  ->  {ty.new_top3:5.1f}%   ({dtop3:+.1f})")
    lines.append("")
    lines.append("  Per-category odds (your team):")
    lines.append(
        f"    {'Cat':<5}{'1st base':>9}{'1st new':>9}{'d1st':>7}"
        f"{'top3 base':>11}{'top3 new':>10}{'dTop3':>8}"
    )
    for c in ty.categories:
        d1 = c.new_first - c.base_first
        d3 = c.new_top3 - c.base_top3
        lines.append(
            f"    {c.category:<5}{c.base_first:>9.1f}{c.new_first:>9.1f}{d1:>+7.1f}"
            f"{c.base_top3:>11.1f}{c.new_top3:>10.1f}{d3:>+8.1f}"
        )

    # Section 2 exists only if draft capital actually moves. A straight
    # player-for-player swap has none, and printing a pick section anyway would
    # credit the deal with VAR that is not in it.
    if ny or sp:
        lines.append("")
        if ny and sp:
            label = "pick swap"
        elif ny:
            label = f"extra 2027 pick (drafted round {ny.drafted_round})"
        else:
            assert sp is not None  # the enclosing `if ny or sp` and `elif ny` leave only this
            label = f"2027 pick given up (drafted round {sp.drafted_round})"
        lines.append(f"2. NEXT YEAR -- {label}")
        lines.append("-" * 72)
        if ny and sp:
            lines.append(
                f"  Receive R{ny.nominal_round} (drafted {ny.drafted_round}) : "
                f"~{ny.expected_var:.2f} VAR"
            )
        if sp:
            lines.append(
                f"  Send    R{sp.nominal_round} (drafted {sp.drafted_round}) : "
                f"~{sp.expected_var:.2f} VAR"
            )
            lines.append(f"  NET pick value      : ~{result.net_pick_var:+.2f} VAR")
        elif ny:
            lines.append(f"  Expected pick value : ~{ny.expected_var:.1f} VAR")
        if ny:
            lines.append(
                f"  (early in the round : ~{ny.early_var:.1f} VAR ; "
                f"keeper-average keeper : ~{_kp(ny.keeper_par)} VAR)"
            )
        lines.append("  VAR is value above a replacement roster spot, so this is roughly the")
        lines.append("  pick's marginal roto-point value next year.")
        lines.append("  Estimate = the 2026 draft-day value distribution at that slot; the")
        lines.append("  specific 2027 player is unknown.")

    lines.append("")
    lines.append("-" * 72)
    # NET, never the gross incoming pick: a swap that receives 3.76 and sends 1.12
    # is worth 2.64, and the verdict is the one line a user decides on.
    pick_clause = f" for pick value worth ~{result.net_pick_var:+.2f} VAR" if (ny or sp) else ""
    if dwin < 0:
        # dtop3 stays SIGNED: it moves independently of win%, and often the other
        # way (trading a high-variance star lowers the ceiling and raises the
        # floor). abs() here read a top-3 GAIN as a loss, contradicting the signed
        # per-category table above in the one line a user acts on.
        lines.append(
            f"You give up ~{abs(dwin):.1f} win% ({dtop3:+.1f} top-3%) this year{pick_clause}."
        )
    else:
        neutral = (
            f"This year is roughly neutral to positive ({dwin:+.1f} win% / {dtop3:+.1f} top-3%)"
        )
        lines.append(
            f"{neutral}, and pick value is worth ~{result.net_pick_var:+.2f} VAR."
            if (ny or sp)
            else f"{neutral}."
        )
    lines.append(
        f"MC: n_iter={ty.n_iter}, seed={ty.seed} (common random numbers across both runs)."
    )
    return "\n".join(lines)
