from datetime import date

import pytest

from fantasy_baseball.lineup.il_return_planner import _activate
from fantasy_baseball.lineup.stash_value import (
    StashResult,
    StashScore,
    _cost_and_drop,
    _marginal_value,
    _open_il_slots,
    _owned_il_stashes,
    _solve_active,
    score_stash_candidates,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.scoring import build_team_sds

EFFECTIVE_DATE = date(2026, 4, 1)

TEAM_NAME = "Test Team"

# Roster slots: 9 hitter slots (C,1B,2B,3B,SS,OF*3,UTIL) + 9 P + 1 BN + 2 IL.
# Nine P slots so the elite arm must crack the top-9 active pitchers.
STASH_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
    "P": 9,
    "BN": 1,
    "IL": 2,
}


def _hitter(name, positions, slot=None, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=HitterStats(
            pa=int(stats.get("ab", 500) * 1.15),
            ab=stats.get("ab", 500),
            h=stats.get("h", 130),
            r=stats.get("r", 70),
            hr=stats.get("hr", 20),
            rbi=stats.get("rbi", 70),
            sb=stats.get("sb", 5),
            avg=stats.get("avg", 0.260),
        ),
        selected_position=Position.parse(slot) if slot else None,
    )


def _arm(name, *, ip, k, slot=None, status="", w=6.0, sv=0.0, er=30.0, bb=20.0, h_allowed=55.0):
    """A pitcher with explicit IP and K. Other rate inputs default so the
    staff shares near-identical ER/BB/H_allowed and K is the lever."""
    era = 9.0 * er / ip if ip else 0.0
    whip = (bb + h_allowed) / ip if ip else 0.0
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        rest_of_season=PitcherStats(
            ip=ip,
            w=w,
            k=k,
            sv=sv,
            er=er,
            bb=bb,
            h_allowed=h_allowed,
            era=era,
            whip=whip,
        ),
        selected_position=Position.parse(slot) if slot else None,
        status=status,
    )


def _full_hitters():
    specs = [
        ("C1", ["C"]),
        ("1B1", ["1B"]),
        ("2B1", ["2B"]),
        ("3B1", ["3B"]),
        ("SS1", ["SS"]),
        ("OFa", ["OF"]),
        ("OFb", ["OF"]),
        ("OFc", ["OF"]),
        ("UT", ["1B"]),
    ]
    out = []
    for name, pos in specs:
        h = _hitter(name, pos, r=75, hr=22, rbi=75, sb=8, avg=0.275, ab=520, h=143)
        h.selected_position = Position.parse(pos[0])
        out.append(h)
    return out


def _make_elite_low_ip_pitcher():
    """~90 IP, sub-3 ERA, high K/9 -- a strikeout monster on limited volume.

    K=130 over 90 IP (13 K/9) beats every mediocre staff arm's K, so adding
    him flips the contended K category and improves the optimized lineup.
    """
    return _arm("Elite", ip=90.0, k=130.0, status="IL15", er=28.0, bb=22.0, h_allowed=58.0)


def _make_replacement_level_pitcher():
    """Worse than every active arm: low IP AND fewest strikeouts. He cannot
    crack the top-9, so he stays in the surplus and his Gain is floored at 0."""
    return _arm("Scrub", ip=40.0, k=20.0, status="IL15", er=20.0, bb=18.0, h_allowed=42.0)


def _opponent_roster(label, *, k_per_arm):
    """A simple opponent: 9 full hitters + 9 mediocre arms (no IL).

    K per arm is tuned (via ``_coupled_standings``) so the user's K category
    sits on a contested margin -- the swap deltas in these tests flip the K
    standings point against exactly one opponent."""
    hitters = []
    specs = [
        ("C", ["C"]),
        ("1B", ["1B"]),
        ("2B", ["2B"]),
        ("3B", ["3B"]),
        ("SS", ["SS"]),
        ("OF1", ["OF"]),
        ("OF2", ["OF"]),
        ("OF3", ["OF"]),
        ("UT", ["1B"]),
    ]
    for nm, pos in specs:
        h = _hitter(f"{label} {nm}", pos, r=72, hr=20, rbi=72, sb=6, avg=0.265, ab=510, h=135)
        h.selected_position = Position.parse(pos[0])
        hitters.append(h)
    arms = [
        _arm(f"{label} SP{i}", ip=150.0, k=k_per_arm, slot="P", er=52.0, bb=40.0, h_allowed=130.0)
        for i in range(1, 10)
    ]
    return [*hitters, *arms]


def _coupled_standings(user_roster):
    """Standings whose USER row is built from ``user_roster`` (coupled).

    Built via ``ProjectedStandings.from_rosters`` so the user's row reflects
    the actual roster -- including any owned IL players, whose ROS the
    displacement model already prices into the anchor (displacement slots the
    high-K stashes in and zeroes the worst mediocre arms). This is the honest
    setup that exposes the double-count bug; a hardcoded ``from_json`` row
    decoupled from the roster would hide it.

    Two mediocre opponents at K totals 710 (Opp A) and 735 (Opp B) bracket the
    user's K so every swap in these tests is decided by the K standings point:

      * No-IL user K=675 (behind both). The elite FA add lifts K to ~734 ->
        passes Opp A only -> +1 point (gain > 0). A scrub can't crack the nine
        (gain 0).
      * IL-full user K=747 (ahead of both, displacement counts Weak+Strong).
        Dropping Weak (K 100) drops K to ~718 -> still beats Opp A, loses Opp B
        -> cost +1. Dropping Strong (K 115) drops K to ~703 -> beats neither ->
        cost +2. So Weak is strictly the cheapest drop (recommended)."""
    rosters = {
        TEAM_NAME: user_roster,
        "Opp A": _opponent_roster("A", k_per_arm=710.0 / 9.0),
        "Opp B": _opponent_roster("B", k_per_arm=735.0 / 9.0),
    }
    return ProjectedStandings.from_rosters(rosters, EFFECTIVE_DATE, fraction_remaining=0.5)


def _mediocre_staff():
    """Nine mediocre arms, all roughly equal IP/ER/BB/H but distinct, modest K."""
    return [
        _arm(f"SP{i}", ip=150.0, k=70.0 + i, slot="P", er=52.0, bb=40.0, h_allowed=130.0)
        for i in range(1, 10)
    ]


@pytest.fixture
def stash_fixture():
    """Return ``(roster, standings, team_sds, roster_slots, team_name)``.

    Roster shape: 9 hitters filling all hitter slots + 9 mediocre starting
    pitchers (K ranging 71..79 over ~150 IP). The nine P slots are exactly
    filled by the nine mediocre arms, so:
      * an ELITE high-K arm (K=130) beats the weakest mediocre arm and cracks
        the active nine -> positive Gain.
      * a REPLACEMENT-level arm (K=20) is worse than all nine -> stays in the
        surplus, after_active == before_active, band mean ~0 -> Gain 0.

    The standings are COUPLED to the roster via ``from_rosters`` (no IL players
    here, so the user's row is just the nine mediocre arms + nine hitters)."""
    roster = [*_full_hitters(), *_mediocre_staff()]
    standings = _coupled_standings(roster)
    team_sds = None
    return roster, standings, team_sds, STASH_SLOTS, TEAM_NAME


@pytest.fixture
def stash_fixture_il_full():
    """A roster with two owned IL stashes filling both IL slots.

    Both IL stashes share the elite FA's IP/ER/BB/H_allowed so K is the only
    lever. K tuning (mediocre active arms 71..79 < Weak 100 < Strong 115 <
    elite FA 130) makes BOTH stashes crack the active nine when activated
    while keeping Weak < Strong. Returns a 6-tuple ending in the weak stash's
    name.

    The standings are COUPLED via ``from_rosters`` over the FULL roster
    (including the two IL stashes), so the displacement model already prices
    their ROS into the user's anchor -- the honest setup that exercises the
    owned-IL drop-cost path."""
    weak = _arm(
        "Weak Stash",
        ip=90.0,
        k=100.0,
        slot="IL",
        status="IL15",
        er=28.0,
        bb=22.0,
        h_allowed=58.0,
    )
    strong = _arm(
        "Strong Stash",
        ip=90.0,
        k=115.0,
        slot="IL",
        status="IL15",
        er=28.0,
        bb=22.0,
        h_allowed=58.0,
    )
    roster = [*_full_hitters(), *_mediocre_staff(), weak, strong]
    standings = _coupled_standings(roster)
    return roster, standings, None, STASH_SLOTS, TEAM_NAME, weak.name


def _band_mean(roster, candidate, standings, team_name, slots, sds):
    before = _solve_active(roster, slots, standings, team_name, sds)
    return _marginal_value(
        candidate,
        before_active=before,
        projected_standings=standings,
        team_name=team_name,
        team_sds=sds,
        fraction_remaining=0.5,
    )


def test_elite_low_volume_arm_has_positive_gain(stash_fixture):
    roster, standings, sds, slots, team = stash_fixture
    elite = _make_elite_low_ip_pitcher()  # ~90 IP, sub-3 ERA, high K/9
    gain = _band_mean(roster, elite, standings, team, slots, sds)
    assert gain > 0.0


def test_scrub_arm_gain_is_floored_at_zero(stash_fixture):
    roster, standings, sds, slots, team = stash_fixture
    scrub = _make_replacement_level_pitcher()  # worse than every active arm
    gain = _band_mean(roster, scrub, standings, team, slots, sds)
    assert gain == 0.0


def test_activate_clears_il_signals():
    p = _arm("IL Guy", ip=90.0, k=130.0, slot="IL", status="IL15")
    cleared = _activate(p)
    assert cleared.status == ""
    assert cleared.selected_position is None
    assert cleared.name == "IL Guy"
    # Original untouched (dataclasses.replace returns a copy).
    assert p.status == "IL15"


def test_stash_score_to_dict_shape():
    s = StashScore(
        name="Blake Snell",
        player_type="pitcher",
        status="IL15",
        owned=False,
        gain=4.2,
        cost=0.0,
        stash_value=4.2,
        band={"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
        recommended_drop=None,
    )
    d = s.to_dict()
    assert d["name"] == "Blake Snell"
    assert d["stash_value"] == 4.2
    assert d["band"]["verdict"] == "real"
    assert d["recommended_drop"] is None


def test_stash_result_to_dict_shape():
    r = StashResult(open_il_slots=1, cutline_rank=2, candidates=[], warning=None)
    d = r.to_dict()
    assert d["open_il_slots"] == 1
    assert d["cutline_rank"] == 2
    assert d["candidates"] == []
    assert d["warning"] is None


@pytest.fixture
def monkeypatched_il_roster(stash_fixture):
    roster, *_ = stash_fixture
    injured = _arm(
        "Injured Owned Arm", ip=80.0, k=95.0, status="IL15", er=25.0, bb=20.0, h_allowed=55.0
    )
    return [*roster, injured]


def test_open_slot_stash_value_equals_gain_and_no_drop(stash_fixture):
    roster, standings, sds, slots, team = stash_fixture  # empty IL
    elite_fa = _make_elite_low_ip_pitcher()
    result = score_stash_candidates(
        roster=roster,
        free_agents=[elite_fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    top = result.candidates[0]
    assert top.name == elite_fa.name
    assert top.cost == 0.0
    assert top.stash_value == top.gain > 0.0
    assert top.recommended_drop is None


def test_il_full_upgrade_recommends_dropping_weakest_stash(stash_fixture_il_full):
    # roster has 2 IL stashes: a strong one and a weak one; a better FA exists.
    roster, standings, sds, slots, team, weak_stash_name = stash_fixture_il_full
    better_fa = _make_elite_low_ip_pitcher()
    result = score_stash_candidates(
        roster=roster,
        free_agents=[better_fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    fa_row = next(c for c in result.candidates if c.name == better_fa.name)
    assert fa_row.recommended_drop == weak_stash_name
    assert fa_row.cost > 0.0
    assert fa_row.stash_value == fa_row.gain - fa_row.cost


def test_owned_candidates_have_zero_cost(stash_fixture_il_full):
    """An owned player already holds his IL slot, so he pays no acquisition
    cost: stash_value == gain, no recommended drop. (Regression: previously the
    FA-acquisition cost was charged to owned players, yielding a confusing
    negative stash_value even for a held arm.)"""
    roster, standings, sds, slots, team, _weak = stash_fixture_il_full
    result = score_stash_candidates(
        roster=roster,
        free_agents=[],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    owned = [c for c in result.candidates if c.owned]
    assert owned, "expected owned IL candidates in the fixture"
    for c in owned:
        assert c.cost == 0.0, f"{c.name}: owned cost should be 0, got {c.cost}"
        assert c.stash_value == c.gain
        assert c.recommended_drop is None


def test_no_injured_players_returns_empty_board(stash_fixture):
    """No owned IL players and no injured FAs -> empty board, no optimizer work."""
    roster, standings, sds, slots, team = stash_fixture  # all healthy
    result = score_stash_candidates(
        roster=roster,
        free_agents=[],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    assert result.candidates == []
    assert result.cutline_rank == slots.get("IL", 0)


def test_open_il_slots_counts_true_il_slots_only(stash_fixture):
    roster, *_ = stash_fixture
    # With an empty IL, both IL slots are open.
    assert _open_il_slots(roster, {"IL": 2}) == 2


def test_owned_il_stashes_uses_is_on_il(monkeypatched_il_roster):
    roster = monkeypatched_il_roster  # one player with status="IL15"
    names = {p.name for p in _owned_il_stashes(roster)}
    assert "Injured Owned Arm" in names


def test_owned_and_fa_player_get_equal_gain():
    """Unification + double-count guard: a player's stash gain must NOT depend
    on whether you already own him. Every candidate is valued against the
    shared healthy-active-lineup baseline (which excludes ALL IL players), so
    scoring an elite closer as an owned IL stash matches scoring the SAME
    closer as a free agent.

    Replaces the old drop-cost regression. If owned players were re-added on
    top of a baseline that already counted them, the owned gain would inflate
    and diverge from the FA gain -- so this equality is also the double-count
    guard. ``team_sds=None`` isolates the scoring path from the (legitimately
    different) per-team SDs of the two rosters.
    """
    closer = _arm(
        "Stash Closer",
        ip=60.0,
        k=80.0,
        sv=30.0,
        status="IL15",
        er=18.0,
        bb=14.0,
        h_allowed=46.0,
    )
    base = [*_moderate_hitters(), *_mediocre_staff()]

    def gain_for(user_roster, free_agents):
        rosters = {
            TEAM_NAME: user_roster,
            "Opp A": _opponent_roster("A", k_per_arm=720.0 / 9.0),
            "Opp B": _opponent_roster("B", k_per_arm=720.0 / 9.0),
        }
        standings = ProjectedStandings.from_rosters(rosters, EFFECTIVE_DATE, fraction_remaining=0.5)
        result = score_stash_candidates(
            roster=user_roster,
            free_agents=free_agents,
            projected_standings=standings,
            roster_slots=STASH_SLOTS,
            team_name=TEAM_NAME,
            team_sds=None,
            fraction_remaining=0.5,
        )
        return next(c for c in result.candidates if c.name == "Stash Closer")

    fa_row = gain_for(base, [closer])  # closer as a free agent
    owned_row = gain_for([*base, closer], [])  # same closer, owned on the IL

    assert fa_row.owned is False and owned_row.owned is True
    assert fa_row.gain > 0.3  # elite closer into a contested SV cat is worth real points
    assert owned_row.gain == pytest.approx(fa_row.gain, abs=0.05)


# ---------------------------------------------------------------------------
# v2: rate-upgrade-over-return-window gain (FA path).
# See docs/superpowers/specs/2026-05-26-stash-value-rate-redesign-design.md
# ---------------------------------------------------------------------------


def _moderate_hitters():
    """Nine user hitters at the SAME level as the opponents (r72/hr20/rbi72/
    sb6/avg.265), so the user is on a coin-flip in every hitting category --
    a rate upgrade tips it. Mirrors the contested production board."""
    specs = [
        ("C1", ["C"]),
        ("1B1", ["1B"]),
        ("2B1", ["2B"]),
        ("3B1", ["3B"]),
        ("SS1", ["SS"]),
        ("OFa", ["OF"]),
        ("OFb", ["OF"]),
        ("OFc", ["OF"]),
        ("UT", ["1B"]),
    ]
    out = []
    for nm, pos in specs:
        h = _hitter(nm, pos, r=72, hr=20, rbi=72, sb=6, avg=0.265, ab=510, h=135)
        h.selected_position = Position.parse(pos[0])
        out.append(h)
    return out


def _contested_rosters():
    return {
        TEAM_NAME: [*_moderate_hitters(), *_mediocre_staff()],
        "Opp A": _opponent_roster("A", k_per_arm=720.0 / 9.0),
        "Opp B": _opponent_roster("B", k_per_arm=720.0 / 9.0),
    }


@pytest.fixture
def contested_fixture():
    """User tied with the field in every hitting category, EV-scored, so any
    real rate upgrade shows positive gain (no discrete-flip tuning needed)."""
    rosters = _contested_rosters()
    standings = ProjectedStandings.from_rosters(rosters, EFFECTIVE_DATE, fraction_remaining=0.5)
    team_sds = build_team_sds(rosters, sd_scale=0.5**0.5)
    return [*_moderate_hitters(), *_mediocre_staff()], standings, team_sds, STASH_SLOTS, TEAM_NAME


def _low_volume_high_rate_hitter():
    """120 ROS AB but elite per-AB rates. His SEASON TOTALS (12 HR) lose to a
    healthy starter's 20, so the old total-volume model never slots him
    (gain 0). The rate model credits his per-AB edge over the 120 AB he'll
    actually play once back."""
    h = _hitter("Rate Bat", ["1B"], ab=120, h=42, r=33, hr=12, rbi=36, sb=10, avg=0.350)
    h.status = "IL15"
    return h


def test_injured_fa_hitter_rate_upgrade_beats_volume(contested_fixture):
    """Core regression: a low-volume, high-rate injured FA hitter scores a
    positive gain even though his season totals trail a healthy starter."""
    roster, standings, sds, slots, team = contested_fixture
    fa = _low_volume_high_rate_hitter()

    result = score_stash_candidates(
        roster=roster,
        free_agents=[fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    row = next(c for c in result.candidates if c.name == "Rate Bat")

    # Old model: exactly 0 (12 season HR can't out-total a 20-HR starter).
    assert row.gain > 0.0
    assert row.band["sd"] > 0.0  # the lineup actually changed -- a real swap


def test_injured_fa_closer_rate_upgrade(contested_fixture):
    """SV is just another per-volume rate: a low-IP, high-SV-rate closer beats
    a starter who gets no saves over the closer's return window."""
    roster, standings, sds, slots, team = contested_fixture
    closer = _arm(
        "Stash Closer", ip=40.0, k=55.0, sv=18.0, status="IL15", er=12.0, bb=12.0, h_allowed=28.0
    )

    result = score_stash_candidates(
        roster=roster,
        free_agents=[closer],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    row = next(c for c in result.candidates if c.name == "Stash Closer")
    assert row.gain > 0.0


def test_uncontested_fa_hitter_upgrade_is_zero(stash_fixture):
    """Leverage still gates: when the user already wins the hitting cats
    outright (.275/22HR vs the field's .265/20HR), a better-rate FA adds no
    marginal roto points -> gain 0."""
    roster, standings, sds, slots, team = stash_fixture
    fa = _low_volume_high_rate_hitter()

    result = score_stash_candidates(
        roster=roster,
        free_agents=[fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    row = next(c for c in result.candidates if c.name == "Rate Bat")
    assert row.gain == 0.0


def test_fa_cost_floored_at_zero():
    """Displacing a net-negative owned IL stash must not CREDIT the candidate.
    Cost floors at 0 -- kills the uniform -0.12 -> +0.12 artifact."""
    weak = _arm("Weak Stash", ip=40.0, k=20.0, slot="IL", status="IL15")
    fa = _arm("Some FA", ip=60.0, k=80.0, status="IL15")

    cost, drop = _cost_and_drop(
        fa,
        gain_by_name={"Weak Stash": -0.12},
        roster=[weak],
        roster_slots={"IL": 1, "P": 9, "BN": 1},
    )
    assert cost == 0.0
    assert drop == "Weak Stash"
