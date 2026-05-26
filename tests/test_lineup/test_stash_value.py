from datetime import date

import pytest

from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band
from fantasy_baseball.lineup.stash_value import (
    StashResult,
    StashScore,
    _activate,
    _counted_pool,
    _marginal_value,
    _open_il_slots,
    _owned_il_stashes,
    _solve_active,
    score_stash_candidates,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import ProjectedStandings

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
        roster=roster,
        roster_slots=slots,
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


def test_open_il_slots_counts_true_il_slots_only(stash_fixture):
    roster, *_ = stash_fixture
    # With an empty IL, both IL slots are open.
    assert _open_il_slots(roster, {"IL": 2}) == 2


def test_owned_il_stashes_uses_is_on_il(monkeypatched_il_roster):
    roster = monkeypatched_il_roster  # one player with status="IL15"
    names = {p.name for p in _owned_il_stashes(roster)}
    assert "Injured Owned Arm" in names


def test_owned_strong_il_arm_gain_is_drop_cost_not_double_count():
    """Regression (spec line 293): an owned IL arm strong enough to crack the
    active nine is ALREADY in the from_rosters standings anchor (displacement
    counts his ROS). His reported Gain must be the il_return-style DROP-COST --
    NOT an inflated add-gain re-applied on top of a row that already has him.

    Pre-fix, ``_marginal_band`` computed (baseline) -> (baseline + him) for
    every candidate, double-counting him for the owned path. This asserts the
    owned candidate's reported gain equals round(-drop_band.mean, 2), where
    drop_band measures activating-him-vs-dropping-him over the counted bodies.

    Opponent K totals (666, 720) are tuned so the double-count and the
    drop-cost DIVERGE: the IL arm displaces SP1 into the anchor at K~734, which
    already beats both opponents (2 pts). The pre-fix add re-applies him on top
    (overshoot to ~793) -> no extra point -> reported gain 0.0; the true
    drop-cost (anchor 734 -> drop him -> 675, losing to Opp B=720) is +1.0. So
    the pre-fix code reports 0.0 and this test goes RED; the fix reports +1.0
    and it goes GREEN.
    """
    strong_il = _arm(
        "Strong IL Arm",
        ip=90.0,
        k=130.0,
        slot="IL",
        status="IL15",
        er=28.0,
        bb=22.0,
        h_allowed=58.0,
    )
    user_roster = [*_full_hitters(), *_mediocre_staff(), strong_il]
    standings = ProjectedStandings.from_rosters(
        {
            TEAM_NAME: user_roster,
            "Opp A": _opponent_roster("A", k_per_arm=666.0 / 9.0),
            "Opp B": _opponent_roster("B", k_per_arm=720.0 / 9.0),
        },
        EFFECTIVE_DATE,
        fraction_remaining=0.5,
    )

    result = score_stash_candidates(
        roster=user_roster,
        free_agents=[],
        projected_standings=standings,
        roster_slots=STASH_SLOTS,
        team_name=TEAM_NAME,
        team_sds=None,
        fraction_remaining=0.5,
    )
    row = next(c for c in result.candidates if c.name == "Strong IL Arm")

    # Expected drop-cost, computed the il_return way: before = lineup WITH him
    # active (matches the anchor), after = baseline WITHOUT him.
    counted = _counted_pool(user_roster, exclude_name="Strong IL Arm")
    baseline_without = _solve_active(counted, STASH_SLOTS, standings, TEAM_NAME, None)
    lineup_with = _solve_active(
        [*counted, _activate(strong_il)], STASH_SLOTS, standings, TEAM_NAME, None
    )
    drop_band = compute_delta_roto_band(
        lineup_with,
        baseline_without,
        standings.field_stats(TEAM_NAME),
        TEAM_NAME,
        0.5,
        projected_standings=standings,
        team_sds=None,
    )
    expected_gain = round(-drop_band.mean, 2)

    # He genuinely cracks the nine, so the drop-cost is real (not ~0).
    assert expected_gain > 0.0
    assert row.gain == expected_gain
