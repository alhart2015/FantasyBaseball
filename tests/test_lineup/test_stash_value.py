from datetime import date

import pytest

from fantasy_baseball.lineup.il_return_planner import _activate
from fantasy_baseball.lineup.stash_value import (
    StashResult,
    StashScore,
    _active_lineup_standings,
    _assign_recommended_drops,
    _cap_candidates,
    _marginal_value,
    _open_il_slots,
    _owned_il_stashes,
    _rank_key,
    _solve_active,
    score_stash_candidates,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
    TeamYtdComponents,
)
from fantasy_baseball.scoring import build_team_sds
from fantasy_baseball.utils.constants import Category

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
    before = _solve_active(roster, slots, standings, team_name, sds, 0.5)
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
        stash_value=4.2,
        band={"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
        recommended_drop=None,
    )
    d = s.to_dict()
    assert d["name"] == "Blake Snell"
    assert d["stash_value"] == 4.2
    assert d["band"]["p_positive"] == 0.91
    assert d["recommended_drop"] is None
    # cost and gain were retired in v3.
    assert "cost" not in d
    assert "gain" not in d


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


def test_open_slot_positive_value_and_no_drop(stash_fixture):
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
    assert top.stash_value > 0.0
    assert top.stash_value == top.band["mean"]  # Value is the band mean
    assert top.recommended_drop is None  # open slot -> nothing to drop


def test_il_full_upgrade_recommends_dropping_weakest_stash(stash_fixture_il_full):
    # roster has 2 owned IL stashes (Weak < Strong); a better FA exists, IL full.
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
    # The FA earns a slot by bumping the WEAKEST owned stash (the one that falls
    # below the cutline), not the stronger one that keeps its slot. Asserted
    # against the fixture's known weak stash -- an independently-correct answer,
    # not a re-derivation of the production selection rule.
    assert fa_row.recommended_drop == weak_stash_name
    assert fa_row.stash_value == fa_row.band["mean"]
    # The stronger owned stash stays above the cutline and is never dropped.
    strong_row = next(c for c in result.candidates if c.owned and c.name != weak_stash_name)
    assert strong_row.recommended_drop is None


def test_owned_candidates_have_no_recommended_drop(stash_fixture_il_full):
    """An owned player already holds his IL slot, so there is nothing to drop
    to keep him and no cost to charge: his Value is just the band mean and his
    recommended_drop is None. (v3: cost retired entirely.)"""
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
        assert c.stash_value == c.band["mean"]
        assert c.recommended_drop is None


def test_multiple_above_cutline_fas_get_distinct_drops(stash_fixture_il_full):
    """Two above-cutline FAs with a full IL each bump a DISTINCT owned stash.

    Regression: the old single shared drop named the one weakest owned stash for
    EVERY FA, implying both FAs could be added by freeing the same slot."""
    roster, standings, sds, slots, team, _weak = stash_fixture_il_full
    elite1 = _arm("Elite One", ip=90.0, k=130.0, status="IL15", er=28.0, bb=22.0, h_allowed=58.0)
    elite2 = _arm("Elite Two", ip=90.0, k=125.0, status="IL15", er=28.0, bb=22.0, h_allowed=58.0)
    result = score_stash_candidates(
        roster=roster,
        free_agents=[elite1, elite2],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    above = result.candidates[: result.cutline_rank]
    assert {c.name for c in above} == {"Elite One", "Elite Two"}  # both earn a slot
    fa_drops = {c.recommended_drop for c in result.candidates if not c.owned}
    # Each FA bumps a different owned stash -- both owned stashes, not one twice.
    assert fa_drops == {"Weak Stash", "Strong Stash"}
    # WORST-first: the top FA bumps the WEAKEST owned stash, so grabbing just him
    # costs the least; the next FA bumps the stronger one.
    assert result.candidates[0].name == "Elite One"
    assert result.candidates[0].recommended_drop == "Weak Stash"
    assert result.candidates[1].recommended_drop == "Strong Stash"


def test_below_cutline_fa_has_no_drop(stash_fixture_il_full):
    """A FA that misses the cutline is not worth adding, so it shows no drop.

    Regression: the old shared drop labeled every FA -- even sub-cutline ones --
    with the weakest owned stash, advising a drop for a player the board's own
    cutline says not to add."""
    roster, standings, sds, slots, team, _weak = stash_fixture_il_full
    scrub_fa = _make_replacement_level_pitcher()  # worse than both owned stashes
    result = score_stash_candidates(
        roster=roster,
        free_agents=[scrub_fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    scrub_row = next(c for c in result.candidates if c.name == scrub_fa.name)
    assert result.candidates.index(scrub_row) >= result.cutline_rank  # below the cutline
    assert scrub_row.recommended_drop is None


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
    assert fa_row.stash_value > 0.3  # elite closer into a contested SV cat is worth real points
    assert owned_row.stash_value == pytest.approx(fa_row.stash_value, abs=0.05)
    # Equal Value AND equal P(helps): same band whether owned or FA.
    assert owned_row.band["p_positive"] == pytest.approx(fa_row.band["p_positive"], abs=0.02)


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
    assert row.stash_value > 0.0
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
    assert row.stash_value > 0.0


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
    assert row.stash_value == 0.0


def _score(name, *, owned, p_positive, mean=1.0):
    """A minimal StashScore for drop-assignment / cap unit tests."""
    return StashScore(
        name=name,
        player_type="pitcher",
        status="IL15",
        owned=owned,
        stash_value=mean,
        band={"mean": mean, "sd": 1.0, "p_positive": p_positive, "verdict": "real"},
        recommended_drop=None,
    )


def test_assign_drops_top_fa_bumps_weakest_owned():
    """IL full, two above-cutline FAs, two below-cutline owned stashes: each FA
    bumps a DISTINCT owned stash, WORST-first -- the top FA bumps the weakest
    owned stash (so adding just him costs the least), the next FA the next-
    weakest. Owned rows carry no drop."""
    scores = [
        _score("FA Best", owned=False, p_positive=0.95),
        _score("FA Next", owned=False, p_positive=0.90),
        _score("Owned Stronger", owned=True, p_positive=0.60),
        _score("Owned Weaker", owned=True, p_positive=0.55),
    ]
    _assign_recommended_drops(scores, cutline_rank=2, open_il_slots=0)
    # Top FA -> weakest owned (not the stronger one ranked just above it).
    assert [s.recommended_drop for s in scores] == [
        "Owned Weaker",
        "Owned Stronger",
        None,
        None,
    ]


def test_assign_drops_skips_above_cutline_owned_keeper():
    """An owned stash above the cutline is a keeper and is never named as a
    drop; the FA bumps a below-cutline owned stash instead."""
    scores = [
        _score("Owned Keeper", owned=True, p_positive=0.95),  # above cutline -> keep
        _score("FA", owned=False, p_positive=0.80),  # above cutline -> needs slot
        _score("Owned Weak", owned=True, p_positive=0.40),  # below cutline -> droppable
    ]
    _assign_recommended_drops(scores, cutline_rank=2, open_il_slots=0)
    fa = next(s for s in scores if s.name == "FA")
    assert fa.recommended_drop == "Owned Weak"  # not the above-cutline keeper
    keeper = next(s for s in scores if s.name == "Owned Keeper")
    assert keeper.recommended_drop is None


def test_assign_drops_open_slot_means_no_drop():
    """An above-cutline FA that fills an open IL slot needs no drop; only the FA
    beyond the open slots bumps an owned stash."""
    scores = [
        _score("FA One", owned=False, p_positive=0.95),
        _score("FA Two", owned=False, p_positive=0.90),
        _score("Owned Weak", owned=True, p_positive=0.40),
    ]
    _assign_recommended_drops(scores, cutline_rank=2, open_il_slots=1)
    fa_one = next(s for s in scores if s.name == "FA One")
    fa_two = next(s for s in scores if s.name == "FA Two")
    assert fa_one.recommended_drop is None  # fills the open slot
    assert fa_two.recommended_drop == "Owned Weak"  # no slot left -> bump


def test_assign_drops_below_cutline_fa_gets_none():
    """A FA below the cutline is not worth adding, so it carries no drop."""
    scores = [
        _score("Owned A", owned=True, p_positive=0.95),
        _score("Owned B", owned=True, p_positive=0.90),
        _score("FA Sub", owned=False, p_positive=0.30),  # below the 2-slot cutline
    ]
    _assign_recommended_drops(scores, cutline_rank=2, open_il_slots=0)
    fa = next(s for s in scores if s.name == "FA Sub")
    assert fa.recommended_drop is None


def test_cap_candidates_keeps_all_owned_even_past_cap():
    """Capping must not hide an owned stash -- it can be another row's
    recommended_drop, so it stays visible. FAs are capped; owned are always
    kept; the result never exceeds the cap."""
    scores = [_score(f"FA {i}", owned=False, p_positive=0.9 - i * 0.01) for i in range(5)] + [
        _score("Owned Tail", owned=True, p_positive=0.10)
    ]
    capped = _cap_candidates(scores, max_candidates=3)
    names = [s.name for s in capped]
    assert "Owned Tail" in names  # owned retained despite ranking past the cap
    assert sum(1 for s in capped if not s.owned) == 2  # FAs capped to (3 - 1 owned)
    assert len(capped) == 3


def test_rank_key_orders_by_p_helps_then_value():
    """Board sorts by P(helps) first: a lower-Value, higher-P(helps) candidate
    ranks ABOVE a higher-Value, lower-P(helps) one (NOT the same order as
    sorting by Value -- see the v3 design doc)."""
    big_shaky = StashScore(
        name="Big Shaky",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=2.0,
        band={"mean": 2.0, "sd": 4.0, "p_positive": 0.69, "verdict": "lean"},
        recommended_drop=None,
    )
    small_sure = StashScore(
        name="Small Sure",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=1.0,
        band={"mean": 1.0, "sd": 0.5, "p_positive": 0.98, "verdict": "real"},
        recommended_drop=None,
    )
    ranked = sorted([big_shaky, small_sure], key=_rank_key, reverse=True)
    assert [s.name for s in ranked] == ["Small Sure", "Big Shaky"]


def test_rank_key_breaks_ties_by_value():
    """Equal P(helps) -> higher Value ranks first (deterministic tie-break)."""
    low_val = StashScore(
        name="Low Val",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=1.0,
        band={"mean": 1.0, "sd": 0.5, "p_positive": 0.90, "verdict": "real"},
        recommended_drop=None,
    )
    high_val = StashScore(
        name="High Val",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=2.0,
        band={"mean": 2.0, "sd": 1.0, "p_positive": 0.90, "verdict": "real"},
        recommended_drop=None,
    )
    ranked = sorted([low_val, high_val], key=_rank_key, reverse=True)
    assert [s.name for s in ranked] == ["High Val", "Low Val"]


def test_synthetic_swap_starter_to_reliever_by_remaining_workload():
    """SP candidate, RP incumbent: the synthetic line subtracts the candidate's
    slot-share of the RP's REMAINING IP, not 60 IP (which would zero the RP and
    overstate the gain) and not the RP's preseason IP.

    Webb returns with 60 ROS / 200 preseason IP, so at full season remaining his
    slot-share is 60/200 = 0.30. The RP (25 ROS IP) loses 0.30 * 25 = 7.5 IP --
    the returning starter's slot-share, applied to the reliever's own remaining
    workload, sets the cross-role window.
    """
    from fantasy_baseball.lineup.stash_value import _synthetic_swap_line

    incumbent_ros = PitcherStats(
        ip=25,
        w=1,
        k=20,
        sv=2,
        er=12,
        bb=10,
        h_allowed=22,
        era=4.32,
        whip=1.28,
    )
    incumbent_pre = PitcherStats(
        ip=65,
        w=3,
        k=60,
        sv=7,
        er=30,
        bb=24,
        h_allowed=58,
        era=4.15,
        whip=1.26,
    )
    incumbent = Player(
        name="RP_Incumbent",
        player_type=PlayerType.PITCHER,
        rest_of_season=incumbent_ros,
        preseason=incumbent_pre,
        selected_position=Position.P,
    )

    candidate_ros = PitcherStats(
        ip=60,
        w=4,
        k=67,
        sv=0,
        er=22,
        bb=15,
        h_allowed=50,
        era=3.30,
        whip=1.08,
    )
    candidate_pre = PitcherStats(
        ip=200,
        w=14,
        k=220,
        sv=0,
        er=72,
        bb=50,
        h_allowed=170,
        era=3.24,
        whip=1.10,
    )
    candidate = Player(
        name="Webb",
        player_type=PlayerType.PITCHER,
        rest_of_season=candidate_ros,
        preseason=candidate_pre,
        selected_position=Position.IL,
    )

    # The ``w`` argument is the legacy direct-IP window. With the new shared
    # helpers, the function derives the correct cross-role window internally
    # from preseason IP; ``w`` is only used as a fallback when preseason data
    # is missing.
    synth = _synthetic_swap_line(incumbent, candidate, w=60.0)

    # Cross-role: slot_share = 60/200 = 0.30 (fraction_remaining defaults to
    # 1.0); window = 0.30 * RP_ros(25) = 7.5 IP. Scale = (25 - 7.5)/25 = 0.70.
    scale = (25.0 - 7.5) / 25.0
    expected_ip = scale * 25.0 + 60.0
    expected_k = scale * 20.0 + 67.0
    assert abs(synth.rest_of_season.ip - expected_ip) < 1e-6
    assert abs(synth.rest_of_season.k - expected_k) < 1e-6


def test_active_lineup_standings_uses_team_ytd_not_player_full_season():
    """REGRESSION (team-YTD projection refactor): the user-row hypothetical
    in the stash board must reflect team_YTD + sum(player.ROS), not the
    per-player full_season sum. Otherwise a mid-season acquisition's
    pre-acquisition YTD inflates the user's pre-stash baseline.

    Test: a roster with one pitcher whose preseason K = 200 and ROS K = 100
    (implying player YTD K = 100 that may not all be the team's YTD).
    Team-YTD K = 50. User row K must equal 50 + 100 = 150, NOT 200 (the
    pre-refactor full_season floor) and NOT 100 (ROS-only).
    """
    # Pitcher with preseason K=200 (full_season), ROS K=100.
    arm = Player(
        name="Mid-Season Pickup",
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        preseason=PitcherStats(
            ip=180.0,
            w=12.0,
            k=200.0,
            sv=0.0,
            er=60.0,
            bb=40.0,
            h_allowed=140.0,
            era=3.00,
            whip=1.00,
        ),
        rest_of_season=PitcherStats(
            ip=90.0,
            w=6.0,
            k=100.0,
            sv=0.0,
            er=30.0,
            bb=20.0,
            h_allowed=70.0,
            era=3.00,
            whip=1.00,
        ),
        selected_position=Position.P,
    )

    # Baseline projected standings: user row at K=0 placeholder, plus a
    # second team so the spread/ranking math is well-defined.
    baseline = ProjectedStandings(
        effective_date=EFFECTIVE_DATE,
        entries=[
            ProjectedStandingsEntry(team_name=TEAM_NAME, stats=CategoryStats()),
            ProjectedStandingsEntry(team_name="Opp A", stats=CategoryStats(k=300.0)),
        ],
    )

    # Team-YTD has only K=50 (much less than the player's implied YTD K=100
    # if the player was picked up mid-season). This is the key signal: the
    # team didn't earn the pre-pickup K total, so the user row must NOT
    # double-count it.
    user_ytd = TeamYtdComponents(k=50.0)

    result = _active_lineup_standings(
        before_active=[arm],
        projected_standings=baseline,
        team_name=TEAM_NAME,
        user_ytd_components=user_ytd,
    )

    user_row = next(e for e in result.entries if e.team_name == TEAM_NAME)
    # Expected: team_YTD.k (50) + ROS.k (100) = 150.
    # Pre-refactor (project_team_stats full_season): 200.
    # ROS-only (no YTD threading): 100.
    assert user_row.stats[Category.K] == pytest.approx(150.0, abs=1e-6)
    # Opponent row is untouched.
    opp_row = next(e for e in result.entries if e.team_name == "Opp A")
    assert opp_row.stats[Category.K] == pytest.approx(300.0, abs=1e-6)


def test_score_stash_candidates_threads_actual_standings_into_user_row():
    """End-to-end: score_stash_candidates derives user_ytd_components from
    actual_standings and the user-row baseline reflects team_YTD + ROS.

    Sanity check (not a P(helps) assertion): the call signature accepts
    actual_standings and the run completes without error. The deeper
    correctness assertion lives in
    :func:`test_active_lineup_standings_uses_team_ytd_not_player_full_season`.
    """
    # No injured players -> empty board, but the call must still accept
    # actual_standings as a keyword.
    roster = [*_full_hitters(), *_mediocre_staff()]
    standings = _coupled_standings(roster)
    result = score_stash_candidates(
        roster=roster,
        free_agents=[],
        projected_standings=standings,
        roster_slots=STASH_SLOTS,
        team_name=TEAM_NAME,
        team_sds=None,
        fraction_remaining=0.5,
        actual_standings=None,
    )
    assert result.candidates == []


def test_score_stash_candidates_warns_when_user_team_missing_from_actual_standings(
    caplog,
):
    """Fix #7: when ``actual_standings`` is non-None but does not contain a
    matching entry for ``team_name``, score_stash_candidates falls back to
    zero YTD components AND emits a warning so the silent collapse to
    ROS-only is observable.

    Use the no-injured-player short-circuit to avoid the full optimizer
    path -- the lookup happens unconditionally before the early return
    once we plumb actual_standings through. We pass at least one injured
    player so the lookup is actually exercised.
    """
    from fantasy_baseball.models.standings import (
        CategoryStats,
        Standings,
        StandingsEntry,
    )

    # An owned IL hitter -- forces the lookup loop to run.
    il_bat = _hitter(
        "InjuredBat",
        positions=["OF"],
        slot="IL",
        ab=200,
        h=58,
        r=30,
        hr=8,
        rbi=28,
        sb=2,
        avg=0.290,
    )
    roster = [*_full_hitters(), *_mediocre_staff(), il_bat]
    standings = _coupled_standings(roster)

    # Mismatched team-name -- "Test Team" is not in entries.
    actual = Standings(
        effective_date=EFFECTIVE_DATE,
        entries=[
            StandingsEntry(
                team_name="DifferentNameEntirely",
                team_key="x",
                rank=1,
                stats=CategoryStats(),
            ),
        ],
    )

    with caplog.at_level("WARNING", logger="fantasy_baseball.lineup.stash_value"):
        score_stash_candidates(
            roster=roster,
            free_agents=[],
            projected_standings=standings,
            roster_slots=STASH_SLOTS,
            team_name=TEAM_NAME,
            team_sds=None,
            fraction_remaining=0.5,
            actual_standings=actual,
        )

    msgs = [r.getMessage() for r in caplog.records]
    assert any(TEAM_NAME in m for m in msgs), (
        f"Expected a warning naming {TEAM_NAME!r}; got: {msgs}"
    )
