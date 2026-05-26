import pytest

from fantasy_baseball.lineup.stash_value import (
    StashResult,
    StashScore,
    _activate,
    _marginal_value,
    _open_il_slots,
    _owned_il_stashes,
    _solve_active,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import ProjectedStandings

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


def _contending_k_standings():
    """Standings where the team is contending in strikeouts.

    All of the fixture's pitchers share identical IP/ER/BB/H_allowed/W/SV, so
    K is the only lever distinguishing them (mirrors
    ``test_il_return_planner._contending_standings``). The optimizer scores the
    team from its active roster's projected totals (``project_team_stats``),
    not the standings row: the nine mediocre arms project ~675 team K and the
    elite-swapped lineup ~734, so an opponent K of 700 makes the elite arm flip
    the contended category (strictly worth starting, no value-neutral tie)
    while the scrub -- who only lowers team K -- never cracks the active nine.
    """
    base = {
        "R": 800,
        "HR": 200,
        "RBI": 800,
        "SB": 100,
        "AVG": 0.260,
        "W": 70,
        "K": 1200,
        "SV": 50,
        "ERA": 3.50,
        "WHIP": 1.20,
        "AB": 5000,
        "H": 1300,
        "IP": 1400,
        "ER": 560,
        "BB": 420,
        "H_ALLOWED": 1300,
    }
    return ProjectedStandings.from_json(
        {
            "effective_date": "2026-04-01",
            "teams": [
                {"name": TEAM_NAME, "stats": dict(base)},
                {"name": "Opponent", "stats": {**base, "K": 700, "SV": 30, "ERA": 3.80}},
            ],
        }
    )


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


@pytest.fixture
def stash_fixture():
    """Return ``(roster, standings, team_sds, roster_slots, team_name)``.

    Roster shape: 9 hitters filling all hitter slots + 9 mediocre starting
    pitchers (K ranging 70..78 over ~150 IP). The nine P slots are exactly
    filled by the nine mediocre arms, so:
      * an ELITE high-K arm (K=130) beats the weakest mediocre arm and cracks
        the active nine -> positive Gain.
      * a REPLACEMENT-level arm (K=20) is worse than all nine -> stays in the
        surplus, after_active == before_active, band mean ~0 -> Gain 0.
    """
    hitters = _full_hitters()
    # Nine mediocre arms, all roughly equal IP/ER/BB/H but distinct, modest K.
    mediocre = [
        _arm(f"SP{i}", ip=150.0, k=70.0 + i, slot="P", er=52.0, bb=40.0, h_allowed=130.0)
        for i in range(1, 10)
    ]
    roster = [*hitters, *mediocre]
    standings = _contending_k_standings()
    team_sds = None
    return roster, standings, team_sds, STASH_SLOTS, TEAM_NAME


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


def test_open_il_slots_counts_true_il_slots_only(stash_fixture):
    roster, *_ = stash_fixture
    # With an empty IL, both IL slots are open.
    assert _open_il_slots(roster, {"IL": 2}) == 2


def test_owned_il_stashes_uses_is_on_il(monkeypatched_il_roster):
    roster = monkeypatched_il_roster  # one player with status="IL15"
    names = {p.name for p in _owned_il_stashes(roster)}
    assert "Injured Owned Arm" in names
