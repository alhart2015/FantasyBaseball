"""Shared fixtures for waiver recommendation integration tests."""

import pytest

from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats


# ---------------------------------------------------------------------------
# Standings: 10-team roto league, mid-season snapshot.
# User Team is rank 5 with intentionally tight gaps in SB and SV
# (close to neighbors) and wide gaps in HR and K.
# ---------------------------------------------------------------------------

STANDINGS = [
    {
        "name": "Bombers",
        "rank": 1,
        "stats": {
            "R": 520, "HR": 165, "RBI": 510, "SB": 95, "AVG": 0.278,
            "W": 58, "K": 900, "SV": 60, "ERA": 3.30, "WHIP": 1.12,
        },
    },
    {
        "name": "Sluggers",
        "rank": 2,
        "stats": {
            "R": 505, "HR": 155, "RBI": 490, "SB": 88, "AVG": 0.274,
            "W": 54, "K": 865, "SV": 54, "ERA": 3.45, "WHIP": 1.16,
        },
    },
    {
        "name": "Mashers",
        "rank": 3,
        "stats": {
            "R": 488, "HR": 148, "RBI": 468, "SB": 82, "AVG": 0.271,
            "W": 51, "K": 840, "SV": 50, "ERA": 3.55, "WHIP": 1.19,
        },
    },
    {
        "name": "Dingers",
        "rank": 4,
        "stats": {
            "R": 470, "HR": 140, "RBI": 450, "SB": 73, "AVG": 0.269,
            "W": 49, "K": 815, "SV": 44, "ERA": 3.65, "WHIP": 1.21,
        },
    },
    {
        "name": "Hart of the Order",
        "rank": 5,
        "stats": {
            "R": 455, "HR": 125, "RBI": 435, "SB": 71,  # SB gap to rank 4: 2
            "AVG": 0.266,
            "W": 46, "K": 790, "SV": 42,                 # SV gap to rank 4: 2
            "ERA": 3.78, "WHIP": 1.24,
        },
    },
    {
        "name": "Fireballers",
        "rank": 6,
        "stats": {
            "R": 438, "HR": 118, "RBI": 415, "SB": 69,  # SB gap below: 2
            "AVG": 0.262,
            "W": 43, "K": 760, "SV": 40,                 # SV gap below: 2
            "ERA": 3.92, "WHIP": 1.27,
        },
    },
    {
        "name": "Aces High",
        "rank": 7,
        "stats": {
            "R": 420, "HR": 110, "RBI": 395, "SB": 58, "AVG": 0.258,
            "W": 40, "K": 730, "SV": 32, "ERA": 4.05, "WHIP": 1.30,
        },
    },
    {
        "name": "Full Count",
        "rank": 8,
        "stats": {
            "R": 405, "HR": 100, "RBI": 378, "SB": 48, "AVG": 0.253,
            "W": 37, "K": 700, "SV": 26, "ERA": 4.25, "WHIP": 1.34,
        },
    },
    {
        "name": "Whiffers",
        "rank": 9,
        "stats": {
            "R": 385, "HR": 90, "RBI": 355, "SB": 35, "AVG": 0.248,
            "W": 33, "K": 665, "SV": 20, "ERA": 4.48, "WHIP": 1.39,
        },
    },
    {
        "name": "Cellar Dwellers",
        "rank": 10,
        "stats": {
            "R": 360, "HR": 78, "RBI": 330, "SB": 22, "AVG": 0.242,
            "W": 29, "K": 625, "SV": 14, "ERA": 4.75, "WHIP": 1.46,
        },
    },
]

USER_TEAM_NAME = "Hart of the Order"


def _hitter(name, positions, r, hr, rbi, sb, avg, ab, *, selected_position=""):
    """Build a hitter Player with realistic stat columns."""
    h = int(round(avg * ab))
    stats = HitterStats(r=r, hr=hr, rbi=rbi, sb=sb, avg=avg, ab=ab, h=h)
    return Player(
        name=name, player_type="hitter", positions=positions,
        ros=stats, selected_position=selected_position,
    )


def _pitcher(name, positions, w, k, sv, era, whip, ip, *, selected_position=""):
    """Build a pitcher Player with realistic stat columns."""
    er = round(era * ip / 9, 1)
    bb = int(round((whip - er / ip) * ip)) if ip > 0 else 0
    # Clamp bb to non-negative (WHIP includes H + BB)
    bb = max(bb, 0)
    h_allowed = int(round(whip * ip - bb))
    stats = PitcherStats(
        w=w, k=k, sv=sv, era=era, whip=whip, ip=ip,
        er=er, bb=bb, h_allowed=h_allowed,
    )
    return Player(
        name=name, player_type="pitcher", positions=positions,
        ros=stats, selected_position=selected_position,
    )


# ---------------------------------------------------------------------------
# Roster: 24 players for "Hart of the Order"
#   - 12 hitters in active slots (C, 1B, 2B, 3B, SS, IF, 4 OF, 2 UTIL)
#   - 9 pitchers in active P slots
#   - 1 hitter on bench
#   - 2 players on IL (1 hitter, 1 pitcher)
#
#   Key constraint: "Pete Alonso" is the ONLY 1B-eligible player.
#   Vladimir Guerrero Jr. is DH-only here (no 1B eligibility).
# ---------------------------------------------------------------------------

ROSTER_PLAYERS = [
    # --- Active hitters ---
    _hitter("J.T. Realmuto", ["C"],           r=55, hr=15, rbi=55, sb=8,  avg=.260, ab=460, selected_position="C"),
    _hitter("Pete Alonso",   ["1B"],          r=80, hr=35, rbi=95, sb=3,  avg=.250, ab=550, selected_position="1B"),  # ONLY 1B
    _hitter("Marcus Semien", ["2B", "SS"],    r=88, hr=25, rbi=78, sb=14, avg=.265, ab=580, selected_position="2B"),
    _hitter("Manny Machado", ["3B"],          r=78, hr=28, rbi=85, sb=10, avg=.270, ab=560, selected_position="3B"),
    _hitter("Trea Turner",   ["SS"],          r=90, hr=22, rbi=72, sb=25, avg=.280, ab=570, selected_position="SS"),
    _hitter("Gleyber Torres",["2B", "SS"],    r=68, hr=18, rbi=65, sb=7,  avg=.258, ab=520, selected_position="IF"),
    _hitter("Aaron Judge",   ["OF"],          r=100, hr=42, rbi=110, sb=8, avg=.290, ab=530, selected_position="OF"),
    _hitter("Julio Rodriguez",["OF"],         r=85, hr=28, rbi=82, sb=22, avg=.272, ab=555, selected_position="OF"),
    _hitter("Kyle Tucker",   ["OF"],          r=92, hr=30, rbi=95, sb=18, avg=.278, ab=560, selected_position="OF"),
    _hitter("Bryan Reynolds", ["OF"],         r=75, hr=22, rbi=70, sb=12, avg=.268, ab=540, selected_position="OF"),
    _hitter("Jose Ramirez",  ["3B"],          r=95, hr=28, rbi=100, sb=20, avg=.275, ab=565, selected_position="UTIL"),
    _hitter("Vladimir Guerrero Jr.", ["DH"],  r=82, hr=32, rbi=92, sb=4,  avg=.285, ab=555, selected_position="UTIL"),
    # --- Bench ---
    _hitter("Andres Gimenez", ["2B", "SS"],   r=62, hr=14, rbi=52, sb=18, avg=.255, ab=490, selected_position="BN"),
    _pitcher("Clay Holmes",   ["RP"],  w=3,  k=55,  sv=12, era=3.70, whip=1.20, ip=58, selected_position="BN"),
    # --- IL hitter ---
    _hitter("Fernando Tatis Jr.", ["OF", "SS"], r=70, hr=25, rbi=68, sb=20, avg=.275, ab=420, selected_position="IL"),
    # --- Active pitchers (9 to fill all P slots) ---
    _pitcher("Zack Wheeler",  ["SP"],  w=13, k=195, sv=0, era=3.05, whip=1.08, ip=190, selected_position="SP"),
    _pitcher("Logan Webb",    ["SP"],  w=12, k=170, sv=0, era=3.25, whip=1.12, ip=185, selected_position="SP"),
    _pitcher("Sonny Gray",    ["SP"],  w=10, k=165, sv=0, era=3.40, whip=1.15, ip=175, selected_position="SP"),
    _pitcher("Luis Castillo", ["SP"],  w=11, k=175, sv=0, era=3.30, whip=1.10, ip=180, selected_position="SP"),
    _pitcher("Pablo Lopez",   ["SP"],  w=10, k=160, sv=0, era=3.45, whip=1.14, ip=172, selected_position="SP"),
    _pitcher("Jordan Romano",  ["RP"], w=3,  k=65,  sv=32, era=2.80, whip=1.02, ip=62, selected_position="RP"),
    _pitcher("Ryan Helsley",   ["RP"], w=4,  k=72,  sv=35, era=2.60, whip=0.98, ip=65, selected_position="RP"),
    _pitcher("Jose Alvarado",  ["RP"], w=3,  k=60,  sv=18, era=3.50, whip=1.18, ip=55, selected_position="RP"),
    _pitcher("Devin Williams",  ["RP"], w=3,  k=68,  sv=28, era=2.90, whip=1.05, ip=60, selected_position="RP"),
    # --- IL pitcher ---
    _pitcher("Shane McClanahan", ["SP"], w=8, k=140, sv=0, era=3.15, whip=1.06, ip=130, selected_position="IL"),
]


# ---------------------------------------------------------------------------
# Free agents: ~30 players, mix of hitters and pitchers
# Some clearly better than worst roster players, some worse.
# ---------------------------------------------------------------------------

FREE_AGENT_PLAYERS = [
    # --- Hitters ---
    _hitter("Luis Arraez",       ["1B", "2B"],     r=72, hr=5,  rbi=52, sb=3,  avg=.310, ab=560),
    _hitter("Yandy Diaz",        ["1B", "3B"],     r=70, hr=18, rbi=68, sb=2,  avg=.282, ab=530),
    _hitter("Ha-Seong Kim",      ["2B", "SS"],     r=65, hr=15, rbi=55, sb=16, avg=.262, ab=510),
    _hitter("Ezequiel Tovar",    ["SS"],           r=68, hr=20, rbi=65, sb=14, avg=.258, ab=540),
    _hitter("Riley Greene",      ["OF"],           r=78, hr=24, rbi=72, sb=10, avg=.270, ab=545),
    _hitter("Lars Nootbaar",     ["OF"],           r=58, hr=14, rbi=48, sb=6,  avg=.252, ab=430),
    _hitter("Spencer Torkelson", ["1B"],           r=52, hr=16, rbi=58, sb=1,  avg=.238, ab=470),
    _hitter("Alec Bohm",         ["3B"],           r=60, hr=15, rbi=62, sb=3,  avg=.268, ab=530),
    _hitter("Tyler O'Neill",     ["OF"],           r=55, hr=22, rbi=55, sb=8,  avg=.242, ab=380),
    _hitter("Sal Frelick",       ["OF"],           r=62, hr=8,  rbi=42, sb=20, avg=.275, ab=480),
    _hitter("Wilmer Flores",     ["1B", "2B", "3B"], r=48, hr=12, rbi=50, sb=1, avg=.260, ab=430),
    _hitter("Jake Cronenworth",  ["1B", "2B"],     r=58, hr=14, rbi=56, sb=4,  avg=.255, ab=510),
    _hitter("J.D. Martinez",     ["DH"],           r=55, hr=20, rbi=68, sb=0,  avg=.262, ab=480),
    _hitter("Jeimer Candelario", ["1B", "3B"],     r=50, hr=16, rbi=55, sb=2,  avg=.248, ab=460),
    _hitter("Joey Meneses",      ["1B"],           r=40, hr=10, rbi=42, sb=1,  avg=.240, ab=400),
    # --- Pitchers (starters) ---
    _pitcher("Brayan Bello",     ["SP"],  w=11, k=168, sv=0, era=3.50, whip=1.18, ip=175),
    _pitcher("Mitch Keller",     ["SP"],  w=10, k=155, sv=0, era=3.65, whip=1.20, ip=170),
    _pitcher("MacKenzie Gore",   ["SP"],  w=9,  k=160, sv=0, era=3.80, whip=1.22, ip=165),
    _pitcher("Bailey Ober",      ["SP"],  w=10, k=150, sv=0, era=3.55, whip=1.14, ip=172),
    _pitcher("Reid Detmers",     ["SP"],  w=8,  k=145, sv=0, era=4.10, whip=1.25, ip=158),
    _pitcher("Andrew Heaney",    ["SP"],  w=7,  k=135, sv=0, era=4.30, whip=1.28, ip=150),
    _pitcher("Cody Bradford",    ["SP"],  w=6,  k=110, sv=0, era=4.50, whip=1.30, ip=130),
    # --- Pitchers (relievers / closers) ---
    _pitcher("Andres Munoz",     ["RP"],  w=4,  k=75,  sv=28, era=2.70, whip=1.00, ip=63),
    _pitcher("Pete Fairbanks",   ["RP"],  w=3,  k=62,  sv=25, era=3.10, whip=1.08, ip=58),
    _pitcher("Paul Sewald",      ["RP"],  w=3,  k=55,  sv=22, era=3.30, whip=1.12, ip=55),
    _pitcher("Jhoan Duran",      ["RP"],  w=4,  k=68,  sv=20, era=3.40, whip=1.10, ip=60),
    _pitcher("Bryan Abreu",      ["RP"],  w=3,  k=70,  sv=15, era=3.20, whip=1.05, ip=62),
    _pitcher("Yennier Cano",     ["RP"],  w=4,  k=55,  sv=8,  era=3.60, whip=1.15, ip=58),
    _pitcher("Tyler Rogers",     ["RP"],  w=3,  k=42,  sv=5,  era=3.80, whip=1.20, ip=60),
    _pitcher("Tim Hill",         ["RP"],  w=2,  k=38,  sv=2,  era=4.10, whip=1.28, ip=52),
]


ROSTER_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "IF": 1,
    "OF": 4,
    "UTIL": 2,
    "P": 9,
    "BN": 2,
    "IL": 2,
}


# ---------------------------------------------------------------------------
# Yahoo-style roster dicts used by detect_open_slots.
# Each entry represents a player's selected position on the Yahoo roster.
# ---------------------------------------------------------------------------

def _yahoo_entry(name, selected_position):
    return {"name": name, "selected_position": selected_position}


YAHOO_ROSTER_FULL = [
    _yahoo_entry("J.T. Realmuto", "C"),
    _yahoo_entry("Pete Alonso", "1B"),
    _yahoo_entry("Marcus Semien", "2B"),
    _yahoo_entry("Manny Machado", "3B"),
    _yahoo_entry("Trea Turner", "SS"),
    _yahoo_entry("Gleyber Torres", "IF"),
    _yahoo_entry("Aaron Judge", "OF"),
    _yahoo_entry("Julio Rodriguez", "OF"),
    _yahoo_entry("Kyle Tucker", "OF"),
    _yahoo_entry("Bryan Reynolds", "OF"),
    _yahoo_entry("Jose Ramirez", "Util"),
    _yahoo_entry("Vladimir Guerrero Jr.", "Util"),
    _yahoo_entry("Andres Gimenez", "BN"),
    _yahoo_entry("Clay Holmes", "BN"),
    _yahoo_entry("Fernando Tatis Jr.", "IL"),
    _yahoo_entry("Zack Wheeler", "SP"),
    _yahoo_entry("Logan Webb", "SP"),
    _yahoo_entry("Sonny Gray", "SP"),
    _yahoo_entry("Luis Castillo", "SP"),
    _yahoo_entry("Pablo Lopez", "SP"),
    _yahoo_entry("Jordan Romano", "RP"),
    _yahoo_entry("Ryan Helsley", "RP"),
    _yahoo_entry("Jose Alvarado", "RP"),
    _yahoo_entry("Devin Williams", "RP"),
    _yahoo_entry("Shane McClanahan", "IL"),
]

# Yahoo roster with 2 IL players and only 7 of 9 pitcher slots filled,
# plus only 1 of 2 bench slots filled (2 open P slots, 1 open BN slot).
YAHOO_ROSTER_WITH_GAPS = [
    _yahoo_entry("J.T. Realmuto", "C"),
    _yahoo_entry("Pete Alonso", "1B"),
    _yahoo_entry("Marcus Semien", "2B"),
    _yahoo_entry("Manny Machado", "3B"),
    _yahoo_entry("Trea Turner", "SS"),
    _yahoo_entry("Gleyber Torres", "IF"),
    _yahoo_entry("Aaron Judge", "OF"),
    _yahoo_entry("Julio Rodriguez", "OF"),
    _yahoo_entry("Kyle Tucker", "OF"),
    _yahoo_entry("Bryan Reynolds", "OF"),
    _yahoo_entry("Jose Ramirez", "Util"),
    _yahoo_entry("Vladimir Guerrero Jr.", "Util"),
    _yahoo_entry("Andres Gimenez", "BN"),
    _yahoo_entry("Fernando Tatis Jr.", "IL"),
    _yahoo_entry("Zack Wheeler", "SP"),
    _yahoo_entry("Logan Webb", "SP"),
    _yahoo_entry("Sonny Gray", "SP"),
    _yahoo_entry("Luis Castillo", "SP"),
    _yahoo_entry("Pablo Lopez", "SP"),
    _yahoo_entry("Jordan Romano", "RP"),
    _yahoo_entry("Ryan Helsley", "RP"),
    # Jose Alvarado, Devin Williams, Clay Holmes dropped — 2 open P slots, 1 open BN slot
    _yahoo_entry("Shane McClanahan", "IL"),
]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standings():
    """10-team roto standings with known stat gaps."""
    return [dict(t) for t in STANDINGS]


@pytest.fixture
def user_team_name():
    return USER_TEAM_NAME


@pytest.fixture
def leverage(standings, user_team_name):
    """Leverage weights derived from the standings fixture."""
    return calculate_leverage(standings, user_team_name)


@pytest.fixture
def roster():
    """Roster of ~24 Player objects, 2 on IL, 1 on BN."""
    import copy
    return [copy.copy(p) for p in ROSTER_PLAYERS]


@pytest.fixture
def active_roster(roster):
    """Only active (non-IL) roster players for waiver swap evaluation."""
    return [p for p in roster if p.selected_position.upper() not in ("IL", "IL+")]


@pytest.fixture
def free_agents():
    """~30 free agents with realistic projections."""
    import copy
    return [copy.copy(p) for p in FREE_AGENT_PLAYERS]


@pytest.fixture
def roster_slots():
    """Roster slot configuration matching league.yaml.example."""
    return dict(ROSTER_SLOTS)


@pytest.fixture
def yahoo_roster_full():
    """Yahoo roster dicts with all slots filled."""
    return [dict(e) for e in YAHOO_ROSTER_FULL]


@pytest.fixture
def yahoo_roster_with_gaps():
    """Yahoo roster dicts with open slots (2 P, 1 BN)."""
    return [dict(e) for e in YAHOO_ROSTER_WITH_GAPS]
