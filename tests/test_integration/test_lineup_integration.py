"""Integration tests for the lineup optimizer and leverage pipelines.

These tests exercise the full flow: standings -> leverage weights ->
weighted SGP -> optimizer assignments, using realistic baseball stat lines.
"""

from datetime import date

import pytest

from fantasy_baseball.lineup.leverage import calculate_leverage, MAX_MEANINGFUL_GAP_MULTIPLIER
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
from fantasy_baseball.models.standings import CategoryStats, StandingsEntry, StandingsSnapshot
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    DEFAULT_ROSTER_SLOTS,
)
from fantasy_baseball.utils.positions import can_fill_slot


def _list_to_snapshot(standings_list: list[dict]) -> StandingsSnapshot:
    """Convert a test standings list[dict] to StandingsSnapshot."""
    return StandingsSnapshot(
        effective_date=date.min,
        entries=[
            StandingsEntry(
                team_name=t["name"],
                team_key=t.get("team_key", ""),
                rank=t.get("rank", 0),
                stats=CategoryStats.from_dict(t.get("stats", {})),
            )
            for t in standings_list
        ],
    )


# ---------------------------------------------------------------------------
# Helpers — realistic player builders
# ---------------------------------------------------------------------------

def _make_hitter(name, positions, r, hr, rbi, sb, avg, ab):
    stats = HitterStats(r=r, hr=hr, rbi=rbi, sb=sb, avg=avg, ab=ab, h=int(avg * ab))
    return Player(
        name=name, positions=positions, player_type="hitter", rest_of_season=stats,
    )


def _make_pitcher(name, positions, w, k, sv, era, whip, ip):
    stats = PitcherStats(
        w=w, k=k, sv=sv, era=era, whip=whip, ip=ip,
        er=era * ip / 9, bb=int(whip * ip * 0.3), h_allowed=int(whip * ip * 0.7),
    )
    return Player(
        name=name, positions=positions, player_type="pitcher", rest_of_season=stats,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def midseason_standings():
    """10-team league at mid-season. User team 'Hart of the Order' is rank 5."""
    return _list_to_snapshot([
        {"name": "Power Hitters", "rank": 1, "stats": {
            "R": 512, "HR": 158, "RBI": 495, "SB": 88, "AVG": 0.274,
            "W": 54, "K": 862, "SV": 58, "ERA": 3.38, "WHIP": 1.14,
        }},
        {"name": "Speed Demons", "rank": 2, "stats": {
            "R": 498, "HR": 142, "RBI": 468, "SB": 102, "AVG": 0.271,
            "W": 51, "K": 838, "SV": 52, "ERA": 3.52, "WHIP": 1.17,
        }},
        {"name": "Mound Masters", "rank": 3, "stats": {
            "R": 471, "HR": 138, "RBI": 452, "SB": 78, "AVG": 0.269,
            "W": 52, "K": 820, "SV": 50, "ERA": 3.45, "WHIP": 1.16,
        }},
        {"name": "Base Bandits", "rank": 4, "stats": {
            "R": 462, "HR": 134, "RBI": 441, "SB": 71, "AVG": 0.266,
            "W": 47, "K": 795, "SV": 44, "ERA": 3.72, "WHIP": 1.23,
        }},
        {"name": "Hart of the Order", "rank": 5, "stats": {
            "R": 453, "HR": 131, "RBI": 432, "SB": 52, "AVG": 0.264,
            "W": 46, "K": 778, "SV": 41, "ERA": 3.78, "WHIP": 1.24,
        }},
        {"name": "Swing Kings", "rank": 6, "stats": {
            "R": 438, "HR": 122, "RBI": 415, "SB": 48, "AVG": 0.261,
            "W": 43, "K": 745, "SV": 36, "ERA": 3.92, "WHIP": 1.27,
        }},
        {"name": "Bullpen Bunch", "rank": 7, "stats": {
            "R": 425, "HR": 116, "RBI": 398, "SB": 42, "AVG": 0.257,
            "W": 40, "K": 718, "SV": 32, "ERA": 4.08, "WHIP": 1.31,
        }},
        {"name": "Strikeout City", "rank": 8, "stats": {
            "R": 405, "HR": 108, "RBI": 382, "SB": 36, "AVG": 0.253,
            "W": 37, "K": 695, "SV": 28, "ERA": 4.25, "WHIP": 1.34,
        }},
        {"name": "Last Rounders", "rank": 9, "stats": {
            "R": 388, "HR": 97, "RBI": 365, "SB": 30, "AVG": 0.249,
            "W": 33, "K": 662, "SV": 22, "ERA": 4.48, "WHIP": 1.39,
        }},
        {"name": "Cellar Dwellers", "rank": 10, "stats": {
            "R": 358, "HR": 82, "RBI": 335, "SB": 22, "AVG": 0.242,
            "W": 29, "K": 625, "SV": 16, "ERA": 4.75, "WHIP": 1.46,
        }},
    ])


@pytest.fixture
def preseason_standings():
    """All zeros — beginning of season, no stats accumulated yet."""
    teams = [
        "Power Hitters", "Speed Demons", "Mound Masters", "Base Bandits",
        "Hart of the Order", "Swing Kings", "Bullpen Bunch", "Strikeout City",
        "Last Rounders", "Cellar Dwellers",
    ]
    zero_stats = {cat: 0 for cat in ALL_CATEGORIES}
    return _list_to_snapshot([
        {"name": name, "rank": i + 1, "stats": dict(zero_stats)}
        for i, name in enumerate(teams)
    ])


@pytest.fixture
def full_hitter_roster():
    """14 hitters — enough to fill all 12 active slots plus 2 bench.

    Uses realistic 2025/2026-caliber projections.
    """
    return [
        # C: Salvador Perez type
        _make_hitter("Sal Perez", ["C"], 58, 24, 78, 1, .253, 490),
        # 1B: Freddie Freeman type
        _make_hitter("Fred Freeman", ["1B"], 95, 28, 92, 10, .295, 565),
        # 2B: Marcus Semien type
        _make_hitter("Marc Semien", ["2B"], 88, 24, 75, 14, .268, 560),
        # 3B: Jose Ramirez type
        _make_hitter("Joe Ramirez", ["3B"], 98, 30, 105, 22, .278, 570),
        # SS: Trea Turner type
        _make_hitter("Trea Turner", ["SS"], 92, 21, 68, 28, .282, 555),
        # IF-eligible (2B/SS): Willy Adames type
        _make_hitter("Will Adames", ["SS", "2B"], 80, 26, 85, 8, .258, 530),
        # OF1: Aaron Judge type
        _make_hitter("Aaron Judge", ["OF"], 110, 48, 125, 4, .290, 540),
        # OF2: Juan Soto type
        _make_hitter("Juan Soto", ["OF"], 105, 35, 100, 6, .288, 520),
        # OF3: Kyle Tucker type
        _make_hitter("Kyle Tucker", ["OF"], 95, 30, 90, 18, .280, 545),
        # OF4: Luis Robert type
        _make_hitter("Luis Robert", ["OF"], 78, 25, 72, 15, .265, 480),
        # UTIL1: Yordan Alvarez type (DH)
        _make_hitter("Yordan Alvarez", ["DH"], 88, 35, 100, 1, .283, 510),
        # UTIL2: Shohei type (DH/OF)
        _make_hitter("Shohei Ohtani", ["DH", "OF"], 102, 42, 110, 15, .285, 545),
        # BN1: weaker OF
        _make_hitter("Bench OF", ["OF"], 55, 12, 45, 8, .248, 380),
        # BN2: backup C/1B
        _make_hitter("Bench UT", ["C", "1B"], 42, 10, 40, 2, .240, 320),
    ]


@pytest.fixture
def full_pitcher_roster():
    """12 pitchers — 9 active P slots, 3 bench."""
    return [
        _make_pitcher("Gerrit Cole", ["SP"], 14, 235, 0, 3.10, 1.05, 195),
        _make_pitcher("Zack Wheeler", ["SP"], 13, 215, 0, 3.25, 1.08, 190),
        _make_pitcher("Corbin Burnes", ["SP"], 12, 200, 0, 3.35, 1.12, 185),
        _make_pitcher("Logan Webb", ["SP"], 12, 175, 0, 3.20, 1.10, 192),
        _make_pitcher("Chris Sale", ["SP"], 11, 195, 0, 3.45, 1.15, 178),
        _make_pitcher("Joe Ryan", ["SP"], 10, 170, 0, 3.55, 1.12, 175),
        _make_pitcher("Emmanuel Clase", ["RP"], 3, 65, 38, 2.15, 0.92, 68),
        _make_pitcher("Josh Hader", ["RP"], 3, 72, 34, 2.55, 1.00, 62),
        _make_pitcher("Devin Williams", ["RP"], 2, 68, 30, 2.40, 0.95, 58),
        _make_pitcher("Ryan Helsley", ["RP"], 3, 70, 32, 2.65, 1.02, 64),
        # Bench pitchers (weaker)
        _make_pitcher("Bench SP1", ["SP"], 7, 120, 0, 4.20, 1.28, 145),
        _make_pitcher("Bench SP2", ["SP"], 6, 110, 0, 4.50, 1.32, 130),
    ]


# ===========================================================================
# LEVERAGE TESTS
# ===========================================================================

class TestLeverageIntegration:
    def test_leverage_weights_sum_to_one(self, midseason_standings):
        """Leverage weights must always sum to 1.0, regardless of standings."""
        leverage = calculate_leverage(midseason_standings, "Hart of the Order")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_leverage_weights_sum_to_one_first_place(self, midseason_standings):
        """Sum-to-one holds for first-place team (no team above)."""
        leverage = calculate_leverage(midseason_standings, "Power Hitters")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_leverage_weights_sum_to_one_last_place(self, midseason_standings):
        """Sum-to-one holds for last-place team (no team below)."""
        leverage = calculate_leverage(midseason_standings, "Cellar Dwellers")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_tied_category_capped(self, midseason_standings):
        """When one category has a near-zero gap (0.01), its leverage
        should not exceed 35% of total weight thanks to the median cap."""
        # Force SB to be nearly tied between user and team above (rank 4)
        import dataclasses
        modified = []
        for entry in midseason_standings.entries:
            if entry.team_name == "Base Bandits":
                entry = dataclasses.replace(entry, stats=dataclasses.replace(entry.stats, sb=52.01))
            elif entry.team_name == "Hart of the Order":
                entry = dataclasses.replace(entry, stats=dataclasses.replace(entry.stats, sb=52.00))
            modified.append(entry)
        tied_standings = StandingsSnapshot(
            effective_date=midseason_standings.effective_date, entries=modified,
        )

        leverage = calculate_leverage(tied_standings, "Hart of the Order", season_progress=1.0)

        assert leverage["SB"] < 0.35, (
            f"SB leverage {leverage['SB']:.4f} exceeds 35% cap despite "
            f"MAX_MEANINGFUL_GAP_MULTIPLIER capping"
        )
        # All other categories combined should still carry majority weight
        non_sb_total = sum(v for k, v in leverage.items() if k != "SB")
        assert non_sb_total > 0.65

    def test_leverage_reflects_standings_gaps(self, midseason_standings):
        """The category with the smallest gap to the team above should have
        higher leverage than the category with the largest gap.

        For 'Hart of the Order' (rank 5) vs 'Base Bandits' (rank 4):
          HR gap: 134 - 131 = 3  (smallest counting gap)
          SB gap: 71 - 52 = 19  (largest counting gap)
        So HR should get higher leverage than SB.
        """
        leverage = calculate_leverage(midseason_standings, "Hart of the Order", season_progress=1.0)

        # HR attack gap (3) is much smaller than SB attack gap (19),
        # so HR leverage should be higher
        assert leverage["HR"] > leverage["SB"], (
            f"HR leverage ({leverage['HR']:.4f}) should exceed SB "
            f"({leverage['SB']:.4f}) because HR has a smaller gap to overtake"
        )

    def test_preseason_equal_weights(self, preseason_standings):
        """With all zeros, gaps are all zero -> leverage should be equal."""
        leverage = calculate_leverage(
            preseason_standings, "Hart of the Order"
        )
        expected = 1.0 / len(ALL_CATEGORIES)  # 0.1 for 10 categories
        for cat in ALL_CATEGORIES:
            assert leverage[cat] == pytest.approx(expected, abs=0.02), (
                f"Pre-season {cat} leverage {leverage[cat]:.4f} deviates "
                f"from expected equal weight {expected:.4f}"
            )


# ===========================================================================
# LINEUP OPTIMIZER TESTS
# ===========================================================================

class TestHitterOptimizerIntegration:
    def test_all_hitter_slots_filled(
        self, midseason_standings, full_hitter_roster
    ):
        """With 14 hitters and 12 active slots, every slot should be assigned."""
        leverage = calculate_leverage(
            midseason_standings, "Hart of the Order"
        )
        lineup = optimize_hitter_lineup(
            full_hitter_roster, leverage, roster_slots=DEFAULT_ROSTER_SLOTS
        )

        # Active hitter slots: C(1), 1B(1), 2B(1), 3B(1), SS(1), IF(1),
        # OF(4), UTIL(2) = 12 total
        expected_slot_count = sum(
            count for pos, count in DEFAULT_ROSTER_SLOTS.items()
            if pos not in ("P", "BN", "IL")
        )

        assert len(lineup) == expected_slot_count, (
            f"Expected {expected_slot_count} assigned slots, got {len(lineup)}: "
            f"{lineup}"
        )

    def test_starters_have_higher_wsgp_than_bench(
        self, midseason_standings, full_hitter_roster
    ):
        """Every starting hitter should have wSGP >= the best bench hitter.

        The Hungarian algorithm may make small tradeoffs for position
        constraints, so we allow a tolerance of 0.5 SGP.
        """
        leverage = calculate_leverage(
            midseason_standings, "Hart of the Order"
        )
        lineup = optimize_hitter_lineup(
            full_hitter_roster, leverage, roster_slots=DEFAULT_ROSTER_SLOTS
        )

        starter_names = set(lineup.values())
        bench_players = [
            h for h in full_hitter_roster if h.name not in starter_names
        ]
        starters = [
            h for h in full_hitter_roster if h.name in starter_names
        ]

        # Compute wSGP for each group
        bench_wsgps = [
            calculate_weighted_sgp(h.rest_of_season, leverage) for h in bench_players
        ]
        starter_wsgps = [
            calculate_weighted_sgp(h.rest_of_season, leverage) for h in starters
        ]

        if not bench_wsgps:
            return  # All players are starting, nothing to check

        best_bench = max(bench_wsgps)
        worst_starter = min(starter_wsgps)

        # Allow small tolerance for position-constraint tradeoffs
        tolerance = 0.5
        assert worst_starter >= best_bench - tolerance, (
            f"Worst starter wSGP ({worst_starter:.3f}) is significantly below "
            f"best bench wSGP ({best_bench:.3f}). The optimizer may be "
            f"misassigning players."
        )

    def test_position_eligibility_respected(
        self, midseason_standings, full_hitter_roster
    ):
        """Every player assigned to a slot must be eligible for that slot."""
        leverage = calculate_leverage(
            midseason_standings, "Hart of the Order"
        )
        lineup = optimize_hitter_lineup(
            full_hitter_roster, leverage, roster_slots=DEFAULT_ROSTER_SLOTS
        )

        # Build name -> positions lookup
        pos_by_name = {
            h.name: h.positions for h in full_hitter_roster
        }

        for slot_key, player_name in lineup.items():
            # Strip suffix from duplicate slot keys like "OF_2", "OF_3"
            base_slot = slot_key.split("_")[0]
            positions = pos_by_name[player_name]
            assert can_fill_slot(positions, base_slot), (
                f"{player_name} (positions={positions}) cannot fill "
                f"slot '{base_slot}' (key='{slot_key}')"
            )

    def test_multi_position_player_optimally_placed(self):
        """A player eligible for SS and UTIL should be placed at SS
        when he is the ONLY SS-eligible player, freeing UTIL for a
        DH-only hitter who cannot play any other position.

        The Hungarian algorithm should recognise that putting the
        multi-position player at UTIL would leave SS empty while a
        DH-only player could have filled UTIL instead.
        """
        leverage = {cat: 0.1 for cat in ALL_CATEGORIES}

        hitters = [
            # Only SS-eligible player on the roster — also UTIL-eligible
            _make_hitter("Bobby Witt", ["SS"], 95, 28, 88, 30, .285, 560),
            # DH-only: can fill UTIL but not any positional slot
            _make_hitter("Yordan Alvarez", ["DH"], 85, 34, 98, 1, .281, 510),
            # Remaining roster: nobody else is SS-eligible
            _make_hitter("Catcher Guy", ["C"], 55, 18, 60, 2, .248, 440),
            _make_hitter("First Base Guy", ["1B"], 78, 26, 84, 3, .265, 520),
            _make_hitter("Second Base Guy", ["2B"], 72, 20, 70, 10, .262, 500),
            _make_hitter("Third Base Guy", ["3B"], 68, 22, 72, 5, .258, 490),
            _make_hitter("IF Guy", ["2B", "3B"], 65, 14, 55, 12, .255, 480),
            _make_hitter("OF Guy 1", ["OF"], 98, 38, 105, 6, .288, 550),
            _make_hitter("OF Guy 2", ["OF"], 88, 28, 82, 14, .275, 530),
            _make_hitter("OF Guy 3", ["OF"], 80, 22, 70, 18, .268, 510),
            _make_hitter("OF Guy 4", ["OF"], 72, 18, 62, 10, .260, 490),
            _make_hitter("Util Filler", ["1B", "DH"], 60, 15, 55, 2, .252, 460),
        ]

        lineup = optimize_hitter_lineup(
            hitters, leverage, roster_slots=DEFAULT_ROSTER_SLOTS
        )

        # Bobby Witt is the only SS-eligible player, so the optimizer
        # must place him at SS (not waste him at UTIL).
        assert lineup.get("SS") == "Bobby Witt", (
            f"Bobby Witt (only SS-eligible player) should be at SS, not at "
            f"another slot. Lineup: {lineup}"
        )
        # Yordan (DH-only) should end up in a UTIL slot
        util_players = [
            v for k, v in lineup.items() if k.startswith("UTIL")
        ]
        assert "Yordan Alvarez" in util_players, (
            f"Yordan Alvarez (DH only) should be in UTIL. "
            f"UTIL players: {util_players}. Lineup: {lineup}"
        )


class TestPitcherOptimizerIntegration:
    def test_pitcher_lineup_respects_slot_count(
        self, midseason_standings, full_pitcher_roster
    ):
        """optimize_pitcher_lineup returns exactly P slots starters."""
        leverage = calculate_leverage(
            midseason_standings, "Hart of the Order"
        )
        p_slots = DEFAULT_ROSTER_SLOTS["P"]  # 9

        starters, bench = optimize_pitcher_lineup(
            full_pitcher_roster, leverage, slots=p_slots
        )

        assert len(starters) == p_slots, (
            f"Expected {p_slots} starters, got {len(starters)}"
        )
        assert len(bench) == len(full_pitcher_roster) - p_slots, (
            f"Expected {len(full_pitcher_roster) - p_slots} bench pitchers, "
            f"got {len(bench)}"
        )

    def test_pitcher_starters_outrank_bench(
        self, midseason_standings, full_pitcher_roster
    ):
        """Every starting pitcher should have wSGP >= every bench pitcher."""
        leverage = calculate_leverage(
            midseason_standings, "Hart of the Order"
        )
        starters, bench = optimize_pitcher_lineup(
            full_pitcher_roster, leverage, slots=DEFAULT_ROSTER_SLOTS["P"]
        )

        if not bench:
            return

        worst_starter_wsgp = min(s["wsgp"] for s in starters)
        best_bench_wsgp = max(b["wsgp"] for b in bench)

        assert worst_starter_wsgp >= best_bench_wsgp, (
            f"Worst starter wSGP ({worst_starter_wsgp:.3f}) < best bench "
            f"wSGP ({best_bench_wsgp:.3f})"
        )
