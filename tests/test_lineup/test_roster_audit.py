import pytest

from fantasy_baseball.lineup.roster_audit import (
    POSITION_POOL_SIZES,
    audit_roster,
    build_position_pools,
    candidates_for_player,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

TEAM_NAME = "Test Team"


def _projected(rows):
    """Wrap a list-of-dicts standings fixture as a ProjectedStandings."""
    return ProjectedStandings.from_json({"effective_date": "2026-04-01", "teams": rows})


def _minimal_standings():
    """A minimal 2-team projected_standings fixture for audit tests.

    The exact stat values don't matter for most tests — we just need the
    structure `compute_delta_roto` can consume (user's team + at least one
    opponent for defense-comfort to compute gaps).
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
    return _projected(
        [
            {"name": TEAM_NAME, "stats": dict(base)},
            {"name": "Opponent", "stats": {**base, "SV": 30, "ERA": 3.80}},
        ]
    )


ROSTER_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
    "P": 3,
    "BN": 2,
    "IL": 0,
}


def _hitter(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=positions,
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
    )


def _pitcher(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=positions,
        rest_of_season=PitcherStats(
            ip=stats.get("ip", 60.0),
            w=stats.get("w", 3.0),
            k=stats.get("k", 60.0),
            sv=stats.get("sv", 0.0),
            er=stats.get("er", 20.0),
            bb=stats.get("bb", 20.0),
            h_allowed=stats.get("h_allowed", 50.0),
            era=stats.get("era", 3.00),
            whip=stats.get("whip", 1.17),
        ),
    )


class TestBuildPositionPools:
    def test_hitter_buckets_by_all_positions(self):
        """A 2B/OF-eligible hitter appears in both 2B and OF pools."""
        multi = _hitter(
            "Multi Pos", ["2B", "OF"], r=80, hr=25, rbi=75, sb=10, avg=0.280, ab=520, h=146
        )
        of_only = _hitter("OF Only", ["OF"], r=70, hr=20, rbi=70, sb=8, avg=0.270, ab=500, h=135)
        pools = build_position_pools([multi, of_only])
        assert multi in pools["2B"]
        assert multi in pools["OF"]
        assert of_only in pools["OF"]
        assert of_only not in pools["2B"]

    def test_pool_sorted_by_sgp_desc_and_truncated_to_top_n(self):
        """Pool is sorted by raw SGP descending and truncated to POSITION_POOL_SIZES[pos]."""
        # Build OF pool of 20 hitters with monotonically decreasing R.
        # POSITION_POOL_SIZES["OF"] is 15, so only the top 15 survive.
        fas = [
            _hitter(f"OF{i}", ["OF"], r=100 - i, hr=25, rbi=75, sb=5, avg=0.270, ab=500, h=135)
            for i in range(20)
        ]
        pools = build_position_pools(fas)
        assert len(pools["OF"]) == POSITION_POOL_SIZES["OF"]
        assert pools["OF"][0].name == "OF0"  # highest R → highest SGP
        assert pools["OF"][-1].name == f"OF{POSITION_POOL_SIZES['OF'] - 1}"

    def test_pitcher_pools(self):
        """SP-eligible pitcher lands in SP pool, RP-eligible in RP pool."""
        sp = _pitcher(
            "SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
        )
        rp = _pitcher(
            "RP", ["RP"], ip=60, w=3, k=60, sv=20, era=3.00, whip=1.17, er=20, bb=20, h_allowed=50
        )
        pools = build_position_pools([sp, rp])
        assert sp in pools["SP"]
        assert rp in pools["RP"]
        assert sp not in pools["RP"]
        assert rp not in pools["SP"]

    def test_empty_fa_list_yields_empty_pools(self):
        pools = build_position_pools([])
        assert set(pools.keys()) == set(POSITION_POOL_SIZES.keys())
        for pos in POSITION_POOL_SIZES:
            assert pools[pos] == []

    def test_pitcher_pools_bucket_yahoo_p_only_by_saves(self):
        """Yahoo returns positions=['P'] for all pitchers in leagues without
        SP/RP slots. SP/RP pools must key on projected saves instead."""
        starter = _pitcher(
            "Starter",
            ["P"],
            ip=180,
            w=12,
            k=180,
            sv=0,
            era=3.20,
            whip=1.10,
            er=64,
            bb=30,
            h_allowed=168,
        )
        closer = _pitcher(
            "Closer",
            ["P"],
            ip=60,
            w=3,
            k=60,
            sv=30,
            era=3.00,
            whip=1.17,
            er=20,
            bb=20,
            h_allowed=50,
        )
        hitter = _hitter("Hitter", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143)
        pools = build_position_pools([starter, closer, hitter])
        assert starter in pools["SP"]
        assert closer in pools["RP"]
        assert starter not in pools["RP"]
        assert closer not in pools["SP"]
        # Hitters never leak into pitcher pools even if the pool is sparse
        assert hitter not in pools["SP"]
        assert hitter not in pools["RP"]


class TestCandidatesForPlayer:
    def test_single_position_hitter_pulls_from_that_pool_only(self):
        """A catcher-only hitter gets candidates from the C pool only."""
        c_fa = _hitter("C FA", ["C"], r=50, hr=12, rbi=55, sb=1, avg=0.250, ab=420, h=105)
        of_fa = _hitter("OF FA", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143)
        pools = build_position_pools([c_fa, of_fa])
        catcher = _hitter("Roster C", ["C"], r=40, hr=8, rbi=40, sb=0, avg=0.230, ab=400, h=92)
        cands = candidates_for_player(catcher, pools)
        assert c_fa in cands
        assert of_fa not in cands

    def test_multi_position_hitter_pulls_from_union(self):
        """A 2B/OF-eligible hitter gets candidates from 2B and OF pools deduped."""
        b2 = _hitter("2B FA", ["2B"], r=70, hr=15, rbi=60, sb=15, avg=0.270, ab=500, h=135)
        of_ = _hitter("OF FA", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143)
        multi = _hitter(
            "Multi FA", ["2B", "OF"], r=75, hr=20, rbi=70, sb=10, avg=0.272, ab=510, h=139
        )
        pools = build_position_pools([b2, of_, multi])
        roster_multi = _hitter(
            "Roster 2B/OF", ["2B", "OF"], r=60, hr=10, rbi=50, sb=5, avg=0.250, ab=480, h=120
        )
        cands = candidates_for_player(roster_multi, pools)
        assert b2 in cands
        assert of_ in cands
        assert multi in cands
        # Deduped: multi appears once even though it's in both pools
        assert sum(1 for c in cands if c.name == "Multi FA") == 1

    def test_pitcher_pulls_from_sp_union_rp(self):
        """A Yahoo roster pitcher (positions=['P']) gets SP pool and RP pool."""
        sp = _pitcher(
            "SP FA", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
        )
        rp = _pitcher(
            "RP FA",
            ["RP"],
            ip=60,
            w=3,
            k=60,
            sv=20,
            era=3.00,
            whip=1.17,
            er=20,
            bb=20,
            h_allowed=50,
        )
        pools = build_position_pools([sp, rp])
        roster_pitcher = _pitcher(
            "Roster P", ["P"], ip=100, w=6, k=80, era=4.00, whip=1.30, er=44, bb=30, h_allowed=100
        )
        cands = candidates_for_player(roster_pitcher, pools)
        assert sp in cands
        assert rp in cands

    def test_hitter_never_gets_pitcher_candidates(self):
        """A hitter (positions=['OF']) never gets candidates from SP/RP pools."""
        sp = _pitcher(
            "SP FA", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
        )
        of_ = _hitter("OF FA", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143)
        pools = build_position_pools([sp, of_])
        hitter = _hitter("Roster OF", ["OF"], r=60, hr=10, rbi=50, sb=5, avg=0.250, ab=480, h=120)
        cands = candidates_for_player(hitter, pools)
        assert sp not in cands
        assert of_ in cands

    def test_lineup_only_slots_are_filtered(self):
        """Positions like 'UTIL' and 'IF' don't contribute pools (not Yahoo source positions)."""
        of_ = _hitter("OF FA", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143)
        pools = build_position_pools([of_])
        # Roster hitter whose positions list is purely lineup-only → no candidates
        util_only = _hitter(
            "Util Only", ["UTIL"], r=60, hr=10, rbi=50, sb=5, avg=0.250, ab=480, h=120
        )
        cands = candidates_for_player(util_only, pools)
        assert cands == []


class TestAuditRoster:
    def test_identifies_upgrade_available(self):
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher(
                "Decent SP",
                ["SP"],
                ip=180,
                w=12,
                k=180,
                era=3.50,
                whip=1.20,
                er=70,
                bb=40,
                h_allowed=176,
            ),
            _pitcher(
                "Decent SP2",
                ["SP"],
                ip=170,
                w=10,
                k=160,
                era=3.60,
                whip=1.22,
                er=68,
                bb=40,
                h_allowed=167,
            ),
            _pitcher(
                "Decent RP",
                ["RP"],
                ip=60,
                w=3,
                k=60,
                era=3.00,
                whip=1.17,
                sv=20,
                er=20,
                bb=20,
                h_allowed=50,
            ),
        ]
        free_agents = [
            _hitter("Better OF", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 3, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        # Should have an entry for every roster player
        assert len(results) == len(roster)

        # The weak OF should have an upgrade identified
        weak_entry = next(e for e in results if e.player == "Weak OF")
        assert weak_entry.best_fa == "Better OF"
        assert weak_entry.gap > 0

    def test_shows_no_better_option(self):
        roster = [
            _hitter("Star OF", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=0.300, ab=550, h=165),
        ]
        free_agents = [
            _hitter("Scrub", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        assert len(results) == 1
        assert results[0].best_fa is None
        assert results[0].gap == 0.0

    def test_rows_sorted_by_top_candidate_delta_roto_descending(self):
        """Rows are sorted by their top candidate's delta_roto.total descending.
        Entries with no upgrade (best_fa=None) sort to the bottom."""
        roster = [
            _hitter("OK 1B", ["1B"], r=60, hr=15, rbi=55, sb=3, avg=0.255, ab=480, h=122),
            _hitter("Bad OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        free_agents = [
            _hitter("Good 1B", ["1B"], r=75, hr=22, rbi=70, sb=5, avg=0.270, ab=520, h=140),
            _hitter("Great OF", ["OF"], r=90, hr=30, rbi=85, sb=10, avg=0.285, ab=550, h=157),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"1B": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        def _sort_key(e):
            if e.best_fa is None or not e.candidates:
                return float("-inf")
            return e.candidates[0]["delta_roto"]["total"]

        keys = [_sort_key(e) for e in results]
        assert keys == sorted(keys, reverse=True)

    def test_empty_free_agents_all_no_upgrade(self):
        roster = [
            _hitter("Solo", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
        ]
        results = audit_roster(
            roster,
            [],
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        assert len(results) == 1
        assert results[0].best_fa is None
        assert results[0].gap == 0.0

    def test_cross_type_swap_pitcher_slot(self):
        """A starter could replace a weak reliever if it produces more team ERoto."""
        roster = [
            _hitter("Hitter", ["OF"], r=80, hr=25, rbi=80, sb=10, avg=0.275, ab=540, h=149),
            _pitcher(
                "Bad RP",
                ["RP"],
                ip=30,
                w=1,
                k=20,
                sv=2,
                era=5.50,
                whip=1.60,
                er=18,
                bb=15,
                h_allowed=33,
            ),
            _pitcher(
                "OK SP",
                ["SP"],
                ip=150,
                w=9,
                k=140,
                era=3.80,
                whip=1.25,
                er=63,
                bb=40,
                h_allowed=148,
            ),
        ]
        free_agents = [
            _pitcher(
                "Good SP",
                ["SP"],
                ip=180,
                w=12,
                k=180,
                era=3.20,
                whip=1.10,
                er=64,
                bb=30,
                h_allowed=168,
            ),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 2, "BN": 1, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        # The bad RP should have the Good SP as best_fa (cross-type upgrade)
        bad_rp_entry = next(e for e in results if e.player == "Bad RP")
        assert bad_rp_entry.best_fa == "Good SP"
        assert bad_rp_entry.gap > 0

    def test_il_players_excluded_from_optimization(self):
        """IL players should not be treated as active starters."""
        il_pitcher = _pitcher(
            "Hurt Closer",
            ["RP"],
            ip=60,
            w=3,
            k=60,
            sv=25,
            era=2.50,
            whip=1.00,
            er=17,
            bb=15,
            h_allowed=45,
        )
        il_pitcher.status = "IL15"

        roster = [
            _hitter("Hitter", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
            _pitcher(
                "Active SP",
                ["SP"],
                ip=150,
                w=9,
                k=140,
                era=3.80,
                whip=1.25,
                er=63,
                bb=40,
                h_allowed=148,
            ),
            il_pitcher,
        ]
        free_agents = [
            _hitter("FA Hitter", ["OF"], r=80, hr=25, rbi=80, sb=10, avg=0.280, ab=540, h=151),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 1, "BN": 1, "IL": 1},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        # IL player should appear with slot="IL" and no upgrade
        il_entry = next(e for e in results if e.player == "Hurt Closer")
        assert il_entry.slot == "IL"
        assert il_entry.best_fa is None

        # Active SP should NOT be recommended to swap for a hitter
        # (would leave 0 active pitchers for 1 P slot)
        sp_entry = next(e for e in results if e.player == "Active SP")
        assert sp_entry.best_fa is None

    def test_pitcher_fa_not_recommended_over_needed_hitter(self):
        """A pitcher FA should not replace a hitter when it would leave
        too few hitters to fill required hitter slots."""
        roster = [
            _hitter("Starter C", ["C"], r=50, hr=12, rbi=45, sb=2, avg=0.240, ab=400, h=96),
            _hitter("Starter OF", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
            _pitcher(
                "Active SP",
                ["SP"],
                ip=150,
                w=9,
                k=140,
                era=3.80,
                whip=1.25,
                er=63,
                bb=40,
                h_allowed=148,
            ),
        ]
        free_agents = [
            _pitcher(
                "Soroka",
                ["SP"],
                ip=170,
                w=11,
                k=160,
                era=3.30,
                whip=1.15,
                er=62,
                bb=35,
                h_allowed=161,
            ),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"C": 1, "OF": 1, "P": 1, "BN": 1, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        # Neither hitter should be recommended to drop for a pitcher —
        # that would leave only 1 hitter for 2 required hitter slots (C + OF).
        for entry in results:
            if entry.player_type == "hitter":
                assert entry.best_fa is None, (
                    f"{entry.player} should not be replaced by pitcher {entry.best_fa}"
                )

    def test_candidates_list_sorted_by_delta_roto(self):
        """candidates list is sorted by delta_roto.total descending."""
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher(
                "SP1", ["SP"], ip=180, w=12, k=180, era=3.50, whip=1.20, er=70, bb=40, h_allowed=176
            ),
            _pitcher(
                "SP2", ["SP"], ip=170, w=10, k=160, era=3.60, whip=1.22, er=68, bb=40, h_allowed=167
            ),
            _pitcher(
                "RP1",
                ["RP"],
                ip=60,
                w=3,
                k=60,
                era=3.00,
                whip=1.17,
                sv=20,
                er=20,
                bb=20,
                h_allowed=50,
            ),
        ]
        free_agents = [
            _hitter("FA OF 1", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
            _hitter("FA OF 2", ["OF"], r=70, hr=22, rbi=75, sb=8, avg=0.270, ab=520, h=140),
            _hitter("FA OF 3", ["OF"], r=60, hr=18, rbi=65, sb=5, avg=0.250, ab=500, h=125),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 3, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        weak_entry = next(e for e in results if e.player == "Weak OF")
        assert len(weak_entry.candidates) >= 2
        # Candidates sorted by delta_roto.total descending
        dr_totals = [c["delta_roto"]["total"] for c in weak_entry.candidates]
        assert dr_totals == sorted(dr_totals, reverse=True)
        # Every candidate dict has both ranking metric (delta_roto) and informational gap
        for c in weak_entry.candidates:
            assert "name" in c
            assert "delta_roto" in c
            assert "gap" in c
            assert "sgp" in c
            assert "player_type" in c
            assert "positions" in c

    def test_no_upgrade_when_all_delta_roto_negative(self):
        """If every candidate has delta_roto.total <= 0, best_fa is None
        but candidates list still contains the (sorted-desc) negative options."""
        roster = [
            _hitter("Star OF", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=0.300, ab=550, h=165),
            _pitcher(
                "SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
            ),
        ]
        free_agents = [
            _hitter("Downgrade", ["OF"], r=40, hr=8, rbi=30, sb=2, avg=0.220, ab=400, h=88),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        star = next(e for e in results if e.player == "Star OF")
        assert star.best_fa is None
        assert star.gap == 0.0
        assert len(star.candidates) == 1
        assert star.candidates[0]["name"] == "Downgrade"
        assert star.candidates[0]["delta_roto"]["total"] <= 0

    def test_pitcher_swap_allowed_when_active_count_below_p_slots(self):
        """Pitcher→pitcher swaps stay feasible even when the active-pitcher
        count is below ``p_slots`` (typical when multiple pitchers are on IL).

        Regression: the cross-type feasibility check used to reject
        ``len(new_pitchers) < p_slots`` on *any* swap where either side was
        a pitcher. Because same-type swaps preserve the count, that check
        blocked every pitcher→pitcher upgrade whenever baseline < p_slots,
        which is the norm for rosters with IL pitchers.
        """
        il_sp = _pitcher(
            "IL SP1", ["SP"], ip=180, w=12, k=170, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
        )
        il_sp.status = "IL15"
        il_sp2 = _pitcher(
            "IL SP2", ["SP"], ip=180, w=12, k=170, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
        )
        il_sp2.status = "IL15"

        # p_slots=3 but only 2 active pitchers after IL exclusion.
        roster = [
            _hitter("OF", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
            _pitcher(
                "Weak SP",
                ["SP"],
                ip=120,
                w=5,
                k=90,
                era=4.80,
                whip=1.45,
                er=64,
                bb=50,
                h_allowed=140,
            ),
            _pitcher(
                "OK SP",
                ["SP"],
                ip=150,
                w=9,
                k=140,
                era=3.80,
                whip=1.25,
                er=63,
                bb=40,
                h_allowed=148,
            ),
            il_sp,
            il_sp2,
        ]
        free_agents = [
            _pitcher(
                "Ace SP",
                ["SP"],
                ip=200,
                w=16,
                k=220,
                era=2.80,
                whip=1.00,
                er=62,
                bb=25,
                h_allowed=175,
            ),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 3, "BN": 1, "IL": 2},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        weak = next(e for e in results if e.player == "Weak SP")
        assert weak.candidates, "weak SP should have at least one candidate"
        assert weak.best_fa == "Ace SP"

    def test_no_cross_type_leakage(self):
        """A hitter row never gets a pitcher candidate, even if pitchers exist in FA pool."""
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher(
                "SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
            ),
        ]
        free_agents = [
            _pitcher(
                "Wacha",
                ["SP"],
                ip=170,
                w=11,
                k=160,
                era=3.40,
                whip=1.15,
                er=64,
                bb=35,
                h_allowed=160,
            ),
            _hitter("FA OF", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        weak = next(e for e in results if e.player == "Weak OF")
        cand_names = {c["name"] for c in weak.candidates}
        assert "Wacha" not in cand_names
        assert "FA OF" in cand_names

    def test_no_best_fa_when_no_upgrade(self):
        """When no FA beats the roster player on deltaRoto, best_fa is None.
        (The candidates list still contains the sorted-desc downgrade options —
        see test_no_upgrade_when_all_delta_roto_negative.)"""
        roster = [
            _hitter("Star OF", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=0.300, ab=550, h=165),
        ]
        free_agents = [
            _hitter("Scrub", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        assert results[0].best_fa is None
        assert results[0].gap == 0.0


class TestAuditILFilterUsesSlotOrStatus:
    def _player(self, name, slot, status="", player_type="hitter"):
        from fantasy_baseball.models.positions import Position

        positions = [Position.OF] if player_type == "hitter" else [Position.P]
        if player_type == "hitter":
            ros = HitterStats(
                pa=500 * 1.15,
                ab=500,
                h=130,
                r=70,
                hr=20,
                rbi=70,
                sb=5,
                avg=0.260,
            )
        else:
            ros = PitcherStats(
                ip=60.0,
                w=3.0,
                k=60.0,
                sv=0.0,
                er=20.0,
                bb=20.0,
                h_allowed=50.0,
                era=3.00,
                whip=1.17,
            )
        return Player(
            name=name,
            player_type=PlayerType(player_type),
            positions=positions,
            rest_of_season=ros,
            selected_position=Position.parse(slot) if slot else None,
            status=status,
        )

    def test_il_slot_with_empty_status_is_filtered(self):
        """A player slotted to IL with no status string set should
        still be excluded from active_roster. Yahoo sometimes omits
        status on freshly-slotted IL players.
        """
        roster = [
            self._player("Healthy OF", "OF"),
            self._player("IL Pitcher", "IL", status="", player_type="pitcher"),
        ]
        free_agents = []
        roster_slots = {
            "OF": 3,
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "IF": 1,
            "UTIL": 2,
            "P": 9,
            "BN": 2,
            "IL": 2,
        }

        entries = audit_roster(
            roster,
            free_agents,
            roster_slots,
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        il_entries = [e for e in entries if e.player == "IL Pitcher"]
        assert len(il_entries) == 1
        assert il_entries[0].slot == "IL"

    def test_il_slot_with_il15_status_is_filtered(self):
        """A player with both IL slot and IL15 status (the typical
        case) is still correctly excluded."""
        roster = [
            self._player("Healthy OF", "OF"),
            self._player("IL Pitcher", "IL", status="IL15", player_type="pitcher"),
        ]
        free_agents = []
        roster_slots = {
            "OF": 3,
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "IF": 1,
            "UTIL": 2,
            "P": 9,
            "BN": 2,
            "IL": 2,
        }

        entries = audit_roster(
            roster,
            free_agents,
            roster_slots,
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        il_entries = [e for e in entries if e.player == "IL Pitcher"]
        assert len(il_entries) == 1
        assert il_entries[0].slot == "IL"

    def test_bn_slot_with_il10_status_is_filtered(self):
        """A player slotted to BN with an IL10 status (the Soto case:
        Yahoo bench-slotted the IL player) is excluded via the
        status check."""
        roster = [
            self._player("Healthy OF", "OF"),
            self._player("Bench IL Hitter", "BN", status="IL10", player_type="hitter"),
        ]
        free_agents = []
        roster_slots = {
            "OF": 3,
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "IF": 1,
            "UTIL": 2,
            "P": 9,
            "BN": 2,
            "IL": 2,
        }

        entries = audit_roster(
            roster,
            free_agents,
            roster_slots,
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )

        il_entries = [e for e in entries if e.player == "Bench IL Hitter"]
        assert len(il_entries) == 1
        assert il_entries[0].slot == "IL"


class TestRegressionFixtures:
    """Lock in fixes for the specific broken-audit cases flagged by the user."""

    def test_herrera_style_negative_delta_roto_yields_no_upgrade(self):
        """Herrera-like catcher whose only C-pool option is a downgrade:
        collapsed row shows no upgrade, expanded candidate list still surfaces
        the negative option (per design clarification c)."""
        roster = [
            # Herrera-ish: decent AVG at modest PA
            _hitter("Herrera", ["C"], r=60, hr=15, rbi=55, sb=2, avg=0.257, ab=486, h=125),
            _hitter("OF Filler", ["OF"], r=70, hr=18, rbi=60, sb=5, avg=0.260, ab=500, h=130),
            _pitcher(
                "SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
            ),
        ]
        free_agents = [
            # Perez-ish: worse AVG, more PA → hurts team AVG, no offsetting gains
            _hitter("Perez", ["C"], r=55, hr=18, rbi=65, sb=1, avg=0.239, ab=554, h=132),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"C": 1, "OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        herrera = next(e for e in results if e.player == "Herrera")
        # Collapsed row shows no upgrade
        assert herrera.best_fa is None
        assert herrera.gap == 0.0
        # Expanded view still has Perez visible with a negative delta_roto
        assert len(herrera.candidates) == 1
        assert herrera.candidates[0]["name"] == "Perez"
        assert herrera.candidates[0]["delta_roto"]["total"] <= 0

    def test_adolis_style_of_row_never_recommends_pitcher(self):
        """Adolis-like OF row must never get a pitcher candidate, even when
        the FA pool is pitcher-heavy and a scoring bug might have ranked
        pitchers above the true best OF upgrade (this was the 'hidden Ward'
        bug)."""
        roster = [
            _hitter("Adolis", ["OF"], r=70, hr=25, rbi=75, sb=6, avg=0.235, ab=541, h=127),
            _pitcher(
                "SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10, er=64, bb=30, h_allowed=168
            ),
        ]
        # FA pool contains pitchers that could have high swap value but are
        # invalid as hitter-row candidates. And one legitimate OF upgrade.
        free_agents = [
            _pitcher(
                "Wacha",
                ["SP"],
                ip=170,
                w=11,
                k=160,
                era=3.40,
                whip=1.15,
                er=64,
                bb=35,
                h_allowed=160,
            ),
            _pitcher(
                "Keller",
                ["SP"],
                ip=175,
                w=11,
                k=165,
                era=3.45,
                whip=1.18,
                er=67,
                bb=38,
                h_allowed=164,
            ),
            _pitcher(
                "Springs",
                ["SP"],
                ip=160,
                w=10,
                k=155,
                era=3.35,
                whip=1.14,
                er=60,
                bb=32,
                h_allowed=150,
            ),
            _pitcher(
                "López",
                ["SP"],
                ip=165,
                w=10,
                k=150,
                era=3.50,
                whip=1.20,
                er=64,
                bb=38,
                h_allowed=158,
            ),
            _pitcher(
                "Cantillo",
                ["SP"],
                ip=155,
                w=9,
                k=145,
                era=3.60,
                whip=1.22,
                er=62,
                bb=40,
                h_allowed=150,
            ),
            _hitter("Ward", ["OF"], r=80, hr=28, rbi=80, sb=10, avg=0.275, ab=569, h=156),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        adolis = next(e for e in results if e.player == "Adolis")
        cand_names = {c["name"] for c in adolis.candidates}
        # Zero pitcher leakage into the hitter row's candidate list
        for pitcher_name in ("Wacha", "Keller", "Springs", "López", "Cantillo"):
            assert pitcher_name not in cand_names, (
                f"{pitcher_name} (pitcher) should not be an Adolis candidate"
            )
        # Ward is the only viable candidate
        assert "Ward" in cand_names
        assert adolis.best_fa == "Ward"


class TestAuditRosterTeamSDs:
    """audit_roster threads team_sds into compute_delta_roto so within-
    uncertainty swaps produce fractional deltas instead of rank flips."""

    def _twelve_team_sb_standings(self):
        cats = {
            "R": 0,
            "HR": 0,
            "RBI": 0,
            "SB": 0,
            "AVG": 0,
            "W": 0,
            "K": 0,
            "SV": 0,
            "ERA": 0,
            "WHIP": 0,
        }
        rows = [
            {"name": TEAM_NAME, "stats": {**cats, "SB": 100}},
            {"name": "Rival", "stats": {**cats, "SB": 99}},
        ] + [{"name": f"T{i}", "stats": {**cats, "SB": 10 + i}} for i in range(10)]
        return _projected(rows)

    def test_team_sds_none_produces_full_rank_flip(self):
        roster = [_hitter("Drop", ["OF"], r=0, hr=0, rbi=0, sb=20, ab=100, h=0, avg=0.0)]
        fas = [_hitter("Add", ["OF"], r=0, hr=0, rbi=0, sb=10, ab=100, h=0, avg=0.0)]
        entries = audit_roster(
            roster,
            fas,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=self._twelve_team_sb_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=None,
        )
        drop_entry = next(e for e in entries if e.player == "Drop")
        assert drop_entry.candidates[0]["delta_roto"]["categories"]["SB"][
            "roto_delta"
        ] == pytest.approx(-1.0)

    def test_team_sds_wide_produces_fractional_delta(self):
        roster = [_hitter("Drop", ["OF"], r=0, hr=0, rbi=0, sb=20, ab=100, h=0, avg=0.0)]
        fas = [_hitter("Add", ["OF"], r=0, hr=0, rbi=0, sb=10, ab=100, h=0, avg=0.0)]
        standings = self._twelve_team_sb_standings()
        team_sds = {e.team_name: dict.fromkeys(ALL_CATEGORIES, 0.0) for e in standings.entries}
        team_sds[TEAM_NAME][Category.SB] = 10.0
        team_sds["Rival"][Category.SB] = 10.0
        entries = audit_roster(
            roster,
            fas,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=standings,
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=team_sds,
        )
        drop_entry = next(e for e in entries if e.player == "Drop")
        sb_delta = drop_entry.candidates[0]["delta_roto"]["categories"]["SB"]["roto_delta"]
        assert abs(sb_delta) < 0.5


class TestAuditRosterBand:
    """audit_roster attaches a MC confidence band to each scored candidate."""

    def test_band_keys_present_on_scored_candidates(self):
        """Every scored candidate dict has 'band' with the expected sub-keys."""
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher(
                "Decent SP",
                ["SP"],
                ip=180,
                w=12,
                k=180,
                era=3.50,
                whip=1.20,
                er=70,
                bb=40,
                h_allowed=176,
            ),
            _pitcher(
                "Decent RP",
                ["RP"],
                ip=60,
                w=3,
                k=60,
                era=3.00,
                whip=1.17,
                sv=20,
                er=20,
                bb=20,
                h_allowed=50,
            ),
        ]
        free_agents = [
            _hitter("Better OF", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
            _hitter("Decent OF", ["OF"], r=60, hr=18, rbi=65, sb=6, avg=0.260, ab=500, h=130),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 2, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=0.6,
        )

        weak_entry = next(e for e in results if e.player == "Weak OF")
        assert weak_entry.candidates, "expected at least one scored candidate"

        required_band_keys = {"mean", "sd", "p_positive", "verdict"}
        for c in weak_entry.candidates:
            assert "band" in c, f"candidate {c['name']} missing 'band' key"
            band = c["band"]
            assert required_band_keys == set(band.keys()), (
                f"band keys mismatch: {set(band.keys())} != {required_band_keys}"
            )
            assert band["sd"] >= 0, f"sd must be non-negative, got {band['sd']}"
            assert 0.0 <= band["p_positive"] <= 1.0
            assert band["verdict"] in {"real", "coin-flip", "downgrade"}


class TestWorstRosterByPosition:
    """worst_roster_by_position picks the lowest-SGP roster player eligible
    at each pool position — the canonical drop target for FA evaluation."""

    def test_single_position_picks_lowest_sgp(self):
        from fantasy_baseball.lineup.roster_audit import worst_roster_by_position

        roster = [
            _hitter("Strong", ["OF"], r=100, hr=30, rbi=100, sb=20, avg=0.290, ab=500, h=145),
            _hitter("Weak", ["OF"], r=40, hr=5, rbi=40, sb=2, avg=0.230, ab=300, h=69),
        ]
        worst = worst_roster_by_position(roster)
        assert worst["OF"].name == "Weak"

    def test_multi_position_player_ranked_at_every_position(self):
        from fantasy_baseball.lineup.roster_audit import worst_roster_by_position

        roster = [
            _hitter("Dual", ["2B", "SS"], r=40, hr=5, rbi=40, sb=2, avg=0.230, ab=300, h=69),
            _hitter("SSOnly", ["SS"], r=100, hr=30, rbi=100, sb=20, avg=0.290, ab=500, h=145),
        ]
        worst = worst_roster_by_position(roster)
        assert worst["2B"].name == "Dual"  # only 2B-eligible player
        assert worst["SS"].name == "Dual"  # lower SGP than SSOnly

    def test_pitchers_bucket_by_saves_threshold(self):
        from fantasy_baseball.lineup.roster_audit import worst_roster_by_position

        roster = [
            _pitcher("Closer", ["P"], ip=60, sv=30, k=70, w=3, era=2.80),
            _pitcher("WeakCloser", ["P"], ip=40, sv=10, k=40, w=1, era=4.20),
            _pitcher("Ace", ["P"], ip=180, sv=0, k=200, w=15, era=2.90),
            _pitcher("WeakSP", ["P"], ip=120, sv=0, k=90, w=5, era=4.80),
        ]
        worst = worst_roster_by_position(roster)
        assert worst["RP"].name == "WeakCloser"
        assert worst["SP"].name == "WeakSP"

    def test_positions_with_no_eligible_player_are_absent(self):
        from fantasy_baseball.lineup.roster_audit import worst_roster_by_position

        roster = [_hitter("OFGuy", ["OF"], r=50, hr=10, rbi=50, sb=5, avg=0.250, ab=400, h=100)]
        worst = worst_roster_by_position(roster)
        assert list(worst.keys()) == ["OF"]  # no C/1B/2B/3B/SS/SP/RP keys
        assert worst["OF"].name == "OFGuy"


class TestFATargetPositions:
    """fa_target_positions drives ΔRoto lookup for the players page."""

    def test_hitter_keeps_only_source_positions(self):
        from fantasy_baseball.lineup.roster_audit import fa_target_positions

        # Util / IF should not produce a ΔRoto target — they're lineup-only slots.
        assert fa_target_positions("hitter", ["2B", "SS", "IF", "Util"], 0.0) == ["2B", "SS"]

    def test_pitcher_sv_below_threshold_is_sp(self):
        from fantasy_baseball.lineup.roster_audit import fa_target_positions

        assert fa_target_positions("pitcher", ["P"], 2.0) == ["SP"]

    def test_pitcher_sv_at_or_above_threshold_is_rp(self):
        from fantasy_baseball.lineup.roster_audit import fa_target_positions

        assert fa_target_positions("pitcher", ["P"], 25.0) == ["RP"]


class TestBandConsistency:
    """Analytic band consistency and performance contracts.

    band["mean"] must equal candidate["delta_roto"]["total"] within 1e-9
    (both are the same EV delta -- the band is not an approximation).
    The analytic band must also be cheap: audit_roster over ~45 FAs
    must complete in well under 3 seconds.
    """

    def _make_standings(self):
        """Minimal 2-team standings for band-consistency tests."""
        base = {
            "R": 700,
            "HR": 180,
            "RBI": 700,
            "SB": 80,
            "AVG": 0.255,
            "W": 65,
            "K": 1100,
            "SV": 45,
            "ERA": 3.60,
            "WHIP": 1.22,
            "AB": 4800,
            "H": 1224,
            "IP": 1350,
            "ER": 540,
            "BB": 405,
            "H_ALLOWED": 1215,
        }
        return _projected(
            [
                {"name": TEAM_NAME, "stats": dict(base)},
                {"name": "Opponent", "stats": {**base, "SB": 50, "HR": 160}},
            ]
        )

    def test_band_mean_equals_delta_roto_total(self):
        """band["mean"] == candidate["delta_roto"]["total"] within 1e-9.

        The analytic band's mean is the EV deltaRoto -- not a separate
        estimate. Both must agree exactly (modulo floating-point round trip
        from .to_dict() rounding to 2 d.p.).
        """
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher(
                "Decent SP",
                ["SP"],
                ip=180,
                w=12,
                k=180,
                era=3.50,
                whip=1.20,
                er=70,
                bb=40,
                h_allowed=176,
            ),
        ]
        free_agents = [
            _hitter("Better OF", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
            _hitter("Decent OF", ["OF"], r=65, hr=18, rbi=65, sb=6, avg=0.260, ab=500, h=130),
        ]
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=self._make_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=0.6,
        )

        weak = next(e for e in results if e.player == "Weak OF")
        assert weak.candidates, "expected scored candidates for Weak OF"

        for c in weak.candidates:
            dr_total = c["delta_roto"]["total"]
            band_mean = c["band"]["mean"]
            # Both are rounded to 2 d.p. by .to_dict() -- tolerance matches
            # the rounding of two independent .round(2) calls.
            assert abs(band_mean - dr_total) < 0.011, (
                f"band.mean {band_mean} != delta_roto.total {dr_total} for "
                f"candidate {c['name']} (should agree within round-trip precision)"
            )

    def test_perf_smoke_45_free_agents(self):
        """audit_roster over ~45 FAs completes in well under 3 seconds.

        The analytic band has no sampling loop so the cost is linear in
        the number of categories (10) and Gauss-Hermite nodes (9),
        not in n_draws. A 45-FA audit should be sub-second on any
        reasonable machine; 3s is a generous ceiling.
        """
        import time

        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher(
                "Decent SP",
                ["SP"],
                ip=180,
                w=12,
                k=180,
                era=3.50,
                whip=1.20,
                er=70,
                bb=40,
                h_allowed=176,
            ),
            _pitcher(
                "Decent RP",
                ["RP"],
                ip=60,
                w=3,
                k=60,
                era=3.00,
                whip=1.17,
                sv=20,
                er=20,
                bb=20,
                h_allowed=50,
            ),
        ]
        # 30 OF + 15 SP free agents = 45 total
        free_agents = [
            _hitter(
                f"FA OF {i}",
                ["OF"],
                r=50 + i,
                hr=10 + i // 3,
                rbi=50 + i,
                sb=3 + i % 5,
                avg=0.245 + i * 0.001,
                ab=450 + i * 2,
                h=int((0.245 + i * 0.001) * (450 + i * 2)),
            )
            for i in range(30)
        ] + [
            _pitcher(
                f"FA SP {i}",
                ["SP"],
                ip=150 + i,
                w=9 + i // 5,
                k=140 + i * 2,
                era=3.20 + i * 0.05,
                whip=1.10 + i * 0.01,
                er=int((3.20 + i * 0.05) * (150 + i) / 9),
                bb=30 + i,
                h_allowed=140 + i,
            )
            for i in range(15)
        ]

        t0 = time.perf_counter()
        results = audit_roster(
            roster,
            free_agents,
            {"OF": 1, "P": 2, "BN": 0, "IL": 0},
            projected_standings=self._make_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=0.6,
        )
        elapsed = time.perf_counter() - t0

        assert len(results) == len(roster), "audit must produce one entry per roster player"
        assert elapsed < 3.0, (
            f"audit_roster over 45 FAs took {elapsed:.2f}s -- analytic band "
            f"must be cheap enough to stay well under 3s"
        )


class TestSgpOverridesFlowThrough:
    """Prove config-sourced sgp_overrides actually reach the SGP math.

    A distinctive override (SB denom 100 makes steals nearly worthless, R
    denom 1 makes runs enormous) must change pool ordering / audit SGP
    relative to the code defaults -- if these pass with the override
    silently dropped, the plumbing is broken.
    """

    def _speed_and_power(self):
        speedy = _hitter("Speedy", ["OF"], r=70, hr=10, rbi=70, sb=10, avg=0.260, ab=500, h=130)
        slugger = _hitter("Slugger", ["OF"], r=70, hr=20, rbi=70, sb=0, avg=0.260, ab=500, h=130)
        return speedy, slugger

    def test_build_position_pools_override_reorders_pool(self):
        speedy, slugger = self._speed_and_power()
        # Defaults: 10 SB / 8 > 10 HR / 9, so Speedy edges Slugger.
        default_pools = build_position_pools([speedy, slugger])
        assert [p.name for p in default_pools["OF"]] == ["Speedy", "Slugger"]
        # SB denom 100 crushes the SB contribution; Slugger takes the lead.
        override_pools = build_position_pools(
            [speedy, slugger], denoms=get_sgp_denominators({"SB": 100.0})
        )
        assert [p.name for p in override_pools["OF"]] == ["Slugger", "Speedy"]

    def test_worst_roster_by_position_override_flips_drop_target(self):
        from fantasy_baseball.lineup.roster_audit import worst_roster_by_position

        speedy, slugger = self._speed_and_power()
        assert worst_roster_by_position([speedy, slugger])["OF"].name == "Slugger"
        worst = worst_roster_by_position(
            [speedy, slugger], denoms=get_sgp_denominators({"SB": 100.0})
        )
        assert worst["OF"].name == "Speedy"

    def test_audit_roster_override_changes_player_sgp(self):
        roster = [
            _hitter("Only OF", ["OF"], r=50, hr=10, rbi=50, sb=5, avg=0.250, ab=400, h=100),
        ]
        kwargs = dict(
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
        )
        slots = {"OF": 1, "P": 0, "BN": 0, "IL": 0}
        default_entry = audit_roster(roster, [], slots, **kwargs)[0]
        override_entry = audit_roster(roster, [], slots, sgp_overrides={"R": 1.0}, **kwargs)[0]
        # R denom 1.0 turns 50 R into 50 SGP; the default (R/20) cannot match.
        assert override_entry.player_sgp > default_entry.player_sgp + 40
