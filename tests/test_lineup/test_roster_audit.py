from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.lineup.roster_audit import audit_roster


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

TEAM_NAME = "Test Team"


def _minimal_standings():
    """A minimal 2-team projected_standings fixture for audit tests.

    The exact stat values don't matter for most tests — we just need the
    structure `compute_delta_roto` can consume (user's team + at least one
    opponent for defense-comfort to compute gaps).
    """
    base = {"R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
            "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20,
            "AB": 5000, "H": 1300, "IP": 1400, "ER": 560, "BB": 420, "H_ALLOWED": 1300}
    return [
        {"name": TEAM_NAME, "stats": dict(base)},
        {"name": "Opponent", "stats": {**base, "SV": 30, "ERA": 3.80}},
    ]


ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "UTIL": 1, "P": 3, "BN": 2, "IL": 0}


def _hitter(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=positions,
        rest_of_season=HitterStats(
            pa=int(stats.get("ab", 500) * 1.15),
            ab=stats.get("ab", 500), h=stats.get("h", 130),
            r=stats.get("r", 70), hr=stats.get("hr", 20),
            rbi=stats.get("rbi", 70), sb=stats.get("sb", 5),
            avg=stats.get("avg", 0.260),
        ),
    )


def _pitcher(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=positions,
        rest_of_season=PitcherStats(
            ip=stats.get("ip", 60.0), w=stats.get("w", 3.0),
            k=stats.get("k", 60.0), sv=stats.get("sv", 0.0),
            er=stats.get("er", 20.0), bb=stats.get("bb", 20.0),
            h_allowed=stats.get("h_allowed", 50.0),
            era=stats.get("era", 3.00), whip=stats.get("whip", 1.17),
        ),
    )


from fantasy_baseball.lineup.roster_audit import build_position_pools, POSITION_POOL_SIZES


class TestBuildPositionPools:
    def test_hitter_buckets_by_all_positions(self):
        """A 2B/OF-eligible hitter appears in both 2B and OF pools."""
        multi = _hitter("Multi Pos", ["2B", "OF"], r=80, hr=25, rbi=75, sb=10, avg=0.280, ab=520, h=146)
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
        assert pools["OF"][0].name == "OF0"   # highest R → highest SGP
        assert pools["OF"][-1].name == f"OF{POSITION_POOL_SIZES['OF'] - 1}"

    def test_pitcher_pools(self):
        """SP-eligible pitcher lands in SP pool, RP-eligible in RP pool."""
        sp = _pitcher("SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                      er=64, bb=30, h_allowed=168)
        rp = _pitcher("RP", ["RP"], ip=60, w=3, k=60, sv=20, era=3.00, whip=1.17,
                      er=20, bb=20, h_allowed=50)
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


from fantasy_baseball.lineup.roster_audit import candidates_for_player


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
        multi = _hitter("Multi FA", ["2B", "OF"], r=75, hr=20, rbi=70, sb=10, avg=0.272, ab=510, h=139)
        pools = build_position_pools([b2, of_, multi])
        roster_multi = _hitter("Roster 2B/OF", ["2B", "OF"], r=60, hr=10, rbi=50, sb=5, avg=0.250, ab=480, h=120)
        cands = candidates_for_player(roster_multi, pools)
        assert b2 in cands
        assert of_ in cands
        assert multi in cands
        # Deduped: multi appears once even though it's in both pools
        assert sum(1 for c in cands if c.name == "Multi FA") == 1

    def test_pitcher_pulls_from_sp_union_rp(self):
        """A Yahoo roster pitcher (positions=['P']) gets SP pool and RP pool."""
        sp = _pitcher("SP FA", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                      er=64, bb=30, h_allowed=168)
        rp = _pitcher("RP FA", ["RP"], ip=60, w=3, k=60, sv=20, era=3.00, whip=1.17,
                      er=20, bb=20, h_allowed=50)
        pools = build_position_pools([sp, rp])
        roster_pitcher = _pitcher("Roster P", ["P"], ip=100, w=6, k=80, era=4.00, whip=1.30,
                                  er=44, bb=30, h_allowed=100)
        cands = candidates_for_player(roster_pitcher, pools)
        assert sp in cands
        assert rp in cands

    def test_hitter_never_gets_pitcher_candidates(self):
        """A hitter (positions=['OF']) never gets candidates from SP/RP pools."""
        sp = _pitcher("SP FA", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                      er=64, bb=30, h_allowed=168)
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
        util_only = _hitter("Util Only", ["UTIL"], r=60, hr=10, rbi=50, sb=5, avg=0.250, ab=480, h=120)
        cands = candidates_for_player(util_only, pools)
        assert cands == []


class TestAuditRoster:
    def test_identifies_upgrade_available(self):
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher("Decent SP", ["SP"], ip=180, w=12, k=180, era=3.50, whip=1.20,
                     er=70, bb=40, h_allowed=176),
            _pitcher("Decent SP2", ["SP"], ip=170, w=10, k=160, era=3.60, whip=1.22,
                     er=68, bb=40, h_allowed=167),
            _pitcher("Decent RP", ["RP"], ip=60, w=3, k=60, era=3.00, whip=1.17,
                     sv=20, er=20, bb=20, h_allowed=50),
        ]
        free_agents = [
            _hitter("Better OF", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 3, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
            roster, free_agents, EQUAL_LEVERAGE,
            {"1B": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
            roster, [], EQUAL_LEVERAGE,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
        )
        assert len(results) == 1
        assert results[0].best_fa is None
        assert results[0].gap == 0.0

    def test_cross_type_swap_pitcher_slot(self):
        """A starter could replace a weak reliever if it produces more team wSGP."""
        roster = [
            _hitter("Hitter", ["OF"], r=80, hr=25, rbi=80, sb=10, avg=0.275, ab=540, h=149),
            _pitcher("Bad RP", ["RP"], ip=30, w=1, k=20, sv=2, era=5.50, whip=1.60,
                     er=18, bb=15, h_allowed=33),
            _pitcher("OK SP", ["SP"], ip=150, w=9, k=140, era=3.80, whip=1.25,
                     er=63, bb=40, h_allowed=148),
        ]
        free_agents = [
            _pitcher("Good SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 2, "BN": 1, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
        )
        # The bad RP should have the Good SP as best_fa (cross-type upgrade)
        bad_rp_entry = next(e for e in results if e.player == "Bad RP")
        assert bad_rp_entry.best_fa == "Good SP"
        assert bad_rp_entry.gap > 0

    def test_il_players_excluded_from_optimization(self):
        """IL players should not be treated as active starters."""
        il_pitcher = _pitcher("Hurt Closer", ["RP"], ip=60, w=3, k=60, sv=25,
                              era=2.50, whip=1.00, er=17, bb=15, h_allowed=45)
        il_pitcher.status = "IL15"

        roster = [
            _hitter("Hitter", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
            _pitcher("Active SP", ["SP"], ip=150, w=9, k=140, era=3.80, whip=1.25,
                     er=63, bb=40, h_allowed=148),
            il_pitcher,
        ]
        free_agents = [
            _hitter("FA Hitter", ["OF"], r=80, hr=25, rbi=80, sb=10, avg=0.280, ab=540, h=151),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 1, "BN": 1, "IL": 1},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
            _pitcher("Active SP", ["SP"], ip=150, w=9, k=140, era=3.80, whip=1.25,
                     er=63, bb=40, h_allowed=148),
        ]
        free_agents = [
            _pitcher("Soroka", ["SP"], ip=170, w=11, k=160, era=3.30, whip=1.15,
                     er=62, bb=35, h_allowed=161),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"C": 1, "OF": 1, "P": 1, "BN": 1, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
        )

        # Neither hitter should be recommended to drop for a pitcher —
        # that would leave only 1 hitter for 2 required hitter slots (C + OF).
        for entry in results:
            if entry.player_type == "hitter":
                assert entry.best_fa is None, (
                    f"{entry.player} should not be replaced by pitcher {entry.best_fa}"
                )

    def test_candidates_list_sorted_by_delta_roto(self):
        """candidates list is sorted by delta_roto.total descending (not wSGP)."""
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher("SP1", ["SP"], ip=180, w=12, k=180, era=3.50, whip=1.20,
                     er=70, bb=40, h_allowed=176),
            _pitcher("SP2", ["SP"], ip=170, w=10, k=160, era=3.60, whip=1.22,
                     er=68, bb=40, h_allowed=167),
            _pitcher("RP1", ["RP"], ip=60, w=3, k=60, era=3.00, whip=1.17,
                     sv=20, er=20, bb=20, h_allowed=50),
        ]
        free_agents = [
            _hitter("FA OF 1", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
            _hitter("FA OF 2", ["OF"], r=70, hr=22, rbi=75, sb=8, avg=0.270, ab=520, h=140),
            _hitter("FA OF 3", ["OF"], r=60, hr=18, rbi=65, sb=5, avg=0.250, ab=500, h=125),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 3, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
            assert "wsgp" in c
            assert "player_type" in c
            assert "positions" in c

    def test_no_upgrade_when_all_delta_roto_negative(self):
        """If every candidate has delta_roto.total <= 0, best_fa is None
        but candidates list still contains the (sorted-desc) negative options."""
        roster = [
            _hitter("Star OF", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=0.300, ab=550, h=165),
            _pitcher("SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        free_agents = [
            _hitter("Downgrade", ["OF"], r=40, hr=8, rbi=30, sb=2, avg=0.220, ab=400, h=88),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
        )
        star = next(e for e in results if e.player == "Star OF")
        assert star.best_fa is None
        assert star.gap == 0.0
        assert len(star.candidates) == 1
        assert star.candidates[0]["name"] == "Downgrade"
        assert star.candidates[0]["delta_roto"]["total"] <= 0

    def test_no_cross_type_leakage(self):
        """A hitter row never gets a pitcher candidate, even if pitchers exist in FA pool."""
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher("SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        free_agents = [
            _pitcher("Wacha", ["SP"], ip=170, w=11, k=160, era=3.40, whip=1.15,
                     er=64, bb=35, h_allowed=160),
            _hitter("FA OF", ["OF"], r=80, hr=25, rbi=75, sb=8, avg=0.275, ab=520, h=143),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 0, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
                ab=500, h=130,
                r=70, hr=20,
                rbi=70, sb=5,
                avg=0.260,
            )
        else:
            ros = PitcherStats(
                ip=60.0, w=3.0,
                k=60.0, sv=0.0,
                er=20.0, bb=20.0,
                h_allowed=50.0,
                era=3.00, whip=1.17,
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
        leverage = {cat: 0.1 for cat in
                    ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
        roster_slots = {"OF": 3, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
                        "IF": 1, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

        entries = audit_roster(
            roster, free_agents, leverage, roster_slots,
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
        leverage = {cat: 0.1 for cat in
                    ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
        roster_slots = {"OF": 3, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
                        "IF": 1, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

        entries = audit_roster(
            roster, free_agents, leverage, roster_slots,
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
        leverage = {cat: 0.1 for cat in
                    ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
        roster_slots = {"OF": 3, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
                        "IF": 1, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

        entries = audit_roster(
            roster, free_agents, leverage, roster_slots,
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
        )

        il_entries = [e for e in entries if e.player == "Bench IL Hitter"]
        assert len(il_entries) == 1
        assert il_entries[0].slot == "IL"


class TestRegressionFixtures:
    """Lock in fixes for the specific broken-audit cases flagged by the user."""

    def test_herrera_style_no_upgrade_shows_no_upgrade_and_surfaces_negative_candidate(self):
        """Herrera-like catcher whose only C-pool option is a downgrade:
        collapsed row shows no upgrade, expanded candidate list still surfaces
        the negative option (per design clarification c)."""
        roster = [
            # Herrera-ish: decent AVG at modest PA
            _hitter("Herrera", ["C"], r=60, hr=15, rbi=55, sb=2, avg=0.257, ab=486, h=125),
            _hitter("OF Filler", ["OF"], r=70, hr=18, rbi=60, sb=5, avg=0.260, ab=500, h=130),
            _pitcher("SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        free_agents = [
            # Perez-ish: worse AVG, more PA → hurts team AVG, no offsetting gains
            _hitter("Perez", ["C"], r=55, hr=18, rbi=65, sb=1, avg=0.239, ab=554, h=132),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"C": 1, "OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
        the FA pool is pitcher-heavy and wSGP might have ranked pitchers above
        the true best OF upgrade (this was the 'hidden Ward' bug)."""
        roster = [
            _hitter("Adolis", ["OF"], r=70, hr=25, rbi=75, sb=6, avg=0.235, ab=541, h=127),
            _pitcher("SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        # FA pool contains pitchers that could have high swap-wSGP but are
        # invalid as hitter-row candidates. And one legitimate OF upgrade.
        free_agents = [
            _pitcher("Wacha", ["SP"], ip=170, w=11, k=160, era=3.40, whip=1.15,
                     er=64, bb=35, h_allowed=160),
            _pitcher("Keller", ["SP"], ip=175, w=11, k=165, era=3.45, whip=1.18,
                     er=67, bb=38, h_allowed=164),
            _pitcher("Springs", ["SP"], ip=160, w=10, k=155, era=3.35, whip=1.14,
                     er=60, bb=32, h_allowed=150),
            _pitcher("López", ["SP"], ip=165, w=10, k=150, era=3.50, whip=1.20,
                     er=64, bb=38, h_allowed=158),
            _pitcher("Cantillo", ["SP"], ip=155, w=9, k=145, era=3.60, whip=1.22,
                     er=62, bb=40, h_allowed=150),
            _hitter("Ward", ["OF"], r=80, hr=28, rbi=80, sb=10, avg=0.275, ab=569, h=156),
        ]
        results = audit_roster(
            roster, free_agents, EQUAL_LEVERAGE,
            {"OF": 1, "P": 1, "BN": 0, "IL": 0},
            projected_standings=_minimal_standings(),
            team_name=TEAM_NAME,
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
