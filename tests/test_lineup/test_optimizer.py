from datetime import date

import pytest

from fantasy_baseball.lineup.optimizer import (
    HitterAssignment,
    PitcherStarter,
    _TeamContext,
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
    team_roto_total,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import (
    Category,
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
    Standings,
    StandingsEntry,
    TeamYtdComponents,
)
from fantasy_baseball.scoring import project_team_stats

CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]


def _hitter(name, positions, r=70, hr=20, rbi=70, sb=10, h=120, ab=450, pa=500):
    avg = (h / ab) if ab > 0 else 0
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=HitterStats(pa=pa, ab=ab, h=h, r=r, hr=hr, rbi=rbi, sb=sb, avg=avg),
        selected_position=Position.parse(positions[0]),
    )


def _pitcher(name, positions, ip=180, w=12, k=180, sv=0, era=3.50, whip=1.20):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=PitcherStats(
            ip=ip,
            w=w,
            k=k,
            sv=sv,
            era=era,
            whip=whip,
            er=era * ip / 9,
            bb=int(whip * ip * 0.3),
            h_allowed=int(whip * ip * 0.7),
        ),
        selected_position=Position.parse(positions[0]),
    )


def _zero_stats():
    return {c: 0.0 for c in CATEGORIES}


def _standing(name: str, **overrides) -> ProjectedStandingsEntry:
    stats = _zero_stats()
    stats.update(overrides)
    return ProjectedStandingsEntry(
        team_name=name,
        stats=CategoryStats.from_dict(stats),
    )


def _standings(*entries: ProjectedStandingsEntry) -> ProjectedStandings:
    return ProjectedStandings(effective_date=date.min, entries=list(entries))


SMALL_ROSTER_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
    "BN": 2,
    "P": 9,
    "IL": 0,
}


class TestBasic:
    def test_fills_all_required_slots(self):
        hitters = [
            _hitter("C1", ["C"]),
            _hitter("1B1", ["1B"]),
            _hitter("2B1", ["2B"]),
            _hitter("3B1", ["3B"]),
            _hitter("SS1", ["SS"]),
            _hitter("OF1", ["OF"]),
            _hitter("OF2", ["OF"]),
            _hitter("OF3", ["OF"]),
            _hitter("OF4", ["OF"]),
            _hitter("UTIL1", ["1B", "OF"]),
        ]
        standings = _standings(
            _standing("Us", R=0, HR=0, RBI=0, SB=0, AVG=0),
            _standing("Rival", R=1, HR=1, RBI=1, SB=1, AVG=0),
        )
        lineup = optimize_hitter_lineup(
            hitters=hitters,
            full_roster=hitters,
            projected_standings=standings,
            team_name="Us",
            roster_slots=SMALL_ROSTER_SLOTS,
        )
        assert isinstance(lineup, list)
        assert len(lineup) == 9  # C, 1B, 2B, 3B, SS, OF, OF, OF, UTIL
        slot_counts: dict = {}
        for a in lineup:
            assert isinstance(a, HitterAssignment)
            slot_counts[a.slot] = slot_counts.get(a.slot, 0) + 1
        assert slot_counts[Position.C] == 1
        assert slot_counts[Position.OF] == 3
        assert slot_counts[Position.UTIL] == 1

    def test_hitter_band_present_when_fraction_remaining_given(self):
        """Passing fraction_remaining attaches a band dict to each assignment."""
        a = _hitter("A", ["OF"], r=80, hr=25)
        b = _hitter("B", ["OF"], r=70, hr=20)
        c = _hitter("C", ["OF"], r=60, hr=15)
        slots = {"OF": 2, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(
            _standing("Us", R=0, HR=0),
            _standing("Rival", R=1, HR=1),
        )
        lineup = optimize_hitter_lineup(
            hitters=[a, b, c],
            full_roster=[a, b, c],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
            fraction_remaining=0.6,
        )
        assert len(lineup) == 2
        d = lineup[0].to_dict()
        assert "band" in d
        assert d["band"] is not None
        band_keys = set(d["band"].keys())
        assert band_keys == {"mean", "sd", "p_positive", "verdict"}

    def test_hitter_band_none_when_fraction_remaining_omitted(self):
        """Without fraction_remaining, band stays None."""
        a = _hitter("A", ["OF"], r=80, hr=25)
        b = _hitter("B", ["OF"], r=70, hr=20)
        slots = {"OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(_standing("Us"), _standing("Rival", R=1, HR=1))
        lineup = optimize_hitter_lineup(
            hitters=[a, b],
            full_roster=[a, b],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        assert lineup[0].band is None
        assert lineup[0].to_dict()["band"] is None


class TestERotoMaximization:
    def test_picks_hitter_who_lifts_category_boundary(self):
        """Scenario: 2 teams + us. We're tied with Rival on SB.
        Hitter A has more HR but we already dominate HR.
        Hitter B has more SB and starting him flips the SB tiebreak.
        Lineup must start Hitter B even though A has higher overall stats.
        """
        a = _hitter("A", ["OF"], r=80, hr=40, rbi=90, sb=5, h=120, ab=450)
        b = _hitter("B", ["OF"], r=70, hr=20, rbi=70, sb=25, h=120, ab=450)
        c = _hitter("C", ["OF"], r=60, hr=18, rbi=60, sb=8, h=120, ab=450)

        slots = {"OF": 1, "BN": 2, "P": 9, "IL": 0}
        full = [a, b, c]

        standings = _standings(
            _standing("Us", R=0, HR=100, RBI=0, SB=0, AVG=0),  # overwritten by loop
            _standing("Rival", R=0, HR=30, RBI=0, SB=20, AVG=0),
            _standing("Third", R=0, HR=20, RBI=0, SB=15, AVG=0),
        )
        lineup = optimize_hitter_lineup(
            hitters=full,
            full_roster=full,
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        assert len(lineup) == 1
        assert lineup[0].name == "B"

    def test_roto_delta_non_negative_for_every_starter(self):
        hitters = [
            _hitter("A", ["OF"], r=80, hr=25),
            _hitter("B", ["OF"], r=70, hr=20),
            _hitter("C", ["OF"], r=60, hr=15),
        ]
        slots = {"OF": 2, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(
            _standing("Us", R=0, HR=0),
            _standing("Rival", R=1, HR=1),
        )
        lineup = optimize_hitter_lineup(
            hitters=hitters,
            full_roster=hitters,
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        for a in lineup:
            assert a.roto_delta >= 0

    def test_feasibility_drives_selection_when_top_subset_infeasible(self):
        """3 catchers, 1 C slot + 1 OF slot: top-two by stats might both be
        catchers, but only 1 can fill C. The optimizer must pick a feasible
        subset (1 C + 1 OF-eligible), not an infeasible one."""
        c1 = _hitter("C1", ["C"], r=90, hr=30, rbi=90)
        c2 = _hitter("C2", ["C"], r=85, hr=28, rbi=85)
        o1 = _hitter("O1", ["OF"], r=60, hr=15, rbi=50)
        hitters = [c1, c2, o1]
        slots = {"C": 1, "OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(_standing("Us"), _standing("Rival", R=1, HR=1, RBI=1))
        lineup = optimize_hitter_lineup(
            hitters=hitters,
            full_roster=hitters,
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        slots_assigned = {a.slot for a in lineup}
        assert Position.C in slots_assigned
        assert Position.OF in slots_assigned
        names = {a.name for a in lineup}
        assert "O1" in names  # only OF-eligible, must start

    def test_il_hitters_excluded_from_optimization(self):
        il = _hitter("IL_Guy", ["OF"], r=100, hr=40, rbi=120)
        il.selected_position = Position.IL
        active = [
            _hitter("A", ["OF"], r=70, hr=20),
            _hitter("B", ["OF"], r=60, hr=15),
        ]
        slots = {"OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(_standing("Us"), _standing("Rival", R=1, HR=1))
        lineup = optimize_hitter_lineup(
            hitters=active,
            full_roster=[*active, il],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        names = {a.name for a in lineup}
        assert "IL_Guy" not in names

    def test_roto_delta_positive_when_starter_is_decisive(self):
        """When we start B (SB-decisive), dropping B must strictly reduce ERoto.
        With the double-counting bug, alt_best would tie or exceed best_total
        because the benched starter still gets counted. The fix ensures B's
        delta is strictly positive."""
        a = _hitter("A", ["OF"], r=80, hr=40, rbi=90, sb=5, h=120, ab=450)
        b = _hitter("B", ["OF"], r=70, hr=20, rbi=70, sb=25, h=120, ab=450)
        c = _hitter("C", ["OF"], r=60, hr=18, rbi=60, sb=8, h=120, ab=450)
        slots = {"OF": 1, "BN": 2, "P": 9, "IL": 0}
        standings = _standings(
            _standing("Us", R=0, HR=100, RBI=0, SB=0, AVG=0),
            _standing("Rival", R=0, HR=30, RBI=0, SB=20, AVG=0),
            _standing("Third", R=0, HR=20, RBI=0, SB=15, AVG=0),
        )
        lineup = optimize_hitter_lineup(
            hitters=[a, b, c],
            full_roster=[a, b, c],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        assert len(lineup) == 1
        assert lineup[0].name == "B"
        assert lineup[0].roto_delta > 0, (
            f"decisive starter B must have positive roto_delta, got {lineup[0].roto_delta}"
        )

    def test_roto_delta_positive_when_irreplaceable(self):
        """Single hitter, one slot -- dropping them leaves no feasible
        replacement. Their delta should reflect their marginal contribution
        (best_total - team_with_slot_empty), not the full best_total."""
        only = _hitter("Only", ["OF"], r=80, hr=30, rbi=90, sb=15, h=120, ab=450)
        slots = {"OF": 1, "BN": 0, "P": 9, "IL": 0}
        standings = _standings(
            _standing("Us"),
            _standing("Rival", R=0, HR=0, RBI=0, SB=0, AVG=0),
        )
        lineup = optimize_hitter_lineup(
            hitters=[only],
            full_roster=[only],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        assert len(lineup) == 1
        # Without Only, no feasible full-size lineup exists -> fallback scores
        # the team with the slot left empty. Delta = best_total - fallback.
        assert lineup[0].roto_delta > 0


class TestPitcherOptimizer:
    def test_returns_requested_starter_count(self):
        pitchers = [
            _pitcher("P1", ["SP"], ip=180, w=15, k=200, era=3.00, whip=1.05),
            _pitcher("P2", ["SP"], ip=170, w=12, k=170, era=3.40, whip=1.15),
            _pitcher("P3", ["SP"], ip=160, w=10, k=150, era=3.80, whip=1.22),
            _pitcher("P4", ["SP"], ip=150, w=8, k=130, era=4.20, whip=1.30),
        ]
        standings = _standings(_standing("Us"), _standing("Rival", W=1, K=1, ERA=0.1, WHIP=0.1))
        starters, bench = optimize_pitcher_lineup(
            pitchers=pitchers,
            full_roster=pitchers,
            projected_standings=standings,
            team_name="Us",
            slots=3,
        )
        assert len(starters) == 3
        assert len(bench) == 1
        for s in starters:
            assert isinstance(s, PitcherStarter)

    def test_picks_highest_impact_pitcher_not_just_highest_stats(self):
        """We're tied with Rival on SV. A has more K but starting him doesn't
        move SV; C has fewer K but his SV contribution flips the tiebreak.
        With only 1 P slot, must start C."""
        a = _pitcher("A", ["SP"], ip=200, w=15, k=230, sv=0, era=3.00, whip=1.05)
        c = _pitcher("C", ["RP"], ip=65, w=3, k=80, sv=35, era=2.50, whip=1.00)
        pitchers = [a, c]
        slots_cfg = 1
        standings = _standings(
            _standing("Us", W=0, K=0, SV=0, ERA=0, WHIP=0),
            _standing("Rival", W=0, K=0, SV=20, ERA=0, WHIP=0),
        )
        starters, _bench = optimize_pitcher_lineup(
            pitchers=pitchers,
            full_roster=pitchers,
            projected_standings=standings,
            team_name="Us",
            slots=slots_cfg,
        )
        assert len(starters) == 1
        assert starters[0].name == "C"

    def test_roto_delta_non_negative_for_every_starter(self):
        pitchers = [
            _pitcher(f"P{i}", ["SP"], ip=180 - i * 10, w=15 - i, k=200 - i * 20) for i in range(5)
        ]
        standings = _standings(_standing("Us"), _standing("Rival", W=1, K=1))
        starters, _ = optimize_pitcher_lineup(
            pitchers=pitchers,
            full_roster=pitchers,
            projected_standings=standings,
            team_name="Us",
            slots=3,
        )
        for s in starters:
            assert s.roto_delta >= 0

    def test_roto_delta_positive_when_starter_is_decisive(self):
        """Mirror of the hitter-side regression test: the dropped starter must
        actually drop out of the counterfactual, otherwise roto_delta can be
        negative or zero. Closer C is decisive on SV; dropping him must
        strictly reduce ERoto."""
        a = _pitcher("A", ["SP"], ip=200, w=15, k=230, sv=0, era=3.00, whip=1.05)
        c = _pitcher("C", ["RP"], ip=65, w=3, k=80, sv=35, era=2.50, whip=1.00)
        pitchers = [a, c]
        standings = _standings(
            _standing("Us"),
            _standing("Rival", W=0, K=0, SV=20, ERA=0, WHIP=0),
        )
        starters, _ = optimize_pitcher_lineup(
            pitchers=pitchers,
            full_roster=pitchers,
            projected_standings=standings,
            team_name="Us",
            slots=1,
        )
        assert len(starters) == 1
        assert starters[0].name == "C"
        assert starters[0].roto_delta > 0, (
            f"decisive closer C must have positive roto_delta, got {starters[0].roto_delta}"
        )

    def test_elite_closer_not_benched_when_il_pitcher_present(self):
        """REGRESSION: an elite low-volume closer must keep his saves in the
        optimizer's objective even when an IL pitcher is on the roster.

        ``team_roto_total`` formerly called ``project_ros_components`` with no
        ``league_context``, so pitcher displacement fell back to the legacy
        SGP picker. That picker selects the lowest-SGP active arm as the
        returning IL pitcher's displacement target -- which is the elite
        low-volume closer (few IP/K/W despite many saves) -- and scales him
        toward zero, erasing his saves. The optimizer then benched the
        closer for a strictly worse arm whenever any IL pitcher was rostered.

        With ``league_context`` threaded through (matching
        ``ProjectedStandings.from_rosters``), pitcher displacement uses the
        pair-swap pool model, which is ROTO-optimal and refuses to displace
        the closer when doing so loses saves. The closer stays started.

        Setup mirrors ``test_picks_highest_impact_pitcher_not_just_highest_stats``
        (closer decisive on SV) but adds an IL starter to the full roster --
        the only trigger for the bug.
        """
        ace = _pitcher("Ace", ["SP"], ip=200, w=15, k=230, sv=0, era=3.00, whip=1.05)
        closer = _pitcher("Closer", ["RP"], ip=65, w=3, k=80, sv=35, era=2.50, whip=1.00)
        il_arm = _pitcher("ILArm", ["SP"], ip=180, w=12, k=180, sv=0, era=3.40, whip=1.15)
        il_arm.selected_position = Position.IL

        pitchers = [ace, closer]  # active pool (IL excluded upstream)
        full_roster = [ace, closer, il_arm]
        standings = _standings(
            _standing("Us"),
            _standing("Rival", W=0, K=0, SV=20, ERA=0, WHIP=0),
        )
        starters, _bench = optimize_pitcher_lineup(
            pitchers=pitchers,
            full_roster=full_roster,
            projected_standings=standings,
            team_name="Us",
            slots=1,
        )
        assert len(starters) == 1
        starter_names = {s.name for s in starters}
        assert "Closer" in starter_names, (
            f"elite closer must stay started with an IL pitcher rostered; "
            f"optimizer benched him and started {starter_names} instead"
        )

    def test_compute_bands_false_suppresses_band_with_real_fraction(self):
        """``compute_bands=False`` skips per-starter bands even when a real
        ``fraction_remaining`` is supplied.

        Callers that need correct in-season displacement sizing but not the
        expensive per-starter bands -- the stash board (`stash_value`), the
        IL-return planner, and `roster_audit` -- pass a real
        ``fraction_remaining`` with ``compute_bands=False``. Before the gate was
        decoupled, those callers passed ``fraction_remaining=None`` to skip
        bands, which silently mis-sized the displacement window. This test pins
        only the band-gate half of that decoupling;
        ``test_fraction_remaining_sizes_il_displacement_window`` pins the
        sizing half (the fraction actually reaching the pool model).
        """
        a = _pitcher("A", ["SP"], ip=200, w=15, k=230, sv=0, era=3.00, whip=1.05)
        c = _pitcher("C", ["RP"], ip=65, w=3, k=80, sv=35, era=2.50, whip=1.00)
        standings = _standings(
            _standing("Us"),
            _standing("Rival", W=0, K=0, SV=20, ERA=0, WHIP=0),
        )
        starters, _ = optimize_pitcher_lineup(
            pitchers=[a, c],
            full_roster=[a, c],
            projected_standings=standings,
            team_name="Us",
            slots=2,
            fraction_remaining=0.5,
            compute_bands=False,
        )
        assert len(starters) == 2
        for s in starters:
            assert s.band is None, (
                f"compute_bands=False must skip bands even with a real "
                f"fraction_remaining; {s.name} got band={s.band}"
            )

    def test_fraction_remaining_sizes_il_displacement_window(self):
        """A real ``fraction_remaining`` must reach the pitcher pool model's
        displacement-window sizing, not just the band.

        The returning IL arm's slot-share denominator in ``swap_window_ip`` is
        ``preseason.ip * fraction_remaining``, so a mid-season fraction (0.4)
        gives the arm a larger slot share and removes MORE of the active arm
        it displaces than a whole-season 1.0 does. With an IL pitcher whose
        preseason IP is set, the two fractions must therefore produce different
        ``roto_delta`` values -- proving ``optimize_pitcher_lineup`` threads the
        fraction into the displacement model (via
        ``LeagueContext.fraction_remaining``) and does not silently drop it to
        1.0 (the bug class behind the season_routes / stash / IL-planner /
        audit callers).

        ``team_sds`` is supplied (continuous Gaussian scoring) so the
        ROTO-optimal picker actually activates the returning arm and exercises
        the swap window; with rank-based scoring against a single rival the
        swap rarely clears the gate and the fraction never reaches
        ``swap_window_ip``. ``compute_bands=False`` so the fraction can affect
        nothing except the displacement window, making the inequality a clean
        witness.
        """
        ace = _pitcher("Ace", ["SP"], ip=200, w=15, k=230, sv=0, era=3.00, whip=1.05)
        closer = _pitcher("Closer", ["RP"], ip=65, w=3, k=80, sv=35, era=2.50, whip=1.00)
        scrub = _pitcher("Scrub", ["SP"], ip=120, w=4, k=90, sv=0, era=5.00, whip=1.55)
        # Returning IL arm: 60 ROS IP against a 180-IP healthy preseason.
        # fr=1.0 -> slot_share 60/180=0.33; fr=0.4 -> 60/72=0.83. The bigger
        # share at 0.4 discounts more of the displaced arm's ROS.
        il_arm = _pitcher("ILArm", ["SP"], ip=60, w=7, k=75, sv=0, era=2.60, whip=0.95)
        il_arm.selected_position = Position.IL
        il_arm.preseason = PitcherStats(
            ip=180, w=12, k=180, sv=0, era=2.60, whip=0.95, er=52.0, bb=51, h_allowed=119
        )

        active = [ace, closer, scrub]
        full_roster = [*active, il_arm]
        standings = _standings(
            _standing("Us", W=9, K=210, SV=20, ERA=3.6, WHIP=1.2),
            _standing("Rival", W=9, K=210, SV=20, ERA=3.6, WHIP=1.2),
        )
        sds = {t: {c: (8.0 if c is Category.K else 3.0) for c in Category} for t in ("Us", "Rival")}

        def _deltas(fr):
            starters, _ = optimize_pitcher_lineup(
                pitchers=active,
                full_roster=full_roster,
                projected_standings=standings,
                team_name="Us",
                slots=2,
                team_sds=sds,
                fraction_remaining=fr,
                compute_bands=False,
            )
            return {s.name: round(s.roto_delta, 6) for s in starters}

        deltas_mid = _deltas(0.4)
        deltas_full = _deltas(1.0)
        assert deltas_mid != deltas_full, (
            "fraction_remaining must change the displacement window (and thus "
            f"roto_delta); got identical {deltas_mid} for fr=0.4 and fr=1.0, "
            "meaning the fraction never reached swap_window_ip"
        )

    def test_pitcher_band_present_when_fraction_remaining_given(self):
        """Passing fraction_remaining attaches a band dict with expected keys."""
        a = _pitcher("A", ["SP"], ip=200, w=15, k=230, sv=0, era=3.00, whip=1.05)
        c = _pitcher("C", ["RP"], ip=65, w=3, k=80, sv=35, era=2.50, whip=1.00)
        pitchers = [a, c]
        standings = _standings(
            _standing("Us"),
            _standing("Rival", W=0, K=0, SV=20, ERA=0, WHIP=0),
        )
        starters, _ = optimize_pitcher_lineup(
            pitchers=pitchers,
            full_roster=pitchers,
            projected_standings=standings,
            team_name="Us",
            slots=2,
            fraction_remaining=0.6,
        )
        assert len(starters) == 2
        d = starters[0].to_dict()
        assert "band" in d
        assert d["band"] is not None
        assert set(d["band"].keys()) == {"mean", "sd", "p_positive", "verdict"}


class TestBandFullRosterOperatingPoint:
    """Verify Fix 2: the band is evaluated at the full-team operating point.

    When a hitter swap is evaluated, the active pitchers are included in both
    the before and after player lists so the analytic band scores a complete
    roster. This means band.mean and roto_delta agree in sign for a decisive
    starter (both > 0, both < 0, or both within a small tolerance of 0).

    The analytic band's mean is the EV deltaRoto derived from the
    projected_standings 'Us' row plus the swap's ROS delta. The 'Us' row
    must be consistent with the roster; otherwise subtracting an on-roster
    hitter yields nonsensical negative stats and the band mean diverges from
    roto_delta. In production the 'Us' row is always built from the roster
    (ProjectedStandings.from_rosters). The fixture here mirrors that by
    calling project_team_stats(full_roster) to build the 'Us' row.
    """

    _TOLERANCE = 0.05

    def test_hitter_band_mean_agrees_in_direction_with_roto_delta(self):
        """A decisive hitter (B has SB advantage that flips a boundary) has a
        positive roto_delta. With the full-roster operating point the band.mean
        must also be positive (or within tolerance of zero -- the point is it
        cannot diverge wildly negative while roto_delta is clearly positive).
        """
        # Hitters: B is decisive on SB, A is strong on HR.
        a = _hitter("A", ["OF"], r=80, hr=40, rbi=90, sb=5, h=120, ab=450)
        b = _hitter("B", ["OF"], r=70, hr=20, rbi=70, sb=25, h=120, ab=450)
        c = _hitter("C", ["OF"], r=60, hr=18, rbi=60, sb=8, h=120, ab=450)

        # Include a pitcher so the hitter-optimizer has a non-trivial other_half
        # to include in the band call (the fix under test: active pitchers anchor
        # the band at the correct full-team operating point).
        p1 = _pitcher("P1", ["SP"], ip=180, w=12, k=180, sv=0, era=3.50, whip=1.20)
        p2 = _pitcher("P2", ["RP"], ip=65, w=3, k=80, sv=25, era=2.80, whip=1.10)

        hitters = [a, b, c]
        full_roster = [a, b, c, p1, p2]

        slots = {"OF": 1, "BN": 2, "P": 9, "IL": 0}

        # Build the "Us" standing row from the actual roster so band.mean
        # (which reads the projected_standings "Us" row) is consistent with
        # roto_delta (which is scored over the same players).
        # project_team_stats sums ROS stats; no displacement needed here
        # because full_roster has no IL/bench markers in the test context.
        us_stats = project_team_stats(full_roster, displacement=False)
        us_stats_dict = us_stats.to_dict()

        standings = _standings(
            ProjectedStandingsEntry(team_name="Us", stats=CategoryStats.from_dict(us_stats_dict)),
            _standing("Rival", R=0, HR=30, RBI=0, SB=20, AVG=0),
            _standing("Third", R=0, HR=20, RBI=0, SB=15, AVG=0),
        )
        lineup = optimize_hitter_lineup(
            hitters=hitters,
            full_roster=full_roster,
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
            fraction_remaining=0.6,
        )

        # Optimizer must still choose B (the decisive SB hitter).
        assert len(lineup) == 1
        starter = lineup[0]
        assert starter.name == "B"
        assert starter.roto_delta > 0, (
            f"decisive starter B must have positive roto_delta, got {starter.roto_delta}"
        )

        band = starter.band
        assert band is not None, "band must be present when fraction_remaining is given"
        band_mean = band["mean"]

        # Both roto_delta and band.mean must be positive (or negligibly small).
        # A large negative band.mean while roto_delta > 0 signals the wrong
        # operating point (the other half of categories was scored as zero).
        assert band_mean > -self._TOLERANCE, (
            f"band.mean {band_mean:.4f} strongly disagrees with roto_delta "
            f"{starter.roto_delta:.4f} -- likely wrong operating point"
        )


def _ytd_standings_with_user_hr(hr: float, team_name: str = "Us") -> Standings:
    """Build a minimal Standings whose user-row YTD HR equals ``hr``.

    Only the user-row entry is meaningful here: the optimizer derives
    ``user_ytd_components`` from that row alone. Two filler rows keep the
    standings object well-formed.
    """
    return Standings(
        effective_date=date(2026, 5, 1),
        entries=[
            StandingsEntry(
                team_name=team_name,
                team_key="t.1",
                rank=1,
                stats=CategoryStats(hr=hr),
            ),
            StandingsEntry(
                team_name="Filler",
                team_key="t.2",
                rank=2,
                stats=CategoryStats(),
            ),
        ],
    )


class TestTeamRotoTotalUsesUserYtdComponents:
    """REGRESSION (team-YTD projection refactor): the user row that
    ``team_roto_total`` builds must be team_YTD + ROS, matching the scale
    of the opponent rows that :func:`ProjectedStandings.from_rosters`
    produces. Before the fix, the optimizer's user row was ROS-only while
    opponents in ``ctx.projected_standings`` were team_YTD + ROS, so the
    user lived in a low-mu region of the score_roto S-curve and
    counting-cat deltas were silently saturated.

    Mirrors the stash board fix in
    :func:`fantasy_baseball.lineup.stash_value._active_lineup_standings`
    -- same plumbing pattern (Standings -> user_ytd_components ->
    team_end_of_season).
    """

    def _hypothetical(self) -> tuple[list[Player], ProjectedStandings]:
        """Single pitcher hypothetical with ROS K=100.

        Three teams in the projected standings so K rank ordering can
        flip when the user's team_YTD K is added:
          - With YTD K=200: user K = 200+100=300, ranks ahead of Mid (250)
            but behind Rival (400) -> 2nd of 3 in K.
          - Without YTD:    user K = 100, last of 3 in K.
        That rank flip is what the team-YTD + ROS fix exposes; the
        pre-fix ROS-only path collapses both scenarios to last-place K
        and the totals are identical.
        """
        arm = _pitcher("Mid-Season Arm", ["P"], ip=200, w=10, k=100, sv=0, era=3.00, whip=1.00)
        rival = ProjectedStandingsEntry(team_name="Rival", stats=CategoryStats(k=400.0))
        mid = ProjectedStandingsEntry(team_name="Mid", stats=CategoryStats(k=250.0))
        # Seed the user row at zero -- team_roto_total overrides it from
        # the hypothetical roster anyway, so its initial value is irrelevant.
        us = ProjectedStandingsEntry(team_name="Us", stats=CategoryStats())
        standings = ProjectedStandings(effective_date=date.min, entries=[us, mid, rival])
        return [arm], standings

    def test_team_roto_total_changes_when_user_ytd_components_provided(self):
        """team_roto_total with team_YTD K=200 differs from team_roto_total
        with no YTD (the pre-fix default).

        Direction: more team_YTD K -> user row's K leapfrogs Mid (K=250) ->
        user climbs the K rank -> higher roto total.
        """
        hypothetical, projected = self._hypothetical()
        ctx_no_ytd = _TeamContext(
            full_roster=hypothetical,
            projected_standings=projected,
            team_name="Us",
        )
        ctx_with_ytd = _TeamContext(
            full_roster=hypothetical,
            projected_standings=projected,
            team_name="Us",
            user_ytd_components=TeamYtdComponents(k=200.0),
        )
        total_no_ytd = team_roto_total(hypothetical, ctx_no_ytd)
        total_with_ytd = team_roto_total(hypothetical, ctx_with_ytd)

        assert total_with_ytd > total_no_ytd, (
            f"team_roto_total with user_ytd_components K=200 "
            f"({total_with_ytd:.4f}) must exceed total without "
            f"({total_no_ytd:.4f}); the pre-fix ROS-only path would equal both"
        )

    def test_team_roto_total_default_is_zero_ytd(self):
        """The default ``user_ytd_components`` (omitted from the
        ``_TeamContext`` constructor) is zero components -- equivalent to
        passing ``TeamYtdComponents()`` explicitly. This pins the preseason
        fallback: callers without a live Standings snapshot get the
        pre-team-YTD ROS-only behavior, so existing pre-season tests don't
        break.
        """
        hypothetical, projected = self._hypothetical()
        ctx_default = _TeamContext(
            full_roster=hypothetical,
            projected_standings=projected,
            team_name="Us",
        )
        ctx_zero = _TeamContext(
            full_roster=hypothetical,
            projected_standings=projected,
            team_name="Us",
            user_ytd_components=TeamYtdComponents(),
        )
        assert team_roto_total(hypothetical, ctx_default) == pytest.approx(
            team_roto_total(hypothetical, ctx_zero)
        )

    def test_optimize_hitter_lineup_accepts_actual_standings(self):
        """optimize_hitter_lineup accepts ``actual_standings`` and runs
        cleanly. End-to-end the marginal roto_delta stays non-negative
        regardless of which YTD scale the user row is built on; the
        deeper user-row-scale assertion lives in
        :func:`test_team_roto_total_changes_when_user_ytd_components_provided`.
        """
        a = _hitter("A", ["OF"], r=80, hr=25, rbi=80, sb=5, h=120, ab=450)
        b = _hitter("B", ["OF"], r=70, hr=10, rbi=70, sb=25, h=120, ab=450)
        slots = {"OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(
            _standing("Us"),
            _standing("Rival", R=0, HR=50, RBI=0, SB=30, AVG=0),
            _standing("Third", R=0, HR=20, RBI=0, SB=15, AVG=0),
        )
        actual = _ytd_standings_with_user_hr(40.0)

        lineup_no_ytd = optimize_hitter_lineup(
            hitters=[a, b],
            full_roster=[a, b],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        lineup_with_ytd = optimize_hitter_lineup(
            hitters=[a, b],
            full_roster=[a, b],
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
            actual_standings=actual,
        )

        assert len(lineup_no_ytd) == 1
        assert len(lineup_with_ytd) == 1
        # Threading actual_standings must not break the marginal computation.
        assert lineup_with_ytd[0].roto_delta >= 0
        assert lineup_no_ytd[0].roto_delta >= 0

    def test_optimize_pitcher_lineup_accepts_actual_standings(self):
        """optimize_pitcher_lineup accepts ``actual_standings`` and runs
        cleanly with the team-YTD + ROS user-row baseline.
        """
        p1 = _pitcher("P1", ["SP"], ip=180, w=12, k=180, era=3.50, whip=1.20)
        p2 = _pitcher("P2", ["SP"], ip=170, w=10, k=160, era=3.80, whip=1.25)
        standings = _standings(
            _standing("Us"),
            _standing("Rival", W=15, K=200, ERA=3.00, WHIP=1.10),
        )
        actual = _ytd_standings_with_user_hr(0.0)

        starters, _ = optimize_pitcher_lineup(
            pitchers=[p1, p2],
            full_roster=[p1, p2],
            projected_standings=standings,
            team_name="Us",
            slots=1,
            actual_standings=actual,
        )
        assert len(starters) == 1
        assert starters[0].roto_delta >= 0

    def test_actual_standings_none_matches_legacy_behavior(self):
        """With ``actual_standings=None`` (the default), the optimizer is
        byte-equivalent to omitting the keyword. Preserves every existing
        pre-season test and legacy caller.
        """
        hitters = [
            _hitter("A", ["OF"], r=80, hr=25),
            _hitter("B", ["OF"], r=70, hr=20),
        ]
        slots = {"OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = _standings(_standing("Us"), _standing("Rival", R=1, HR=1))

        legacy = optimize_hitter_lineup(
            hitters=hitters,
            full_roster=hitters,
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
        )
        explicit_none = optimize_hitter_lineup(
            hitters=hitters,
            full_roster=hitters,
            projected_standings=standings,
            team_name="Us",
            roster_slots=slots,
            actual_standings=None,
        )
        assert [a.name for a in legacy] == [a.name for a in explicit_none]
        assert legacy[0].roto_delta == pytest.approx(explicit_none[0].roto_delta)
