"""Integration tests for Monte Carlo simulation and roto scoring pipelines.

Verifies that simulate_season and score_roto work correctly end-to-end
with realistic 10-team rosters, each with distinct category strengths.
"""

from unittest.mock import patch

import numpy as np
import pytest

from fantasy_baseball.scoring import project_team_stats, score_roto_dict
from fantasy_baseball.simulation import simulate_season
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    INVERSE_STATS,
)

# ---------------------------------------------------------------------------
# Helpers to build realistic player dicts
# ---------------------------------------------------------------------------


def _hitter(name, r, hr, rbi, sb, avg, ab=550):
    h = round(avg * ab)
    return {
        "name": name,
        "player_type": "hitter",
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "h": h,
        "ab": ab,
    }


def _pitcher(name, w, k, sv, era, whip, ip=180):
    er = round(era * ip / 9)
    total_baserunners = round(whip * ip)
    bb = round(total_baserunners * 0.35)
    h_allowed = total_baserunners - bb
    return {
        "name": name,
        "player_type": "pitcher",
        "w": w,
        "k": k,
        "sv": sv,
        "ip": ip,
        "er": er,
        "bb": bb,
        "h_allowed": h_allowed,
    }


# ---------------------------------------------------------------------------
# Fixture: 10 teams with different strengths, ~15 players each
# ---------------------------------------------------------------------------


def _build_team_rosters():
    """Build 10 teams with ~15 players each (mix of hitters and pitchers).

    Each team is designed with a distinct strength area so that no single
    team dominates all categories, creating realistic competitive balance.
    """
    rosters = {}

    # Team 1 - HR heavy (power bats, less speed)
    rosters["hr_heavy"] = [
        _hitter("Aaron Judge", 100, 45, 120, 5, 0.275),
        _hitter("Pete Alonso", 80, 40, 110, 2, 0.255),
        _hitter("Kyle Schwarber", 90, 38, 95, 3, 0.230),
        _hitter("Matt Olson", 85, 35, 100, 1, 0.260),
        _hitter("Yordan Alvarez", 85, 33, 105, 0, 0.280),
        _hitter("Marcell Ozuna", 75, 30, 90, 1, 0.270),
        _hitter("Rhys Hoskins", 70, 28, 85, 2, 0.240),
        _hitter("Anthony Rizzo", 60, 22, 75, 1, 0.250),
        _hitter("Ryan Mountcastle", 65, 25, 80, 2, 0.265),
        _hitter("Joc Pederson", 55, 20, 55, 0, 0.235),
        _pitcher("Zack Wheeler", 14, 210, 0, 3.10, 1.05),
        _pitcher("Logan Webb", 12, 170, 0, 3.30, 1.10),
        _pitcher("Sonny Gray", 10, 180, 0, 3.40, 1.15),
        _pitcher("Josh Hader", 3, 70, 30, 2.80, 0.95, ip=60),
        _pitcher("Ryan Helsley", 2, 65, 28, 2.50, 1.00, ip=60),
    ]

    # Team 2 - SB heavy (speed demons)
    rosters["sb_heavy"] = [
        _hitter("Elly De La Cruz", 90, 20, 65, 60, 0.250),
        _hitter("Bobby Witt Jr", 95, 25, 80, 40, 0.280),
        _hitter("Trea Turner", 85, 18, 65, 30, 0.275),
        _hitter("Ronald Acuna Jr", 90, 22, 70, 45, 0.285),
        _hitter("Cedric Mullins", 70, 15, 55, 30, 0.260),
        _hitter("Esteury Ruiz", 60, 5, 35, 50, 0.245),
        _hitter("Jose Caballero", 50, 8, 40, 35, 0.230),
        _hitter("Corbin Carroll", 75, 18, 60, 35, 0.265),
        _hitter("Jazz Chisholm", 70, 16, 55, 25, 0.255),
        _hitter("Jorge Mateo", 45, 8, 35, 30, 0.225),
        _pitcher("Gerrit Cole", 15, 220, 0, 2.90, 1.00),
        _pitcher("Spencer Strider", 13, 230, 0, 3.00, 1.05),
        _pitcher("Tyler Glasnow", 11, 200, 0, 3.20, 1.08),
        _pitcher("Devin Williams", 2, 75, 25, 2.00, 0.90, ip=60),
        _pitcher("Andres Munoz", 3, 70, 22, 2.60, 1.00, ip=60),
    ]

    # Team 3 - AVG heavy (contact hitters, balanced counting)
    rosters["avg_heavy"] = [
        _hitter("Luis Arraez", 70, 5, 50, 3, 0.320, ab=600),
        _hitter("Freddie Freeman", 85, 22, 90, 5, 0.310, ab=600),
        _hitter("Mookie Betts", 90, 25, 85, 12, 0.305),
        _hitter("Steven Kwan", 75, 8, 50, 10, 0.305, ab=580),
        _hitter("Corey Seager", 80, 28, 85, 3, 0.295),
        _hitter("Vladimir Guerrero Jr", 80, 26, 90, 2, 0.300),
        _hitter("Rafael Devers", 78, 27, 88, 3, 0.290),
        _hitter("Bo Bichette", 70, 18, 70, 8, 0.290),
        _hitter("Yandy Diaz", 65, 15, 65, 1, 0.285),
        _hitter("Xander Bogaerts", 60, 14, 60, 3, 0.275),
        _pitcher("Corbin Burnes", 13, 195, 0, 3.20, 1.08),
        _pitcher("Framber Valdez", 12, 165, 0, 3.30, 1.15),
        _pitcher("Joe Musgrove", 10, 155, 0, 3.50, 1.12),
        _pitcher("Emmanuel Clase", 3, 55, 32, 2.20, 0.90, ip=65),
        _pitcher("Edwin Diaz", 2, 80, 28, 2.60, 1.05, ip=60),
    ]

    # Team 4 - K/pitching heavy (aces and strikeouts)
    rosters["k_heavy"] = [
        _hitter("Shohei Ohtani", 85, 30, 90, 10, 0.270),
        _hitter("Juan Soto", 90, 28, 85, 5, 0.280),
        _hitter("Bryce Harper", 80, 25, 80, 5, 0.270),
        _hitter("Jose Ramirez", 85, 28, 95, 15, 0.275),
        _hitter("Marcus Semien", 75, 22, 70, 12, 0.260),
        _hitter("Brandon Nimmo", 70, 18, 65, 5, 0.265),
        _hitter("Wilyer Abreu", 60, 14, 55, 8, 0.255),
        _hitter("Nick Castellanos", 65, 20, 75, 2, 0.260),
        _hitter("Alex Bregman", 70, 18, 70, 3, 0.265),
        _hitter("J.P. Crawford", 55, 8, 45, 5, 0.250),
        _pitcher("Max Scherzer", 12, 210, 0, 3.30, 1.08),
        _pitcher("Kevin Gausman", 13, 205, 0, 3.10, 1.05),
        _pitcher("Dylan Cease", 11, 215, 0, 3.60, 1.20),
        _pitcher("Pablo Lopez", 12, 190, 0, 3.40, 1.10),
        _pitcher("Kenley Jansen", 3, 65, 25, 3.20, 1.10, ip=60),
    ]

    # Team 5 - SV heavy (closers galore, solid bats and SP to stay competitive)
    rosters["sv_heavy"] = [
        _hitter("Mike Trout", 80, 28, 80, 5, 0.270),
        _hitter("Julio Rodriguez", 80, 24, 80, 22, 0.270),
        _hitter("Wander Franco", 75, 20, 70, 12, 0.280),
        _hitter("CJ Abrams", 75, 18, 60, 28, 0.265),
        _hitter("Ozzie Albies", 80, 22, 75, 14, 0.270),
        _hitter("Cal Raleigh", 60, 25, 78, 1, 0.240),
        _hitter("Ke'Bryan Hayes", 60, 14, 60, 10, 0.270),
        _hitter("Tommy Edman", 65, 12, 50, 20, 0.260),
        _hitter("Tyler O'Neill", 60, 24, 65, 6, 0.250),
        _hitter("Josh Naylor", 65, 20, 75, 2, 0.265),
        _pitcher("Cristian Javier", 11, 175, 0, 3.40, 1.10),
        _pitcher("Jordan Montgomery", 11, 150, 0, 3.50, 1.15),
        _pitcher("Robert Suarez", 3, 65, 35, 2.20, 0.90, ip=65),
        _pitcher("Felix Bautista", 3, 70, 30, 2.30, 0.95, ip=60),
        _pitcher("Alexis Diaz", 3, 75, 28, 2.50, 1.00, ip=62),
    ]

    # Team 6 - R heavy (runs scored focus)
    rosters["r_heavy"] = [
        _hitter("Marcus Semien 2", 105, 22, 70, 15, 0.270),
        _hitter("Francisco Lindor", 100, 25, 85, 20, 0.275),
        _hitter("Manny Machado", 90, 28, 90, 5, 0.270),
        _hitter("Trea Turner 2", 95, 20, 65, 25, 0.275),
        _hitter("Kyle Tucker", 95, 28, 90, 18, 0.280),
        _hitter("Randy Arozarena", 80, 20, 70, 15, 0.265),
        _hitter("Spencer Torkelson", 65, 18, 60, 2, 0.245),
        _hitter("Jeremy Pena", 70, 15, 55, 12, 0.260),
        _hitter("Gleyber Torres", 75, 20, 65, 5, 0.260),
        _hitter("Lane Thomas", 70, 15, 55, 12, 0.255),
        _pitcher("Shane McClanahan", 12, 185, 0, 3.20, 1.05),
        _pitcher("MacKenzie Gore", 10, 170, 0, 3.50, 1.15),
        _pitcher("Nick Pivetta", 9, 165, 0, 3.80, 1.20),
        _pitcher("Craig Kimbrel", 3, 60, 25, 3.20, 1.10, ip=55),
        _pitcher("Jordan Romano", 2, 55, 23, 3.00, 1.08, ip=55),
    ]

    # Team 7 - ERA/WHIP heavy (pitching ratios, low-count bats)
    rosters["era_heavy"] = [
        _hitter("Gunnar Henderson", 85, 25, 80, 10, 0.270),
        _hitter("Adley Rutschman", 70, 18, 75, 2, 0.275),
        _hitter("Dansby Swanson", 70, 20, 70, 10, 0.260),
        _hitter("Matt Chapman", 65, 22, 70, 3, 0.250),
        _hitter("Jonah Heim", 50, 15, 60, 1, 0.245),
        _hitter("Christian Walker", 65, 25, 80, 3, 0.255),
        _hitter("Isaac Paredes", 55, 18, 65, 1, 0.250),
        _hitter("Andres Gimenez", 65, 12, 55, 18, 0.270),
        _hitter("Bryson Stott", 55, 10, 50, 12, 0.260),
        _hitter("Nico Hoerner", 60, 8, 45, 15, 0.270),
        _pitcher("Blake Snell", 12, 200, 0, 2.60, 0.95, ip=170),
        _pitcher("Yoshinobu Yamamoto", 13, 175, 0, 2.80, 0.98),
        _pitcher("Justin Verlander", 11, 160, 0, 3.00, 1.00),
        _pitcher("Zac Gallen", 12, 175, 0, 3.10, 1.05),
        _pitcher("Camilo Doval", 2, 60, 24, 2.70, 1.00, ip=60),
    ]

    # Team 8 - W heavy (workhorse pitchers, wins)
    rosters["w_heavy"] = [
        _hitter("Cody Bellinger", 70, 20, 70, 10, 0.260),
        _hitter("Nolan Arenado", 65, 22, 80, 2, 0.265),
        _hitter("Will Smith C", 60, 18, 65, 1, 0.260),
        _hitter("Ketel Marte", 75, 22, 75, 8, 0.275),
        _hitter("Luis Robert Jr", 70, 25, 75, 15, 0.265),
        _hitter("Seiya Suzuki", 60, 18, 65, 8, 0.270),
        _hitter("Brandon Marsh", 55, 12, 50, 5, 0.260),
        _hitter("Nolan Jones", 60, 20, 65, 8, 0.255),
        _hitter("Austin Riley", 75, 30, 90, 2, 0.270),
        _hitter("Ha-seong Kim", 55, 10, 45, 12, 0.255),
        _pitcher("Aaron Nola", 15, 195, 0, 3.30, 1.08),
        _pitcher("Julio Urias", 14, 170, 0, 3.40, 1.10),
        _pitcher("Marcus Stroman", 12, 140, 0, 3.60, 1.18),
        _pitcher("George Kirby", 14, 180, 0, 3.20, 1.05),
        _pitcher("Scott Barlow", 3, 55, 22, 3.10, 1.10, ip=60),
    ]

    # Team 9 - RBI heavy (run producers)
    rosters["rbi_heavy"] = [
        _hitter("Vladimir Guerrero 2", 80, 30, 110, 2, 0.285),
        _hitter("Rafael Devers 2", 78, 28, 105, 3, 0.280),
        _hitter("Nolan Arenado 2", 70, 25, 95, 2, 0.270),
        _hitter("Matt Olson 2", 80, 32, 105, 1, 0.255),
        _hitter("Pete Alonso 2", 75, 35, 100, 1, 0.250),
        _hitter("Eloy Jimenez", 50, 18, 70, 0, 0.260, ab=400),
        _hitter("Anthony Santander", 70, 25, 85, 3, 0.260),
        _hitter("Christian Encarnacion", 55, 15, 60, 8, 0.255),
        _hitter("Willson Contreras", 55, 18, 70, 1, 0.255),
        _hitter("Salvador Perez", 55, 22, 80, 1, 0.250),
        _pitcher("Yu Darvish", 12, 185, 0, 3.40, 1.10),
        _pitcher("Charlie Morton", 10, 175, 0, 3.70, 1.20),
        _pitcher("Nestor Cortes", 10, 155, 0, 3.60, 1.15),
        _pitcher("Pete Fairbanks", 2, 60, 25, 3.00, 1.05, ip=55),
        _pitcher("Clay Holmes", 2, 55, 22, 3.30, 1.15, ip=55),
    ]

    # Team 10 - Balanced (no extreme strength, average everywhere)
    rosters["balanced"] = [
        _hitter("Willy Adames", 75, 22, 75, 10, 0.260),
        _hitter("Lars Nootbaar", 60, 15, 55, 5, 0.265),
        _hitter("Teoscar Hernandez", 70, 25, 80, 5, 0.260),
        _hitter("Whit Merrifield", 65, 12, 55, 15, 0.270),
        _hitter("Anthony Volpe", 70, 18, 60, 15, 0.255),
        _hitter("Hunter Renfroe", 55, 22, 65, 2, 0.245),
        _hitter("Andrew Benintendi", 60, 12, 55, 5, 0.275),
        _hitter("Brandon Drury", 55, 18, 60, 3, 0.255),
        _hitter("Josh Jung", 60, 20, 65, 3, 0.260),
        _hitter("MJ Melendez", 50, 16, 55, 5, 0.240),
        _pitcher("Luis Castillo", 12, 185, 0, 3.30, 1.10),
        _pitcher("Mitch Keller", 11, 170, 0, 3.50, 1.15),
        _pitcher("Jeffrey Springs", 10, 155, 0, 3.60, 1.18),
        _pitcher("David Bednar", 3, 60, 26, 2.80, 1.00, ip=58),
        _pitcher("Paul Sewald", 2, 55, 22, 3.20, 1.10, ip=55),
    ]

    return rosters


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def team_rosters():
    return _build_team_rosters()


# ---------------------------------------------------------------------------
# Monte Carlo simulation tests
# ---------------------------------------------------------------------------


class TestMonteCarloDeterminism:
    """Seed-based reproducibility."""

    def test_simulation_deterministic_with_seed(self, team_rosters):
        """Two runs with the same seed produce identical team_stats."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        stats1, inj1 = simulate_season(team_rosters, rng1)
        stats2, inj2 = simulate_season(team_rosters, rng2)

        for team in team_rosters:
            for cat in ALL_CATEGORIES:
                key = cat.value
                assert stats1[team][key] == pytest.approx(
                    stats2[team][key],
                    abs=1e-12,
                ), f"{team} {key}: {stats1[team][key]} != {stats2[team][key]}"

        # Injuries should also be identical
        for team in team_rosters:
            assert len(inj1[team]) == len(inj2[team])
            for (n1, f1), (n2, f2) in zip(inj1[team], inj2[team]):
                assert n1 == n2
                assert f1 == pytest.approx(f2, abs=1e-12)


class TestWinRateDistribution:
    """Statistical properties of Monte Carlo win distribution."""

    NUM_SIMS = 500

    def _run_sims(self, team_rosters, seed=12345):
        """Run NUM_SIMS simulations and count first-place finishes."""
        rng = np.random.default_rng(seed)
        wins = {team: 0 for team in team_rosters}

        for _ in range(self.NUM_SIMS):
            stats, _ = simulate_season(team_rosters, rng)
            roto = score_roto_dict(stats)
            # Find the winner (highest total roto points)
            winner = max(roto, key=lambda t: roto[t]["total"])
            wins[winner] += 1

        return wins

    def test_win_rates_sum_to_num_teams(self, team_rosters):
        """Total first-place finishes across all teams must equal NUM_SIMS.

        Exactly one team wins each simulation, so the sum of all wins
        must equal the number of simulations.
        """
        wins = self._run_sims(team_rosters)
        total_wins = sum(wins.values())
        assert total_wins == self.NUM_SIMS, (
            f"Total wins {total_wins} != {self.NUM_SIMS} sims. Wins by team: {wins}"
        )

    def test_no_team_has_zero_or_hundred_percent_win_rate(self, team_rosters):
        """With 10 teams over 500 sims, at least 8 teams should win once
        and no team should win every time. Per-stat variance means some
        teams (e.g. balanced) may rarely win against specialists.
        """
        wins = self._run_sims(team_rosters)
        teams_with_wins = sum(1 for c in wins.values() if c > 0)
        assert teams_with_wins >= 8, (
            f"Only {teams_with_wins}/10 teams won at least once in "
            f"{self.NUM_SIMS} sims. Distribution: {wins}"
        )
        for team, count in wins.items():
            assert count < self.NUM_SIMS, (
                f"Team '{team}' won all {self.NUM_SIMS} sims. Distribution: {wins}"
            )


class TestInjuryEffects:
    """Injury system reduces stats and replacement players contribute."""

    NUM_SIMS = 200

    def test_injuries_reduce_stats(self, team_rosters):
        """Sims with injury_prob=0 should produce higher average counting
        stats than sims with normal injury probability.
        """
        rng_normal = np.random.default_rng(999)
        rng_no_inj = np.random.default_rng(999)

        counting_cats = ["R", "HR", "RBI", "SB", "W", "K", "SV"]
        normal_totals = {cat: 0.0 for cat in counting_cats}
        no_inj_totals = {cat: 0.0 for cat in counting_cats}

        no_injury_probs = {"pitcher": 0.0, "hitter": 0.0}

        for _ in range(self.NUM_SIMS):
            # Normal run
            stats_n, _ = simulate_season(team_rosters, rng_normal)
            for team in team_rosters:
                for cat in counting_cats:
                    normal_totals[cat] += stats_n[team][cat]

            # No injury run (patch INJURY_PROB to zero)
            with patch(
                "fantasy_baseball.simulation.INJURY_PROB",
                no_injury_probs,
            ):
                stats_ni, injuries_ni = simulate_season(
                    team_rosters,
                    rng_no_inj,
                )
            for team in team_rosters:
                for cat in counting_cats:
                    no_inj_totals[cat] += stats_ni[team][cat]

            # Verify no injuries occurred in the patched run
            for team in team_rosters:
                assert len(injuries_ni[team]) == 0, (
                    f"Injuries should not occur with prob=0, but {team} had {injuries_ni[team]}"
                )

        # With injuries, aggregate counting stats should be lower
        for cat in counting_cats:
            assert normal_totals[cat] < no_inj_totals[cat], (
                f"Category {cat}: normal={normal_totals[cat]:.1f} should be "
                f"less than no-injury={no_inj_totals[cat]:.1f}"
            )

    def test_replacement_players_contribute_during_injury(self, team_rosters):
        """When a player is injured, the team's stats should not drop to
        zero in that player's categories -- replacement backfill works.
        """
        # Force all players to be injured by setting injury probability to 1.0
        full_injury = {"pitcher": 1.0, "hitter": 1.0}
        rng = np.random.default_rng(7777)

        with patch("fantasy_baseball.simulation.INJURY_PROB", full_injury):
            stats, injuries = simulate_season(team_rosters, rng)

        # Every team should have injuries
        for team in team_rosters:
            assert len(injuries[team]) > 0, f"Team '{team}' should have injuries with prob=1.0"

        # Even with all players injured, stats should not be zero because
        # replacement players backfill during the missed fraction
        for team in team_rosters:
            ts = stats[team]
            assert ts["R"] > 0, f"{team} R should be > 0 with replacement"
            assert ts["HR"] > 0, f"{team} HR should be > 0 with replacement"
            assert ts["K"] > 0, f"{team} K should be > 0 with replacement"
            assert ts["W"] > 0, f"{team} W should be > 0 with replacement"
            # AVG should be a valid positive number
            assert ts["AVG"] > 0, f"{team} AVG should be > 0 with replacement"
            # ERA/WHIP should not be the default 99 sentinel
            assert ts["ERA"] < 99, f"{team} ERA should be < 99 with replacement"
            assert ts["WHIP"] < 99, f"{team} WHIP should be < 99 with replacement"


class TestRateStatPlausibility:
    """Rate stats should remain within plausible bounds over many sims."""

    NUM_SIMS = 300

    def test_rate_stats_remain_plausible(self, team_rosters):
        """Over many sims, no team should have extreme rate stats.

        ERA < 1.0 or > 15.0, AVG < .150 or > .350, WHIP < 0.5 or > 3.0
        would all indicate a bug in the variance/aggregation logic.
        """
        rng = np.random.default_rng(54321)

        for sim_idx in range(self.NUM_SIMS):
            stats, _ = simulate_season(team_rosters, rng)
            for team, ts in stats.items():
                avg = ts["AVG"]
                era = ts["ERA"]
                whip = ts["WHIP"]

                assert 0.150 <= avg <= 0.350, (
                    f"Sim {sim_idx}, {team}: AVG={avg:.4f} out of [.150, .350] range"
                )
                assert 1.0 <= era <= 15.0, (
                    f"Sim {sim_idx}, {team}: ERA={era:.3f} out of [1.0, 15.0] range"
                )
                assert 0.5 <= whip <= 3.0, (
                    f"Sim {sim_idx}, {team}: WHIP={whip:.4f} out of [0.5, 3.0] range"
                )


# ---------------------------------------------------------------------------
# Roto scoring tests
# ---------------------------------------------------------------------------


class TestRotoScoring:
    """Tests for the roto scoring system."""

    @pytest.fixture
    def team_stats(self, team_rosters):
        """Project stats for all 10 teams (no variance, just raw projections).

        Returns string-keyed dicts at the ``score_roto`` I/O boundary —
        ``CategoryStats`` itself requires ``Category`` enum indexing.
        """
        return {
            team: project_team_stats(players).to_dict() for team, players in team_rosters.items()
        }

    def test_roto_points_sum_correctly(self, team_stats):
        """Each team's total roto points must equal the sum of its
        per-category point values.
        """
        roto = score_roto_dict(team_stats)
        for team, scores in roto.items():
            expected_total = sum(scores[f"{cat.value}_pts"] for cat in ALL_CATEGORIES)
            assert scores["total"] == pytest.approx(expected_total, abs=1e-9), (
                f"Team '{team}': total={scores['total']:.2f} != sum of cat_pts={expected_total:.2f}"
            )

    def test_inverse_stats_scored_correctly(self, team_stats):
        """Lower ERA should yield MORE roto points than higher ERA.
        Similarly for WHIP.
        """
        roto = score_roto_dict(team_stats)
        teams = list(team_stats.keys())

        for cat in INVERSE_STATS:
            # Find teams with the best (lowest) and worst (highest) raw value
            key = cat.value
            best_team = min(teams, key=lambda t: team_stats[t][key])
            worst_team = max(teams, key=lambda t: team_stats[t][key])

            best_pts = roto[best_team][f"{key}_pts"]
            worst_pts = roto[worst_team][f"{key}_pts"]

            assert best_pts > worst_pts, (
                f"Inverse stat {key}: best team '{best_team}' "
                f"(val={team_stats[best_team][key]:.3f}, pts={best_pts:.1f}) "
                f"should have more roto pts than worst team '{worst_team}' "
                f"(val={team_stats[worst_team][key]:.3f}, pts={worst_pts:.1f})"
            )

    def test_ten_teams_max_100_points(self, team_stats):
        """With 10 teams and 10 categories, max possible total = 100.

        (10 points in each of 10 categories). No team should exceed this.
        Also, the sum of all teams' totals must equal 10 * (1+2+...+10) = 550
        (conservation of roto points).
        """
        roto = score_roto_dict(team_stats)
        n_teams = len(team_stats)
        max_possible = n_teams * len(ALL_CATEGORIES)  # 10 * 10 = 100

        for team, scores in roto.items():
            assert scores["total"] <= max_possible + 1e-9, (
                f"Team '{team}': total={scores['total']:.1f} exceeds max possible {max_possible}"
            )

        # Conservation: sum of all teams' totals must equal
        # num_categories * (1 + 2 + ... + n_teams)
        # = 10 * (10 * 11 / 2) = 550
        expected_league_total = len(ALL_CATEGORIES) * (n_teams * (n_teams + 1) / 2)
        actual_league_total = sum(scores["total"] for scores in roto.values())
        assert actual_league_total == pytest.approx(
            expected_league_total,
            abs=1e-6,
        ), (
            f"League total {actual_league_total:.2f} != "
            f"expected {expected_league_total:.0f} "
            f"(conservation of roto points violated)"
        )
