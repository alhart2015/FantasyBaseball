import pandas as pd
import pytest

from fantasy_baseball.sgp.replacement import (
    calculate_replacement_levels,
    calculate_replacement_rates,
    find_replacement_players,
)
from fantasy_baseball.utils.positions import is_hitter


def _make_player_pool():
    hitters = []
    for i in range(15):
        hitters.append(
            {
                "name": f"Catcher_{i}",
                "positions": ["C"],
                "total_sgp": 20.0 - i,
                "player_type": "hitter",
            }
        )
    for i in range(15):
        hitters.append(
            {
                "name": f"FirstBase_{i}",
                "positions": ["1B"],
                "total_sgp": 25.0 - i,
                "player_type": "hitter",
            }
        )
    for i in range(50):
        hitters.append(
            {
                "name": f"Outfielder_{i}",
                "positions": ["OF"],
                "total_sgp": 30.0 - i * 0.5,
                "player_type": "hitter",
            }
        )
    pitchers = []
    for i in range(100):
        pitchers.append(
            {
                "name": f"Pitcher_{i}",
                "positions": ["SP"] if i < 70 else ["RP"],
                "total_sgp": 25.0 - i * 0.2,
                "player_type": "pitcher",
            }
        )
    return pd.DataFrame(hitters + pitchers)


class TestReplacementLevels:
    def test_catcher_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["C"] == pytest.approx(10.0)

    def test_first_base_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["1B"] == pytest.approx(15.0)

    def test_of_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["OF"] == pytest.approx(10.0)

    def test_pitcher_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["P"] == pytest.approx(7.0)

    def test_all_starter_positions_have_levels(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert "C" in levels
        assert "OF" in levels
        assert "P" in levels


def _make_pool_with_stats():
    """Build a pool with 130 hitters and 100 pitchers carrying rate-stat columns."""
    hitters = []
    for i in range(130):
        ab = 500 - i
        avg = 0.280 - i * 0.0005
        h = int(ab * avg)
        hitters.append(
            {
                "name": f"Hitter_{i}",
                "positions": ["OF"] if i % 2 == 0 else ["1B"],
                "total_sgp": 30.0 - i * 0.2,
                "player_type": "hitter",
                "ab": ab,
                "h": h,
                "avg": avg,
                # Pitcher columns needed so DataFrame is uniform
                "ip": 0.0,
                "er": 0,
                "bb": 0,
                "h_allowed": 0,
                "era": 0.0,
                "whip": 0.0,
            }
        )

    pitchers = []
    for i in range(100):
        ip = 180.0 - i
        era = 3.50 + i * 0.02
        er = era * ip / 9
        whip = 1.15 + i * 0.005
        bb = int(whip * ip * 0.35)
        h_allowed = int(whip * ip - bb)
        pitchers.append(
            {
                "name": f"Pitcher_{i}",
                "positions": ["SP"],
                "total_sgp": 25.0 - i * 0.2,
                "player_type": "pitcher",
                "ab": 0,
                "h": 0,
                "avg": 0.0,
                "ip": ip,
                "er": er,
                "bb": bb,
                "h_allowed": h_allowed,
                "era": era,
                "whip": whip,
            }
        )

    return pd.DataFrame(hitters + pitchers)


class TestReplacementRates:
    def test_returns_era_whip_avg(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert "era" in rates
        assert "whip" in rates
        assert "avg" in rates

    def test_era_between_reasonable_bounds(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert 3.0 < rates["era"] < 6.0

    def test_whip_between_reasonable_bounds(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert 1.0 < rates["whip"] < 2.0

    def test_avg_between_reasonable_bounds(self):
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        assert 0.200 < rates["avg"] < 0.300

    def test_empty_pitcher_pool_uses_defaults(self):
        """When no pitchers exist, ERA and WHIP fall back to hardcoded defaults."""
        pool = _make_pool_with_stats()
        hitters_only = pool[pool["player_type"] == "hitter"].copy()
        rates = calculate_replacement_rates(hitters_only)
        assert rates["era"] == pytest.approx(4.50)
        assert rates["whip"] == pytest.approx(1.35)

    def test_empty_hitter_pool_uses_defaults(self):
        """When no hitters exist, AVG falls back to hardcoded default."""
        pool = _make_pool_with_stats()
        pitchers_only = pool[pool["player_type"] == "pitcher"].copy()
        rates = calculate_replacement_rates(pitchers_only)
        assert rates["avg"] == pytest.approx(0.250)

    def test_zero_ip_pitchers_excluded_from_band(self):
        """Pitchers with 0 IP in the band are filtered out, preventing division by zero."""
        pool = _make_pool_with_stats()
        # Set IP to 0 for pitchers near the replacement threshold
        pitcher_mask = pool["player_type"] == "pitcher"
        pitcher_indices = pool[pitcher_mask].sort_values("total_sgp", ascending=False).index
        # Zero out IP for pitchers 85-95 (around the 90-starter threshold)
        for idx in pitcher_indices[85:96]:
            pool.loc[idx, "ip"] = 0.0
            pool.loc[idx, "er"] = 0
            pool.loc[idx, "bb"] = 0
            pool.loc[idx, "h_allowed"] = 0
        rates = calculate_replacement_rates(pool)
        assert rates["era"] > 0

    def test_closers_only_pool(self):
        """A pool of only RP-eligible pitchers still produces valid rates."""
        pool = _make_pool_with_stats()
        # Change all SP to RP
        pitcher_mask = pool["player_type"] == "pitcher"
        pool.loc[pitcher_mask, "positions"] = pool.loc[pitcher_mask, "positions"].apply(
            lambda _: ["RP"]
        )
        rates = calculate_replacement_rates(pool)
        assert 3.0 < rates["era"] < 6.0
        assert 1.0 < rates["whip"] < 2.0

    def test_sp_only_pool(self):
        """A pool of only SP-eligible pitchers produces valid rates (already the default)."""
        pool = _make_pool_with_stats()
        rates = calculate_replacement_rates(pool)
        # All pitchers in the helper are SP, so this is just SP-only
        assert 3.0 < rates["era"] < 6.0
        assert 1.0 < rates["whip"] < 2.0


class TestFindReplacementPlayers:
    def test_returns_marginal_player_per_position(self):
        """For a position with N starter slots, the replacement player
        is the (N+1)-th best at that position by total_sgp — the
        marginal-just-not-a-starter."""
        pool = _make_player_pool()
        # _make_player_pool creates 15 catchers with total_sgp = 20, 19, ..., 6.
        # Default STARTERS_PER_POSITION has C with positive demand; pass an
        # explicit starters dict so the test is independent of league config.
        starters = {"C": 5}
        reps = find_replacement_players(pool, starters)
        # 6th-best catcher (index 5) → total_sgp = 20 - 5 = 15.0
        assert reps["C"]["name"] == "Catcher_5"
        assert reps["C"]["total_sgp"] == pytest.approx(15.0)

    def test_returns_player_for_each_position(self):
        pool = _make_player_pool()
        starters = {"C": 1, "1B": 1, "OF": 1, "P": 1}
        reps = find_replacement_players(pool, starters)
        for pos in ("C", "1B", "OF", "P"):
            assert pos in reps
            assert reps[pos]["name"]  # has a name
            assert "total_sgp" in reps[pos]

    def test_skips_if_meta_slot(self):
        """IF is a meta-slot in calculate_replacement_levels (handled via
        UTIL/positional fallback in calculate_var). find_replacement_players
        follows the same convention so its output drops cleanly into the same
        downstream lookup logic."""
        pool = _make_player_pool()
        starters = {"IF": 1, "OF": 1}
        reps = find_replacement_players(pool, starters)
        assert "IF" not in reps
        assert "OF" in reps

    def test_util_uses_total_hitter_starters(self):
        """UTIL replacement is the marginal hitter at depth =
        positional_hitter_starters + util_starters across the full
        hitter pool. Mirrors calculate_replacement_levels behavior."""
        pool = _make_player_pool()
        # Pool has 15 C + 15 1B + 50 OF = 80 hitters total.
        # With C=5, 1B=5, OF=10, UTIL=5 → positional=20, total_hitter_starters=25.
        # The 26th-best hitter overall (index 25 by total_sgp desc) is the UTIL replacement.
        starters = {"C": 5, "1B": 5, "OF": 10, "UTIL": 5}
        reps = find_replacement_players(pool, starters)
        assert "UTIL" in reps
        all_hitters_sorted = (
            pool[pool["positions"].apply(is_hitter)]
            .sort_values("total_sgp", ascending=False)
            .reset_index(drop=True)
        )
        expected_name = all_hitters_sorted.iloc[25]["name"]
        assert reps["UTIL"]["name"] == expected_name

    def test_pitcher_pool_includes_sp_and_rp(self):
        """The P bucket pulls from any P/SP/RP-eligible row, mirroring
        calculate_replacement_levels._get_eligible_players."""
        pool = _make_player_pool()
        # Pool has 100 pitchers (70 SP, 30 RP) with total_sgp = 25, 24.8, ..., -3.0.
        starters = {"P": 50}
        reps = find_replacement_players(pool, starters)
        # 51st-best pitcher (index 50) → total_sgp = 25 - 50*0.2 = 15.0
        assert reps["P"]["total_sgp"] == pytest.approx(15.0)

    def test_falls_back_to_worst_eligible_when_pool_too_small(self):
        """When fewer eligible players exist than the demand, the
        replacement is the worst eligible player — same behavior as
        calculate_replacement_levels' `eligible.iloc[-1]` branch."""
        # Tiny pool: 3 catchers, demand 10.
        pool = pd.DataFrame(
            [
                {
                    "name": f"C_{i}",
                    "positions": ["C"],
                    "total_sgp": 10.0 - i,
                    "player_type": "hitter",
                }
                for i in range(3)
            ]
        )
        starters = {"C": 10}
        reps = find_replacement_players(pool, starters)
        assert reps["C"]["name"] == "C_2"  # worst eligible
