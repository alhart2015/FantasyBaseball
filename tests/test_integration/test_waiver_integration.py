"""Integration tests for the waiver recommendation pipeline.

Exercises the full path: standings -> leverage -> weighted SGP -> scan_waivers,
with realistic 10-team roto data and a 22-player roster.
"""

import pytest
import pandas as pd

from fantasy_baseball.lineup.waivers import (
    detect_open_slots,
    evaluate_pickup,
    scan_waivers,
)
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.positions import can_cover_slots, is_pitcher, is_hitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hitter_slot_types():
    """Hitter slot names that are not bench/IL/pitcher."""
    return {"C", "1B", "2B", "3B", "SS", "IF", "OF", "UTIL"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWaiverRecsNotEmpty:
    """With open slots and available FAs, recommendations should not be empty."""

    def test_waiver_recs_not_empty_when_free_agents_available(
        self, active_roster, free_agents, leverage, roster_slots
    ):
        results = scan_waivers(
            roster=active_roster,
            free_agents=free_agents,
            leverage=leverage,
            max_results=10,
            roster_slots=roster_slots,
        )
        assert len(results) > 0, (
            "Expected at least one waiver recommendation when free agents "
            "are available and some are better than roster tail-enders"
        )


class TestOpenSlotDetection:
    """detect_open_slots should return correct typed counts."""

    def test_full_roster_has_no_open_slots(
        self, yahoo_roster_full, roster_slots
    ):
        open_h, open_p, open_bn = detect_open_slots(
            yahoo_roster_full, roster_slots
        )
        assert open_h == 0, f"Expected 0 open hitter slots, got {open_h}"
        assert open_p == 0, f"Expected 0 open pitcher slots, got {open_p}"
        assert open_bn == 0, f"Expected 0 open bench slots, got {open_bn}"

    def test_open_slot_detection_matches_roster(
        self, yahoo_roster_with_gaps, roster_slots
    ):
        """Roster with 2 IL + 7/9 pitchers + 1/2 bench -> 2 open P, 1 open BN."""
        open_h, open_p, open_bn = detect_open_slots(
            yahoo_roster_with_gaps, roster_slots
        )
        assert open_h == 0, f"Expected 0 open hitter slots, got {open_h}"
        assert open_p == 2, f"Expected 2 open pitcher slots, got {open_p}"
        assert open_bn == 1, f"Expected 1 open bench slot, got {open_bn}"


class TestDropPreservesPositionalCoverage:
    """When only one player can play 1B, never recommend dropping them."""

    def test_drop_preserves_positional_coverage(
        self, active_roster, free_agents, leverage, roster_slots
    ):
        results = scan_waivers(
            roster=active_roster,
            free_agents=free_agents,
            leverage=leverage,
            max_results=20,
            roster_slots=roster_slots,
        )

        # Identify the sole 1B player on the active roster.
        # Pete Alonso is the ONLY player with "1B" in positions (Vlad Jr has
        # "1B" too but can also fill UTIL -- however both have 1B so let's
        # verify which are 1B-eligible and confirm coverage).
        sole_1b_names = set()
        players_with_1b = [
            p for p in active_roster
            if "1B" in p.get("positions", [])
            and p.get("player_type") == "hitter"
        ]

        for candidate in players_with_1b:
            # Would dropping this player leave us unable to cover 1B?
            remaining_positions = [
                list(p.get("positions", []))
                for p in active_roster
                if p["name"] != candidate["name"]
                and p.get("player_type") == "hitter"
            ]
            if not can_cover_slots(remaining_positions, roster_slots):
                sole_1b_names.add(candidate["name"])

        # Pete Alonso should be identified as the sole irreplaceable 1B
        assert "Pete Alonso" in sole_1b_names, (
            "Expected Pete Alonso to be the sole irreplaceable 1B provider"
        )

        # Now verify no recommendation drops a sole-1B player without
        # providing 1B coverage via the add.
        for rec in results:
            drop_name = rec["drop"]
            if drop_name.startswith("(empty"):
                continue
            if drop_name not in sole_1b_names:
                continue

            # If a sole-1B player IS dropped, the add player must cover 1B.
            # Find the add player's positions from free_agents.
            add_fa = next(
                (fa for fa in free_agents if fa["name"] == rec["add"]), None
            )
            assert add_fa is not None

            # Build post-swap roster and verify coverage
            post_swap_positions = [
                list(p.get("positions", []))
                for p in active_roster
                if p["name"] != drop_name
                and p.get("player_type") == "hitter"
            ]
            post_swap_positions.append(list(add_fa.get("positions", [])))
            assert can_cover_slots(post_swap_positions, roster_slots), (
                f"Dropping {drop_name} for {rec['add']} leaves a position "
                f"hole -- can_cover_slots returned False"
            )


class TestCategoryDirectionMatchesRawStats:
    """For every swap, categories marked as 'gain' must actually be gains.

    This catches the Arraez/Murakami class of bugs where the direction of
    a category contribution is reversed.
    """

    def test_category_direction_matches_raw_stats(
        self, active_roster, free_agents, leverage, roster_slots
    ):
        results = scan_waivers(
            roster=active_roster,
            free_agents=free_agents,
            leverage=leverage,
            max_results=10,
            roster_slots=roster_slots,
        )

        for rec in results:
            if rec["drop"].startswith("(empty"):
                continue  # pure adds have no drop-side comparison

            add_fa = next(
                (fa for fa in free_agents if fa["name"] == rec["add"]), None
            )
            drop_p = next(
                (p for p in active_roster if p["name"] == rec["drop"]), None
            )
            assert add_fa is not None
            assert drop_p is not None

            player_type = add_fa.get("player_type")

            if player_type == "hitter":
                counting_stats = [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]
            elif player_type == "pitcher":
                counting_stats = [("W", "w"), ("K", "k"), ("SV", "sv")]
            else:
                continue

            categories = rec.get("categories", {})

            for cat_name, col in counting_stats:
                cat_sgp_delta = categories.get(cat_name, 0)
                raw_add = add_fa.get(col, 0)
                raw_drop = drop_p.get(col, 0)
                raw_delta = raw_add - raw_drop

                # Direction must match: if the SGP delta says the add is
                # better in this category, the raw stat should agree.
                if cat_sgp_delta > 0.001:
                    assert raw_delta > 0, (
                        f"{rec['add']} vs {rec['drop']}: {cat_name} SGP "
                        f"delta is positive ({cat_sgp_delta:.3f}) but raw "
                        f"stat delta is {raw_delta} ({raw_add} - {raw_drop})"
                    )
                elif cat_sgp_delta < -0.001:
                    assert raw_delta < 0, (
                        f"{rec['add']} vs {rec['drop']}: {cat_name} SGP "
                        f"delta is negative ({cat_sgp_delta:.3f}) but raw "
                        f"stat delta is {raw_delta} ({raw_add} - {raw_drop})"
                    )

            # For rate stats, the SGP formula uses volume-weighted marginal
            # value (IP or AB weighted), so a player with a slightly worse
            # raw rate but much more volume can still produce a positive SGP
            # delta.  We therefore compare the marginal contribution, not the
            # raw rate.
            if player_type == "hitter":
                # AVG: marginal_hits = (avg - replacement_avg) * ab
                add_avg_marginal = (add_fa.get("avg", 0) - 0.250) * add_fa.get("ab", 0)
                drop_avg_marginal = (drop_p.get("avg", 0) - 0.250) * drop_p.get("ab", 0)
                avg_delta = categories.get("AVG", 0)
                if avg_delta > 0.001:
                    assert add_avg_marginal > drop_avg_marginal, (
                        f"{rec['add']} vs {rec['drop']}: AVG SGP delta is "
                        f"positive ({avg_delta:.3f}) but add marginal hits "
                        f"({add_avg_marginal:.1f}) <= drop ({drop_avg_marginal:.1f})"
                    )
                elif avg_delta < -0.001:
                    assert add_avg_marginal < drop_avg_marginal, (
                        f"{rec['add']} vs {rec['drop']}: AVG SGP delta is "
                        f"negative ({avg_delta:.3f}) but add marginal hits "
                        f"({add_avg_marginal:.1f}) >= drop ({drop_avg_marginal:.1f})"
                    )

            elif player_type == "pitcher":
                # ERA/WHIP use (replacement_rate - player_rate) * IP.
                # Positive marginal = better than replacement.
                for cat_name, col, repl_rate, divisor in [
                    ("ERA", "era", 4.50, 9),
                    ("WHIP", "whip", 1.35, 1),
                ]:
                    cat_sgp_delta = categories.get(cat_name, 0)
                    add_ip = add_fa.get("ip", 0)
                    drop_ip = drop_p.get("ip", 0)
                    add_marginal = (repl_rate - add_fa.get(col, 0)) * add_ip / divisor
                    drop_marginal = (repl_rate - drop_p.get(col, 0)) * drop_ip / divisor
                    if cat_sgp_delta > 0.001:
                        assert add_marginal > drop_marginal, (
                            f"{rec['add']} vs {rec['drop']}: {cat_name} SGP "
                            f"delta is positive ({cat_sgp_delta:.3f}) but "
                            f"add marginal ({add_marginal:.2f}) <= "
                            f"drop marginal ({drop_marginal:.2f})"
                        )
                    elif cat_sgp_delta < -0.001:
                        assert add_marginal < drop_marginal, (
                            f"{rec['add']} vs {rec['drop']}: {cat_name} SGP "
                            f"delta is negative ({cat_sgp_delta:.3f}) but "
                            f"add marginal ({add_marginal:.2f}) >= "
                            f"drop marginal ({drop_marginal:.2f})"
                        )


class TestNoScheduleScalingAsymmetry:
    """If schedule scaling is applied, no counting stat should exceed 2x unscaled."""

    def test_no_schedule_scaling_asymmetry(self, free_agents, leverage):
        """Verify that weighted SGP doesn't scale any player's effective
        counting-stat contribution more than 2x from their raw projection.

        This catches bugs where a schedule multiplier is applied to the add
        side but not the drop side (or vice versa).
        """
        from fantasy_baseball.sgp.denominators import get_sgp_denominators

        denoms = get_sgp_denominators()

        for fa in free_agents:
            wsgp = calculate_weighted_sgp(fa, leverage, denoms=denoms)
            player_type = fa.get("player_type")

            if player_type == "hitter":
                counting = [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]
            elif player_type == "pitcher":
                counting = [("W", "w"), ("K", "k"), ("SV", "sv")]
            else:
                continue

            # Sum of unweighted counting SGP
            raw_counting_sgp = sum(
                fa.get(col, 0) / denoms[cat] for cat, col in counting
            )

            # The weighted total can redistribute emphasis but should not
            # make the contribution more than 2x the raw counting total.
            # This would indicate an asymmetric scaling bug.
            if raw_counting_sgp > 0.5:
                assert wsgp < raw_counting_sgp * 2.0, (
                    f"{fa['name']}: weighted SGP ({wsgp:.2f}) exceeds 2x "
                    f"raw counting SGP ({raw_counting_sgp:.2f}), suggesting "
                    f"asymmetric schedule scaling"
                )


class TestPureAddsMatchSlotType:
    """When there are open pitcher slots, pure adds should be pitchers."""

    def test_pure_adds_match_slot_type(
        self, active_roster, free_agents, leverage, roster_slots
    ):
        results = scan_waivers(
            roster=active_roster,
            free_agents=free_agents,
            leverage=leverage,
            max_results=20,
            open_pitcher_slots=2,
            open_hitter_slots=0,
            open_bench_slots=0,
            roster_slots=roster_slots,
        )

        pure_pitcher_adds = [
            r for r in results if r["drop"] == "(empty pitcher slot)"
        ]
        assert len(pure_pitcher_adds) > 0, (
            "Expected at least one pure pitcher add with 2 open pitcher slots"
        )

        for rec in pure_pitcher_adds:
            add_fa = next(
                (fa for fa in free_agents if fa["name"] == rec["add"]), None
            )
            assert add_fa is not None
            assert add_fa.get("player_type") == "pitcher", (
                f"Pure pitcher-slot add '{rec['add']}' is not a pitcher "
                f"(type={add_fa.get('player_type')})"
            )
            assert is_pitcher(list(add_fa.get("positions", []))), (
                f"Pure pitcher-slot add '{rec['add']}' has no pitcher "
                f"positions: {add_fa.get('positions')}"
            )

    def test_pure_hitter_adds_are_hitters(
        self, active_roster, free_agents, leverage, roster_slots
    ):
        """Symmetric check: open hitter slots should be filled by hitters."""
        results = scan_waivers(
            roster=active_roster,
            free_agents=free_agents,
            leverage=leverage,
            max_results=20,
            open_pitcher_slots=0,
            open_hitter_slots=2,
            open_bench_slots=0,
            roster_slots=roster_slots,
        )

        pure_hitter_adds = [
            r for r in results if r["drop"] == "(empty hitter slot)"
        ]
        assert len(pure_hitter_adds) > 0, (
            "Expected at least one pure hitter add with 2 open hitter slots"
        )

        for rec in pure_hitter_adds:
            add_fa = next(
                (fa for fa in free_agents if fa["name"] == rec["add"]), None
            )
            assert add_fa is not None
            assert add_fa.get("player_type") == "hitter", (
                f"Pure hitter-slot add '{rec['add']}' is not a hitter "
                f"(type={add_fa.get('player_type')})"
            )


class TestRecommendationsSortedByGain:
    """Results must be sorted descending by sgp_gain."""

    def test_recommendations_sorted_by_gain(
        self, active_roster, free_agents, leverage, roster_slots
    ):
        results = scan_waivers(
            roster=active_roster,
            free_agents=free_agents,
            leverage=leverage,
            max_results=20,
            roster_slots=roster_slots,
        )
        assert len(results) >= 2, (
            "Need at least 2 recommendations to verify sort order"
        )
        gains = [r["sgp_gain"] for r in results]
        for i in range(len(gains) - 1):
            assert gains[i] >= gains[i + 1], (
                f"Results not sorted: index {i} gain={gains[i]:.4f} < "
                f"index {i+1} gain={gains[i+1]:.4f}"
            )
