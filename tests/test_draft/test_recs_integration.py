"""Unit tests for recs_integration.monte_carlo_roto_totals + compute_standings_cache."""

from __future__ import annotations

from datetime import date

from fantasy_baseball.draft.recs_integration import (
    compute_standings_cache,
    monte_carlo_roto_totals,
)
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category


def _three_team_standings(hr_spread: tuple[int, int, int]) -> ProjectedStandings:
    """Build a minimal 3-team ProjectedStandings with the given HR spread.

    All other categories are identical across teams so ranks are driven
    by HR alone — easy to reason about for the SD assertions.
    """
    entries = []
    for team_name, hr in zip(("A", "B", "C"), hr_spread, strict=True):
        entries.append(
            ProjectedStandingsEntry(
                team_name=team_name,
                stats=CategoryStats(
                    r=600,
                    hr=hr,
                    rbi=600,
                    sb=80,
                    avg=0.260,
                    w=60,
                    k=1200,
                    sv=40,
                    era=3.80,
                    whip=1.20,
                ),
            )
        )
    return ProjectedStandings(effective_date=date(2026, 1, 1), entries=entries)


def _uniform_sds(sd_per_cat: float) -> dict[str, dict[Category, float]]:
    """Every team has the same SD for every category."""
    return {t: {cat: sd_per_cat for cat in ALL_CATEGORIES} for t in ("A", "B", "C")}


def test_mc_sd_larger_for_close_race_than_blowout():
    """When HR totals are close, projection noise frequently flips the
    HR rank — per-team totals swing more, so SDs should be noticeably
    larger than in a blowout where ranks are stable."""
    # Close race: A=180, B=182, C=184 — HR rank easily flips under noise.
    close = monte_carlo_roto_totals(
        _three_team_standings((180, 182, 184)),
        _uniform_sds(sd_per_cat=10.0),
        n_iters=500,
        seed=42,
    )
    # Blowout: A=100, B=200, C=300 — HR rank is locked in under the same noise.
    blowout = monte_carlo_roto_totals(
        _three_team_standings((100, 200, 300)),
        _uniform_sds(sd_per_cat=10.0),
        n_iters=500,
        seed=42,
    )

    close_sd_avg = sum(s for _, s in close.values()) / 3
    blowout_sd_avg = sum(s for _, s in blowout.values()) / 3
    assert close_sd_avg > blowout_sd_avg
    # Sanity: SDs should be roto-points-scale, not raw-stat-scale. In a
    # 3-team league the per-team total roto range is 10..30, so SD > 10
    # would be suspicious. The old quadrature-of-raw-SDs bug produced
    # values above the scale max.
    assert close_sd_avg < 5.0


def test_mc_sd_zero_when_team_sds_zero():
    """Zero projection SD → deterministic ranks → zero total SD."""
    result = monte_carlo_roto_totals(
        _three_team_standings((100, 200, 300)),
        _uniform_sds(sd_per_cat=0.0),
        n_iters=200,
        seed=0,
    )
    for _, sd in result.values():
        assert sd == 0.0


def test_mc_mean_approximates_expected_rank_sum():
    """With zero noise, each team's total equals rank-per-cat summed
    across all categories. With ``hr`` the only varying category and
    everything else tied, rank-averaging on ties puts every team at
    ``2.0`` per tied cat; HR gives 1, 2, 3. Total = 2.0*(N-1) + HR_rank."""
    n_cats = len(ALL_CATEGORIES)
    expected_tied_points = 2.0 * (n_cats - 1)  # every non-HR cat is a 3-way tie

    result = monte_carlo_roto_totals(
        _three_team_standings((100, 200, 300)),
        _uniform_sds(sd_per_cat=0.0),
        n_iters=100,
        seed=0,
    )
    # A has worst HR → rank 1, C has best → rank 3.
    assert result["A"][0] == expected_tied_points + 1.0
    assert result["B"][0] == expected_tied_points + 2.0
    assert result["C"][0] == expected_tied_points + 3.0


def test_compute_standings_cache_schema():
    """Cache entries must expose total.point_estimate / total.sd at the
    top level and keep per-category EVs under ``categories``."""
    standings = _three_team_standings((150, 200, 250))
    team_sds = _uniform_sds(sd_per_cat=5.0)
    cache = compute_standings_cache(standings, team_sds, mc_iters=100, mc_seed=7)

    assert set(cache.keys()) == {"A", "B", "C"}
    for _team, entry in cache.items():
        assert "total" in entry
        assert "point_estimate" in entry["total"]
        assert "sd" in entry["total"]
        assert entry["total"]["sd"] >= 0.0
        assert isinstance(entry["categories"], dict)
        # Every category present; each has a point_estimate float.
        for cat in ALL_CATEGORIES:
            cat_entry = entry["categories"][cat.value]
            assert "point_estimate" in cat_entry
            assert isinstance(cat_entry["point_estimate"], float)
