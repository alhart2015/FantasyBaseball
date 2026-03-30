"""Integration test: load projections -> blend -> calculate SGP -> rank by VAR.

Migrated from tests/test_integration.py to resolve the naming conflict
between that standalone module and the tests/test_integration/ package.
"""
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent.parent / "fixtures"


def test_full_pipeline(fixtures_dir):
    """End-to-end: blend projections, calculate SGP, compute VAR, rank players."""
    # Step 1: Blend projections
    hitters, pitchers, _ = blend_projections(
        fixtures_dir,
        systems=["steamer", "zips"],
    )
    assert len(hitters) == 4
    assert len(pitchers) == 3

    # Step 2: Calculate SGP for each player
    # Add mock position data by name (order may vary by groupby key)
    hitter_positions = {
        "Aaron Judge": ["OF"],
        "Mookie Betts": ["OF"],
        "Adley Rutschman": ["C"],
        "Marcus Semien": ["2B", "SS"],
    }
    pitcher_positions = {
        "Gerrit Cole": ["SP"],
        "Emmanuel Clase": ["RP"],
        "Corbin Burnes": ["SP"],
    }
    hitters["positions"] = hitters["name"].map(hitter_positions)
    pitchers["positions"] = pitchers["name"].map(pitcher_positions)

    for idx, row in hitters.iterrows():
        hitters.loc[idx, "total_sgp"] = calculate_player_sgp(row)
    for idx, row in pitchers.iterrows():
        pitchers.loc[idx, "total_sgp"] = calculate_player_sgp(row)

    # Step 3: Build player pool and calculate replacement levels
    pool = pd.concat([hitters, pitchers], ignore_index=True)
    assert len(pool) == 7

    # With only 7 players, replacement levels will use the last player
    # Use smaller starters_per_position for this test
    small_starters = {"C": 1, "OF": 2, "SS": 1, "2B": 1, "P": 2}
    levels = calculate_replacement_levels(pool, small_starters)

    # Step 4: Calculate VAR for each player
    vars_list = []
    for _, player in pool.iterrows():
        var = calculate_var(player, levels)
        vars_list.append({"name": player["name"], "var": var})

    rankings = (
        pd.DataFrame(vars_list)
        .sort_values("var", ascending=False)
        .reset_index(drop=True)
    )

    # Step 5: Verify rankings make sense
    assert len(rankings) == 7
    # Top player should have positive VAR
    assert rankings.iloc[0]["var"] > 0
    # Rankings should be sorted descending
    assert rankings.iloc[0]["var"] >= rankings.iloc[-1]["var"]
    # Aaron Judge should have the highest VAR among hitters
    # (high HR, R, RBI in a 7-player pool)
    hitter_rankings = rankings[rankings["name"].isin(
        ["Aaron Judge", "Mookie Betts", "Adley Rutschman", "Marcus Semien"]
    )]
    assert hitter_rankings.iloc[0]["name"] == "Aaron Judge"


def test_pipeline_with_keepers(fixtures_dir):
    """Verify keepers can be removed from the player pool."""
    hitters, pitchers, _ = blend_projections(
        fixtures_dir,
        systems=["steamer"],
    )
    hitters["positions"] = [["OF"], ["OF"], ["C"], ["2B", "SS"]]
    pitchers["positions"] = [["SP"], ["RP"], ["SP"]]

    pool = pd.concat([hitters, pitchers], ignore_index=True)
    assert len(pool) == 7

    # Remove keepers
    keepers = ["Aaron Judge", "Gerrit Cole"]
    pool = pool[~pool["name"].isin(keepers)]
    assert len(pool) == 5
    assert "Aaron Judge" not in pool["name"].values
