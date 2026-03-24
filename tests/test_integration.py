"""Integration test: load projections -> blend -> calculate SGP -> rank by VAR."""
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var


def test_full_pipeline(fixtures_dir):
    """End-to-end: blend projections, calculate SGP, compute VAR, rank players."""
    # Step 1: Blend projections
    hitters, pitchers = blend_projections(
        fixtures_dir,
        systems=["steamer", "zips"],
    )
    assert len(hitters) == 4
    assert len(pitchers) == 3

    # Step 2: Calculate SGP for each player
    # Add mock position data (in real use, this comes from Yahoo API)
    hitters["positions"] = [["OF"], ["OF"], ["C"], ["2B", "SS"]]
    pitchers["positions"] = [["SP"], ["RP"], ["SP"]]

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
    # Aaron Judge should be near the top (high HR, R, RBI)
    judge_rank = rankings[rankings["name"] == "Aaron Judge"].index[0]
    assert judge_rank <= 3  # Top 4 (7-player fixture pool)


def test_pipeline_with_keepers(fixtures_dir):
    """Verify keepers can be removed from the player pool."""
    hitters, pitchers = blend_projections(
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
