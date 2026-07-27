import pandas as pd

from fantasy_baseball.keepers.calibration import PAIR_YEARS, YearPair, survivorship


def test_pair_years_are_the_three_usable_ones() -> None:
    # 2025->2026 needs a complete 2026 season; 2021->2022 has no ZiPS 2021 on disk.
    assert PAIR_YEARS == (2022, 2023, 2024)


def test_survivorship_counts_players_who_kept_playing() -> None:
    pair = YearPair(
        year=2022,
        base=pd.DataFrame({"hr_pa": [0.04, 0.03]}, index=[1, 2]),
        residual=pd.DataFrame({"hr_pa": [0.01, -0.01]}, index=[1, 2]),
        target=pd.DataFrame({"hr_pa": [0.045, float("nan")]}, index=[1, 2]),
        realized_pt=pd.Series([500.0, 300.0], index=[1, 2]),
        target_pt=pd.Series([550.0, 0.0], index=[1, 2]),
    )
    out = survivorship([pair], threshold=100.0)
    row = out.iloc[0]
    assert row["n_in_year"] == 2
    assert row["n_survived"] == 1
    assert row["survival_rate"] == 0.5
