"""Role bucketing for the pitcher backtest (#313).

`roles()` is the only new inference in the backtest harness: everything else it does is
delegated to the two estimators. It is tested here because its edge cases are silent --
a season with zero games divides by zero, and a mid-season trade arrives as two rows
that must be re-summed before the starter share means anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest_trajectory import roles


def _season(mlbam_id: int, season: int, *, starts: float, games: float, sv: float) -> dict:
    return {
        "mlbam_id": mlbam_id,
        "season": season,
        "starts": starts,
        "games": games,
        "sv": sv,
    }


def test_starter_and_reliever_split_on_starts_share():
    panel = pd.DataFrame(
        [
            _season(1, 2020, starts=30, games=32, sv=0),  # 0.94 -> SP
            _season(2, 2020, starts=0, games=60, sv=0),  # 0.00 -> RP
        ]
    )
    out = roles(panel)
    assert out[(1, 2020)] == "SP"
    assert out[(2, 2020)] == "RP"


def test_starter_share_boundary_is_inclusive():
    """STARTER_SHARE is 0.5 and the comparison is `>=`, so an exact half is a starter."""
    panel = pd.DataFrame(
        [
            _season(1, 2020, starts=15, games=30, sv=0),  # exactly 0.5
            _season(2, 2020, starts=14, games=30, sv=0),  # just under
        ]
    )
    out = roles(panel)
    assert out[(1, 2020)] == "SP"
    assert out[(2, 2020)] == "RP"


def test_closer_needs_the_save_threshold_and_loses_to_starting():
    panel = pd.DataFrame(
        [
            _season(1, 2020, starts=0, games=65, sv=20),  # at the threshold -> closer
            _season(2, 2020, starts=0, games=65, sv=19),  # one short -> RP
            # A starter who also picked up saves is still a starter: the share is
            # checked first, so a swingman cannot be bucketed as a closer.
            _season(3, 2020, starts=25, games=30, sv=25),
        ]
    )
    out = roles(panel)
    assert out[(1, 2020)] == "closer"
    assert out[(2, 2020)] == "RP"
    assert out[(3, 2020)] == "SP"


def test_zero_games_does_not_divide_by_zero():
    """A 0-game row must bucket as RP rather than raising or producing NaN."""
    panel = pd.DataFrame([_season(1, 2020, starts=0, games=0, sv=0)])
    out = roles(panel)
    assert out[(1, 2020)] == "RP"


def test_split_season_is_summed_before_the_share_is_taken():
    """A traded starter arrives as two half-seasons.

    Each half alone is 12/16 = 0.75, which happens to still be a starter -- so the
    fixture makes the halves disagree on their own: 2/12 (RP) and 20/22 (SP). Summed
    they are 22/34 = 0.65, a starter. Reading either row alone gets it wrong.
    """
    panel = pd.DataFrame(
        [
            _season(1, 2020, starts=2, games=12, sv=0),
            _season(1, 2020, starts=20, games=22, sv=0),
        ]
    )
    out = roles(panel)
    assert len(out) == 1, "split season must collapse to one row"
    assert out[(1, 2020)] == "SP"


def test_split_season_sums_saves_too():
    """Two half-seasons of 11 saves each is a 22-save closer, not two 11-save relievers."""
    panel = pd.DataFrame(
        [
            _season(1, 2020, starts=0, games=30, sv=11),
            _season(1, 2020, starts=0, games=32, sv=11),
        ]
    )
    assert roles(panel)[(1, 2020)] == "closer"


def test_missing_role_columns_raise_rather_than_bucketing_everyone_as_rp():
    panel = pd.DataFrame([{"mlbam_id": 1, "season": 2020, "starts": 30, "games": 32}])
    with pytest.raises(KeyError, match="sv"):
        roles(panel)


def test_era_normalized_panel_is_refused():
    """The closer cut counts REAL saves, so an era-normalized frame must be rejected.

    `era_normalize` rescales `sv_ip` and `panel.score` rebuilds `sv` from it, so a
    20-save threshold applied to a normalized frame is a threshold on restated saves.
    The frames are otherwise interchangeable to look at, which is exactly why this is a
    refusal rather than a comment -- it was a live bug caught in review on #326.
    """
    panel = pd.DataFrame(
        [
            {
                "mlbam_id": 1,
                "season": 2020,
                "starts": 0,
                "games": 65,
                "sv": 20,
                "era_factor_sv_ip": 0.94,
            }
        ]
    )
    with pytest.raises(ValueError, match="needs the RAW panel"):
        roles(panel)
