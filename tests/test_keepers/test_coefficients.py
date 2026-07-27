"""Hold the shipped constants against the study's own output.

Without this, `coefficients.py` is twelve hand-typed floats that drift silently
from the CSVs the moment the study is re-run (e.g. when the 2025->2026 pair
opens after the 2026 season). A failure here means the study moved and the
shipped policy has not been updated deliberately -- it is not a flaky test.
"""

from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.coefficients import POLICIES

ANALYSIS = Path(__file__).resolve().parents[2] / "data" / "analysis"


def _study(player_type: str) -> pd.DataFrame:
    path = ANALYSIS / f"keeper_calibration_{player_type}.csv"
    if not path.exists():  # pragma: no cover - only when the study output is absent
        pytest.skip(f"{path.name} not present; run scripts/keeper_calibration.py")
    frame = pd.read_csv(path)
    return frame.loc[frame["estimator"] == "fitted-k"].set_index("column")


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_shipped_coefficients_match_the_study(player_type: str) -> None:
    policy = POLICIES[player_type]
    study = _study(player_type)
    assert set(policy.coefficients) == set(study.index), "coefficient set drifted from the study"
    for column, k in policy.coefficients.items():
        row = study.loc[column]
        if row["verdict"] == "pass":
            assert k == pytest.approx(float(row["k_full"]), abs=5e-4), column
        else:
            # A fallback ships the winning ENDPOINT, not the fitted value.
            expected = float(row["verdict"].removeprefix("fallback:k="))
            assert k == pytest.approx(expected), column


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_no_shipped_coefficient_amplifies(player_type: str) -> None:
    # Spec 6.2 requirement 7: a coefficient above 1 would amplify the residual.
    assert all(0.0 <= k <= 1.0 for k in POLICIES[player_type].coefficients.values())


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_policy_records_the_n0_the_coefficients_are_conditional_on(player_type: str) -> None:
    # k and the shrink are multiplicative, so a policy without its n0 is unusable.
    policy = POLICIES[player_type]
    assert policy.n0 > 0
    assert policy.ramp_width > 0
    assert policy.gate > 0
