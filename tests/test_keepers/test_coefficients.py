"""Hold the shipped constants against the study's own output.

Without this, `coefficients.py` is thirteen hand-typed floats that drift silently
from the CSVs the moment the study is re-run (e.g. when the 2025->2026 pair
opens after the 2026 season). A failure here means the study moved and the
shipped policy has not been updated deliberately -- it is not a flaky test.
"""

from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.coefficients import POLICIES, policy_from_study

ANALYSIS = Path(__file__).resolve().parents[2] / "data" / "analysis"


def _report(player_type: str) -> pd.DataFrame:
    path = ANALYSIS / f"keeper_calibration_{player_type}.csv"
    if not path.exists():  # pragma: no cover - only when the study output is absent
        pytest.skip(f"{path.name} not present; run scripts/keeper_calibration.py")
    return pd.read_csv(path)


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_shipped_policy_matches_the_study(player_type: str) -> None:
    """The coefficients, n0 and gate, derived by the library's own rule.

    Deriving rather than re-implementing the comparison means n0 and gate are
    checked too. Re-running the study at a different n0 previously moved every k
    while `POLICIES` silently kept the old n0, which the module docstring calls
    meaningless.

    `ramp_width` and `pt_col` are NOT checked here: the study does not measure
    them, so they are fed in from the policy under test and cannot fail. They are
    covered by the serve_weights tests below.
    """
    policy = POLICIES[player_type]
    derived = policy_from_study(
        _report(player_type), pt_col=policy.pt_col, ramp_width=policy.ramp_width
    )
    assert dict(policy.coefficients) == dict(derived.coefficients)
    assert policy.n0 == derived.n0
    assert policy.gate == derived.gate


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


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_serve_weights_shrink_the_rates_but_not_playing_time(player_type: str) -> None:
    """Spec 5.3: the shrink applies to rate residuals only.

    Folding the playing-time column at the rate weight is the failure this method
    exists to prevent -- it silently attenuates the PT move and nothing raises.
    """
    policy = POLICIES[player_type]
    well_above_gate = pd.Series([policy.gate * 4], index=[1])
    weights = policy.serve_weights(well_above_gate)
    assert set(weights) == set(policy.coefficients)
    assert weights[policy.pt_col].iloc[0] == pytest.approx(1.0)
    rate_cols = [c for c in policy.coefficients if c != policy.pt_col]
    for col in rate_cols:
        assert 0.0 < weights[col].iloc[0] < 1.0, col
        assert weights[col].iloc[0] < weights[policy.pt_col].iloc[0], col


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_serve_weights_use_the_ramp_not_the_hard_gate(player_type: str) -> None:
    # Just above the gate the fold must be nearly off, not fully on -- that is the
    # 78.7% / 44.6% cliff the ramp exists to remove (finding B.7).
    policy = POLICIES[player_type]
    just_above = pd.Series([policy.gate + 1.0], index=[1])
    assert policy.serve_weights(just_above)[policy.pt_col].iloc[0] < 0.05


@pytest.mark.parametrize("player_type", ["hitter", "pitcher"])
def test_serve_weights_do_not_fold_a_player_below_the_gate(player_type: str) -> None:
    policy = POLICIES[player_type]
    below = pd.Series([policy.gate - 1.0], index=[1])
    assert all(w.iloc[0] == 0.0 for w in policy.serve_weights(below).values())


def test_policy_from_study_ships_the_endpoint_for_a_fallback() -> None:
    report = pd.DataFrame(
        {
            "estimator": ["fitted-k", "fitted-k"],
            "column": ["hr_pa", "pa"],
            "verdict": ["pass", "fallback:k=1"],
            "k_full": [0.4944, 0.6461],
            "n0": [200.0, 200.0],
            "gate": [100.0, 100.0],
        }
    )
    policy = policy_from_study(report, pt_col="pa", ramp_width=100.0)
    assert policy.coefficients["hr_pa"] == pytest.approx(0.494)  # fitted value
    assert policy.coefficients["pa"] == 1.0  # the endpoint that won, not 0.6461


def test_policy_from_study_rejects_a_report_mixing_fit_settings() -> None:
    report = pd.DataFrame(
        {
            "estimator": ["fitted-k", "fitted-k"],
            "column": ["hr_pa", "pa"],
            "verdict": ["pass", "pass"],
            "k_full": [0.4, 0.6],
            "n0": [200.0, 400.0],
            "gate": [100.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="mixes fit settings"):
        policy_from_study(report, pt_col="pa", ramp_width=100.0)


def test_chosen_estimator_name_matches_the_library() -> None:
    # `policy_from_study` matches report rows on CHOSEN_ESTIMATOR, but the name is
    # defined by the estimator class. If they drift, the study emits a report the
    # library cannot read -- and the failure is an empty policy, not an error.
    from fantasy_baseball.keepers.calibration import ShrunkTransfer
    from fantasy_baseball.keepers.coefficients import CHOSEN_ESTIMATOR

    assert ShrunkTransfer.name == CHOSEN_ESTIMATOR
