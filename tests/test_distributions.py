"""Unit tests for the MC distribution-builder module."""

import json

import numpy as np

from fantasy_baseball.distributions import (
    GRID_POINTS,
    _silverman_bandwidth,
    build_continuous_metric,
)


def test_continuous_metric_shared_grid_and_shape():
    rng = np.random.default_rng(0)
    team_samples = {
        "A": rng.normal(100.0, 8.0, 500),
        "B": rng.normal(130.0, 8.0, 500),
    }
    out = build_continuous_metric(team_samples)
    assert len(out["x"]) == GRID_POINTS
    assert set(out["teams"]) == {"A", "B"}
    for name in ("A", "B"):
        assert len(out["teams"][name]["y"]) == GRID_POINTS
    # x is the single shared grid (one list, every team sampled on it).
    assert isinstance(out["x"], list)
    # Medians track the samples.
    assert abs(out["teams"]["A"]["median"] - 100.0) < 5.0
    assert abs(out["teams"]["B"]["median"] - 130.0) < 5.0


def test_continuous_metric_density_integrates_to_one():
    rng = np.random.default_rng(1)
    out = build_continuous_metric({"A": rng.normal(50.0, 5.0, 800)})
    x = np.array(out["x"])
    y = np.array(out["teams"]["A"]["y"])
    assert abs(float(np.trapezoid(y, x)) - 1.0) < 0.05


def test_continuous_metric_is_json_serializable_plain_floats():
    rng = np.random.default_rng(2)
    out = build_continuous_metric({"A": rng.normal(0.0, 1.0, 100)})
    json.dumps(out)  # raises TypeError if any numpy types leaked
    assert isinstance(out["x"][0], float)
    assert isinstance(out["teams"]["A"]["y"][0], float)
    assert isinstance(out["teams"]["A"]["median"], float)


def test_near_constant_input_is_finite_and_normalized():
    # One near-deterministic team plus a spread team (so pooled range > 0 and
    # the metric-relative bandwidth floor is positive). A near-constant team
    # SHOULD render tight; the contract here is "finite, normalized, no NaN",
    # not artificial width.
    samples = {
        "tight": np.full(200, 50.0),
        "wide": np.linspace(40.0, 60.0, 200),
    }
    out = build_continuous_metric(samples)
    y = np.array(out["teams"]["tight"]["y"])
    assert np.all(np.isfinite(y))
    x = np.array(out["x"])
    assert abs(float(np.trapezoid(y, x)) - 1.0) < 0.1
    # Peak sits at the constant value.
    assert abs(float(x[int(np.argmax(y))]) - 50.0) < 2.0


def test_sentinel_values_are_dropped_before_grid():
    # ERA-like samples with a few 99.0 zero-IP sentinels; they must not stretch
    # the grid or add mass near 99.
    samples = {"A": np.array([3.1, 3.2, 3.3, 3.0, 3.4, 99.0, 99.0])}
    out = build_continuous_metric(samples, sentinel=99.0)
    assert max(out["x"]) < 10.0  # 99 dropped, grid stays near the real data


def test_silverman_bandwidth_zero_for_constant_input():
    assert _silverman_bandwidth(np.full(10, 5.0)) == 0.0


from fantasy_baseball.distributions import build_discrete_metric


def test_discrete_metric_shared_support_and_pmf():
    team_samples = {
        "A": [11, 11, 12, 12, 12],
        "B": [1, 2, 2, 3],
    }
    out = build_discrete_metric(team_samples)
    # Shared x = sorted union of observed values across BOTH teams.
    assert out["x"] == [1.0, 2.0, 3.0, 11.0, 12.0]
    for name in ("A", "B"):
        p = out["teams"][name]["p"]
        assert len(p) == len(out["x"])
        assert abs(sum(p) - 1.0) < 1e-9
    # A never realized 1/2/3 -> zeros there.
    assert out["teams"]["A"]["p"][:3] == [0.0, 0.0, 0.0]
    # mean = sum(x * p): A is (11*2 + 12*3)/5 = 11.6
    assert abs(out["teams"]["A"]["mean"] - 11.6) < 1e-9


def test_discrete_metric_half_integer_support_from_ties():
    # A tie produces a 0.5-step point value; it must appear in the shared support.
    out = build_discrete_metric({"A": [11.5, 11.5, 12.0], "B": [1.0, 1.0, 1.0]})
    assert 11.5 in out["x"]
    assert out["x"] == sorted(out["x"])


def test_discrete_metric_json_serializable():
    out = build_discrete_metric({"A": [1, 2, 3], "B": [3, 3, 3]})
    json.dumps(out)
    assert isinstance(out["x"][0], float)
    assert isinstance(out["teams"]["A"]["p"][0], float)
    assert isinstance(out["teams"]["A"]["mean"], float)
