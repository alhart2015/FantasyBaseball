"""The band calibration: the multiplier math, the artifact, and the ordering contract.

The coverage GUARANTEE is asserted here too (`test_every_cell_holds_its_nominal_tails`),
because a conformal correction whose tails do not land on nominal is not a weaker
correction -- it is a band lying about a measured quantity, which is the exact failure
`calibration` was written to remove.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.calibration import (
    ALPHA,
    BUCKET_LABELS,
    MAX_HORIZON,
    TARGETS,
    BandCalibration,
    bucket_of,
    build_table,
    conformal_multipliers,
    span_frame,
    span_target,
)


def _frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Held-out predictions whose band is deliberately WRONG, in both directions.

    Well-supported rows get a band 1.5x too wide and thin rows one 2x too narrow, which is
    the shape the real measurement found. A fixture whose band is already right cannot
    tell a working correction from `return 1.0`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        support = 0.5 if i % 2 else 0.05
        offered = 12.0 if i % 2 else 4.0  # too wide when supported, too narrow when not
        for horizon in range(1, MAX_HORIZON + 1):
            predicted = float(rng.normal(10, 4))
            rows.append(
                {
                    "pool": "hitter",
                    "mlbam_id": i,
                    "season": 2000 + (i % 20),
                    "horizon": horizon,
                    "support": support,
                    "predicted": predicted,
                    "p10": predicted - offered,
                    "p90": predicted + offered,
                    # True spread is 8.0 either way, so the offered band is wrong by a
                    # known factor and the fitted multiplier has a value to recover.
                    "actual": predicted + float(rng.normal(0, 8.0)),
                }
            )
    return pd.DataFrame(rows)


def test_bucket_edges_are_closed_below_and_nan_reads_as_supported() -> None:
    """NaN is the comp matchers, where no line was fitted -- see `bucket_of`."""
    assert bucket_of(0.0) == "<30%"
    assert bucket_of(0.2999) == "<30%"
    assert bucket_of(0.30) == ">30%"
    assert bucket_of(1.0) == ">30%"
    assert bucket_of(float("nan")) == ">30%"


def test_a_multiplier_recovers_a_known_wrong_width() -> None:
    """A band 2x too narrow around a known spread wants roughly 2x back."""
    rng = np.random.default_rng(1)
    predicted = np.full(5000, 10.0)
    actual = predicted + rng.normal(0, 8.0, 5000)
    half = 8.0 * 1.2816 / 2.0  # half the honest 10th/90th percentile offset
    lo, hi, n = conformal_multipliers(predicted, predicted - half, predicted + half, actual)
    assert n == 5000
    assert lo == pytest.approx(2.0, abs=0.1)
    assert hi == pytest.approx(2.0, abs=0.1)


def test_a_zero_width_side_is_dropped_rather_than_dividing_by_zero() -> None:
    """`shape`'s containment clamp can pin one side to `predicted` (0.04% of rows)."""
    predicted = np.array([10.0, 10.0, 10.0])
    p10 = np.array([10.0, 8.0, 8.0])  # first row's lower side has no width
    p90 = np.array([12.0, 12.0, 12.0])
    lo, hi, n = conformal_multipliers(predicted, p10, p90, np.array([9.0, 9.0, 11.0]))
    assert n == 2, "the degenerate row is excluded, not divided by"
    assert np.isfinite(lo) and np.isfinite(hi)


def test_every_cell_holds_its_nominal_tails() -> None:
    """THE GUARANTEE. Each tail must hold `ALPHA` on the rows the cell was fitted on.

    In-sample by construction, which is the point: the conformal quantile makes this
    arithmetic, so a miss means the table is being APPLIED differently from how it was
    FITTED. That is the bug this module has already produced once -- see the module
    docstring on the span-ordering error, which read 13-17% here against a nominal 10%.
    """
    frame = _frame()
    table = build_table(frame, panel_vintage="test", window_years=50)
    for target in TARGETS:
        cells = span_frame(frame, target)
        for label in BUCKET_LABELS:
            sub = cells[cells["support"].map(bucket_of) == label]
            if sub.empty:
                continue
            corrected = [
                table.apply(p, lo, hi, pool="hitter", target=target, support=s)
                for p, lo, hi, s in zip(
                    sub["predicted"], sub["p10"], sub["p90"], sub["support"], strict=True
                )
            ]
            actual = sub["actual"].to_numpy()
            below = (actual < np.array([c[0] for c in corrected])).mean()
            above = (actual > np.array([c[1] for c in corrected])).mean()
            assert below == pytest.approx(ALPHA, abs=0.015), f"{target}/{label} lower tail"
            assert above == pytest.approx(ALPHA, abs=0.015), f"{target}/{label} upper tail"


def test_the_span_multiplier_is_fitted_against_a_raw_sum() -> None:
    """The ordering contract, pinned.

    `s{k}` must be usable on a sum of RAW yearly bands with nothing applied first. Fitting
    it against already-corrected years produced a table that looked right and was several
    points off in use, so this asserts the property directly rather than trusting the
    call order at each site.
    """
    frame = _frame()
    table = build_table(frame, panel_vintage="test", window_years=50)
    spans = span_frame(frame, "s3")

    raw = [
        table.apply(p, lo, hi, pool="hitter", target="s3", support=s)
        for p, lo, hi, s in zip(
            spans["predicted"], spans["p10"], spans["p90"], spans["support"], strict=True
        )
    ]
    actual = spans["actual"].to_numpy()
    assert (actual < np.array([r[0] for r in raw])).mean() == pytest.approx(ALPHA, abs=0.015)

    # And the WRONG order is genuinely wrong, so the test above is not vacuous: correcting
    # the years first and then summing overshoots, because both corrections land.
    years = frame[frame["horizon"].isin({1, 2, 3})].copy()
    corrected = [
        table.apply_year(p, lo, hi, pool="hitter", horizon=int(h), support=s)
        for p, lo, hi, h, s in zip(
            years["predicted"],
            years["p10"],
            years["p90"],
            years["horizon"],
            years["support"],
            strict=True,
        )
    ]
    years["p10"] = [c[0] for c in corrected]
    years["p90"] = [c[1] for c in corrected]
    twice = span_frame(years, "s3")
    doubled = [
        table.apply(p, lo, hi, pool="hitter", target="s3", support=s)
        for p, lo, hi, s in zip(
            twice["predicted"], twice["p10"], twice["p90"], twice["support"], strict=True
        )
    ]
    below_twice = (twice["actual"].to_numpy() < np.array([d[0] for d in doubled])).mean()
    assert abs(below_twice - ALPHA) > 0.02, (
        "double-correcting must visibly miss, or the raw-sum assertion proves nothing"
    )


def test_a_span_the_table_cannot_price_is_left_alone() -> None:
    """Non-contiguous or over-long ranges get no multiplier, never a neighbouring one."""
    assert span_target((1, 2, 3)) == "s3"
    assert span_target((1,)) == "s1"
    assert span_target((2, 3, 4)) is None, "must start at 1"
    assert span_target((1, 3)) is None, "must be contiguous"
    assert span_target(tuple(range(1, MAX_HORIZON + 2))) is None, "past the fitted range"
    assert span_target(()) is None


def test_an_unknown_cell_is_the_identity_not_an_error() -> None:
    """A board render must not die on one unpriced row -- see `scale`."""
    table = BandCalibration(
        panel_vintage="v",
        alpha=ALPHA,
        window_years=8,
        multipliers={},
        curves={},
        fallbacks={},
    )
    assert table.scale(pool="hitter", target="s3", support=0.5) == (1.0, 1.0)
    assert table.apply(10.0, 5.0, 15.0, pool="nobody", target="s3", support=0.5) == (5.0, 15.0)


def test_a_mismatched_panel_vintage_is_refused(tmp_path) -> None:
    """The silent-rot guard. A table fitted on another panel prints a scale nothing
    measured, and every number still looks entirely normal."""
    path = tmp_path / "band_calibration.json"
    BandCalibration(
        panel_vintage="hitter_pt_panel_2000_2020.csv",
        alpha=ALPHA,
        window_years=8,
        multipliers={},
        curves={},
        fallbacks={},
    ).save(path)
    assert BandCalibration.load(path).panel_vintage == "hitter_pt_panel_2000_2020.csv"
    assert BandCalibration.load(path, panel_vintage="hitter_pt_panel_2000_2020.csv")
    with pytest.raises(ValueError, match=r"[Rr]egenerate"):
        BandCalibration.load(path, panel_vintage="hitter_pt_panel_2000_2026.csv")


def test_the_artifact_round_trips(tmp_path) -> None:
    frame = _frame(n=600)
    table = build_table(frame, panel_vintage="v1", window_years=50)
    path = tmp_path / "band_calibration.json"
    table.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["window_years"] == 50
    back = BandCalibration.load(path)
    assert back.multipliers == table.multipliers
    assert back.scale(pool="hitter", target="s3", support=0.05) == table.scale(
        pool="hitter", target="s3", support=0.05
    )


def test_the_shipped_table_prices_every_target_and_bucket() -> None:
    """The artifact in `data/trajectory/` must cover everything a surface can render.

    A missing cell is the identity, so a gap here would not raise -- it would quietly
    print one uncorrected row among corrected ones.
    """
    from fantasy_baseball.trajectory.calibration import load_shipped

    table = load_shipped()
    if table is None:
        pytest.skip("no band_calibration.json built in this checkout")
    for pool in ("hitter", "pitcher"):
        for target in TARGETS:
            cells = table.multipliers[pool][target]
            assert set(cells) == set(BUCKET_LABELS), f"{pool}/{target}"
            for lo, hi in cells.values():
                assert 0.5 < lo < 2.0 and 0.5 < hi < 2.0, (
                    f"{pool}/{target} multiplier {lo}/{hi} is outside anything measured; "
                    "a table this far off is a fitting failure, not a correction"
                )


def test_the_curve_and_the_band_edges_cannot_disagree() -> None:
    """A threshold sitting on the corrected p90 must read as exactly `ALPHA`.

    The two are the same measurement -- the band edges are the `ALPHA` and `1 - ALPHA`
    points of the score curve -- so they are fitted off one set of rows. This asserts the
    identity holds through both code paths, because "the band says 10% and the
    probability says 14%" is the kind of contradiction a reader would never resolve.
    """
    table = build_table(_frame(), panel_vintage="test", window_years=50)
    predicted, p10, p90 = 10.0, 0.0, 20.0
    for support in (0.05, 0.5):
        lo_m, hi_m = table.scale(pool="hitter", target="s3", support=support)
        at_p90 = table.exceedance(
            predicted + hi_m * (p90 - predicted),
            predicted,
            p10,
            p90,
            pool="hitter",
            target="s3",
            support=support,
        )
        at_p10 = table.exceedance(
            predicted - lo_m * (predicted - p10),
            predicted,
            p10,
            p90,
            pool="hitter",
            target="s3",
            support=support,
        )
        assert at_p90 == pytest.approx(ALPHA, abs=0.01)
        assert at_p10 == pytest.approx(1 - ALPHA, abs=0.01)


def test_exceedance_is_monotone_and_bounded() -> None:
    """Higher bar, lower probability -- and never outside [0, 1]."""
    table = build_table(_frame(), panel_vintage="test", window_years=50)
    probs = [
        table.exceedance(t, 10.0, 0.0, 20.0, pool="hitter", target="s3", support=0.5)
        for t in range(-40, 60, 5)
    ]
    assert all(p is not None for p in probs)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True), "a higher threshold must not be likelier"


def test_exceedance_returns_none_rather_than_guessing() -> None:
    """An unknown cell or a degenerate band has no measured answer, so it gives none."""
    table = build_table(_frame(n=600), panel_vintage="test", window_years=50)
    assert table.exceedance(5.0, 10.0, 0.0, 20.0, pool="nobody", target="s3", support=0.5) is None
    # Lower side pinned to `predicted` by the containment clamp: no width to normalise by.
    assert table.exceedance(5.0, 10.0, 10.0, 20.0, pool="hitter", target="s3", support=0.5) is None


def test_a_curve_grid_change_is_refused_rather_than_silently_misread(tmp_path) -> None:
    """The curve is meaningless without the levels it was sampled at, and a grid edit is
    exactly the change that would not fail on its own."""
    path = tmp_path / "band_calibration.json"
    build_table(_frame(n=600), panel_vintage="v1", window_years=50).save(path)
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["curve_levels"] = blob["curve_levels"][:-1]
    path.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(ValueError, match=r"curve levels"):
        BandCalibration.load(path)


def test_a_shipped_artifact_loads_where_the_panel_it_was_fitted_on_is_absent(tmp_path) -> None:
    """THE DEPLOYED SHAPE. `.gitignore` ships `band_calibration.json` and NOT the panel
    CSVs it was fitted on, so Render has the artifact and an otherwise empty
    `data/trajectory/`. `panel_vintage_of` raises FileNotFoundError there, and it is
    called from `load_shipped`, which `sweep.totals` calls on EVERY board render -- so
    the vintage guard took every trajectory page to a 500 instead of the graceful
    degradation the ignore rule's own comment promises.

    The guard exists to catch a calibration paired with the WRONG panel. With no panel
    present there is no pairing to be wrong about, so the check is skipped, not fatal --
    and it must still raise for the build scripts, which are about to read the panel.
    """
    from fantasy_baseball.trajectory.calibration import panel_vintage_of

    assert panel_vintage_of(tmp_path, missing_ok=True) is None
    with pytest.raises(FileNotFoundError):
        panel_vintage_of(tmp_path)

    path = tmp_path / "band_calibration.json"
    build_table(_frame(n=600), panel_vintage="v1", window_years=50).save(path)
    assert BandCalibration.load(path, panel_vintage=None).panel_vintage == "v1", (
        "a None vintage means 'nothing to compare against', not 'compare against None'"
    )
