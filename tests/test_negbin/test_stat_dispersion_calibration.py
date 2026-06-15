import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from calibrate_stat_dispersion import (
    POISSON_SENTINEL,
    bucket_diagnostic,
    build_residuals,
    fit_banded_dispersion,
    fit_dispersion,
    interval_coverage,
    loso_coverage,
    role_stable_sv,
)

from fantasy_baseball.utils.dispersion import resolve_dispersion_r


def test_fit_dispersion_recovers_known_r():
    rng = np.random.default_rng(0)
    mu = np.full(20_000, 12.0)
    true_r = 2.5
    p = true_r / (true_r + mu)
    x = rng.negative_binomial(true_r, p)
    r_hat = fit_dispersion(x, mu)
    assert abs(r_hat - true_r) < 0.3


def test_fit_dispersion_clamps_to_poisson_when_underdispersed():
    rng = np.random.default_rng(1)
    mu = np.full(20_000, 8.0)
    x = rng.poisson(mu)  # var == mean -> no overdispersion
    assert fit_dispersion(x, mu) == POISSON_SENTINEL


def test_interval_coverage_matches_nominal_when_model_is_correct():
    rng = np.random.default_rng(2)
    mu = np.full(50_000, 15.0)
    r = 3.0
    p = r / (r + mu)
    x = rng.negative_binomial(r, p)
    cov80 = interval_coverage(x, mu, r, level=0.80)
    assert abs(cov80 - 0.80) < 0.03


def test_interval_coverage_handles_poisson_sentinel():
    # ppf-based central intervals for discrete distributions are conservative:
    # the actual coverage >= the nominal level rather than approximately equal.
    # For Poisson(6) at level=0.50, ppf(0.25)=4 and ppf(0.75)=8 enclose ~70%
    # of the mass, so we assert that coverage is at least 0.50 and within 0.25.
    rng = np.random.default_rng(3)
    mu = np.full(50_000, 6.0)
    x = rng.poisson(mu)
    cov50 = interval_coverage(x, mu, POISSON_SENTINEL, level=0.50)
    assert cov50 >= 0.50
    assert abs(cov50 - 0.50) < 0.25


def _write_csv(path, rows, header):
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_build_residuals_conditions_on_realized_pt(tmp_path, monkeypatch):
    # One hitter: proj 600 PA / 30 SB (rate 0.05/PA); realized 300 PA, 18 SB.
    # mu = 0.05 * 300 = 15.0; actual_count = 18.
    proj_dir = tmp_path / "projections" / "2022"
    proj_dir.mkdir(parents=True)
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    # Proj CSVs must carry ALL counting columns -- pandas merge only suffixes
    # overlapping columns, so a partial proj would leave R/HR/RBI/H unsuffixed
    # and build_residuals' m["R_proj"] lookup would KeyError.
    proj_header = "PA,R,HR,RBI,SB,H,MLBAMID"
    proj_row = "600,90,25,85,30,150,111"
    _write_csv(proj_dir / "steamer-hitters.csv", [proj_row], proj_header)
    _write_csv(proj_dir / "zips-hitters.csv", [proj_row], proj_header)
    h_header = "Name,Team,G,PA,AB,H,HR,R,RBI,SB,AVG,MLBAMID"
    _write_csv(stats_dir / "hitters-2022.csv", ["A,X,75,300,270,80,12,45,40,18,.296,111"], h_header)

    monkeypatch.setattr("calibrate_stat_dispersion.PROJ_DIR", tmp_path / "projections")
    monkeypatch.setattr("calibrate_stat_dispersion.STATS_DIR", stats_dir)
    monkeypatch.setattr("calibrate_stat_dispersion.YEARS", [2022])

    res = build_residuals("hitters")
    # All five correlated hitter keys must be produced (proves the loop covers
    # every key, not just sb).
    assert set(res) == {"r", "hr", "rbi", "sb", "h"}
    sb = res["sb"]
    assert sb["year"].tolist() == [2022]
    assert sb["actual"].tolist() == [18.0]
    assert abs(sb["mu"].iloc[0] - 15.0) < 1e-6


def test_loso_coverage_returns_per_fold_table():
    rng = np.random.default_rng(4)
    frames = []
    for yr in (2022, 2023, 2024):
        mu = rng.uniform(5, 25, 4000)
        r = 3.0
        x = rng.negative_binomial(r, r / (r + mu))
        frames.append(pd.DataFrame({"year": yr, "actual": x.astype(float), "mu": mu}))
    data = pd.concat(frames, ignore_index=True)
    table = loso_coverage(data, levels=(0.50, 0.80))
    assert set(table["held_out"]) == {2022, 2023, 2024}
    assert abs(table["cov_0.80"].mean() - 0.80) < 0.05


def test_bucket_diagnostic_flags_scalar_r_as_adequate_when_true():
    # Data generated with a single r across all mu buckets -> the per-observation
    # Pearson statistic is ~1.0 in every bucket (immune to within-bucket mu spread).
    rng = np.random.default_rng(5)
    mu = rng.uniform(2, 40, 30_000)
    r = 3.0
    x = rng.negative_binomial(r, r / (r + mu))
    diag = bucket_diagnostic(pd.DataFrame({"actual": x.astype(float), "mu": mu}), r, n_bins=4)
    assert diag["pearson"].between(0.85, 1.15).all()


def test_role_stable_sv_keeps_only_stable_closers():
    df = pd.DataFrame(
        {
            "year": [2022, 2022, 2022],
            "actual": [30.0, 2.0, 28.0],
            "mu": [29.0, 25.0, 27.0],
            "proj_sv": [30.0, 28.0, 26.0],
            "actual_sv": [30.0, 2.0, 28.0],
        }
    )
    out = role_stable_sv(df)
    # row1 (proj 30/act 30) and row3 (proj 26/act 28) are stable closers;
    # row2 (proj 28/act 2) lost the job -> excluded (CLOSER_SV_THRESHOLD=20).
    assert out["actual"].tolist() == [30.0, 28.0]


def test_fit_banded_dispersion_fits_per_band_and_lowers_pearson():
    # Mean-dependent overdispersion: low mu -> small r (heavy spread), high mu ->
    # large r (tight). A single scalar r cannot fit both; bands can.
    rng = np.random.default_rng(9)
    lo_mu = rng.uniform(1, 5, 8000)
    hi_mu = rng.uniform(20, 40, 8000)
    lo = rng.negative_binomial(0.5, 0.5 / (0.5 + lo_mu))
    hi = rng.negative_binomial(8.0, 8.0 / (8.0 + hi_mu))
    df = pd.DataFrame(
        {
            "actual": np.concatenate([lo, hi]).astype(float),
            "mu": np.concatenate([lo_mu, hi_mu]),
        }
    )
    bands = fit_banded_dispersion(df, n_bands=4)
    assert bands[-1][0] == float("inf")
    r_elem = resolve_dispersion_r(bands, df["mu"].to_numpy())
    diag = bucket_diagnostic(df, r_elem, n_bins=4)
    assert diag["pearson"].between(0.6, 1.6).all()


def test_fit_banded_dispersion_collapses_thin_data_to_single_band():
    rng = np.random.default_rng(13)
    mu = rng.uniform(20, 35, 40)  # only 40 rows -> < min_per_band -> 1 band
    r = 5.0
    x = rng.negative_binomial(r, r / (r + mu))
    bands = fit_banded_dispersion(pd.DataFrame({"actual": x.astype(float), "mu": mu}))
    assert len(bands) == 1
    assert bands[0][0] == float("inf")
