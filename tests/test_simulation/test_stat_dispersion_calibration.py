import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from calibrate_stat_dispersion import (
    POISSON_SENTINEL,
    build_residuals,
    fit_dispersion,
    interval_coverage,
)


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
