"""Tests for the projection-rate reader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.streaks.data.projections import (
    PROJECTION_PA_FLOOR,
    discover_projection_files,
    load_projection_rates,
)


def _write_proj_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols = ["Name", "PA", "HR", "SB", "MLBAMID"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_discover_projection_files_no_suffix(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    (base / "steamer-hitters.csv").touch()
    (base / "zips-hitters.csv").touch()
    (base / "steamer-pitchers.csv").touch()  # ignored: pitcher file
    files = discover_projection_files(tmp_path, season=2024)
    assert sorted(p.name for p in files) == ["steamer-hitters.csv", "zips-hitters.csv"]


def test_discover_projection_files_with_year_suffix(tmp_path: Path) -> None:
    base = tmp_path / "2025"
    base.mkdir()
    (base / "steamer-hitters-2025.csv").touch()
    (base / "zips-hitters-2025.csv").touch()
    files = discover_projection_files(tmp_path, season=2025)
    assert sorted(p.name for p in files) == [
        "steamer-hitters-2025.csv",
        "zips-hitters-2025.csv",
    ]


def test_load_projection_rates_blends_two_systems(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    # Player 100: 30 HR / 600 PA in steamer (=0.05/PA), 36 HR / 600 PA in zips (=0.06/PA).
    # Mean: 0.055 HR/PA. SB: 12/600 (=0.02) and 18/600 (=0.03) -> 0.025 SB/PA.
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [{"Name": "P", "PA": 600, "HR": 30, "SB": 12, "MLBAMID": 100}],
    )
    _write_proj_csv(
        base / "zips-hitters.csv",
        [{"Name": "P", "PA": 600, "HR": 36, "SB": 18, "MLBAMID": 100}],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    assert len(rates) == 1
    r = rates[0]
    assert r.player_id == 100
    assert r.season == 2024
    assert r.hr_per_pa == pytest.approx(0.055, rel=1e-6)
    assert r.sb_per_pa == pytest.approx(0.025, rel=1e-6)
    assert r.n_systems == 2


def test_load_projection_rates_emits_single_system_player(tmp_path: Path) -> None:
    """A player appearing in only one of the two systems is still emitted with n_systems=1."""
    base = tmp_path / "2024"
    base.mkdir()
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [{"Name": "Solo", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": 200}],
    )
    _write_proj_csv(
        base / "zips-hitters.csv",
        [{"Name": "Other", "PA": 400, "HR": 12, "SB": 8, "MLBAMID": 300}],
    )
    rates_by_id = {r.player_id: r for r in load_projection_rates(tmp_path, season=2024)}
    assert rates_by_id[200].n_systems == 1
    assert rates_by_id[200].hr_per_pa == pytest.approx(20 / 500, rel=1e-6)
    assert rates_by_id[300].n_systems == 1


def test_load_projection_rates_filters_below_pa_floor(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [
            {"Name": "Reg", "PA": 600, "HR": 30, "SB": 5, "MLBAMID": 1},
            {"Name": "Filler", "PA": PROJECTION_PA_FLOOR - 1, "HR": 1, "SB": 0, "MLBAMID": 2},
        ],
    )
    _write_proj_csv(
        base / "zips-hitters.csv",
        [{"Name": "Reg", "PA": 600, "HR": 28, "SB": 6, "MLBAMID": 1}],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    ids = {r.player_id for r in rates}
    assert 1 in ids
    assert 2 not in ids  # below floor in steamer; not in zips at all -> dropped


def test_load_projection_rates_drops_rows_without_mlbamid(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [
            {"Name": "Has", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": 1},
            {"Name": "NoID", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": ""},
        ],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    assert {r.player_id for r in rates} == {1}


def test_load_projection_rates_skips_files_missing_required_columns(tmp_path: Path) -> None:
    """A CSV missing PA/HR/SB columns is skipped (warning) without raising."""
    base = tmp_path / "2024"
    base.mkdir()
    # File 1: missing HR column entirely.
    pd.DataFrame(
        [{"Name": "X", "PA": 500, "SB": 5, "MLBAMID": 1}], columns=["Name", "PA", "SB", "MLBAMID"]
    ).to_csv(base / "broken-hitters.csv", index=False)
    # File 2: well-formed.
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [{"Name": "Y", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": 2}],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    # Only the well-formed file's player should be emitted.
    assert {r.player_id for r in rates} == {2}


def test_load_projection_rates_blends_dense_categories(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    # Steamer: 600 PA, 30 HR, 10 SB, 90 R, 100 RBI, .280 AVG.
    # ZiPS:    600 PA, 36 HR, 14 SB, 100 R, 110 RBI, .300 AVG.
    # Blended rates: HR=0.055/PA, SB=0.020/PA, R=0.158333/PA, RBI=0.175/PA, AVG=0.290.
    pd.DataFrame(
        [
            {
                "Name": "P",
                "PA": 600,
                "HR": 30,
                "SB": 10,
                "R": 90,
                "RBI": 100,
                "AVG": 0.280,
                "MLBAMID": 100,
            }
        ],
        columns=["Name", "PA", "HR", "SB", "R", "RBI", "AVG", "MLBAMID"],
    ).to_csv(base / "steamer-hitters.csv", index=False)
    pd.DataFrame(
        [
            {
                "Name": "P",
                "PA": 600,
                "HR": 36,
                "SB": 14,
                "R": 100,
                "RBI": 110,
                "AVG": 0.300,
                "MLBAMID": 100,
            }
        ],
        columns=["Name", "PA", "HR", "SB", "R", "RBI", "AVG", "MLBAMID"],
    ).to_csv(base / "zips-hitters.csv", index=False)

    rates = load_projection_rates(tmp_path, season=2024)
    assert len(rates) == 1
    r = rates[0]
    assert r.hr_per_pa == pytest.approx(0.055, rel=1e-6)
    assert r.sb_per_pa == pytest.approx(0.020, rel=1e-6)
    assert r.r_per_pa == pytest.approx((90 + 100) / 2 / 600, rel=1e-6)
    assert r.rbi_per_pa == pytest.approx((100 + 110) / 2 / 600, rel=1e-6)
    assert r.avg == pytest.approx(0.290, rel=1e-6)
    assert r.n_systems == 2


def test_load_projection_rates_handles_missing_dense_columns(tmp_path: Path) -> None:
    """If a CSV is missing R/RBI/AVG columns (older fixtures), the loader emits
    the rate row with NULL in those fields rather than crashing."""
    base = tmp_path / "2024"
    base.mkdir()
    # Old-style CSV: only Name/PA/HR/SB/MLBAMID.
    pd.DataFrame(
        [{"Name": "P", "PA": 600, "HR": 30, "SB": 12, "MLBAMID": 100}],
        columns=["Name", "PA", "HR", "SB", "MLBAMID"],
    ).to_csv(base / "steamer-hitters.csv", index=False)
    rates = load_projection_rates(tmp_path, season=2024)
    assert len(rates) == 1
    assert rates[0].r_per_pa is None
    assert rates[0].rbi_per_pa is None
    assert rates[0].avg is None
