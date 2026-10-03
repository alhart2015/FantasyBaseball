"""The run comparison reads a run folder's config and scored frames."""

import json

import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.features import TARGETS
from scripts import compare_hitter_ros_runs


def _scored(ours_err, blend_err, **unit):
    rows = []
    for system, err in (("ours", ours_err), ("fg_blend", blend_err)):
        for s in TARGETS:
            rows.append({"player_id": 1, "system": system, "stat": s, "abs_err": err, **unit})
    return pd.DataFrame(rows)


def test_run_row(tmp_path, monkeypatch):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "seasons": [
                    {"best_epoch": 2, "val_loss": [0.9, 0.8, 0.85]},
                    {"best_epoch": 4, "val_loss": [0.7, 0.6]},
                ]
            }
        )
    )
    pd.concat([_scored(10.0, 9.0, season=2024), _scored(8.0, 9.0, season=2025)]).to_parquet(
        run / "scored_preseason.parquet"
    )
    pd.concat(
        [
            _scored(5.0, 6.0, season=2026, snapshot="2026-06-04"),
            _scored(1.0, 9.0, season=2026, snapshot="2026-09-01"),  # outside the window
        ]
    ).to_parquet(run / "scored_snapshots.parquet")
    monkeypatch.setattr(compare_hitter_ros_runs, "RUNS", tmp_path)

    row = compare_hitter_ros_runs.compare(["r1"], "2026-06-01", "2026-07-31").loc["r1"]
    assert row["best_epoch"] == 3 and row["val_loss"] == pytest.approx(0.7)
    assert row["pre_hr"] == pytest.approx(9.0) and row["pre_gap_hr"] == pytest.approx(0.0)
    assert row["mid_gap_r"] == pytest.approx(-1.0)
