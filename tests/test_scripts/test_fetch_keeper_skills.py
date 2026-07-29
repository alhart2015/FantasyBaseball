"""Unit tests for the keeper-skills ingest script's local helpers.

The derivation lives in `fantasy_baseball.keepers.skills` and the name repair in
`fantasy_baseball.utils.name_utils`; both are tested next to their own modules.
What is script-local is snapshot selection, the MLBAM->park-factor bridge, and
attaching names.
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts import fetch_keeper_skills
from scripts.fetch_keeper_skills import build_park_factors, latest_ros_dir, with_names

ESCAPED_ACUNA = "Luisangel Acu\\xc3\\xb1a"


@pytest.fixture
def projections(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(fetch_keeper_skills, "PROJECTIONS_DIR", tmp_path)
    return tmp_path / "2026" / "rest_of_season"


def _write_snapshot(root: Path, name: str, **frame: list) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    pd.DataFrame(frame).to_csv(directory / "steamer-hitters.csv", index=False)


# --- snapshot selection ----------------------------------------------------


def test_latest_snapshot_picked_by_parsed_date_not_string_sort(projections: Path):
    """An undatable helper dir sorts above every dated one as a raw string; it
    must not shadow a fresh snapshot."""
    for name in ("2026-06-25", "2026-07-27", "manual-latest"):
        _write_snapshot(projections, name, MLBAMID=[1], Team=["COL"])
    assert latest_ros_dir(2026).name == "2026-07-27"


def test_dated_suffix_still_parses(projections: Path):
    for name in ("2026-06-25", "2026-07-27-manual"):
        _write_snapshot(projections, name, MLBAMID=[1], Team=["COL"])
    assert latest_ros_dir(2026).name == "2026-07-27-manual"


def test_missing_season_dir_is_not_an_error(projections: Path):
    assert latest_ros_dir(2026) is None


# --- park factor bridge ----------------------------------------------------


def test_park_factors_keyed_by_mlbam(projections: Path):
    _write_snapshot(projections, "2026-07-27", MLBAMID=[11, 22], Team=["COL", "SDP"])
    factors = build_park_factors(2026, "hitters")
    assert factors.loc[11] == pytest.approx(1.13)  # Coors inflates
    assert factors.loc[22] == pytest.approx(0.92)  # Petco suppresses


def test_unknown_team_falls_back_to_neutral(projections: Path):
    _write_snapshot(projections, "2026-07-27", MLBAMID=[11], Team=["ZZZ"])
    assert build_park_factors(2026, "hitters").loc[11] == pytest.approx(1.0)


def test_a_non_projection_csv_is_skipped_not_fatal(projections: Path):
    """A usecols mismatch means the file is not a projection export."""
    _write_snapshot(projections, "2026-07-27", MLBAMID=[11], Team=["COL"])
    (projections / "2026-07-27" / "notes-hitters.csv").write_text("a,b\n1,2\n")
    assert build_park_factors(2026, "hitters").loc[11] == pytest.approx(1.13)


def test_no_snapshot_yields_none_so_adjustment_goes_neutral(projections: Path):
    projections.mkdir(parents=True)
    assert build_park_factors(2026, "hitters") is None


# --- name attachment -------------------------------------------------------


def test_with_names_prepends_name_keyed_by_mlbam():
    skills = pd.DataFrame({"pa": [600, 500]}, index=pd.Index([11, 22], name="mlbam_id"))
    source = pd.DataFrame({"mlbID": [22, 11], "Name": ["Ronald Acuna Jr.", "Andrew Abbott"]})
    out = with_names(skills, source)
    assert list(out.columns) == ["name", "pa"]
    assert out.loc[11, "name"] == "Andrew Abbott"
    assert out.loc[22, "name"] == "Ronald Acuna Jr."


def test_with_names_does_not_repair_because_ingest_already_did():
    """If this ever fails, repair has been lost from `bref` -- the cache would be
    corrupt too, which this test cannot see."""
    skills = pd.DataFrame({"pa": [600]}, index=pd.Index([11], name="mlbam_id"))
    source = pd.DataFrame({"mlbID": [11], "Name": [ESCAPED_ACUNA]})
    assert with_names(skills, source).loc[11, "name"] == ESCAPED_ACUNA


def test_with_names_blanks_an_unmatched_id_rather_than_dropping_it():
    skills = pd.DataFrame({"pa": [600]}, index=pd.Index([999], name="mlbam_id"))
    source = pd.DataFrame({"mlbID": [11], "Name": ["Andrew Abbott"]})
    out = with_names(skills, source)
    assert len(out) == 1
    assert out.loc[999, "name"] == ""
    assert out.loc[999, "pa"] == 600
