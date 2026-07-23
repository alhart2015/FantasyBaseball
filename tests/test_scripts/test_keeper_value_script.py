import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_value as script


def test_load_zips_year_missing_raises_with_url(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        script.load_zips_year(tmp_path, 2027)
    assert "fangraphs.com" in str(exc.value)
    assert "2027" in str(exc.value)


def test_load_zips_year_loads_present(tmp_path):
    d = tmp_path / "2027"
    d.mkdir()
    pd.DataFrame(
        [{"Name": "A B", "AB": 500, "H": 150, "HR": 30, "R": 90, "RBI": 95, "SB": 10, "AVG": 0.300}]
    ).to_csv(d / "zips-hitters.csv", index=False)
    # FanGraphs (and ZiPS) exports use "SO" for strikeouts; PITCHING_COLUMN_MAP
    # normalizes SO -> k. A "K" header would NOT be recognized.
    pd.DataFrame(
        [{"Name": "C D", "IP": 180, "W": 14, "SO": 200, "ERA": 3.2, "WHIP": 1.05, "SV": 0}]
    ).to_csv(d / "zips-pitchers.csv", index=False)
    hitters, pitchers = script.load_zips_year(tmp_path, 2027)
    assert not hitters.empty and not pitchers.empty


def test_is_candidate_matches_by_normalized_name():
    from fantasy_baseball.utils.name_utils import normalize_name

    norms = {normalize_name(n) for n in ["Julio Rodriguez"]}
    assert script.is_candidate("Julio Rodriguez", norms) is True
    assert script.is_candidate("Some Other", norms) is False
