from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.bref import fetch_bref_batting, fetch_bref_pitching


def test_batting_raw_passthrough_no_rename(tmp_path: Path):
    raw = pd.DataFrame({"mlbID": [1], "PA": [600], "R": [90], "OPS": [0.800]})
    out = fetch_bref_batting(tmp_path, 2026, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)
    assert (tmp_path / "bref_batting_2026.csv").exists()


def test_pitching_strike_rates_not_converted(tmp_path: Path):
    raw = pd.DataFrame({"mlbID": [1], "StL": [0.16], "StS": [0.10], "IP": [180.1]})
    out = fetch_bref_pitching(tmp_path, 2026, fetcher=lambda: raw)
    assert out["StS"].iloc[0] == 0.10  # NOT rescaled to 10.0
    assert out["StL"].iloc[0] == 0.16
    assert out["IP"].iloc[0] == 180.1  # baseball notation preserved, not 180.33


def test_empty_pull_refuses_to_cache(tmp_path: Path):
    """A blocked/403 pull must not overwrite or create a cache -- the whole point
    of routing these through fetch_or_cache."""
    try:
        fetch_bref_pitching(tmp_path, 2026, fetcher=lambda: pd.DataFrame())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on empty pull")
    assert not (tmp_path / "bref_pitching_2026.csv").exists()


def test_public_api_reexports():
    import fantasy_baseball.keepers as k

    for name in ("fetch_bref_batting", "fetch_bref_pitching"):
        assert hasattr(k, name), f"{name} not re-exported"
        assert name in k.__all__, f"{name} missing from __all__"
