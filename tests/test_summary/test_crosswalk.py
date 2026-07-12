from pathlib import Path

import pandas as pd

from fantasy_baseball.summary.crosswalk import build_typed_name_to_mlbam, player_group


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_crosswalk_is_type_namespaced_and_avoids_collisions(tmp_path):
    season_dir = tmp_path / "2026"
    _write_csv(season_dir / "steamer-hitters.csv", [{"Name": "Will Smith", "MLBAMID": 111}])
    _write_csv(season_dir / "steamer-pitchers.csv", [{"Name": "Will Smith", "MLBAMID": 222}])

    xmap = build_typed_name_to_mlbam(tmp_path, season=2026)

    assert xmap[("will smith", "hitter")] == 111
    assert xmap[("will smith", "pitcher")] == 222


def test_crosswalk_skips_rows_missing_mlbamid(tmp_path):
    season_dir = tmp_path / "2026"
    _write_csv(season_dir / "atc-hitters.csv", [{"Name": "No Id", "MLBAMID": ""}])
    xmap = build_typed_name_to_mlbam(tmp_path, season=2026)
    assert ("no id", "hitter") not in xmap


def test_crosswalk_drops_same_type_same_name_collision(tmp_path):
    # Two DIFFERENT pitchers who normalize to "luis garcia" cannot be told apart
    # by name, so the key is dropped (not first-write-won) to avoid emitting the
    # wrong player's line. A same-id repeat (same person across systems) is kept.
    season_dir = tmp_path / "2026"
    _write_csv(
        season_dir / "steamer-pitchers.csv",
        [{"Name": "Luis Garcia", "MLBAMID": 111}, {"Name": "Luis Garcia", "MLBAMID": 222}],
    )
    _write_csv(season_dir / "steamer-hitters.csv", [{"Name": "Solo Guy", "MLBAMID": 333}])
    _write_csv(season_dir / "atc-hitters.csv", [{"Name": "Solo Guy", "MLBAMID": 333}])

    xmap = build_typed_name_to_mlbam(tmp_path, season=2026)

    assert ("luis garcia", "pitcher") not in xmap  # ambiguous -> dropped
    assert xmap[("solo guy", "hitter")] == 333  # same id across systems -> kept


def test_crosswalk_skips_a_csv_missing_required_columns(tmp_path):
    # A malformed pitcher CSV (no MLBAMID column) must not crash the whole map;
    # the good hitter file still resolves.
    season_dir = tmp_path / "2026"
    _write_csv(season_dir / "steamer-hitters.csv", [{"Name": "Good Hitter", "MLBAMID": 555}])
    _write_csv(season_dir / "steamer-pitchers.csv", [{"Name": "Bad Row", "WrongCol": 1}])
    xmap = build_typed_name_to_mlbam(tmp_path, season=2026)
    assert xmap[("good hitter", "hitter")] == 555
    assert ("bad row", "pitcher") not in xmap


def test_player_group_classification():
    assert player_group(["1B", "OF"]) == ["hitting"]
    assert player_group(["SP"]) == ["pitching"]
    assert player_group(["RP", "P"]) == ["pitching"]
    assert sorted(player_group(["DH", "SP"])) == ["hitting", "pitching"]  # two-way
