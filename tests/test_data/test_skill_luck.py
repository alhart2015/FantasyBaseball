from pathlib import Path
import pandas as pd
from fantasy_baseball.data import skill_luck


def _fake_register():
    return pd.DataFrame(
        {
            "key_mlbam": [665742, 700, float("nan")],  # Soto, junk, unmatched
            "key_fangraphs": [20123, float("nan"), 55],
            "name_first": ["Juan", "No", "No"],
            "name_last": ["Soto", "Fg", "Mlbam"],
        }
    )


def test_load_id_map_drops_unmatched_and_caches(tmp_path: Path):
    m = skill_luck.load_id_map(tmp_path, fetcher=_fake_register)
    # only the fully-identified row survives
    assert list(m["key_mlbam"]) == [665742]
    assert list(m["key_fangraphs"]) == [20123]

    # second call with a fetcher that would raise must hit the cache, not the network
    def _boom():
        raise AssertionError("must not refetch")

    m2 = skill_luck.load_id_map(tmp_path, fetcher=_boom)
    assert list(m2["key_mlbam"]) == [665742]
