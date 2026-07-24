"""Season-level skill/luck data layer: FanGraphs rates + age, Statcast xStats,
and the MLBAM<->FanGraphs id map, cached to data/skill_luck/. Fetch-on-miss,
never overwrite a good cache with an empty/failed pull. See
docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

ID_MAP_FILE = "id_map_mlbam_fg.csv"


def _read_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None


def load_id_map(
    cache_dir: Path, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    path = cache_dir / ID_MAP_FILE
    cached = _read_cached(path)
    if cached is not None:
        return cached
    if fetcher is None:
        from pybaseball import chadwick_register  # local import: keeps module import-safe

        fetcher = chadwick_register
    reg = fetcher()
    out = (
        reg[["key_mlbam", "key_fangraphs"]]
        .dropna()
        .astype({"key_mlbam": int, "key_fangraphs": int})
    )
    if out.empty:
        raise RuntimeError("chadwick_register returned no usable id rows; refusing to cache empty")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
