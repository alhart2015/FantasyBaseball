"""Season-level skill/luck data layer: FanGraphs rates + age, Statcast xStats,
and the MLBAM<->FanGraphs id map, cached to data/skill_luck/. Fetch-on-miss,
never overwrite a good cache with an empty/failed pull. See
docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fantasy_baseball.analysis.breakout import SkillLuckRow  # shared shape from Task 0

ID_MAP_FILE = "id_map_mlbam_fg.csv"


def _read_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None


def fetch_or_cache(path: Path, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    cached = _read_cached(path)
    if cached is not None:
        return cached
    df = fetcher()
    if df is None or df.empty:
        raise RuntimeError(
            f"fetch for {path.name} returned empty; refusing to overwrite/write cache"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# Explicit source-column -> SkillLuckRow-field maps. If a source key is missing we
# raise KeyError (fail loud) rather than silently emit a NaN column.
_FG_HIT_RENAME = {
    "IDfg": "key_fangraphs",
    "Age": "age",
    "K%": "k_pct",
    "BB%": "bb_pct",
    "BABIP": "babip",
    "HR/FB": "hr_fb",
    "Contact%": "contact_pct",
    "PA": "pa",
}


def _rename_strict(df: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise KeyError(f"source frame missing expected columns {missing}; got {list(df.columns)}")
    return df[list(rename)].rename(columns=rename)


def load_fg_hitters(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    def _default() -> pd.DataFrame:
        from pybaseball import batting_stats  # local import: keeps module import-safe

        return batting_stats(year, qual=1)

    raw = fetch_or_cache(cache_dir / f"fg_h_{year}.csv", fetcher or _default)
    return _rename_strict(raw, _FG_HIT_RENAME)


_STATCAST_XHIT_RENAME = {
    "player_id": "mlbam",
    "woba": "woba",
    "est_woba": "xwoba",
    "ba": "ba",
    "est_ba": "xba",
    "slg": "slg",
    "est_slg": "xslg",
}
_STATCAST_BARREL_RENAME = {"player_id": "mlbam", "brl_percent": "barrel_pct"}


def load_statcast_hitters(
    cache_dir: Path,
    year: int,
    *,
    xstats_fetcher: Callable[[], pd.DataFrame] | None = None,
    barrels_fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    def _x() -> pd.DataFrame:
        from pybaseball import (  # local import: keeps module import-safe
            statcast_batter_expected_stats,
        )

        return statcast_batter_expected_stats(year, minPA=1)

    def _b() -> pd.DataFrame:
        from pybaseball import (  # local import: keeps module import-safe
            statcast_batter_exitvelo_barrels,
        )

        return statcast_batter_exitvelo_barrels(year, minBBE=1)

    x = _rename_strict(
        fetch_or_cache(cache_dir / f"sc_x_h_{year}.csv", xstats_fetcher or _x),
        _STATCAST_XHIT_RENAME,
    )
    b = _rename_strict(
        fetch_or_cache(cache_dir / f"sc_brl_h_{year}.csv", barrels_fetcher or _b),
        _STATCAST_BARREL_RENAME,
    )
    # barrel_pct arrives as a percent (0-100) on Savant; normalize to a share.
    b = b.assign(barrel_pct=b["barrel_pct"].astype(float) / 100.0)
    return x.merge(b, on="mlbam", how="left")


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


_FG_PITCH_RENAME = {
    "IDfg": "key_fangraphs",
    "Age": "age",
    "K%": "k_pct",
    "BB%": "bb_pct",
    "IP": "ip",
}


def load_fg_pitchers(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    def _default() -> pd.DataFrame:
        from pybaseball import pitching_stats  # local import: keeps module import-safe

        return pitching_stats(year, qual=1)

    raw = fetch_or_cache(cache_dir / f"fg_p_{year}.csv", fetcher or _default)
    return _rename_strict(raw, _FG_PITCH_RENAME)


_STATCAST_XPITCH_RENAME = {
    "player_id": "mlbam",
    "woba": "woba",
    "est_woba": "xwoba",
}


def load_statcast_pitchers(
    cache_dir: Path,
    year: int,
    *,
    xstats_fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    def _x() -> pd.DataFrame:
        from pybaseball import (  # local import: keeps module import-safe
            statcast_pitcher_expected_stats,
        )

        return statcast_pitcher_expected_stats(year, minPA=1)

    return _rename_strict(
        fetch_or_cache(cache_dir / f"sc_x_p_{year}.csv", xstats_fetcher or _x),
        _STATCAST_XPITCH_RENAME,
    )


@dataclass(frozen=True)
class CoverageReport:
    matched: int  # FG rows with Statcast xStats joined
    fg_only: int  # FG rows without Statcast
    unmatched_fg: list[int]  # FG ids absent from the id map


def _f(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def build_hitter_skill_luck(
    cache_dir: Path, year: int, *, fetchers: dict[str, Callable[[], pd.DataFrame]] | None = None
) -> tuple[dict[int, SkillLuckRow], CoverageReport]:
    fetchers = fetchers or {}
    id_map = load_id_map(cache_dir, fetcher=fetchers.get("id_map"))
    fg = load_fg_hitters(cache_dir, year, fetcher=fetchers.get("fg"))
    sc = load_statcast_hitters(
        cache_dir,
        year,
        xstats_fetcher=fetchers.get("sc_x"),
        barrels_fetcher=fetchers.get("sc_brl"),
    )
    fg2m = dict(
        zip(
            id_map["key_fangraphs"].astype(int),
            id_map["key_mlbam"].astype(int),
            strict=True,
        )
    )
    sc_by_mlbam = {int(r.mlbam): r for r in sc.itertuples(index=False)}
    rows: dict[int, SkillLuckRow] = {}
    matched = fg_only = 0
    unmatched: list[int] = []
    for r in fg.itertuples(index=False):
        fgid = int(r.key_fangraphs)
        mlbam = fg2m.get(fgid)
        if mlbam is None:
            unmatched.append(fgid)
            continue
        s = sc_by_mlbam.get(mlbam)
        if s is not None:
            matched += 1
        else:
            fg_only += 1
        rows[fgid] = SkillLuckRow(
            mlbam=mlbam,
            player_type="hitter",
            pa=float(r.pa),
            ip=0.0,
            age=_f(r.age),
            barrel_pct=_f(getattr(s, "barrel_pct", None)) if s else None,
            xslg=_f(getattr(s, "xslg", None)) if s else None,
            slg=_f(getattr(s, "slg", None)) if s else None,
            xba=_f(getattr(s, "xba", None)) if s else None,
            ba=_f(getattr(s, "ba", None)) if s else None,
            babip=_f(r.babip),
            xwoba=_f(getattr(s, "xwoba", None)) if s else None,
            woba=_f(getattr(s, "woba", None)) if s else None,
            k_pct=_f(r.k_pct),
            bb_pct=_f(r.bb_pct),
        )
    return rows, CoverageReport(matched, fg_only, unmatched)


def build_pitcher_skill_luck(
    cache_dir: Path, year: int, *, fetchers: dict[str, Callable[[], pd.DataFrame]] | None = None
) -> tuple[dict[int, SkillLuckRow], CoverageReport]:
    fetchers = fetchers or {}
    id_map = load_id_map(cache_dir, fetcher=fetchers.get("id_map"))
    fg = load_fg_pitchers(cache_dir, year, fetcher=fetchers.get("fg"))
    sc = load_statcast_pitchers(cache_dir, year, xstats_fetcher=fetchers.get("sc_x"))
    fg2m = dict(
        zip(
            id_map["key_fangraphs"].astype(int),
            id_map["key_mlbam"].astype(int),
            strict=True,
        )
    )
    sc_by_mlbam = {int(r.mlbam): r for r in sc.itertuples(index=False)}
    rows: dict[int, SkillLuckRow] = {}
    matched = fg_only = 0
    unmatched: list[int] = []
    for r in fg.itertuples(index=False):
        fgid = int(r.key_fangraphs)
        mlbam = fg2m.get(fgid)
        if mlbam is None:
            unmatched.append(fgid)
            continue
        s = sc_by_mlbam.get(mlbam)
        if s is not None:
            matched += 1
        else:
            fg_only += 1
        rows[fgid] = SkillLuckRow(
            mlbam=mlbam,
            player_type="pitcher",
            pa=0.0,
            ip=float(r.ip),
            age=_f(r.age),
            barrel_pct=None,
            xslg=None,
            slg=None,
            xba=None,
            ba=None,
            babip=None,
            xwoba=_f(getattr(s, "xwoba", None)) if s else None,
            woba=_f(getattr(s, "woba", None)) if s else None,
            k_pct=_f(r.k_pct),
            bb_pct=_f(r.bb_pct),
        )
    return rows, CoverageReport(matched, fg_only, unmatched)
