"""Season-level skill/luck data layer: MLB Stats API roto lines + derived rates
and Statcast xStats, cached to data/skill_luck/, keyed by MLBAM id. Fetch-on-miss,
never overwrite a good cache with an empty/failed pull.

Data sources (both public, no auth, MLBAM-native -- no FanGraphs / Cloudflare):
- MLB Stats API (statsapi.mlb.com): season counting line (H/HR/R/RBI/SB/AVG/PA;
  W/SV/IP/K/ERA/WHIP) plus the raw components to DERIVE K%, BB%, BABIP.
- Baseball Savant (via pybaseball): expected stats (xwOBA/xBA/xSLG) + barrel%.

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.analysis.breakout import SkillLuckRow  # shared shape from Task 0

_MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
_MLB_PAGE = 1000


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


def _rename_strict(df: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise KeyError(f"source frame missing expected columns {missing}; got {list(df.columns)}")
    return df[list(rename)].rename(columns=rename)


def _f(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _sf(s: Any, field: str) -> float | None:
    """Statcast expected-stat field as float-or-None; None when the player had no
    matched Statcast row (`s is None`)."""
    return _f(getattr(s, field, None)) if s is not None else None


def _num(v) -> float:
    """Parse a numeric field from the MLB Stats API / cache, defensively.

    Rate stats arrive as STRINGS and can be a non-numeric sentinel ('.---',
    '-.--', '---') for PA>0/AB=0 players (walk-only hitters, old NL pitchers);
    blank cache cells read back as NaN. Returns 0.0 for anything unparseable
    (including NaN) instead of raising or propagating NaN.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(f) else f


def _getf(r: Any, field: str) -> float:
    """MLB-line counting/rate field as a safe float -- 0.0 when the column is absent
    or a non-numeric sentinel (see _num). Wrapping every field keeps a later-added
    column from silently reintroducing the sentinel-string crash _num guards."""
    return _num(getattr(r, field, 0))


def _parse_ip(raw) -> float:
    """MLB Stats API innings-pitched uses thirds notation ('177.2' = 177 + 2/3).
    Non-numeric sentinels (e.g. '.---') and NaN cache cells parse to 0.0 rather
    than raising.
    """
    if raw is None or pd.isna(raw):
        return 0.0
    s = str(raw)
    try:
        if "." in s:
            whole, frac = s.split(".", 1)
            return int(whole) + (int(frac[:1]) / 3.0)
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# MLB Stats API season lines (roto counting + derived rates), keyed by MLBAM.
# ---------------------------------------------------------------------------


def _mlb_fetch_season(group: str, year: int) -> pd.DataFrame:
    """Paginated season-stats leaderboard for `group` ('hitting'|'pitching'),
    one combined row per player, keyed by MLBAM. Local import keeps the module
    import-safe (no network at import time)."""
    import requests

    rows: list[dict] = []
    offset = 0
    while True:
        params: dict[str, str | int] = {
            "stats": "season",
            "group": group,
            "season": year,
            "sportId": 1,
            "playerPool": "all",
            "limit": _MLB_PAGE,
            "offset": offset,
        }
        resp = requests.get(_MLB_STATS_URL, params=params, timeout=60)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        splits = stats[0]["splits"] if stats else []
        if not splits:
            break
        for sp in splits:
            rows.append({"mlbam": sp["player"]["id"], **sp["stat"]})
        if len(splits) < _MLB_PAGE:
            break
        offset += _MLB_PAGE
    return pd.DataFrame(rows)


def load_mlb_hitters(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Per-hitter season line keyed by MLBAM: pa/ab/h/hr/r/rbi/sb/avg plus derived
    k_pct, bb_pct, babip. Cached raw; rates derived on load."""
    raw = fetch_or_cache(
        cache_dir / f"mlb_h_{year}.csv", fetcher or (lambda: _mlb_fetch_season("hitting", year))
    )
    out: list[dict] = []
    for r in raw.itertuples(index=False):
        pa = _getf(r, "plateAppearances")
        if pa <= 0:
            continue
        ab = _getf(r, "atBats")
        h = _getf(r, "hits")
        hr = _getf(r, "homeRuns")
        so = _getf(r, "strikeOuts")
        bb = _getf(r, "baseOnBalls")
        sf = _getf(r, "sacFlies")
        denom = ab - so - hr + sf
        out.append(
            {
                "mlbam": int(r.mlbam),
                "pa": pa,
                "ab": ab,
                "h": h,
                "hr": hr,
                "r": _getf(r, "runs"),
                "rbi": _getf(r, "rbi"),
                "sb": _getf(r, "stolenBases"),
                "avg": _getf(r, "avg"),
                "k_pct": so / pa,
                "bb_pct": bb / pa,
                "babip": (h - hr) / denom if denom > 0 else float("nan"),
            }
        )
    return pd.DataFrame(out)


def load_mlb_pitchers(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Per-pitcher season line keyed by MLBAM: ip/w/sv/k/era/whip plus derived
    k_pct, bb_pct (per batter faced)."""
    raw = fetch_or_cache(
        cache_dir / f"mlb_p_{year}.csv", fetcher or (lambda: _mlb_fetch_season("pitching", year))
    )
    out: list[dict] = []
    for r in raw.itertuples(index=False):
        ip = _parse_ip(getattr(r, "inningsPitched", 0))
        if ip <= 0:
            continue
        tbf = _getf(r, "battersFaced")
        so = _getf(r, "strikeOuts")
        bb = _getf(r, "baseOnBalls")
        out.append(
            {
                "mlbam": int(r.mlbam),
                "ip": ip,
                "w": _getf(r, "wins"),
                "sv": _getf(r, "saves"),
                "k": so,
                "era": _getf(r, "era"),
                "whip": _getf(r, "whip"),
                "k_pct": so / tbf if tbf > 0 else float("nan"),
                "bb_pct": bb / tbf if tbf > 0 else float("nan"),
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Statcast expected stats (Baseball Savant), keyed by MLBAM.
# ---------------------------------------------------------------------------

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
_STATCAST_XPITCH_RENAME = {"player_id": "mlbam", "woba": "woba", "est_woba": "xwoba"}


def load_statcast_hitters(
    cache_dir: Path,
    year: int,
    *,
    xstats_fetcher: Callable[[], pd.DataFrame] | None = None,
    barrels_fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    def _x() -> pd.DataFrame:
        from pybaseball import statcast_batter_expected_stats

        return statcast_batter_expected_stats(year, minPA=1)

    def _b() -> pd.DataFrame:
        from pybaseball import statcast_batter_exitvelo_barrels

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


def load_statcast_pitchers(
    cache_dir: Path,
    year: int,
    *,
    xstats_fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    def _x() -> pd.DataFrame:
        from pybaseball import statcast_pitcher_expected_stats

        return statcast_pitcher_expected_stats(year, minPA=1)

    return _rename_strict(
        fetch_or_cache(cache_dir / f"sc_x_p_{year}.csv", xstats_fetcher or _x),
        _STATCAST_XPITCH_RENAME,
    )


# ---------------------------------------------------------------------------
# Join into per-player-season SkillLuckRow, keyed by MLBAM.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    matched: int  # MLB-line players with Statcast xStats joined
    no_xstats: int  # MLB-line players without a Statcast row (small sample / pre-2015)


def _join_and_count(
    mlb: pd.DataFrame,
    sc: pd.DataFrame,
    build_row: Callable[[int, Any, Any], SkillLuckRow],
) -> tuple[dict[int, SkillLuckRow], CoverageReport]:
    """Join `mlb` season lines to `sc` Statcast rows by MLBAM, counting coverage.
    `build_row(mlbam, r, s)` builds the SkillLuckRow for one player (`s` is the
    matched Statcast namedtuple, or None when unmatched)."""
    sc_by_mlbam = {int(r.mlbam): r for r in sc.itertuples(index=False)}
    rows: dict[int, SkillLuckRow] = {}
    matched = no_xstats = 0
    for r in mlb.itertuples(index=False):
        mlbam = int(r.mlbam)
        s = sc_by_mlbam.get(mlbam)
        if s is not None:
            matched += 1
        else:
            no_xstats += 1
        rows[mlbam] = build_row(mlbam, r, s)
    return rows, CoverageReport(matched, no_xstats)


def build_hitter_skill_luck(
    cache_dir: Path, year: int, *, fetchers: dict[str, Callable[[], pd.DataFrame]] | None = None
) -> tuple[dict[int, SkillLuckRow], CoverageReport]:
    fetchers = fetchers or {}
    mlb = load_mlb_hitters(cache_dir, year, fetcher=fetchers.get("mlb"))
    sc = load_statcast_hitters(
        cache_dir,
        year,
        xstats_fetcher=fetchers.get("sc_x"),
        barrels_fetcher=fetchers.get("sc_brl"),
    )
    return _join_and_count(
        mlb,
        sc,
        lambda mlbam, r, s: SkillLuckRow(
            mlbam=mlbam,
            player_type="hitter",
            pa=float(r.pa),
            ip=0.0,
            age=None,  # age is only marcel's mild adjustment; not sourced (marcel handles None)
            barrel_pct=_sf(s, "barrel_pct"),
            xslg=_sf(s, "xslg"),
            slg=_sf(s, "slg"),
            xba=_sf(s, "xba"),
            ba=_sf(s, "ba"),
            babip=_f(r.babip),
            xwoba=_sf(s, "xwoba"),
            woba=_sf(s, "woba"),
            k_pct=_f(r.k_pct),
            bb_pct=_f(r.bb_pct),
        ),
    )


def build_pitcher_skill_luck(
    cache_dir: Path, year: int, *, fetchers: dict[str, Callable[[], pd.DataFrame]] | None = None
) -> tuple[dict[int, SkillLuckRow], CoverageReport]:
    fetchers = fetchers or {}
    mlb = load_mlb_pitchers(cache_dir, year, fetcher=fetchers.get("mlb"))
    sc = load_statcast_pitchers(cache_dir, year, xstats_fetcher=fetchers.get("sc_x"))
    return _join_and_count(
        mlb,
        sc,
        lambda mlbam, r, s: SkillLuckRow(
            mlbam=mlbam,
            player_type="pitcher",
            pa=0.0,
            ip=float(r.ip),
            age=None,
            barrel_pct=None,
            xslg=None,
            slg=None,
            xba=None,
            ba=None,
            babip=None,
            xwoba=_sf(s, "xwoba"),
            woba=_sf(s, "woba"),
            k_pct=_f(r.k_pct),
            bb_pct=_f(r.bb_pct),
        ),
    )
