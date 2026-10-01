"""Calibrate the IN-SEASON playing-time curves used by the ROS Monte Carlo.

``calibrate_playing_time.py`` fits how FULL-SEASON playing time deviates from a
preseason projection. Applied to a rest-of-season window, that shape is wrong: over
a short window a player is mostly either healthy (about his projection) or out for
the rest of it, so real outcomes pile up near zero and near one. The full-season
shape puts too much mass in the middle and too little at zero. Measured on 2026
active-slot players (issue #393): the MC simulated 22% of pitchers losing half their
remaining time vs 14% real, and 2% losing it all vs 5% real.

This script measures the real in-season distribution and emits
``ROS_PLAYING_TIME_QUANTILES`` for ``utils/constants.py``.

Method:
  - Seasons 2023-2025, MLB per-game logs (synced with the app's own box-score sync
    into ``data/cache/pt_logs-{year}.db``; a no-op when already complete).
  - Checkpoints with 75/60/45/30/15% of the season left. Season start is the first
    day with a full slate, which skips the Seoul/Tokyo openers.
  - Population mirrors "on an active fantasy slot": a real workload (full-season
    pace below) AND appeared in a game in the 10 days (15 for starters) before the
    checkpoint -- healthy and in role.
  - Expected rest-of-season volume = 50/50 blend of preseason pace (steamer+zips
    mean) and year-to-date pace, times the share of season left. Historical ROS
    projections were not saved; this blend matched the 2026 real ROS projections'
    low tail best (under 50% of expected: 8% vs 9% hitters, 15% vs 14% pitchers).
  - Roles: hitters, SP, RP (``role_from_ip`` on the blended full-season IP).
    Floors: 450 PA, 120 IP SP, 50 IP RP.
  - Quantiles of actual / expected at ``ROS_PT_LEVELS``, each horizon pooled with
    its neighbors (half weight) to steady the extreme quantiles.

Usage:
    python scripts/calibrate_ros_playing_time.py            # sync if needed, fit, print
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.utils.constants import ROS_PT_LEVELS, role_from_ip

YEARS = (2023, 2024, 2025)
HORIZONS = (0.75, 0.60, 0.45, 0.30, 0.15)
CACHE = PROJECT_ROOT / "data" / "cache"
FLOORS = {"H": 450.0, "SP": 120.0, "RP": 50.0}
LOOKBACK_DAYS = {"H": 10, "SP": 15, "RP": 10}


def _sync(year: int) -> Path:
    """Sync one season's per-game logs into its own scratch store (idempotent)."""
    db = CACHE / f"pt_logs-{year}.db"
    os.environ["FANTASY_LOCAL_KV_PATH"] = str(db)
    os.environ.pop("RENDER", None)
    import importlib

    from fantasy_baseball.data import kv_store

    importlib.reload(kv_store)
    from fantasy_baseball.data.mlb_game_logs import sync_game_logs

    sync_game_logs(kv_store.get_kv(), year, progress_cb=lambda m: print(f"  {year}: {m}"))
    return db


def _logs(db: Path, year: int) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {"hitting": {}, "pitching": {}}
    con = sqlite3.connect(db)
    for key, val in con.execute(
        "select key, value from kv where key like ?", (f"game_logs:{year}:%:%",)
    ):
        parts = key.split(":")
        if len(parts) == 4 and parts[3] in out:
            out[parts[3]][parts[2]] = json.loads(val).get("games", [])
    return out


def _preseason(year: int) -> dict[str, dict[str, float]]:
    base = PROJECT_ROOT / "data" / "projections" / str(year)
    out: dict[str, dict[str, float]] = {}
    for kind, vol in (("hitters", "PA"), ("pitchers", "IP")):
        frames = []
        for system in ("steamer", "zips"):
            path = next(
                p
                for p in (base / f"{system}-{kind}.csv", base / f"{system}-{kind}-{year}.csv")
                if p.exists()
            )
            d = pd.read_csv(
                path, encoding="utf-8-sig", usecols=lambda c, v=vol: c in ("MLBAMID", v)
            )
            frames.append(d.dropna(subset=["MLBAMID"]))
        m = pd.concat(frames).groupby("MLBAMID")[vol].mean()
        out[kind] = {str(int(k)): float(v) for k, v in m.items()}
    return out


def build_windows() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        logs = _logs(_sync(year), year)
        per_day: dict[str, int] = {}
        for games in logs["hitting"].values():
            for g in games:
                per_day[g["date"]] = per_day.get(g["date"], 0) + 1
        start = date.fromisoformat(min(d for d, n in per_day.items() if n >= 150))
        end = date.fromisoformat(max(per_day))
        span = (end - start).days
        pre = _preseason(year)
        for f in HORIZONS:
            d = end - timedelta(days=round(f * span))
            ds, elapsed = d.isoformat(), (d - start).days / span
            for group, kind, vkey in (("hitting", "hitters", "pa"), ("pitching", "pitchers", "ip")):
                for pid, pre_vol in pre[kind].items():
                    games = logs[group].get(pid, [])
                    ytd = sum(g.get(vkey, 0) for g in games if g["date"] < ds)
                    full = 0.5 * pre_vol + 0.5 * (ytd / elapsed)
                    role = "H" if group == "hitting" else role_from_ip(full)
                    if full < FLOORS[role]:
                        continue
                    since = (d - timedelta(days=LOOKBACK_DAYS[role])).isoformat()
                    if not any(since <= g["date"] < ds for g in games):
                        continue
                    expected = full * f
                    actual = sum(g.get(vkey, 0) for g in games if g["date"] >= ds)
                    rows.append(dict(year=year, f=f, role=role, ratio=actual / expected))
        print(f"{year}: {start} -> {end}, {len(rows)} windows so far")
    return pd.DataFrame(rows)


def fit(windows: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    table: dict[str, list[dict[str, object]]] = {}
    for role, key in (("H", "hitters"), ("SP", "SP"), ("RP", "RP")):
        sub = windows[windows.role == role]
        points = []
        for i, f in enumerate(HORIZONS):
            parts = [(sub[sub.f == f].ratio.to_numpy(), 1.0)]
            for j in (i - 1, i + 1):
                if 0 <= j < len(HORIZONS):
                    parts.append((sub[sub.f == HORIZONS[j]].ratio.to_numpy(), 0.5))
            vals = np.concatenate([p for p, _ in parts])
            wts = np.concatenate([np.full(len(p), w) for p, w in parts])
            order = np.argsort(vals)
            cum = np.cumsum(wts[order]) / wts.sum()
            q = [round(float(np.interp(lev, cum, vals[order])), 3) for lev in ROS_PT_LEVELS]
            points.append({"f": f, "n": len(sub[sub.f == f]), "q": q})
        table[key] = sorted(points, key=lambda p: p["f"])
    return table


def main() -> None:
    windows = build_windows()
    table = fit(windows)
    print("\nROS_PLAYING_TIME_QUANTILES = {")
    for key, points in table.items():
        print(f'    "{key}": [')
        for p in points:
            print(f'        {{"f": {p["f"]}, "q": {p["q"]}}},  # n={p["n"]}')
        print("    ],")
    print("}")


if __name__ == "__main__":
    main()
