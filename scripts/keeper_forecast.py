"""Forecast a player's 2027 (and 2028) 5x5 line, folding 2026 into the out-year ZiPS.

    forecast = zips_out_year_baseline_effect
             = zips_2026 + drift + centered_aging + S * (blend_2026 - zips_2026)

Every term is measured, not chosen:

* `zips_2026` is the baseline. It is the SAME 2026-03-25 model run as the 2027/2028
  out-years, so the gap below is exactly "what ZiPS did not know", with no vintage
  mismatch smuggled in.
* `blend_2026` is the live full-season blend (actual-to-date + rest-of-season) from
  the KV store -- a season two-thirds finished made comparable to a full-season
  projection.
* `S` and `drift` come from `scripts/keeper_persistence.py`, fit on 2022->23,
  2023->24 and 2024->25.
* `centered_aging` is `zips_2027 - zips_2026`, DEMEANED. See
  `keepers.persistence.centered_aging` for why only the spread is used.

Reads live data from Upstash (read-only). Off Render, `get_kv()` would return the
possibly-stale local SQLite, so this sets RENDER=true first, as `refresh_remote.py`
does.

Usage:
    python scripts/keeper_forecast.py                     # 2027, both pools
    python scripts/keeper_forecast.py --year 2028
    python scripts/keeper_forecast.py --no-aging          # ablate the out-year term
    python scripts/keeper_forecast.py --players "Juan Soto,Bobby Witt Jr."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    HITTER_RATES,
    PITCHER_PT,
    PITCHER_RATES,
)
from fantasy_baseball.keepers.blend import parse_blend
from fantasy_baseball.keepers.persistence import (
    Share,
    centered_aging,
    fit_share,
    fold_forecast,
    gap,
)
from fantasy_baseball.keepers.vintages import load_vintage

PROJECTIONS = PROJECT_ROOT / "data" / "projections"
BASE_YEAR = 2026

POOLS = {
    "hitter": {"pt": HITTER_PT, "rates": HITTER_RATES, "volume": "PA"},
    "pitcher": {"pt": PITCHER_PT, "rates": PITCHER_RATES, "volume": "IP"},
}


def fetch_blend() -> dict:
    """Live full-season blend from Upstash. Read-only."""
    os.environ["RENDER"] = "true"
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv

    raw = build_explicit_upstash_kv().get(f"cache:{CacheKey.FULL_SEASON_PROJECTIONS.value}")
    if raw is None:
        raise RuntimeError("no full_season_projections blob in Upstash; run the ROS refresh first")
    blob = json.loads(raw) if isinstance(raw, str) else raw
    # Prod wraps every blob in a `_meta`/`_data` envelope; a local store may not.
    payload = blob.get("_data", blob)
    meta = blob.get("_meta", {})
    print(
        f"  blend snapshot: {meta.get('_ros_snapshot_date', '?')} "
        f"(written {str(meta.get('_written_at', '?'))[:10]})"
    )
    return payload


def _dedupe(frame: pd.DataFrame, pt: str) -> pd.DataFrame:
    ordered = frame.sort_values(pt, ascending=False)
    return ordered.loc[~ordered.index.duplicated(keep="first")]


def load_shares(kind: str, args: argparse.Namespace) -> dict[str, Share]:
    """Refit S and the drift here rather than pasting constants.

    `keeper_persistence` owns the fit; importing its loaders keeps ONE definition of
    the transition panels, so the shares applied below cannot drift from the shares
    that were validated. Slower than a cached table by a couple of seconds, and worth
    it -- a stale constant is the exact failure the previous model died of.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from keeper_persistence import (
        TRANSITIONS,
        _fit_column,
        _pooled,
        build_transition,
        build_volume_transition,
    )

    pool = POOLS[kind]
    pt = str(pool["pt"])
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip

    rates = _pooled(
        [build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0] for y, _ in TRANSITIONS]
    )
    volume = _pooled([build_volume_transition(y, kind, min_pt=min_pt) for y, _ in TRANSITIONS])
    shares = {pt: _fit_column(volume, pt, pt)}
    shares.update({col: _fit_column(rates, col, pt) for col in pool["rates"]})
    return shares


def forecast_pool(
    kind: str, target_year: int, payload: dict, args: argparse.Namespace
) -> pd.DataFrame:
    pool = POOLS[kind]
    pt = str(pool["pt"])
    columns = (pt, *pool["rates"])

    base = _dedupe(load_vintage(BASE_YEAR, PROJECTIONS, kind), pt)
    out = _dedupe(load_vintage(target_year, PROJECTIONS, kind), pt)
    observed = parse_blend(payload, kind)
    shares = load_shares(kind, args)

    # Everyone the live blend says is a real 2026 contributor and whom the baseline
    # covers. The floor keeps the fringe/minor-league bulk of the blob out.
    floor = args.min_pa if kind == "hitter" else args.min_ip
    idx = observed.index[observed[pt] >= floor].intersection(base.index)

    result = pd.DataFrame(index=idx)
    for col in columns:
        g = gap(observed.loc[idx, col], base.loc[idx, col])
        aging = None
        if not args.no_aging:
            aging = centered_aging(
                out[col].reindex(idx), base.loc[idx, col], weights=observed.loc[idx, pt]
            )
        result[col] = fold_forecast(base.loc[idx, col], g, shares[col], aging)
        result[f"{col}_gap"] = g
    # A rate cannot go negative, and neither can volume. The linear fold can push a
    # near-zero rate under zero for a player whose 2026 collapsed; clip rather than
    # emit a negative HR rate that would silently subtract from a keeper's line.
    for col in columns:
        result[col] = result[col].clip(lower=0.0)
    return result


def to_counting(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reconstruct the scored 5x5 categories from the forecast rate/volume pair."""
    if kind == "hitter":
        pa = frame[HITTER_PT]
        return pd.DataFrame(
            {
                "PA": pa,
                "R": frame["r_pa"] * pa,
                "HR": frame["hr_pa"] * pa,
                "RBI": frame["rbi_pa"] * pa,
                "SB": frame["sb_pa"] * pa,
                "AVG": frame["h_ab"],
            },
            index=frame.index,
        )
    ip = frame[PITCHER_PT]
    return pd.DataFrame(
        {
            "IP": ip,
            "W": frame["w_ip"] * ip,
            "SV": frame["sv_ip"] * ip,
            "K": frame["k_ip"] * ip,
            "ERA": frame["er_ip"] * 9.0,
            "WHIP": frame["bb_ip"] + frame["h_ip"],
        },
        index=frame.index,
    )


def _names(payload: dict, kind: str) -> pd.Series:
    key = "hitters" if kind == "hitter" else "pitchers"
    recs = payload[key]
    frame = pd.DataFrame.from_records(recs)[
        ["mlbam_id", "name", "pa" if kind == "hitter" else "ip"]
    ]
    frame = frame.dropna(subset=["mlbam_id"])
    vol = "pa" if kind == "hitter" else "ip"
    frame = frame.sort_values(vol, ascending=False)
    frame["mlbam_id"] = frame["mlbam_id"].astype(int)
    return frame.drop_duplicates("mlbam_id").set_index("mlbam_id")["name"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2027, choices=(2027, 2028))
    parser.add_argument("--min-pa", type=float, default=300)
    parser.add_argument("--min-ip", type=float, default=50)
    # Must match keeper_persistence: these define the rate-fit sample. See its main().
    parser.add_argument("--min-next-pa", type=float, default=250)
    parser.add_argument("--min-next-ip", type=float, default=50)
    parser.add_argument("--no-aging", action="store_true", help="ablate the out-year term")
    parser.add_argument("--players", help="comma-separated names to show")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--pool", choices=("hitter", "pitcher"))
    args = parser.parse_args()

    print(f"Fetching the live {BASE_YEAR} full-season blend...")
    payload = fetch_blend()

    wanted = {n.strip().lower() for n in args.players.split(",")} if args.players else None
    for kind in [args.pool] if args.pool else ["hitter", "pitcher"]:
        frame = forecast_pool(kind, args.year, payload, args)
        counting = to_counting(frame, kind)
        counting.insert(0, "name", _names(payload, kind).reindex(counting.index))

        volume = str(POOLS[kind]["volume"])
        sort_col = "R" if kind == "hitter" else "K"
        view = counting.sort_values(sort_col, ascending=False)
        if wanted:
            view = view.loc[view["name"].str.lower().isin(wanted)]
        else:
            view = view.head(args.top)

        label = "no aging term" if args.no_aging else "with out-year aging"
        print(f"\n{'=' * 82}")
        print(f"{kind.upper()}S -- {args.year} forecast ({label}), {len(counting)} players")
        print(f"{'=' * 82}")
        cols = [c for c in view.columns if c != "name"]
        print(f"{'name':<24}" + "".join(f"{c:>9}" for c in cols))
        print("-" * 82)
        for _, row in view.iterrows():
            name = str(row["name"])[:23]
            cells = "".join(
                f"{row[c]:>9.3f}" if c in {"AVG", "WHIP", "ERA"} else f"{row[c]:>9.1f}"
                for c in cols
            )
            print(f"{name:<24}{cells}")
        print(
            f"  ({volume} floor {args.min_pa if kind == 'hitter' else args.min_ip:g} on the 2026 blend)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
