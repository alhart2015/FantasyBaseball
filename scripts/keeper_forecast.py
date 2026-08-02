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

# Accented names (Sanchez, Luzardo) mangle under the cp1252 Windows default.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    HITTER_RATES,
    PITCHER_PT,
    PITCHER_RATES,
)
from fantasy_baseball.keepers.blend import parse_blend

PT_PANEL_DIR = PROJECT_ROOT / "data" / "playing_time"


def _panel_path(kind: str) -> Path | None:
    """Newest `{kind}_pt_panel_{start}_{end}.csv`, or None if none is built.

    Globbed rather than hardcoded: `build_pt_panel.py` names its output from
    --start/--end with --end defaulting to the CURRENT year, so a pinned
    `_2010_2026.csv` would be orphaned by the next rebuild and the board would
    silently fall back to the one-year gap volume model this curve replaced.
    """
    matches = sorted(PT_PANEL_DIR.glob(f"{kind}_pt_panel_*.csv"))
    return matches[-1] if matches else None


from fantasy_baseball.keepers.persistence import (
    Share,
    centered_aging,
    fit_share,
    fold_forecast,
    gap,
)
from fantasy_baseball.keepers.playing_time import (
    FEATURES,
    build_features,
    fit_curve,
    lag_panel,
    per_appearance,
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


# Volume floor for FITTING the curve. Deliberately a constant and deliberately NOT
# --min-pa/--min-ip: those select which players the board DISPLAYS, and the training
# population must not move with them. A review flagged the two as "desynced" and
# threading the display floor through here was measured to be actively harmful --
# at a 650 PA floor the `role` coefficient inverts to -58.6 and at 700 `vol1` goes
# negative, i.e. more playing time last year predicts less next year. Raising the
# floor also strips out the injury-shortened seasons `shortfall` and `role` exist to
# handle. The separation is the design; do not couple them again.
TRAIN_FLOOR = {"hitter": 300.0, "pitcher": 50.0}


def volume_forecast(
    kind: str, target_year: int, observed: pd.Series, include_exits: bool = True
) -> pd.Series | None:
    """Projected `target_year` PA (hitters) or IP (pitchers) from career history.

    Supersedes the one-year gap term for both pools. `observed` is the CURRENT season's
    full-season blend, used as the first lag; the older lags, age and role come from the
    2010-present panel. Returns None when the panel is absent, so the caller falls back
    to the gap model rather than failing.

    For a two-years-out target the curve is applied twice, its own output feeding back
    as the first lag. That is an extrapolation: the curve was fit one year ahead, and
    iterating it compounds its error.
    """
    path = _panel_path(kind)
    if path is None:
        print(f"  WARNING: no {kind} playing-time panel; falling back to the gap model")
        return None
    print(f"  {kind} playing-time panel: {path.name}")
    panel = pd.read_csv(path)
    rows = lag_panel(panel, kind, min_recent=TRAIN_FLOOR[kind], include_exits=include_exits)
    curve = fit_curve(rows[list(FEATURES[kind])], rows["target"], kind)
    latest = int(panel.loc[~panel["partial_season"].astype(bool), "season"].max())
    volume = "pa" if kind == "hitter" else "ip"

    def series_for(year: int, column: str) -> pd.Series:
        sub = panel.loc[panel["season"] == year].set_index("mlbam_id")[column]
        return sub.loc[~sub.index.duplicated()].reindex(observed.index)

    vol2 = series_for(BASE_YEAR - 1, volume)
    vol3 = series_for(BASE_YEAR - 2, volume)
    # Age comes from the BASE year, not the last COMPLETED one. Sourcing it from
    # `latest` gave NaN to every player whose first season is the base year -- 107 of
    # them in the 2026 panel, 10 already past 300 PA -- and that NaN propagated through
    # the whole 5x5 line, silently dropping exactly the young debutants a keeper league
    # values most. The base-year row exists for anyone the blend covers; `latest` stays
    # as a fallback for a player the base season somehow missed.
    age = (series_for(BASE_YEAR, "age") + (target_year - BASE_YEAR)).fillna(
        series_for(latest, "age") + (target_year - latest)
    )
    # Role comes from the base season's PARTIAL panel row, not the blend. The blend
    # carries full-season volume but its `g` field is REST-OF-SEASON games, so volume/g
    # off the blend is nonsense (600+ PA over 46 games). A per-appearance rate is
    # readable off two thirds of a season anyway.
    base_vol = series_for(BASE_YEAR, volume).fillna(0.0)
    base_app = series_for(BASE_YEAR, "games").fillna(0.0)
    role = per_appearance(base_vol, base_app, kind)
    start_share = None
    if kind == "pitcher":
        starts = series_for(BASE_YEAR, "starts").fillna(0.0)
        start_share = (starts / base_app.where(base_app > 0)).fillna(0.0)

    # One application of the curve per year from the base season to the target. `age` is
    # advanced to the TARGET year, so each step walks it back to the season it actually
    # projects: for a 2028 target, step 0 projects 2027. Role and start share carry
    # forward unchanged -- a batting-order slot or a rotation job is far stickier than a
    # workload, and projecting a change in either would be inventing information.
    vol1, projected = observed, None
    steps = target_year - BASE_YEAR
    for step in range(steps):
        built = build_features(vol1, vol2, vol3, age - (steps - step - 1), role, kind, start_share)
        projected = curve.predict(built)
        vol1, vol2, vol3 = projected, vol1, vol2
    return projected


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
    # Volume comes from the multi-year career curve for BOTH pools now, not the
    # one-year gap term. Falls back to the gap model if a panel is missing.
    curve_volume = volume_forecast(
        kind, target_year, observed.loc[idx, pt], not getattr(args, "no_exit_rows", False)
    )
    for col in columns:
        g = gap(observed.loc[idx, col], base.loc[idx, col])
        aging = None
        if not args.no_aging:
            aging = centered_aging(
                out[col].reindex(idx), base.loc[idx, col], weights=observed.loc[idx, pt]
            )
        folded = fold_forecast(base.loc[idx, col], g, shares[col], aging)
        if col == pt and curve_volume is not None:
            # Per-player fallback, not all-or-nothing: a player the curve cannot score
            # (missing lags, missing age) must not land a NaN in the counting line and
            # vanish from the board without a word.
            result[col] = curve_volume.reindex(idx).fillna(folded)
        else:
            result[col] = folded
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
                # Carried so the AVG term is scaled by the at-bats THIS player is
                # forecast to take. ab_pa spans 0.80-0.96 across the pool, so a league
                # constant misprices high-walk and high-contact bats in opposite
                # directions by more than adjacent keeper candidates are separated by.
                "AB": frame["ab_pa"] * pa,
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
    parser.add_argument(
        "--no-exit-rows",
        action="store_true",
        help="ablate the survivorship correction: do not train on career endings",
    )
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
