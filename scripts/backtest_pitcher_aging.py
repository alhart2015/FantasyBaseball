"""Score the pitcher age corrections (pitcher_ros.aging, #503) on held-out seasons.

Rows: pitchers with 50+ IP the season before. Our rates: --rates-run's week-0
projections (each season predicted by a net trained only on earlier seasons). PT and
role: Steamer's preseason file (projected IP and GS / G). Actuals: the MLB pitching
leaderboards. For each season, the corrections are fit on the other seasons (their
knots chosen leave-one-season-out within those), then applied to it.

Prints, per role and stat, the held-out weighted squared error with and without the
correction (all pitchers, and 36+) with how sure the corrected one is better (0-100,
bootstrap over pitchers), and the mean miss by age group after the correction.

    python scripts/backtest_pitcher_aging.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from backtest_hitter_ranges import CACHE, PEOPLE, pt_file

from fantasy_baseball.pitcher_ros import aging
from fantasy_baseball.pt_model import range_inputs as ri

RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
# Rows: pitchers with this many IP the season before (known before the season).
MIN_PRIOR_IP = 50.0
OLD = 36.0
AGE_BINS = (0.0, 26.99, 29.99, 32.99, 35.99, 99.0)
AGE_LABELS = ("<=26", "27-29", "30-32", "33-35", "36+")
BOOTSTRAPS = 2000


def season_age(birth: pd.Series, season: int) -> pd.Series:
    """Age on June 30 of ``season``."""
    return (pd.Timestamp(f"{season}-06-30") - birth).dt.days / 365.25


def aging_rows(seasons: list[int], run: str) -> pd.DataFrame:
    """One row per pitcher-season (shared with scripts/keeper_report.py): season, age,
    role, our rates (p_*), Steamer's projected IP and the actuals (a_*; 0 if he didn't
    pitch)."""
    pitching = ri.load_pitching_seasons(CACHE, {s - 1 for s in seasons} | set(seasons))
    rates = ri.load_pitcher_rates(RUNS / run)
    people = pd.read_csv(PEOPLE, usecols=["id", "birthDate"]).set_index("id")
    birth = pd.to_datetime(people["birthDate"])
    parts = []
    for season in seasons:
        prev = pitching[season - 1]
        ids = prev.index[prev["ip"] >= MIN_PRIOR_IP]
        pt = ri.load_pitcher_pt(pt_file("steamer", season, "pitchers")).reindex(ids)
        d = pd.DataFrame(index=ids)
        d["season"] = season
        d["age"] = season_age(birth.reindex(ids), season).to_numpy()
        d["role"] = aging.role(pt["gs_share"])
        d["proj_ip"] = pt["ip"]
        actual = pitching[season].reindex(ids).fillna(0.0)
        for c in ("ip", "bf", "k", "w", "er", "h", "bb"):
            d[f"a_{c}"] = actual[c]
        own = (
            rates.xs(season, level="season").reindex(ids)
            if season in rates.index.get_level_values("season")
            else None
        )
        for c in ("k", "w", "era", "whip"):
            d[f"p_{c}"] = own[f"p_{c}"] if own is not None else np.nan
        parts.append(d.reset_index(names="player_id"))
    return pd.concat(parts, ignore_index=True)


def sure_better(
    err_new: np.ndarray, err_old: np.ndarray, players: np.ndarray, rng: np.random.Generator
) -> float:
    gain = pd.Series(err_old - err_new).groupby(players).sum().to_numpy()
    draws = rng.integers(0, len(gain), (BOOTSTRAPS, len(gain)))
    return float((gain[draws].sum(axis=1) > 0).mean() * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--rates-run", default="035-saves-net")
    args = parser.parse_args()
    seasons = sorted(args.seasons)
    rows = aging_rows(seasons, args.rates_run)
    rows = rows[rows["age"].notna() & rows["role"].notna()].reset_index(drop=True)
    print(f"rows: {len(rows)} pitcher-seasons, 36+: {int((rows['age'] >= OLD).sum())}")

    # Held out: each season corrected by fits on the others.
    adj_rates, adj_ip = [], []
    for s in seasons:
        train, test = rows[rows["season"] != s], rows[rows["season"] == s]
        rfit, ifit = aging.fit_rates(train), aging.fit_innings(train)
        r = test[["p_k", "p_w", "p_era", "p_whip"]].rename(columns=lambda c: c[2:])
        adj = aging.adjust_rates(r, test["age"], test["role"], rfit)
        adj_rates.append(adj.add_prefix("q_"))
        adj_ip.append(
            aging.adjust_innings(test["proj_ip"], test["age"], test["role"], ifit).rename("q_ip")
        )
    held = rows.join(pd.concat(adj_rates)).join(pd.concat(adj_ip))
    rng = np.random.default_rng(0)
    old = (held["age"] >= OLD).to_numpy()

    print("\nheld-out weighted squared error: corrected vs ours (lower = better), how sure better")
    bf, ip = held["a_bf"].to_numpy(dtype=float), held["a_ip"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = {
            "k": held["a_k"] / bf,
            "w": held["a_w"] / bf,
            "era": 9 * held["a_er"] / ip,
            "whip": (held["a_h"] + held["a_bb"]) / ip,
        }
    for r in aging.ROLES:
        in_role = (held["role"] == r).to_numpy()
        for stat, weight in aging.RATE_WEIGHT.items():
            w = held[weight].to_numpy(dtype=float)
            a = actual[stat].to_numpy(dtype=float)
            ok = in_role & np.isfinite(a) & (w > 0) & held[f"p_{stat}"].notna().to_numpy()
            e_old = w * (a - held[f"p_{stat}"].to_numpy(dtype=float)) ** 2
            e_new = w * (a - held[f"q_{stat}"].to_numpy(dtype=float)) ** 2
            parts = []
            for label, sel in (("all", ok), ("36+", ok & old)):
                change = e_new[sel].sum() / e_old[sel].sum() - 1
                sure = sure_better(e_new[sel], e_old[sel], held["player_id"].to_numpy()[sel], rng)
                parts.append(f"{label} {change:+.2%} ({sure:.0f})")
            print(f"  {r:8s} {stat:4s} " + " | ".join(parts))
        sel = in_role & (held["proj_ip"].to_numpy(dtype=float) >= aging.MIN_PROJECTED_IP)
        e_old = (ip - held["proj_ip"].to_numpy(dtype=float)) ** 2
        e_new = (ip - held["q_ip"].to_numpy(dtype=float)) ** 2
        parts = []
        for label, s in (("all", sel), ("36+", sel & old)):
            change = e_new[s].sum() / e_old[s].sum() - 1
            sure = sure_better(e_new[s], e_old[s], held["player_id"].to_numpy()[s], rng)
            parts.append(f"{label} {change:+.2%} ({sure:.0f})")
        print(f"  {r:8s} IP   " + " | ".join(parts))

    held["grp"] = pd.cut(held["age"], list(AGE_BINS), labels=list(AGE_LABELS))
    print("\nmean miss by age, actual minus projection: ours -> corrected (K per 180 IP)")
    for r in aging.ROLES:
        g = held[held["role"] == r]
        lines = []
        for label, sub in g.groupby("grp", observed=True):
            bfs, ips = sub["a_bf"].to_numpy(dtype=float), sub["a_ip"].to_numpy(dtype=float)
            ok_bf, ok_ip = bfs > 0, ips > 0
            k_old = np.average((sub["a_k"] / bfs - sub["p_k"])[ok_bf], weights=bfs[ok_bf]) * 756
            k_new = np.average((sub["a_k"] / bfs - sub["q_k"])[ok_bf], weights=bfs[ok_bf]) * 756
            e_old = np.average((9 * sub["a_er"] / ips - sub["p_era"])[ok_ip], weights=ips[ok_ip])
            e_new = np.average((9 * sub["a_er"] / ips - sub["q_era"])[ok_ip], weights=ips[ok_ip])
            pj = sub["proj_ip"] >= aging.MIN_PROJECTED_IP
            i_old = (sub["a_ip"][pj] / sub["proj_ip"][pj]).clip(upper=aging.RATIO_CAP).mean()
            i_new = (sub["a_ip"][pj] / sub["q_ip"][pj]).clip(upper=aging.RATIO_CAP).mean()
            lines.append(
                f"    {label:6s} n={len(sub):4d}  K {k_old:+.1f} -> {k_new:+.1f}  ERA {e_old:+.2f} -> {e_new:+.2f}"
                f"  IP share {i_old:.2f} -> {i_new:.2f}"
            )
        print(f"  {r}s")
        print("\n".join(lines))

    print("\nfit on every season (what a new season uses):")
    print(
        "\n".join(
            "  " + x
            for x in aging.describe(
                aging.fit_rates(rows), [(r, s) for r in aging.ROLES for s in aging.RATE_WEIGHT]
            )
        )
    )
    print("\n".join("  " + x for x in aging.describe(aging.fit_innings(rows), aging.ROLES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
