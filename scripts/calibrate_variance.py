"""Calibrate Monte Carlo stat variance from historical projections vs actuals.

Compares Steamer + ZiPS blended projections against actual MLB stats for
2022-2024 to compute empirical variance per stat category. Also produces
a correlation matrix for future covariance-based simulation.

Usage:
    python scripts/calibrate_variance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.utils.name_utils import normalize_name

PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"

YEARS = [2022, 2023, 2024]

# Minimum thresholds: only include players with enough playing time
# in BOTH projections and actuals to be meaningful comps
MIN_PROJ_PA = 200
MIN_ACTUAL_PA = 200
MIN_PROJ_IP = 40
MIN_ACTUAL_IP = 40

HITTER_STATS = ["R", "HR", "RBI", "SB"]
PITCHER_COUNTING = ["W", "SO", "SV"]
# Rate stats computed from components
PITCHER_RATE_COMPONENTS = ["ER", "BB", "H"]  # for ERA and WHIP

# Minimum projected values to include in ratio calculations —
# avoids extreme ratios when projected value is near zero
MIN_PROJ_STAT = {
    "R": 30, "HR": 8, "RBI": 30, "SB": 5,
    "W": 3, "SO": 40, "SV": 5, "IP": 40,
    "ER": 15, "BB": 15, "H": 40,
}

# Only include players who played 60-140% of projected PA/IP —
# since the injury model handles playing time separately, we want
# performance-only variance from players who roughly played as expected
PA_RATIO_RANGE = (0.60, 1.40)
IP_RATIO_RANGE = (0.60, 1.40)


def load_projection(proj_dir: Path, system: str, player_type: str) -> pd.DataFrame:
    """Load a single projection CSV and normalize columns."""
    patterns = [
        proj_dir / f"{system}-{player_type}.csv",
        proj_dir / f"{system}_{player_type}.csv",
    ]
    # Also try year-suffixed
    for f in sorted(proj_dir.glob(f"{system}-{player_type}*.csv")):
        patterns.append(f)

    path = None
    for p in patterns:
        if p.exists():
            path = p
            break
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    # Standardize column names
    rename = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc == "so":
            rename[col] = "SO"
        elif lc == "mlbamid":
            rename[col] = "MLBAMID"
    df = df.rename(columns=rename)
    return df


def blend_projections(proj_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and blend steamer + zips 50/50 for a given year directory."""
    systems = ["steamer", "zips"]

    all_hitters = []
    all_pitchers = []
    for sys_name in systems:
        h = load_projection(proj_dir, sys_name, "hitters")
        p = load_projection(proj_dir, sys_name, "pitchers")
        if not h.empty:
            all_hitters.append(h)
        if not p.empty:
            all_pitchers.append(p)

    if not all_hitters:
        return pd.DataFrame(), pd.DataFrame()

    def blend(dfs, id_col="MLBAMID"):
        combined = pd.concat(dfs, ignore_index=True)
        if id_col not in combined.columns:
            return pd.DataFrame()
        combined = combined.dropna(subset=[id_col])
        combined[id_col] = combined[id_col].astype(int)
        # Average numeric columns per player
        numeric = combined.select_dtypes(include=[np.number])
        numeric[id_col] = combined[id_col]
        numeric["Name"] = combined["Name"]
        result = numeric.groupby(id_col).mean(numeric_only=True).reset_index()
        # Keep first name
        names = combined.groupby(id_col)["Name"].first().reset_index()
        result = result.merge(names, on=id_col, how="left", suffixes=("_drop", ""))
        if "Name_drop" in result.columns:
            result = result.drop(columns=["Name_drop"])
        return result

    blended_h = blend(all_hitters) if all_hitters else pd.DataFrame()
    blended_p = blend(all_pitchers) if all_pitchers else pd.DataFrame()
    return blended_h, blended_p


def load_actuals(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load actual stats for a year."""
    h_path = STATS_DIR / f"hitters-{year}.csv"
    p_path = STATS_DIR / f"pitchers-{year}.csv"

    h = pd.read_csv(h_path, encoding="utf-8-sig") if h_path.exists() else pd.DataFrame()
    p = pd.read_csv(p_path, encoding="utf-8-sig") if p_path.exists() else pd.DataFrame()
    return h, p


def compute_hitter_residuals(proj: pd.DataFrame, actual: pd.DataFrame, year: int) -> pd.DataFrame:
    """Match projected vs actual hitters and compute per-stat residuals.

    Residual = actual/projected - 1 (so 0 = perfect prediction).
    """
    if proj.empty or actual.empty:
        return pd.DataFrame()

    # Match on MLBAMID
    proj = proj.copy()
    actual = actual.copy()
    proj["MLBAMID"] = proj["MLBAMID"].astype(int)
    actual["MLBAMID"] = actual["MLBAMID"].astype(int)

    merged = proj.merge(actual, on="MLBAMID", suffixes=("_proj", "_act"))

    # Filter: meaningful playing time in both
    merged = merged[merged["PA_proj"] >= MIN_PROJ_PA]
    merged = merged[merged["PA_act"] >= MIN_ACTUAL_PA]

    # Filter: only players who played roughly as projected (60-140% of PA)
    # This isolates performance variance from playing time variance,
    # since the injury model handles playing time separately
    pa_ratio = merged["PA_act"] / merged["PA_proj"]
    merged = merged[(pa_ratio >= PA_RATIO_RANGE[0]) & (pa_ratio <= PA_RATIO_RANGE[1])]

    rows = []
    for _, row in merged.iterrows():
        name = row.get("Name_proj", row.get("Name_act", "?"))
        entry = {"year": year, "name": name, "MLBAMID": row["MLBAMID"],
                 "PA_proj": row["PA_proj"], "PA_act": row["PA_act"]}

        for stat in HITTER_STATS:
            proj_val = float(row.get(f"{stat}_proj", row.get(stat, 0)) or 0)
            act_val = float(row.get(f"{stat}_act", 0) or 0)
            min_proj = MIN_PROJ_STAT.get(stat, 1)
            if proj_val >= min_proj:
                entry[f"{stat}_ratio"] = act_val / proj_val
                entry[f"{stat}_residual"] = act_val / proj_val - 1.0
            else:
                entry[f"{stat}_ratio"] = np.nan
                entry[f"{stat}_residual"] = np.nan

        # AVG: use H/AB ratio
        ab_proj = float(row.get("AB_proj", row.get("AB", 0)) or 0)
        h_proj = float(row.get("H_proj", row.get("H", 0)) or 0)
        ab_act = float(row.get("AB_act", 0) or 0)
        h_act = float(row.get("H_act", 0) or 0)
        avg_proj = h_proj / ab_proj if ab_proj > 0 else 0
        avg_act = h_act / ab_act if ab_act > 0 else 0
        if avg_proj > 0:
            entry["AVG_ratio"] = avg_act / avg_proj
            entry["AVG_residual"] = avg_act / avg_proj - 1.0
        else:
            entry["AVG_ratio"] = np.nan
            entry["AVG_residual"] = np.nan

        rows.append(entry)

    return pd.DataFrame(rows)


def compute_pitcher_residuals(proj: pd.DataFrame, actual: pd.DataFrame, year: int) -> pd.DataFrame:
    """Match projected vs actual pitchers and compute per-stat residuals."""
    if proj.empty or actual.empty:
        return pd.DataFrame()

    proj = proj.copy()
    actual = actual.copy()
    proj["MLBAMID"] = proj["MLBAMID"].astype(int)
    actual["MLBAMID"] = actual["MLBAMID"].astype(int)

    merged = proj.merge(actual, on="MLBAMID", suffixes=("_proj", "_act"))

    # Filter
    ip_proj_col = "IP_proj" if "IP_proj" in merged.columns else "IP"
    ip_act_col = "IP_act" if "IP_act" in merged.columns else "IP"
    merged = merged[merged[ip_proj_col] >= MIN_PROJ_IP]
    merged = merged[merged[ip_act_col] >= MIN_ACTUAL_IP]

    # Filter: only players who pitched roughly as projected (60-140% of IP)
    ip_ratio = merged[ip_act_col] / merged[ip_proj_col]
    merged = merged[(ip_ratio >= IP_RATIO_RANGE[0]) & (ip_ratio <= IP_RATIO_RANGE[1])]

    rows = []
    for _, row in merged.iterrows():
        name = row.get("Name_proj", row.get("Name_act", "?"))
        ip_proj = float(row.get("IP_proj", row.get("IP", 0)) or 0)
        ip_act = float(row.get("IP_act", 0) or 0)
        entry = {"year": year, "name": name, "MLBAMID": row["MLBAMID"],
                 "IP_proj": ip_proj, "IP_act": ip_act}

        # IP itself
        if ip_proj >= MIN_PROJ_STAT.get("IP", 1):
            entry["IP_ratio"] = ip_act / ip_proj
            entry["IP_residual"] = ip_act / ip_proj - 1.0
        else:
            entry["IP_ratio"] = np.nan
            entry["IP_residual"] = np.nan

        # Counting stats
        for stat in PITCHER_COUNTING:
            proj_val = float(row.get(f"{stat}_proj", row.get(stat, 0)) or 0)
            act_val = float(row.get(f"{stat}_act", 0) or 0)
            min_proj = MIN_PROJ_STAT.get(stat, 1)
            if proj_val >= min_proj:
                entry[f"{stat}_ratio"] = act_val / proj_val
                entry[f"{stat}_residual"] = act_val / proj_val - 1.0
            else:
                entry[f"{stat}_ratio"] = np.nan
                entry[f"{stat}_residual"] = np.nan

        # Rate components: ER, BB, H (for ERA/WHIP)
        for stat in PITCHER_RATE_COMPONENTS:
            proj_val = float(row.get(f"{stat}_proj", row.get(stat, 0)) or 0)
            act_val = float(row.get(f"{stat}_act", 0) or 0)
            min_proj = MIN_PROJ_STAT.get(stat, 1)
            if proj_val >= min_proj:
                entry[f"{stat}_ratio"] = act_val / proj_val
                entry[f"{stat}_residual"] = act_val / proj_val - 1.0
            else:
                entry[f"{stat}_ratio"] = np.nan
                entry[f"{stat}_residual"] = np.nan

        # ERA and WHIP as rate stats
        era_proj = float(row.get("ERA_proj", row.get("ERA", 0)) or 0)
        era_act = float(row.get("ERA_act", 0) or 0)
        if era_proj > 0:
            entry["ERA_ratio"] = era_act / era_proj
            entry["ERA_residual"] = era_act / era_proj - 1.0

        whip_proj = float(row.get("WHIP_proj", row.get("WHIP", 0)) or 0)
        whip_act = float(row.get("WHIP_act", 0) or 0)
        if whip_proj > 0:
            entry["WHIP_ratio"] = whip_act / whip_proj
            entry["WHIP_residual"] = whip_act / whip_proj - 1.0

        rows.append(entry)

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("PROJECTION VARIANCE CALIBRATION")
    print(f"Years: {YEARS}")
    print(f"Min PA (hitters): {MIN_ACTUAL_PA}  |  Min IP (pitchers): {MIN_ACTUAL_IP}")
    print("=" * 70)

    all_h_residuals = []
    all_p_residuals = []

    for year in YEARS:
        print(f"\n--- {year} ---")
        proj_dir = PROJECTIONS_DIR / str(year)
        if not proj_dir.exists():
            print(f"  No projection directory for {year}, skipping")
            continue

        proj_h, proj_p = blend_projections(proj_dir)
        act_h, act_p = load_actuals(year)

        if proj_h.empty:
            print(f"  No hitter projections for {year}")
        else:
            h_res = compute_hitter_residuals(proj_h, act_h, year)
            print(f"  Hitters matched: {len(h_res)}")
            all_h_residuals.append(h_res)

        if proj_p.empty:
            print(f"  No pitcher projections for {year}")
        else:
            p_res = compute_pitcher_residuals(proj_p, act_p, year)
            print(f"  Pitchers matched: {len(p_res)}")
            all_p_residuals.append(p_res)

    if not all_h_residuals and not all_p_residuals:
        print("\nNo data to analyze!")
        return

    # ── Hitter Results ────────────────────────────────────────────────
    if all_h_residuals:
        h_df = pd.concat(all_h_residuals, ignore_index=True)
        print("\n" + "=" * 70)
        print(f"HITTER VARIANCE (n={len(h_df)} player-seasons)")
        print("=" * 70)

        hitter_cats = [*HITTER_STATS, "AVG"]
        print(f"\n  {'Stat':>5} {'Mean':>7} {'SD':>7} {'Median':>7} {'P10':>7} {'P90':>7}  Current")
        print("  " + "-" * 60)
        for stat in hitter_cats:
            col = f"{stat}_residual"
            vals = h_df[col].dropna()
            if len(vals) < 10:
                continue
            mean = vals.mean()
            sd = vals.std()
            med = vals.median()
            p10 = vals.quantile(0.10)
            p90 = vals.quantile(0.90)
            current = 0.10
            print(f"  {stat:>5} {mean:>+7.3f} {sd:>7.3f} {med:>+7.3f} {p10:>+7.3f} {p90:>+7.3f}  {current:.2f}")

        # Overall hitter sigma (mean across counting stats)
        counting_sds = []
        for stat in HITTER_STATS:
            col = f"{stat}_residual"
            vals = h_df[col].dropna()
            if len(vals) >= 10:
                counting_sds.append(vals.std())
        if counting_sds:
            overall_h = np.mean(counting_sds)
            print(f"\n  Overall hitter counting stat SD: {overall_h:.3f}")
            print("  Current STAT_VARIANCE['hitter']: 0.100")
            print(f"  Recommended:                     {overall_h:.3f}")

        # Correlation matrix
        print("\n  Residual correlation matrix (hitters):")
        corr_cols = [f"{s}_residual" for s in hitter_cats]
        corr_df = h_df[corr_cols].dropna()
        if len(corr_df) >= 20:
            corr = corr_df.corr()
            labels = hitter_cats
            print(f"  {'':>5}", end="")
            for label in labels:
                print(f" {label:>7}", end="")
            print()
            for i, row_label in enumerate(labels):
                print(f"  {row_label:>5}", end="")
                for j, _col_label in enumerate(labels):
                    val = corr.iloc[i, j]
                    print(f" {val:>+7.3f}", end="")
                print()

    # ── Pitcher Results ───────────────────────────────────────────────
    if all_p_residuals:
        p_df = pd.concat(all_p_residuals, ignore_index=True)
        print("\n" + "=" * 70)
        print(f"PITCHER VARIANCE (n={len(p_df)} player-seasons)")
        print("=" * 70)

        pitcher_cats = ["W", "SO", "SV", "IP", "ER", "BB", "H", "ERA", "WHIP"]
        print(f"\n  {'Stat':>5} {'Mean':>7} {'SD':>7} {'Median':>7} {'P10':>7} {'P90':>7}  Current")
        print("  " + "-" * 60)
        for stat in pitcher_cats:
            col = f"{stat}_residual"
            if col not in p_df.columns:
                continue
            vals = p_df[col].dropna()
            if len(vals) < 10:
                print(f"  {stat:>5} (n={len(vals)}, too few)")
                continue
            mean = vals.mean()
            sd = vals.std()
            med = vals.median()
            p10 = vals.quantile(0.10)
            p90 = vals.quantile(0.90)
            current = 0.18
            print(f"  {stat:>5} {mean:>+7.3f} {sd:>7.3f} {med:>+7.3f} {p10:>+7.3f} {p90:>+7.3f}  {current:.2f}")

        # Overall pitcher sigma
        p_counting = ["W", "SO", "IP", "ER", "BB", "H"]
        counting_sds = []
        for stat in p_counting:
            col = f"{stat}_residual"
            if col not in p_df.columns:
                continue
            vals = p_df[col].dropna()
            if len(vals) >= 10:
                counting_sds.append((stat, vals.std()))
        if counting_sds:
            overall_p = np.mean([sd for _, sd in counting_sds])
            print("\n  Per-stat pitcher counting SDs:")
            for stat, sd in counting_sds:
                print(f"    {stat:>4}: {sd:.3f}")
            print(f"\n  Overall pitcher counting stat SD: {overall_p:.3f}")
            print("  Current STAT_VARIANCE['pitcher']: 0.180")
            print(f"  Recommended:                      {overall_p:.3f}")

        # SV separately (closers only — very different from SP stats)
        sv_col = "SV_residual"
        if sv_col in p_df.columns:
            sv_vals = p_df[sv_col].dropna()
            if len(sv_vals) >= 5:
                print(f"\n  SV variance (n={len(sv_vals)}): SD={sv_vals.std():.3f}")
                print("  (Saves are far more volatile than other pitching stats)")

        # Correlation matrix
        print("\n  Residual correlation matrix (pitchers):")
        p_corr_stats = ["W", "SO", "SV", "ERA", "WHIP"]
        corr_cols = [f"{s}_residual" for s in p_corr_stats if f"{s}_residual" in p_df.columns]
        corr_df = p_df[corr_cols].dropna()
        if len(corr_df) >= 20:
            corr = corr_df.corr()
            labels = [c.replace("_residual", "") for c in corr_cols]
            print(f"  {'':>5}", end="")
            for label in labels:
                print(f" {label:>7}", end="")
            print()
            for i, row_label in enumerate(labels):
                print(f"  {row_label:>5}", end="")
                for j in range(len(labels)):
                    val = corr.iloc[i, j]
                    print(f" {val:>+7.3f}", end="")
                print()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Recommended STAT_VARIANCE updates")
    print("=" * 70)
    if all_h_residuals:
        h_df = pd.concat(all_h_residuals, ignore_index=True)
        h_sds = []
        for stat in HITTER_STATS:
            vals = h_df[f"{stat}_residual"].dropna()
            if len(vals) >= 10:
                h_sds.append(vals.std())
        if h_sds:
            rec_h = np.mean(h_sds)
            print(f"\n  Hitter:  current=0.10  ->  recommended={rec_h:.3f}")
    if all_p_residuals:
        p_df = pd.concat(all_p_residuals, ignore_index=True)
        p_sds = []
        for stat in ["W", "SO", "IP", "ER", "BB", "H"]:
            col = f"{stat}_residual"
            if col in p_df.columns:
                vals = p_df[col].dropna()
                if len(vals) >= 10:
                    p_sds.append(vals.std())
        if p_sds:
            rec_p = np.mean(p_sds)
            print(f"  Pitcher: current=0.18  ->  recommended={rec_p:.3f}")

    print()


if __name__ == "__main__":
    main()
