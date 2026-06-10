"""Find ROS playing-time disagreements across the 5 projection sources.

For each player, compare the playing-time signal (PA for hitters, IP for
pitchers) across atc / oopsy / steamer / the-bat-x / zips. Flag players where
ONE source diverges most from the consensus (median) of the other four --
the signature of a source that ignores an injury everyone else priced in.

Usage: python scripts/ros_injury_divergence.py [YYYY-MM-DD]
"""

import csv
import sys
from pathlib import Path
from statistics import median

SOURCES = ["atc", "oopsy", "steamer", "the-bat-x", "zips"]


def load(path, key_col):
    """Return {playerid: (name, team, value)} for one source CSV."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r.get("PlayerId")
            if not pid:
                continue
            raw = r.get(key_col)
            if raw is None or raw == "":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            name = r.get("NameASCII") or r.get("Name") or ""
            out[pid] = (name, r.get("Team", ""), val)
    return out


def analyze(
    date_dir,
    kind,
    key_col,
    min_consensus,
    top_n,
    min_others_med=0.0,
    max_others_med=float("inf"),
    positive_only=False,
    min_outlier_val=0.0,
    title_suffix="",
):
    per_source = {}
    for src in SOURCES:
        path = date_dir / f"{src}-{kind}.csv"
        per_source[src] = load(path, key_col)

    # union of all player ids
    all_ids = set()
    for d in per_source.values():
        all_ids.update(d.keys())

    rows = []
    for pid in all_ids:
        vals = {}
        name = team = ""
        for src in SOURCES:
            if pid in per_source[src]:
                n, t, v = per_source[src][pid]
                vals[src] = v
                name = name or n
                team = team or t
        # require the player to appear in at least 4 of 5 so "consensus" means something
        if len(vals) < 4:
            continue
        # only care about players with meaningful projected playing time somewhere
        if max(vals.values()) < min_consensus:
            continue
        # for each source present, compare it to the median of the OTHERS
        max_dev = 0.0
        outlier_src = None
        outlier_val = None
        others_med = None
        for src in vals:
            others = [vals[s] for s in vals if s != src]
            if len(others) < 3:
                continue
            med = median(others)
            dev = vals[src] - med
            if abs(dev) > abs(max_dev):
                max_dev = dev
                outlier_src = src
                outlier_val = vals[src]
                others_med = med
        if outlier_src is None:
            continue
        if others_med is not None and not (min_others_med <= others_med <= max_others_med):
            continue
        if positive_only and max_dev <= 0:
            continue
        if outlier_val is not None and outlier_val < min_outlier_val:
            continue
        rows.append(
            {
                "name": name,
                "team": team,
                "outlier": outlier_src,
                "outlier_val": outlier_val,
                "others_med": others_med,
                "dev": max_dev,
                "vals": vals,
            }
        )

    # rank by absolute deviation
    rows.sort(key=lambda r: abs(r["dev"]), reverse=True)

    label = "PA" if kind == "hitters" else "IP"
    print(f"\n{'=' * 100}")
    print(f"TOP {top_n} {kind.upper()} DISAGREEMENTS{title_suffix} (signal = {label}, ROS)")
    print(f"{'=' * 100}")
    hdr = (
        f"{'Player':<24}{'Tm':<5}{'outlier':<11}{'its ' + label:>9}{'others med':>12}{'dev':>9}   "
        + "  ".join(f"{s[:4]:>6}" for s in SOURCES)
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:top_n]:
        cells = []
        for s in SOURCES:
            cells.append(f"{r['vals'][s]:6.0f}" if s in r["vals"] else f"{'-':>6}")
        sign = "+" if r["dev"] > 0 else "-"
        print(
            f"{r['name'][:23]:<24}{r['team']:<5}{r['outlier']:<11}"
            f"{r['outlier_val']:9.0f}{r['others_med']:12.0f}{sign}{abs(r['dev']):8.0f}   "
            + "  ".join(cells)
        )

    # how often is each source the high outlier (projecting MORE than consensus)?
    high = {s: 0 for s in SOURCES}
    for r in rows[:top_n]:
        if r["dev"] > 0:
            high[r["outlier"]] += 1
    print(f"\nAmong top {top_n}, times each source projects MORE than the other-4 consensus:")
    for s in SOURCES:
        print(f"  {s:<12} {high[s]}")
    return rows


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-10"
    base = Path("data/projections/2026/rest_of_season") / date
    if not base.exists():
        print(f"No such dir: {base}")
        sys.exit(1)
    # Pass 1: raw biggest gaps (dominated by prospects/depth allocation)
    analyze(base, "hitters", "PA", min_consensus=150, top_n=30)
    analyze(base, "pitchers", "IP", min_consensus=40, top_n=30)

    # Pass 2: ESTABLISHED REGULARS only -- the other-4 consensus is high, so a
    # divergence here is an injury/role change, not a prospect playing-time guess.
    analyze(
        base,
        "hitters",
        "PA",
        min_consensus=150,
        top_n=30,
        min_others_med=250,
        title_suffix=" [ESTABLISHED REGULARS: others-med PA>=250]",
    )
    analyze(
        base,
        "pitchers",
        "IP",
        min_consensus=40,
        top_n=30,
        min_others_med=60,
        title_suffix=" [ESTABLISHED REGULARS: others-med IP>=60]",
    )

    # Pass 3: INJURY-IGNORE CANDIDATES -- one source projects a near-full ROS load
    # while the other-4 consensus has clearly CUT the player (partial season).
    # others-med in a "cut regular" band; outlier projects a full slate.
    analyze(
        base,
        "hitters",
        "PA",
        min_consensus=150,
        top_n=30,
        min_others_med=80,
        max_others_med=300,
        positive_only=True,
        min_outlier_val=280,
        title_suffix=" [INJURY-IGNORE CANDIDATES: others cut to 80-300 PA, outlier>=280]",
    )
    analyze(
        base,
        "pitchers",
        "IP",
        min_consensus=40,
        top_n=30,
        min_others_med=20,
        max_others_med=80,
        positive_only=True,
        min_outlier_val=70,
        title_suffix=" [INJURY-IGNORE CANDIDATES: others cut to 20-80 IP, outlier>=70]",
    )


if __name__ == "__main__":
    main()
