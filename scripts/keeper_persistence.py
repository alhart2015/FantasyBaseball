"""Fit and validate the per-stat persistence share S from historical seasons.

S answers: of the gap between a player's projection and what he actually did, how
much repeats next year? See `keepers/persistence.py` for the model and
`docs/keeper-value-teardown-2026-08-01.md` for why the rate/volume split matters.

Data used (local only, no network):

    projection for year Y   data/projections/{Y}/zips-{hitters,pitchers}.csv
    actual for year Y       data/stats/{hitters,pitchers}-{Y}.csv

Transitions available: 2022->23, 2023->24, 2024->25.

Usage:
    python scripts/keeper_persistence.py                 # fit + leave-one-out validation
    python scripts/keeper_persistence.py --counting      # add the blended counting-stat fit
    python scripts/keeper_persistence.py --terciles      # is a CONSTANT S good enough?
    python scripts/keeper_persistence.py --min-pa 400 --min-ip 80
"""

from __future__ import annotations

import argparse
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
    index_by_mlbam,
)
from fantasy_baseball.keepers.persistence import (
    HITTER_COUNTING,
    PITCHER_COUNTING,
    Share,
    apply_reliability_share,
    apply_share,
    evaluate_shares,
    fit_counting_share,
    fit_reliability_share,
    fit_share,
    gap,
    rmse,
)
from fantasy_baseball.keepers.vintages import decompose_hitters, decompose_pitchers

PROJECTIONS = PROJECT_ROOT / "data" / "projections"
STATS = PROJECT_ROOT / "data" / "stats"

TRANSITIONS = ((2022, 2023), (2023, 2024), (2024, 2025))

POOLS = {
    "hitter": {
        "pt": HITTER_PT,
        "rates": HITTER_RATES,
        "counting": HITTER_COUNTING,
        "file": "hitters",
        "decompose": decompose_hitters,
    },
    "pitcher": {
        "pt": PITCHER_PT,
        "rates": PITCHER_RATES,
        "counting": PITCHER_COUNTING,
        "file": "pitchers",
        "decompose": decompose_pitchers,
    },
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _projection_path(year: int, stem: str) -> Path:
    exact = PROJECTIONS / str(year) / f"zips-{stem}.csv"
    if exact.exists():
        return exact
    matches = sorted((PROJECTIONS / str(year)).glob(f"zips-{stem}-*.csv"))
    if not matches:
        raise FileNotFoundError(f"no ZiPS {stem} export for {year}")
    return matches[-1]


def load_rates(year: int, kind: str, *, source: str) -> pd.DataFrame:
    """Decomposed rate/PT frame for one (year, pool, source), indexed by mlbam_id.

    `decompose_*` works on BOTH sources because the ZiPS export and our actuals CSVs
    carry the same counting-stat column names (PA/AB/H/HR/R/RBI/SB, IP/SO/W/SV/ER/BB/H)
    keyed by MLBAMID. That is why there is no separate actuals decomposer.
    """
    pool = POOLS[kind]
    path = (
        _projection_path(year, str(pool["file"]))
        if source == "projection"
        else STATS / f"{pool['file']}-{year}.csv"
    )
    frame = pool["decompose"](_read(path))  # type: ignore[operator]
    # A ZiPS export can list a player twice (multi-team rows). Keep the highest-volume
    # entry rather than the first, so the survivor is the real line and not a 12-PA stub.
    pt = str(pool["pt"])
    ordered = frame.sort_values(pt, ascending=False)
    return ordered.loc[~ordered.index.duplicated(keep="first")]


def load_counts(year: int, kind: str, *, source: str) -> pd.DataFrame:
    """Raw counting stats for the 5x5 categories, indexed by mlbam_id."""
    pool = POOLS[kind]
    path = (
        _projection_path(year, str(pool["file"]))
        if source == "projection"
        else STATS / f"{pool['file']}-{year}.csv"
    )
    raw = index_by_mlbam(_read(path), "MLBAMID")
    cols = [*pool["counting"], "PA" if kind == "hitter" else "IP"]
    out = raw[cols].astype(float)
    volume = "PA" if kind == "hitter" else "IP"
    ordered = out.sort_values(volume, ascending=False)
    return ordered.loc[~ordered.index.duplicated(keep="first")]


def build_transition(
    year: int, kind: str, *, min_pt: float, min_next_pt: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aligned (projection_Y, actual_Y, actual_Y+1) panel plus a sample-attrition record.

    Columns are suffixed `_proj`, `_obs`, `_next`. The filter is deliberately asymmetric:
    `min_pt` on year Y selects the population we would actually consider keeping, while
    `min_next_pt` on Y+1 is only high enough to make a rate meaningful. Requiring a Y+1
    appearance at all is a SURVIVORSHIP filter, so the counts are reported, not hidden.

    Measured, the filter DEFLATES the volume share rather than inflating it (hitters
    0.771 -> 0.655, pitchers 0.622 -> 0.446; reproduce with `run_survivorship`). The
    intuition that wash-outs are players whose good seasons failed to repeat is simply
    wrong: they are mostly players who ALREADY played less than projected, and who then
    fall to zero -- a negative gap followed by a larger negative gap, which steepens the
    slope. `build_volume_transition` is the corrected sample.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    proj = load_rates(year, kind, source="projection")
    obs = load_rates(year, kind, source="actual")
    nxt = load_rates(year + 1, kind, source="actual")

    qualified = obs.index[obs[pt] >= min_pt]
    survived = qualified.intersection(nxt.index[nxt[pt] >= min_next_pt])
    idx = survived.intersection(proj.index)
    attrition = {
        "qualified_year_Y": len(qualified),
        "survived_to_Y1": len(survived),
        "also_projected": len(idx),
    }

    parts = {}
    for name, frame in (("proj", proj), ("obs", obs), ("next", nxt)):
        for col in (pt, *pool["rates"]):
            parts[f"{col}_{name}"] = frame.loc[idx, col]
    return pd.DataFrame(parts, index=idx), attrition


def build_counting_transition(
    year: int, kind: str, *, min_pt: float, min_next_pt: float
) -> pd.DataFrame:
    pool = POOLS[kind]
    volume = "PA" if kind == "hitter" else "IP"
    proj = load_counts(year, kind, source="projection")
    obs = load_counts(year, kind, source="actual")
    nxt = load_counts(year + 1, kind, source="actual")
    idx = (
        obs.index[obs[volume] >= min_pt]
        .intersection(nxt.index[nxt[volume] >= min_next_pt])
        .intersection(proj.index)
    )
    parts = {}
    for name, frame in (("proj", proj), ("obs", obs), ("next", nxt)):
        for col in (*pool["counting"], volume):
            parts[f"{col}_{name}"] = frame.loc[idx, col]
    return pd.DataFrame(parts, index=idx)


def _pooled(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack transitions. The index repeats across transitions by design -- the same
    player contributes one observation per transition -- so it is reset to keep the
    row count honest."""
    return pd.concat(frames, ignore_index=True)


def _fit_column(panel: pd.DataFrame, col: str, pt: str) -> Share:
    return fit_share(
        gap(panel[f"{col}_obs"], panel[f"{col}_proj"]),
        gap(panel[f"{col}_next"], panel[f"{col}_proj"]),
        column=col,
        weights=panel[f"{pt}_obs"],
    )


def run_fit(
    kind: str,
    panels: dict[tuple[int, int], pd.DataFrame],
    volume_panels: dict[tuple[int, int], pd.DataFrame],
) -> dict[str, Share]:
    """The shipped shares: volume off the survivorship-corrected panel, rates off the
    survivor panel. Those are different samples on purpose -- see `build_volume_transition`
    for why volume must count the players who vanished and rates must not."""
    pool = POOLS[kind]
    pt = str(pool["pt"])
    pooled = _pooled(list(panels.values()))
    fits = {pt: _fit_column(_pooled(list(volume_panels.values())), pt, pt)}
    fits.update({col: _fit_column(pooled, col, pt) for col in pool["rates"]})

    print(
        f"\n{'=' * 78}\n{kind.upper()}S -- persistence share S, pooled over {len(panels)} transitions"
    )
    print(f"{'=' * 78}")
    print(f"{'column':<10} {'S':>7} {'+/-':>6} {'intercept':>11} {'r2':>7} {'n':>6}  signal")
    print("-" * 78)
    for col, fit in fits.items():
        flag = "yes" if fit.separable_from_zero else "NO -- gap is noise"
        label = f"{col} (volume)" if col == pt else col
        print(
            f"{label:<10} {fit.share:>7.3f} {fit.stderr:>6.3f} {fit.intercept:>11.5f} "
            f"{fit.r2:>7.3f} {fit.n:>6}  {flag}"
        )
    return fits


def run_validation(
    kind: str,
    panels: dict[tuple[int, int], pd.DataFrame],
    volume_panels: dict[tuple[int, int], pd.DataFrame],
) -> None:
    """Leave-one-transition-out: fit on the others, score on the held-out one.

    With three transitions this is three folds. Reported per fold rather than averaged,
    because the spread ACROSS folds is the noise floor -- a difference between models
    smaller than the spread between folds is not a difference.

    The volume column is validated on the SURVIVORSHIP-CORRECTED panel, matching what
    `run_fit` ships. Folding it over the survivor-only rate panels validated a share
    the pipeline never uses (0.42-0.59 against a shipped 0.771) and stamped a verdict
    on the wrong estimator.

    **The folds are not independent.** With three consecutive transitions, T1's response
    is built from the same realized season that forms T2's regressor, and the same
    players recur in every fold, so shared sampling noise flows between train and test.
    Read these numbers as an UPPER BOUND on out-of-sample skill, not a clean estimate.
    Fixing that needs more seasons, not different code.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    columns = (pt, *pool["rates"])
    print(f"\n{kind.upper()}S -- leave-one-transition-out RMSE (weighted by year-Y volume)")
    print(f"{'-' * 78}")
    print(
        f"{'column':<10} {'holdout':<12} {'S(fit)':>7} {'s=0':>10} {'s=1':>10} {'fitted':>10}  verdict"
    )
    print("-" * 78)
    for col in columns:
        # Volume off the corrected panel, rates off the survivor panel -- the same
        # split `run_fit` ships. See `build_volume_transition` for why they differ.
        source = volume_panels if col == pt else panels
        for held, panel in source.items():
            train = _pooled([p for k, p in source.items() if k != held])
            fit = _fit_column(train, col, pt)
            scores = evaluate_shares(
                panel[f"{col}_proj"],
                gap(panel[f"{col}_obs"], panel[f"{col}_proj"]),
                panel[f"{col}_next"],
                fit,
                weights=panel[f"{pt}_obs"],
            )
            best = min(scores, key=lambda k: scores[k])
            verdict = "fitted wins" if best == "fitted" else f"{best} wins"
            print(
                f"{col:<10} {held[0]}->{str(held[1])[2:]:<7} {fit.share:>7.3f} "
                f"{scores['s0']:>10.4f} {scores['s1']:>10.4f} {scores['fitted']:>10.4f}  {verdict}"
            )
        print()


def run_counting(kind: str, args: argparse.Namespace) -> None:
    """The blended counting-stat fit, for comparison against the rate fit."""
    pool = POOLS[kind]
    volume = "PA" if kind == "hitter" else "IP"
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip
    frames = [
        build_counting_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)
        for y, _ in TRANSITIONS
    ]
    pooled = _pooled(frames)
    print(f"\n{kind.upper()}S -- COUNTING-stat S (volume+rate blended) vs the matching RATE S")
    print(f"{'-' * 78}")
    print(f"{'stat':<8} {'S(counting)':>12} {'S(rate)':>9} {'S(volume)':>10}   read")
    print("-" * 78)

    rate_panels = {
        (y, y + 1): build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0]
        for y, _ in TRANSITIONS
    }
    rate_pooled = _pooled(list(rate_panels.values()))
    pt = str(pool["pt"])
    # Volume off the SURVIVORSHIP-CORRECTED panel, matching run_fit and run_validation.
    # Scoring it on the survivor-only rate panel printed 0.655/0.446 under a column
    # headed "S(volume)" while the pipeline ships 0.771/0.622 -- one run, two different
    # numbers for the same quantity under the same name.
    vol_fit = _fit_column(
        _pooled([build_volume_transition(y, kind, min_pt=min_pt) for y, _ in TRANSITIONS]), pt, pt
    )

    for stat, rate_col in pool["counting"].items():  # type: ignore[union-attr]
        counting = fit_counting_share(
            pooled[f"{stat}_proj"],
            pooled[f"{stat}_obs"],
            pooled[f"{stat}_next"],
            column=stat,
            weights=pooled[f"{volume}_obs"],
        )
        rate_fit = _fit_column(rate_pooled, rate_col, pt)
        drift = counting.share - rate_fit.share
        read = "counting inflated by volume" if drift > 0.05 else "close -- little PT leakage"
        print(
            f"{stat:<8} {counting.share:>12.3f} {rate_fit.share:>9.3f} "
            f"{vol_fit.share:>10.3f}   {read}"
        )


def run_vs_fresh(kind: str, args: argparse.Namespace) -> None:
    """Score the S-adjusted STALE projection against a real FRESH one.

    This is the calibration that matters for the live use case. For 2027 we hold only
    a ZiPS out-year generated before the 2026 season -- there is no fresh 2027 line to
    fall back on. Historically we DO have one (ZiPS Y+1, built with year-Y information),
    so this measures what a fresh projection would have been worth and therefore how
    much the S-adjustment recovers of it.

    `fresh` is the ceiling, `s0` (the untouched stale projection) is the floor, and the
    fraction of that span the fitted model closes is the headline number.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip

    print(f"\n{kind.upper()}S -- S-adjusted STALE projection vs a real FRESH projection")
    print(f"{'-' * 78}")
    print(f"{'column':<10} {'stale (s0)':>11} {'fitted':>10} {'fresh Y+1':>11} {'gap closed':>11}")
    print("-" * 78)

    panels = {
        (y, y + 1): build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0]
        for y, _ in TRANSITIONS
    }
    for col in (pt, *pool["rates"]):
        stale_e, fit_e, fresh_e = [], [], []
        for (y, y1), panel in panels.items():
            train = _pooled([p for k, p in panels.items() if k != (y, y1)])
            fit = _fit_column(train, col, pt)
            g = gap(panel[f"{col}_obs"], panel[f"{col}_proj"])
            w = panel[f"{pt}_obs"]
            fresh = load_rates(y1, kind, source="projection")[col].reindex(panel.index)
            stale_e.append(
                rmse(
                    apply_share(
                        panel[f"{col}_proj"], g, Share(col, 0.0, fit.intercept, 0, 0.0, 0.0)
                    ),
                    panel[f"{col}_next"],
                    w,
                )
            )
            fit_e.append(rmse(apply_share(panel[f"{col}_proj"], g, fit), panel[f"{col}_next"], w))
            fresh_e.append(rmse(fresh, panel[f"{col}_next"], w))
        stale, fitted, fresh_m = (sum(v) / len(v) for v in (stale_e, fit_e, fresh_e))
        span = stale - fresh_m
        closed = (
            f"{100.0 * (stale - fitted) / span:.0f}%" if span > 1e-12 else "n/a (fresh no better)"
        )
        print(f"{col:<10} {stale:>11.4f} {fitted:>10.4f} {fresh_m:>11.4f} {closed:>11}")


def build_volume_transition(year: int, kind: str, *, min_pt: float) -> pd.DataFrame:
    """Volume panel that KEEPS the players who washed out.

    The rate panels have to drop a player who did not play in Y+1 -- he has no
    observable rate. Volume has no such excuse: "he took 0 plate appearances" is a
    real, informative observation, and dropping it biases the volume share DOWNWARD
    (0.771 -> 0.655 for hitters, 0.622 -> 0.446 for pitchers) while making the drift
    term far too optimistic. So this panel starts from everyone who qualified in year Y
    and reindexes Y+1 onto them.

    Absent from the Y+1 file means BELOW its floor (50 PA / 10 IP), not necessarily
    zero, so filling 0.0 slightly understates those players. That is deliberate and
    conservative: it errs toward treating a vanished player as vanished, and the
    alternative (dropping him) errs by pretending he never existed.

    Why the rate fits stay conditional on playing: expected value is volume * rate, so
    a player who does not play contributes nothing whatever his rate would have been.
    Pricing the risk of not playing belongs in the VOLUME term -- which is exactly what
    this panel makes honest -- and "how good is he when he plays" is the right thing
    for the rate term to answer.
    """
    pt = str(POOLS[kind]["pt"])
    proj = load_rates(year, kind, source="projection")
    obs = load_rates(year, kind, source="actual")
    nxt = load_rates(year + 1, kind, source="actual")
    idx = obs.index[obs[pt] >= min_pt].intersection(proj.index)
    return pd.DataFrame(
        {
            f"{pt}_proj": proj.loc[idx, pt],
            f"{pt}_obs": obs.loc[idx, pt],
            f"{pt}_next": nxt[pt].reindex(idx).fillna(0.0),
        },
        index=idx,
    )


def run_survivorship(kind: str, args: argparse.Namespace) -> None:
    """Volume S with and without the players who washed out."""
    pool = POOLS[kind]
    pt = str(pool["pt"])
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip

    survivors = _pooled(
        [build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0] for y, _ in TRANSITIONS]
    )
    everyone = _pooled([build_volume_transition(y, kind, min_pt=min_pt) for y, _ in TRANSITIONS])
    biased, corrected = _fit_column(survivors, pt, pt), _fit_column(everyone, pt, pt)

    print(f"\n{kind.upper()}S -- survivorship correction on the {pt} (volume) share")
    print(f"{'-' * 78}")
    print(f"{'sample':<34} {'n':>6} {'S':>8} {'intercept':>11}")
    print("-" * 78)
    print(
        f"{'survivors only (biased)':<34} {biased.n:>6} {biased.share:>8.3f} "
        f"{biased.intercept:>11.2f}"
    )
    print(
        f"{'all year-Y qualifiers (corrected)':<34} {corrected.n:>6} {corrected.share:>8.3f} "
        f"{corrected.intercept:>11.2f}"
    )
    print(
        f"\n  bias from dropping wash-outs: S {biased.share - corrected.share:+.3f}, "
        f"drift {biased.intercept - corrected.intercept:+.1f} {pt}"
    )


def run_reliability(kind: str, args: argparse.Namespace) -> None:
    """Constant S vs the reliability form S(n) = s_max * n/(n+k), leave-one-out.

    The extra parameter has to pay for itself on held-out data. Where it does not, the
    constant is the right answer and the tercile wobble was noise.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    min_pt = args.min_pa if kind == "hitter" else args.min_ip
    min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip

    rate_panels = {
        (y, y + 1): build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)[0]
        for y, _ in TRANSITIONS
    }
    vol_panels = {
        (y, y + 1): build_volume_transition(y, kind, min_pt=min_pt) for y, _ in TRANSITIONS
    }

    print(f"\n{kind.upper()}S -- constant S vs reliability S(n) = s_max * n/(n+k)")
    print(f"{'-' * 78}")
    print(f"{'column':<10} {'s_max':>7} {'k':>8} {'const RMSE':>11} {'S(n) RMSE':>11}   verdict")
    print("-" * 78)

    for col in (pt, *pool["rates"]):
        # The volume column uses the survivorship-corrected panel; rates cannot.
        panels = vol_panels if col == pt else rate_panels
        const_err, rel_err, params = [], [], []
        for held, panel in panels.items():
            train = _pooled([p for k, p in panels.items() if k != held])
            g_tr = gap(train[f"{col}_obs"], train[f"{col}_proj"])
            gn_tr = gap(train[f"{col}_next"], train[f"{col}_proj"])
            const = fit_share(g_tr, gn_tr, column=col, weights=train[f"{pt}_obs"])
            rel = fit_reliability_share(
                g_tr, gn_tr, train[f"{pt}_obs"], column=col, weights=train[f"{pt}_obs"]
            )
            params.append((rel.s_max, rel.k))
            g = gap(panel[f"{col}_obs"], panel[f"{col}_proj"])
            w, truth = panel[f"{pt}_obs"], panel[f"{col}_next"]
            const_err.append(rmse(apply_share(panel[f"{col}_proj"], g, const), truth, w))
            rel_err.append(
                rmse(
                    apply_reliability_share(panel[f"{col}_proj"], g, panel[f"{pt}_obs"], rel),
                    truth,
                    w,
                )
            )
        c_mean, r_mean = sum(const_err) / len(const_err), sum(rel_err) / len(rel_err)
        s_max = sum(p[0] for p in params) / len(params)
        k_hat = sum(p[1] for p in params) / len(params)
        better = (c_mean - r_mean) / c_mean if c_mean > 0 else 0.0
        verdict = f"S(n) wins ({better:+.1%})" if r_mean < c_mean else "constant is enough"
        print(f"{col:<10} {s_max:>7.3f} {k_hat:>8.1f} {c_mean:>11.4f} {r_mean:>11.4f}   {verdict}")


def run_terciles(kind: str, panels: dict[tuple[int, int], pd.DataFrame]) -> None:
    """Refit S within playing-time terciles.

    A constant S assumes a 250-PA gap is as believable as a 650-PA gap. It is not:
    the small-sample gap carries more noise, so its true share should be LOWER. If S
    rises monotonically across terciles, a constant is the wrong shape and the fix is
    a reliability form, S(n) = n / (n + k), fit per stat.
    """
    pool = POOLS[kind]
    pt = str(pool["pt"])
    pooled = _pooled(list(panels.values()))
    edges = pooled[f"{pt}_obs"].quantile([1 / 3, 2 / 3]).to_list()
    bands = {
        f"low (<{edges[0]:.0f})": pooled[pooled[f"{pt}_obs"] < edges[0]],
        "mid": pooled[(pooled[f"{pt}_obs"] >= edges[0]) & (pooled[f"{pt}_obs"] < edges[1])],
        f"high (>{edges[1]:.0f})": pooled[pooled[f"{pt}_obs"] >= edges[1]],
    }
    print(f"\n{kind.upper()}S -- S by year-Y volume tercile (is a CONSTANT S the right shape?)")
    print(f"{'-' * 78}")
    header = "".join(f"{name:>20}" for name in bands)
    print(f"{'column':<10}{header}   shape")
    print("-" * 78)
    for col in (pt, *pool["rates"]):
        shares = []
        for band in bands.values():
            try:
                shares.append(_fit_column(band, col, pt).share)
            except ValueError:
                shares.append(float("nan"))
        cells = "".join(f"{s:>20.3f}" for s in shares)
        rising = shares[0] < shares[1] < shares[2]
        shape = "RISING -- constant S too crude" if rising else "flat/mixed"
        print(f"{col:<10}{cells}   {shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-pa", type=float, default=300, help="year-Y hitter PA floor")
    parser.add_argument("--min-ip", type=float, default=50, help="year-Y pitcher IP floor")
    # Year-Y+1 floors chosen by held-out prediction, not by eye: fit at 50/150/250/350
    # (hitters) and 20/50/80 (pitchers), each scored on the SAME population of players
    # who reach real playing time. Pitchers: 50 IP wins on all five rates. Hitters:
    # converges at 250-350; 250 is best-or-tied on four of six and keeps 81% of the
    # sample. A LOW floor does not just add noise, it biases the DRIFT -- a 60-PA bad
    # season is a playing-time outcome the volume term already prices, and letting it
    # into the rate fit charges the same player twice (h_ab drift -0.0088 at 50 vs
    # -0.0037 at 250, against a real decline of about -0.0013).
    parser.add_argument("--min-next-pa", type=float, default=250, help="year-Y+1 hitter PA floor")
    parser.add_argument("--min-next-ip", type=float, default=50, help="year-Y+1 pitcher IP floor")
    parser.add_argument("--counting", action="store_true", help="add the blended counting fit")
    parser.add_argument("--terciles", action="store_true", help="refit S by volume tercile")
    parser.add_argument(
        "--vs-fresh", action="store_true", help="score against a real fresh Y+1 projection"
    )
    parser.add_argument(
        "--reliability", action="store_true", help="constant S vs S(n)=s_max*n/(n+k)"
    )
    parser.add_argument("--pool", choices=("hitter", "pitcher"), help="restrict to one pool")
    args = parser.parse_args()

    kinds = [args.pool] if args.pool else ["hitter", "pitcher"]
    for kind in kinds:
        min_pt = args.min_pa if kind == "hitter" else args.min_ip
        min_next = args.min_next_pa if kind == "hitter" else args.min_next_ip
        panels: dict[tuple[int, int], pd.DataFrame] = {}
        print(f"\n### {kind}s: sample attrition (floor {min_pt:g} in year Y)")
        for y, y1 in TRANSITIONS:
            panel, attrition = build_transition(y, kind, min_pt=min_pt, min_next_pt=min_next)
            panels[(y, y1)] = panel
            lost = attrition["qualified_year_Y"] - attrition["survived_to_Y1"]
            pct = 100.0 * lost / max(attrition["qualified_year_Y"], 1)
            print(
                f"  {y}->{y1}: {attrition['qualified_year_Y']} qualified, "
                f"{attrition['survived_to_Y1']} survived ({lost} lost, {pct:.0f}% "
                f"-- rate fits use survivors only), {attrition['also_projected']} also projected"
            )
        volume_panels = {
            (y, y1): build_volume_transition(y, kind, min_pt=min_pt) for y, y1 in TRANSITIONS
        }
        run_fit(kind, panels, volume_panels)
        run_survivorship(kind, args)
        run_validation(kind, panels, volume_panels)
        if args.counting:
            run_counting(kind, args)
        if args.vs_fresh:
            run_vs_fresh(kind, args)
        if args.reliability:
            run_reliability(kind, args)
        if args.terciles:
            run_terciles(kind, panels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
