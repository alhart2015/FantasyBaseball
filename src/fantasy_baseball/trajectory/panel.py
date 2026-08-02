"""Load the historical season panel and score every player-season in SGP.

Input is `scripts/build_pt_panel.py --out-dir data/trajectory` output: one row per
(mlbam_id, season) over each player's observed career span, absent seasons explicitly
NaN. This module turns that into one SGP number per observed season.

Three cleaning steps, each of which changes the answer materially:

* **Each pool keeps only the seasons the player was USED in that role** (`_in_role`).
  Before the universal DH in 2022, NL pitchers taking a handful of plate appearances
  are 530-720 rows a season -- roughly 45% of the pre-2022 hitter panel -- at ~10 PA
  and near-zero SGP; left in, they halve every pre-2022 age bin and manufacture an
  aging trend out of a 2022 rule change. The mirror holds on the mound, where 655
  position-player mop-up outings are guaranteed all-zero forward paths. Role is
  decided by volume rather than by MLBAM's career-long primary position, so a genuine
  two-way season lands in BOTH pools and is scored separately in each.
* **Short schedules are scaled to 162 games.** 2020's 60-game season would otherwise
  read as a career-wide collapse at whatever age a player happened to be. Scaling
  counting volume by `162 / scheduled_games` keeps the season in the career instead of
  punching a hole in it, which matters because `comps` indexes forward by season offset
  and would score a missing year as a zero.
* **In-progress seasons are excluded from the comp pool.** Every 2026 row is
  `partial_season`; a two-thirds season is not a career year. 2026 is still loaded,
  because it is the QUERY side -- see `prorate_partial`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# NOT data/playing_time/. `keeper_forecast._panel_path` globs that directory and ranks
# on (end, -start), so a wider panel dropped in beside the keeper model's 2010-2026 one
# silently becomes its playing-time training set. Separate directory, no interaction.
DEFAULT_PANEL_DIR = PROJECT_ROOT / "data" / "trajectory"

FULL_SCHEDULE = 162

#: Volume at which a season counts as real work in a role the player's MLBAM primary
#: position says is not his. Both are measured cuts between two separate populations,
#: not tuned parameters -- see `_in_role`.
MIN_TWO_WAY_IP = 10.0
MIN_TWO_WAY_PA = 100

#: Counting stats reconstructed from the panel's stored rates, as
#: ``{output_column: rate_column}``. Volume (``pa`` / ``ip``) is the multiplier.
HITTER_COUNTS = {"hr": "hr_pa", "r": "r_pa", "rbi": "rbi_pa", "sb": "sb_pa"}
PITCHER_COUNTS = {"k": "k_ip", "w": "w_ip", "sv": "sv_ip"}


def _display_path(path: Path) -> Path:
    """Repo-relative if it is inside the repo, absolute otherwise.

    `path` may be outside the repo (a `tmp_path` in tests, or an explicit --panel-dir),
    where a plain `relative_to` raises and buries the actionable message. `walk_up=True`
    would handle it but is 3.12+, and this package supports 3.11.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def widest_newest(paths: Iterable[Path]) -> Path | None:
    """The `..._{start}_{end}.csv` with the newest end year, then the widest span.

    Ranking on the PARSED years, never on the filename: a raw string sort puts
    `_2010_2026` above `_2000_2026`, which silently picks the narrower file. Mirrors
    `keeper_forecast._panel_path`'s rank so a rebuild with a different span is picked
    up rather than orphaning a hardcoded name. Paths whose stem does not end in two
    integers are EXCLUDED, not merely ranked low -- a stray backup or a partial write
    would otherwise be returned whenever it is the only candidate.
    """

    def span(path: Path) -> tuple[int, int] | None:
        try:
            start, end = (int(x) for x in path.stem.rsplit("_", 2)[-2:])
        except ValueError:
            return None
        return (end, -start)

    candidates = [(key, path) for path in paths if (key := span(path)) is not None]
    return max(candidates)[1] if candidates else None


def panel_path(kind: str, panel_dir: Path | None = None) -> Path:
    """The widest-span panel for `kind`, newest end year first."""
    directory = panel_dir or DEFAULT_PANEL_DIR
    found = widest_newest(directory.glob(f"{kind}_pt_panel_*.csv"))
    if found is None:
        shown = _display_path(directory)
        raise FileNotFoundError(
            f"no {kind} panel in {directory}. Build it with:\n"
            f"    python scripts/build_pt_panel.py --start 2000 --end 2026 --out-dir {shown}"
        )
    return found


def score(df: pd.DataFrame, kind: str, sgp_overrides: SgpOverrides | None = None) -> pd.DataFrame:
    """Rebuild counting lines from the stored rates and add an `sgp` column.

    Called once on load and again after `era.era_normalize` rescales the rates, so the
    two paths cannot disagree about how a rate becomes a score.
    """
    df = _reconstruct(df.copy(), kind)
    if df.empty:
        # A row-wise apply over zero rows returns a DataFrame, not a Series, and the
        # assignment then dies with `Cannot set a DataFrame with multiple columns to the
        # single column sgp` -- an error naming a column the caller never mentioned.
        df["sgp"] = pd.Series(dtype=float)
        return df
    denoms = get_sgp_denominators(sgp_overrides)
    df["sgp"] = df.apply(lambda row: calculate_player_sgp(row, denoms=denoms), axis=1)
    return df


def _reconstruct(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rebuild counting lines from the panel's stored per-PA / per-IP rates."""
    if kind == "hitter":
        df["ab"] = df["pa"] * df["ab_pa"]
        df["avg"] = df["h_ab"]
        for out, rate in HITTER_COUNTS.items():
            df[out] = df["pa"] * df[rate]
        df["player_type"] = PlayerType.HITTER
    else:
        for out, rate in PITCHER_COUNTS.items():
            df[out] = df["ip"] * df[rate]
        df["era"] = df["er_ip"] * 9.0
        df["whip"] = df["bb_ip"] + df["h_ip"]
        df["player_type"] = PlayerType.PITCHER
    return df


def _in_role(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Keep the seasons in which the player was actually USED in `kind`'s role.

    Membership is a question about usage, not about `is_pitcher` -- which is
    `primary_position == "P"`, a single label MLBAM assigns for a whole CAREER. Taking
    it at face value is wrong in both directions, and both were measured on the shipped
    panel:

    * It admits mop-up work. 655 pitcher-seasons belong to position players, median
      1.0 IP, and since a position player never pitches again each one is a guaranteed
      all-zero forward path -- 6-8% of a low-SGP cohort, biasing exactly the
      fringe-reliever query this tool exists to answer.
    * It discards real seasons. A converted player carries `P` for life, so Jason
      Lane's 561-PA and Anthony Gose's 535-PA hitting seasons were being dropped from
      the hitter pool, as were every one of Ohtani's 130-166 IP pitching seasons from
      the pitcher pool.

    So a confirmed out-of-role season is admitted only on real volume. The thresholds
    separate the two populations cleanly rather than splitting a continuum: position
    players pitching sit at a 4.3 IP 95th percentile against 44 IP median for actual
    pitchers, and pitchers batting sit at an 85 PA 99th percentile (a full-season NL
    starter's workload) against 228 PA median for position players. Above the cuts sit
    only genuine two-way careers -- Ohtani, Rick Ankiel before his conversion. The one
    knowing casualty is Dontrelle Willis's 101-PA 2005, a good-hitting pitcher landing
    just over the line.

    A season can qualify for BOTH pools, and should: this league drafts and scores a
    two-way player as two separate assets, so each half gets its own trajectory.

    An UNCONFIRMED position (NaN, a player `people` never returned) is kept in either
    pool rather than dropped -- losing a real season is worse than carrying a rare
    unclassifiable one. NaN also makes the column object-dtype, where `~series` raises
    `TypeError: bad operand type for unary ~: 'float'`; `.ne()` is dtype-safe.
    """
    if kind == "hitter":
        return df[df["is_pitcher"].ne(True) | (df["pa"] >= MIN_TWO_WAY_PA)].copy()
    return df[df["is_pitcher"].ne(False) | (df["ip"] >= MIN_TWO_WAY_IP)].copy()


def _scale_short_schedules(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Scale a short season's VOLUME up to a 162-game schedule. Rates are untouched --
    a 60-game season's per-PA rates are already on the right scale, only the
    accumulation is short."""
    volume = "pa" if kind == "hitter" else "ip"
    scale = FULL_SCHEDULE / df["scheduled_games"]
    # A partial (in-progress) season has a FULL schedule and short accumulation, which
    # this ratio cannot see. `prorate_partial` handles those; they are excluded from
    # the comp pool by default.
    df[volume] = df[volume] * scale
    df["games"] = df["games"] * scale
    return df


def load_scored_panel(
    kind: str,
    *,
    panel_dir: Path | None = None,
    sgp_overrides: SgpOverrides | None = None,
    include_partial: bool = False,
) -> pd.DataFrame:
    """One row per observed player-season with an `sgp` column.

    `include_partial` keeps in-progress seasons (2026), which are the query side of a
    trajectory lookup and must not be averaged into the comp pool.
    """
    if kind not in ("hitter", "pitcher"):
        raise ValueError(f"kind must be 'hitter' or 'pitcher', got {kind!r}")

    df = pd.read_csv(panel_path(kind, panel_dir))
    df = df[df["observed"]].copy()
    if not include_partial:
        df = df[~df["partial_season"]].copy()

    volume = "pa" if kind == "hitter" else "ip"
    df = df[df[volume].notna() & (df[volume] > 0)].copy()

    # Age first: a player absent from the MLB `people` response has neither a birth date
    # nor a position, so this removes most unknown-position rows before anything has to
    # reason about them.
    df = df[df["age"].notna()].copy()
    df["age"] = df["age"].astype(int)

    df = _in_role(df, kind)

    if df.empty:
        raise ValueError(
            f"no usable {kind} seasons in {panel_path(kind, panel_dir)} after filtering "
            f"(observed, {'' if include_partial else 'complete, '}{volume} > 0, known age, "
            f"used as a {kind}). The panel is empty or "
            "malformed -- rebuild it with scripts/build_pt_panel.py."
        )

    df = _scale_short_schedules(df, kind)
    return score(df, kind, sgp_overrides).reset_index(drop=True)


def season_elapsed_fraction(df: pd.DataFrame, season: int) -> float:
    """How much of an in-progress season has been played. HITTER PANEL ONLY.

    Estimated as the busiest player's games over a full schedule. An everyday regular
    plays nearly every team game, so the max is a good read on the calendar and needs
    no second data source. Clipped to (0, 1]: a completed season returns 1.0.

    **Must be handed the hitter panel even when pacing a pitcher.** How much of the
    season has elapsed is a calendar fact about the league, not a property of the pool.
    In the PITCHER panel `games` is `stat.gamesPitched` -- APPEARANCES, not team games
    played -- so the busiest pitcher is a reliever at ~57 appearances and this would
    read 2026 as 35% elapsed against the true 70%, roughly DOUBLING every pitcher's
    projected pace and matching him against comps a full SGP tier too high. The guard
    below rejects a pitcher panel rather than silently returning that number.
    """
    if "pa" not in df.columns:
        raise ValueError(
            "season_elapsed_fraction needs the HITTER panel: in the pitcher panel "
            "`games` counts appearances, not team games played, and the fraction "
            "comes out roughly half of the truth. Elapsed season is a league fact -- "
            "pass the hitter panel even when pacing a pitcher."
        )
    rows = df[df["season"] == season]
    if rows.empty:
        raise ValueError(f"season {season} is not in the panel")
    if not rows["games"].notna().any():
        # Without this the clip below is a no-op -- min(max(nan, 1e-6), 1.0) is nan,
        # since both comparisons are False -- and the nan surfaces a frame later as
        # "fraction must be in (0, 1]" from prorate_partial, pointing at the pace
        # calculation instead of at the missing games data that actually caused it.
        raise ValueError(
            f"season {season} has no usable `games` values, so the elapsed fraction "
            "cannot be estimated; rebuild the panel with scripts/build_pt_panel.py"
        )
    fraction = float(rows["games"].max()) / FULL_SCHEDULE
    return min(max(fraction, 1e-6), 1.0)


def prorate_partial(sgp: float, fraction: float) -> float:
    """A partial season's SGP at a full-season pace.

    Straight-line: SGP is counting-loaded, so pace scales with playing time. This is
    the "on track to produce 13 SGP" step -- it assumes the rate holds and the player
    stays healthy, which is exactly the assumption the trajectory then prices.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return sgp / fraction
