"""Anchor the in-progress season on YTD actuals plus a rest-of-season projection (#348).

Every other surface on the season dashboard -- projected standings, the Monte Carlo,
SPoE, leverage, waiver and trade evaluation, deltaRoto -- already prices a player off
the ROS blend. The trajectory board was the last thing deriving its own current-season
figure, by dividing season-to-date SGP by the LEAGUE's elapsed fraction. That is right
for a player who has taken every one of his team's games and wrong in a specific way for
one who did not: six weeks on the IL get scaled by 1/0.70 and baked in as his RATE, so
he is priced as a worse player rather than as a healthy one who lost time.

**The YTD half comes from the PANEL, not from `game_log_totals`.** Two definitions of
"2026 so far" exist and they do not agree exactly. Every training row's current season
comes from the panel, so an anchor whose realized half came from Upstash would put a
seam between the query row and the rows it is fitted against -- and it would make the
board unbuildable without a network. Only the REMAINDER is taken from the projection.

**The two halves are only disjoint if the snapshot is current.** The realized half runs
to whenever the panel was built and the remainder was projected from the snapshot date,
so any gap between them is counted twice -- worst for the healthy full-time players who
have the most remaining games. Taking the blend at face value is a deliberate call
(Hart, 2026-08-09) and there is no guard here, which is exactly why this says so: the
alternative is a module that reads as though the combination were disjoint by
construction. `docs/trajectory-ros-anchor-movers-2026-08-09.md` measures it.

**Combine the LINE, never the scores.** `calculate_player_sgp` prices AVG, ERA and WHIP,
which are rates; `ytd_sgp + ros_sgp` is meaningless. The counting stats are added and the
panel's rate x volume schema is rebuilt from the totals -- `panel._reconstruct` run
backwards -- so a combined AVG is the combined hits over the combined at-bats.

**Injection must precede `era.era_normalize`.** Every training row lives on the
2023-2025 reference scale and a raw FanGraphs line lives on the raw current-season run
environment; combining after normalization would put a half-normalized anchor against
fully-normalized comps. The caller is also responsible for deriving the current season's
era factor from the ACTUAL rows -- see `era_factors`' `factors` argument -- because
projections are regressed toward the mean, so league rates computed off injected rows
sit closer to the reference than the season really did.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

#: ``panel_rate_column -> (ROS counting column, what the rate is per)``.
#:
#: The denominator names a counting total in the SAME combined line, which is what makes
#: AVG come out as combined H over combined AB rather than as an average of two averages.
#: `ab_pa` is the structural PA-to-AB ratio the panel stores; `h_ab` IS batting average.
_HITTER_RATES = {
    "ab_pa": ("ab", "pa"),
    "h_ab": ("h", "ab"),
    "hr_pa": ("hr", "pa"),
    "r_pa": ("r", "pa"),
    "rbi_pa": ("rbi", "pa"),
    "sb_pa": ("sb", "pa"),
}

#: Same shape for pitchers. `er_ip * 9` is ERA and `bb_ip + h_ip` is WHIP -- see
#: `panel._reconstruct` -- so both follow from the combined line for free.
_PITCHER_RATES = {
    "k_ip": ("k", "ip"),
    "w_ip": ("w", "ip"),
    "sv_ip": ("sv", "ip"),
    "er_ip": ("er", "ip"),
    "bb_ip": ("bb", "ip"),
    # FanGraphs' hits-allowed column is `h_allowed` once `data/fangraphs.py` has renamed
    # it -- `h` on a pitcher frame is a different stat's name in the hitter map.
    "h_ip": ("h_allowed", "ip"),
}

#: ``kind -> (volume column, rate spec)``. Volume is the panel's own `pa` / `ip`, which
#: is also the ROS frame's column name for the same quantity.
_SPEC = {"hitter": ("pa", _HITTER_RATES), "pitcher": ("ip", _PITCHER_RATES)}


@dataclass(frozen=True)
class RosBlend:
    """One ROS snapshot, blended, with the date it was taken.

    The date is carried rather than re-derived because it is PROVENANCE the board has to
    stamp: an anchor is only as good as the snapshot it came from, the FanGraphs fetch is
    Cloudflare-403 blocked so a snapshot can sit for a while, and a reader comparing this
    board against a fresher one needs to be able to see which vintage each was built on.
    """

    snapshot_date: date
    frames: dict[str, pd.DataFrame]


def load_ros_blend(
    projections_dir: Path,
    season: int,
    systems: list[str],
    weights: dict[str, float] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> RosBlend:
    """Blend the newest rest-of-season snapshot for `season`, in memory.

    THE SAME blend the dashboard runs on -- `blend_projections` with no normalizer, over
    the snapshot `latest_ros_snapshot` picks -- so the board and the standings cannot end
    up pricing one player off two different projections. It stops short of
    `blend_and_cache_ros`, which reads `game_log_totals`, writes two KV blobs, and
    refuses a snapshot older than `ROS_SNAPSHOT_STALE_DAYS`. The first two do not apply:
    this path is offline by construction (the panel it anchors is a gitignored local
    build) and takes its realized half from the panel. The third is a DECISION rather
    than an oversight -- the board takes the newest snapshot on disk at face value and
    stamps its date for the reader (Hart, 2026-08-09). Named here because a list of
    omissions that quietly skips the one with consequences is worse than no list.

    `progress_cb` is the ONLY channel `blend_projections` reports through: a system that
    fails to parse is caught and skipped there, and every projection-quality exclusion is
    announced there. Passing None discards both, so a truncated CSV shifts every anchor
    silently -- which is why all three entry points supply one. It is a PARAMETER rather
    than a `print` in here because this is library code: an importer that is not a
    console script would otherwise inherit both the noise and, on this Windows box, the
    cp1252 exposure the scripts opt out of with `sys.stdout.reconfigure`.

    Raises:
        FileNotFoundError: no datable snapshot dir for `season`.
    """
    from fantasy_baseball.data.projections import blend_projections
    from fantasy_baseball.data.ros_pipeline import latest_ros_snapshot, ros_snapshot_root

    newest = latest_ros_snapshot(projections_dir, season)
    if newest is None:
        raise FileNotFoundError(
            f"no dated rest-of-season snapshot under "
            f"{ros_snapshot_root(projections_dir, season)}, so the {season} anchor has "
            f"no remainder to add. Stage one with scripts/ingest_ros_export.py."
        )
    snapshot, snapshot_date = newest
    hitters, pitchers, _quality = blend_projections(
        snapshot, systems, weights, progress_cb=progress_cb, normalizer=None
    )
    return RosBlend(snapshot_date, {"hitter": hitters, "pitcher": pitchers})


@dataclass(frozen=True)
class AnchoredPanels:
    """Both pools, era-normalized, with the in-progress season anchored. And the receipts.

    `season` is the base season, read off the HITTER panel: `panel_path` picks the newest
    file for each pool INDEPENDENTLY, so a hitter panel rebuilt through 2026 against a
    pitcher panel still ending 2025 produces an empty pitcher pool rather than two boards
    on two different years. Only `push_trajectory_board.py` refuses that outright; the
    CLI board and the single-player lookup simply find nothing for that pool.

    `no_ros` is who was DROPPED for having no rest-of-season row, per pool. It is
    returned rather than logged because the page has to be able to say how many went --
    silence there reads as "the model priced him low".
    """

    season: int
    #: The snapshot the anchor came from, or None when the base season is already
    #: COMPLETE -- a panel rebuilt in the offseason has no remainder to project, so no
    #: snapshot is read and nothing is injected. A reader must show the difference:
    #: "anchored on the 2026-07-21 projection" and "2026 is over" are different boards.
    snapshot_date: date | None
    panels: dict[str, pd.DataFrame]
    no_ros: dict[str, list[int]]

    def comps(self, kind: str) -> pd.DataFrame:
        """`kind`'s FITTING pool: the same panel minus its in-progress season.

        The comp pool must not contain the anchored season -- part of it is a forecast,
        and averaging that into the population a player is matched against would fit the
        model on its own projection.

        DERIVED, never loaded again. A second `load_scored_panel` re-reads a 4.7MB CSV
        and runs another full scoring pass to produce this frame minus its partial rows,
        and it would have to re-anchor to stay consistent with the query. Verified
        identical on both pools: same ids, same seasons, max |sgp diff| 0.0.

        Lives here because all three entry points need it and all three had spelled it
        out with their own justifying comment -- three chances for one rule to drift.
        """
        panel = self.panels[kind]
        return panel[~panel["partial_season"]].reset_index(drop=True)


def load_anchored_panels(
    *,
    systems: list[str],
    weights: dict[str, float] | None = None,
    panel_dir: Path | None = None,
    projections_dir: Path | None = None,
    sgp_overrides=None,
    era_adjust: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> AnchoredPanels:
    """The panels every trajectory surface is built from, in the one order that is right.

    ONE spelling of the sequence, because the ORDER is the requirement (#348) and it is
    silent when wrong. Three entry points need it -- the pushed board, the CLI board and
    the single-player lookup -- and a board whose anchor was injected after normalization,
    or whose base-season era factor came off projected rates, renders exactly like one
    built correctly.

    Both pools are always loaded, even when the caller wants one: the base season is a
    property of the hitter panel, so a pitcher-only board still has to read it.

    `era_adjust=False` returns the anchored panels on their RAW season scales, for
    `player_trajectory.py --no-era-adjust`. It turns off the restatement, never the
    anchor: what a player will finish the season with and what environment that is quoted in are
    separate questions, and the flag only ever meant the second one.
    """
    from .era import era_factors, era_normalize
    from .panel import PROJECT_ROOT, load_scored_panel

    raw = {
        kind: load_scored_panel(
            kind, panel_dir=panel_dir, sgp_overrides=sgp_overrides, include_partial=True
        )
        for kind in _SPEC
    }
    season = int(raw["hitter"]["season"].max())
    # A COMPLETE base season needs no anchor and must not read a snapshot: in the
    # offseason the newest snapshot belongs to the season that just finished, and adding
    # it would bolt a remainder onto a year that has none. Decided on the hitter panel,
    # the same frame the base season itself comes from.
    in_progress = bool(raw["hitter"].loc[raw["hitter"]["season"] == season, "partial_season"].any())
    blend = (
        load_ros_blend(
            projections_dir
            if projections_dir is not None
            else PROJECT_ROOT / "data" / "projections",
            season,
            systems,
            weights,
            progress_cb=progress_cb,
        )
        if in_progress
        else None
    )

    panels: dict[str, pd.DataFrame] = {}
    no_ros: dict[str, list[int]] = {}
    for kind, panel in raw.items():
        # BEFORE the injection. `era_normalize` derives each season's factor from the
        # league rates of the rows it is handed, and the injected rows are part
        # projection -- see its `factors` argument. Computed on both anchor branches so
        # the offseason path cannot quietly become a second definition of the table.
        factors = era_factors(panel, kind) if era_adjust else None
        anchored, no_ros[kind] = (
            anchor_full_season(
                panel,
                blend.frames[kind],
                kind=kind,
                season=season,
                sgp_overrides=sgp_overrides,
            )
            if blend is not None
            else (panel, [])
        )
        panels[kind] = (
            era_normalize(anchored, kind, sgp_overrides=sgp_overrides, factors=factors)
            if era_adjust
            else anchored
        )
    return AnchoredPanels(
        season, blend.snapshot_date if blend is not None else None, panels, no_ros
    )


def _ros_by_id(ros: pd.DataFrame, columns: list[str], kind: str) -> pd.DataFrame:
    """The ROS frame indexed by MLBAM id, with the counting columns the spec needs.

    ID-KEYED, and there is no name fallback by design. The FanGraphs exports carry
    `MLBAMID`, `data/fangraphs.py` renames it to `mlbam_id` at load and the blend carries
    it through, and coverage is effectively total (4747/4748 hitter rows, 749/749 pitcher
    rows on the 2026-07-21 snapshot). A normalized name is not unique -- 58 hitters in
    this panel share one with somebody else -- so a name fallback would buy a handful of
    joins at the price of silently anchoring a player on a namesake's projection.
    """
    missing = [c for c in columns if c not in ros.columns]
    if missing:
        raise KeyError(
            f"the {kind} ROS blend is missing {missing}, so a full-season line cannot be "
            f"built from it. The FanGraphs export schema has probably shifted -- check "
            f"the column maps in fantasy_baseball/data/fangraphs.py."
        )
    if "mlbam_id" not in ros.columns:
        raise KeyError(
            f"the {kind} ROS blend carries no mlbam_id, so it can only be joined by "
            f"name -- which is not unique. Check that the export includes MLBAMID."
        )
    frame = ros[ros["mlbam_id"].notna()].copy()
    frame["mlbam_id"] = frame["mlbam_id"].astype(int)
    duplicated = sorted(set(frame.loc[frame["mlbam_id"].duplicated(), "mlbam_id"]))
    if duplicated:
        raise ValueError(
            f"two ROS rows share one mlbam_id in the {kind} blend: {duplicated[:5]}. "
            f"Adding both would inflate the remainder by a whole projection, and every "
            f"number downstream would still render normally."
        )
    return frame.set_index("mlbam_id")[columns].astype(float)


def anchor_full_season(
    panel: pd.DataFrame,
    ros: pd.DataFrame,
    *,
    kind: str,
    season: int,
    sgp_overrides=None,
) -> tuple[pd.DataFrame, list[int]]:
    """Replace `season`'s rows with a YTD + ROS full-season line. Returns the panel and
    the ids that had no ROS row.

    The returned frame is on the panel's OWN schema and is FULLY SCORED: the rates carry
    the combined line, and `panel.score` is re-run over the rows that moved so the
    reconstructed counting columns and `sgp` agree with them. Without that the frame
    would leave `ab`, `hr`, `era` and `sgp` describing the season-to-date line while `pa`
    and the rates described the full one -- and `era.league_rates` weights by `ab`, so a
    factor table built off this frame would be weighted by the wrong volumes. The real
    path re-scores anyway inside `era_normalize`; this makes the frame honest for anyone
    who does not, at the cost of one scoring pass over a few hundred rows.

    WHAT IS NOT TOUCHED, and why each one would be a defect:

    * `games` / `starts`. Appearances are a calendar and role fact about what HAPPENED.
      `season_elapsed_fraction` dates the league off the busiest hitter's games, and
      `_pitcher_slots` reads `starts / games` from a SETTLED season to choose between the
      SP and RP floors. Writing a projection into either would read the season as
      complete and let a forecast pick a replacement level.
    * `partial_season`. The comp pool is `~partial_season`; an anchored row that lost the
      flag would be fitted AGAINST as though it were a realized career year.
    * every other season. They are the training data, and they are already on the
      reference era scale.

    Args:
        panel: A scored panel including the in-progress season.
        ros: The blended rest-of-season frame for the same pool.
        kind: ``"hitter"`` or ``"pitcher"``.
        season: The in-progress season to anchor.

    Raises:
        KeyError: the ROS frame is missing a column the line needs.
        ValueError: `kind` is unknown, one id has two ROS rows, or the panel carries a
            split (traded) in-progress season.
    """
    if kind not in _SPEC:
        raise ValueError(f"kind must be 'hitter' or 'pitcher', got {kind!r}")
    volume, rates = _SPEC[kind]

    # IN-PROGRESS rows only, not simply "the base season". `build_pt_panel._live_seasons`
    # flags a season partial iff `year >= today.year`, so a panel rebuilt in January
    # un-flags the season that just ended -- and that season is still the newest one in
    # the file. Adding a remainder to a season that HAS no remainder would bolt a whole
    # projection onto a complete year, and the snapshot it came from would be last
    # season's. The flag is the only thing that knows the difference.
    target = (panel["season"] == season) & panel["partial_season"].astype(bool)
    current = panel[target]
    if current.empty:
        return panel.copy(), []
    split = sorted(set(current.loc[current["mlbam_id"].duplicated(), "mlbam_id"].astype(int)))
    if split:
        raise ValueError(
            f"the {kind} panel has a split (traded) {season} season for {split[:5]}, so "
            f"the whole remaining-season projection would be added to EACH half. Collapse "
            f"the season before anchoring it."
        )

    needed = [volume] + [numerator for numerator, _ in rates.values()]
    # The denominators are counting totals in the same line, so anything a rate divides
    # by has to be reconstructable too -- `h_ab` needs `ab`, which is itself `pa * ab_pa`.
    remainder = _ros_by_id(ros, sorted(set(needed)), kind)

    ids = current["mlbam_id"].astype(int)
    matched = ids.isin(remainder.index)
    dropped = sorted(set(ids[~matched]))

    kept = current[matched.to_numpy()].copy()
    if kept.empty:
        return panel[~target].copy(), dropped
    add = remainder.loc[kept["mlbam_id"].astype(int)]

    # AFTER the join, over the rows that will actually be added. A NaN does NOT surface
    # as a NaN score: measured, a NaN `hr` produced a 9.34 SGP row whose home runs
    # contributed nothing -- an ordinary-looking number that clears every downstream
    # gate, which is the failure this repo ranks worst.
    #
    # Checked HERE and not over the whole blended frame. That frame is the entire
    # FanGraphs export (~4,748 hitter rows against ~600 anchored players), so a guard
    # across all of it let one malformed row for a deep-minors player nobody prices
    # abort the whole board build, with no override -- and the join two lines up would
    # have discarded that row anyway.
    holes = add.columns[add.isna().any()].tolist()
    if holes:
        who = add.index[add[holes].isna().any(axis=1)].tolist()
        raise ValueError(
            f"the {kind} ROS blend has NaN in {holes} for {len(who)} anchored player(s) "
            f"(e.g. mlbam {who[:5]}), so their full-season line would score as though "
            f"those categories were zero -- and it would look like an ordinary number. "
            f"Re-stage the snapshot rather than blending a partial one."
        )

    # The realized half, reconstructed from what the panel stores. `_reconstruct` does
    # exactly this for scoring; it is spelled again here because that one writes display
    # columns onto the frame and this needs the totals as values.
    #
    # ORDER-DEPENDENT, deliberately: `h_ab` divides by `ab`, which `ab_pa` produces one
    # entry earlier. The spec dicts are ordered so every denominator is already built,
    # and the guard says so rather than letting a reordering raise a bare KeyError.
    ytd: dict[str, np.ndarray] = {volume: kept[volume].astype(float).to_numpy()}
    for rate, (numerator, denominator) in rates.items():
        if denominator not in ytd:
            raise ValueError(
                f"{kind} rate {rate!r} is per {denominator!r}, which no earlier entry "
                f"builds. Reorder the spec so a denominator precedes what divides by it."
            )
        ytd[numerator] = ytd[denominator] * kept[rate].astype(float).to_numpy()

    combined = {stat: ytd[stat] + add[stat].to_numpy() for stat in ytd}

    kept[volume] = combined[volume]
    for rate, (numerator, denominator) in rates.items():
        below = combined[denominator]
        # A zero denominator is unreachable on real data (the panel keeps only volume > 0
        # rows and the projection floors at 1 PA / 1 IP) but the ratio is not guarded
        # anywhere downstream: an inf rate would multiply straight into a counting line
        # and score as a career year. 0.0 is the honest value for "no at-bats, no hits".
        kept[rate] = np.divide(
            combined[numerator], below, out=np.zeros_like(below, dtype=float), where=below > 0
        )

    from .panel import score

    # Row order preserved: `kept` holds the panel's own index, so sorting on it puts each
    # anchored row back where it was rather than moving the whole season to the end.
    return pd.concat([panel[~target], score(kept, kind, sgp_overrides)]).sort_index(), dropped
