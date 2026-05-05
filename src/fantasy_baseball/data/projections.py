import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.roster import Roster
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher

from .fangraphs import _find_file, load_projection_set

if TYPE_CHECKING:
    from fantasy_baseball.data.projection_quality import QualityReport

logger = logging.getLogger(__name__)

# Counting stats to blend directly (weighted average)
HITTING_COUNTING_COLS: list[str] = ["r", "hr", "rbi", "sb", "h", "ab", "pa"]
PITCHING_COUNTING_COLS: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed", "gs"]


def normalize_rest_of_season_to_full_season(
    df: pd.DataFrame,
    game_log_totals: dict[int, dict],
    player_type: str,
) -> pd.DataFrame:
    """Convert remaining-games ROS projections to full-season totals.

    All FanGraphs ROS exports (steamer, the-bat-x, zips, atc, oopsy) publish
    remaining-games-only projections — verified empirically on 2026-04-10 by
    comparing PA values across snapshots:

        system     Apr 1 PA   Apr 10 PA   delta
        zips       604        579         -25
        steamer    633        607         -26
        atc        624        596         -28
        the-bat-x  638        603         -35
        oopsy      624        596         -28

    All five systems decreased by roughly 9 games' worth of PA over the 9
    elapsed games (Yankees played ~9 games during that window), proving they
    are all rest-of-season-only.

    For each player with a matching mlbam_id in game_log_totals, adds the
    actual season-to-date counting stats to the ROS counting stats so the
    result represents a full-season projection. Rate stats (AVG, ERA, WHIP)
    are NOT touched here — they get recomputed downstream from blended
    counting components in _blend_hitters / _blend_pitchers.

    Players without a matching mlbam_id are left unchanged. This affects
    prospects and recent callups whose mlbam_id wasn't in our roster
    or wasn't matched at projection-load time.

    Returns a new DataFrame (does not mutate the input).
    """
    if not game_log_totals or "mlbam_id" not in df.columns:
        return df.copy()

    result = df.copy()
    counting_cols = (
        HITTING_COUNTING_COLS if player_type == PlayerType.HITTER else PITCHING_COUNTING_COLS
    )

    # Coerce counting columns to float64 BEFORE adding actuals. Some
    # projection systems publish whole-number IP (zips, atc) so pandas
    # infers int64 from those columns; game logs sum IP as fractional
    # thirds (e.g. 180.6667 = 180⅔), and pandas 2.x+ refuses to write a
    # float into an int64 column. Casting upfront avoids the TypeError.
    for col in counting_cols:
        if col in result.columns:
            result[col] = result[col].astype("float64")

    for idx, row in result.iterrows():
        mid = row.get("mlbam_id")
        if pd.isna(mid):
            continue
        mid = int(mid)
        actuals = game_log_totals.get(mid)
        if actuals is None:
            continue
        for col in counting_cols:
            if col in result.columns and col in actuals:
                result.at[idx, col] = row[col] + actuals[col]

    return result


def validate_projections_dir(projections_dir: Path, systems: list[str]) -> None:
    """Validate that the projections directory exists and contains expected CSV files.

    Raises FileNotFoundError with an actionable message if the directory is
    missing or if no projection files can be found for the requested systems.
    """
    if not projections_dir.exists():
        raise FileNotFoundError(
            f"Projections directory not found: {projections_dir}\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Create the directory: mkdir -p {projections_dir}\n"
            f"  2. Download projection CSVs from FanGraphs:\n"
            f"     https://www.fangraphs.com/projections\n"
            f"  3. Export hitter and pitcher CSVs for each system ({', '.join(systems)})\n"
            f"  4. Save them as e.g. steamer-hitters.csv, steamer-pitchers.csv"
        )

    if not projections_dir.is_dir():
        raise FileNotFoundError(
            f"Projections path exists but is not a directory: {projections_dir}"
        )

    csv_files = list(projections_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {projections_dir}\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Download projection CSVs from FanGraphs:\n"
            f"     https://www.fangraphs.com/projections\n"
            f"  2. Export hitter and pitcher CSVs for each system ({', '.join(systems)})\n"
            f"  3. Save them as e.g. steamer-hitters.csv, steamer-pitchers.csv"
        )

    # Check each requested system has at least one file (hitters or pitchers)
    missing_systems = []
    for system in systems:
        hit_file = _find_file(projections_dir, system, "hitters")
        pit_file = _find_file(projections_dir, system, "pitchers")
        if hit_file is None and pit_file is None:
            missing_systems.append(system)

    if missing_systems:
        found_files = [f.name for f in csv_files]
        raise FileNotFoundError(
            f"No projection files found for system(s): {', '.join(missing_systems)}\n"
            f"\n"
            f"Directory {projections_dir} contains: {', '.join(found_files)}\n"
            f"\n"
            f"Expected files like:\n"
            + "\n".join(f"  - {s}-hitters.csv / {s}-pitchers.csv" for s in missing_systems)
            + f"\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Download the missing projections from FanGraphs:\n"
            f"     https://www.fangraphs.com/projections\n"
            f"  2. Select each system and export hitter + pitcher CSVs\n"
            f"  3. Save them in {projections_dir}"
        )


def blend_projections(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
    roster_names: set[str] | None = None,
    progress_cb=None,
    normalizer=None,
) -> tuple[pd.DataFrame, pd.DataFrame, "QualityReport | None"]:
    """Blend multiple projection systems into weighted averages.

    Counting stats are blended directly. Rate stats (AVG, ERA, WHIP)
    are recomputed from blended component stats.

    Runs pre-blend quality checks when 2+ systems are loaded. Excludes
    stat columns flagged as outliers (e.g., a system with all-zero SV).

    Returns (hitters_df, pitchers_df, quality_report). quality_report is
    None if fewer than 2 systems were loaded.
    """
    from fantasy_baseball.data.projection_quality import check_projection_quality

    validate_projections_dir(projections_dir, systems)

    if weights is None:
        weights = {s: 1.0 / len(systems) for s in systems}

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Load all systems
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_hitters: list[pd.DataFrame] = []
    all_pitchers: list[pd.DataFrame] = []

    for system in systems:
        if progress_cb:
            progress_cb(f"Loading {system} from {projections_dir.name}")
        try:
            hitters, pitchers = load_projection_set(projections_dir, system)
        except Exception as exc:
            import traceback

            if progress_cb:
                progress_cb(f"ERROR loading {system}: {type(exc).__name__}: {exc}")
                progress_cb(f"ERROR {system} traceback: {traceback.format_exc().splitlines()[-5:]}")
            continue
        if normalizer is not None:
            try:
                hitters, pitchers = normalizer(system, hitters, pitchers)
            except Exception as exc:
                import traceback

                if progress_cb:
                    progress_cb(f"ERROR normalizing {system}: {type(exc).__name__}: {exc}")
                    progress_cb(
                        f"ERROR {system} traceback: {traceback.format_exc().splitlines()[-5:]}"
                    )
                continue
        system_dfs[system] = (hitters, pitchers)
        w = weights.get(system, 0)
        if not hitters.empty:
            hitters = hitters.copy()
            hitters["_weight"] = w
            hitters["_system"] = system
            all_hitters.append(hitters)
        if not pitchers.empty:
            pitchers = pitchers.copy()
            pitchers["_weight"] = w
            pitchers["_system"] = system
            all_pitchers.append(pitchers)

    # Run quality checks
    report = None
    if len(system_dfs) >= 2:
        report = check_projection_quality(system_dfs, roster_names)
        if progress_cb:
            for warning in report.warnings:
                progress_cb(f"QUALITY: {warning}")

        # Apply exclusions: zero out excluded stat columns so they don't contribute
        if report.exclusions:
            for df_list, stat_source in [
                (all_hitters, HITTING_COUNTING_COLS),
                (all_pitchers, PITCHING_COUNTING_COLS),
            ]:
                for df in df_list:
                    df_system = df["_system"].iloc[0] if not df.empty else None
                    if df_system and df_system in report.exclusions:
                        excluded = report.exclusions[df_system]
                        for stat in excluded:
                            if stat in df.columns and stat in stat_source:
                                df[stat] = float("nan")

    # Clean up _system column before blending
    for df in all_hitters + all_pitchers:
        if "_system" in df.columns:
            df.drop(columns=["_system"], inplace=True)

    blended_hitters = _blend_hitters(all_hitters)
    blended_pitchers = _blend_pitchers(all_pitchers)
    return blended_hitters, blended_pitchers, report


def _blend_players(
    dfs: list[pd.DataFrame],
    counting_cols: list[str],
    player_type: str,
) -> pd.DataFrame:
    """Vectorized projection blending shared by hitters and pitchers.

    Instead of iterating per group with pandas Series arithmetic,
    this pre-multiplies all counting stats by their normalized weight
    and uses a single groupby().sum() to aggregate.
    """
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Group by fg_id when available (robust against name variations
    # across systems, e.g. accented vs ASCII). Fall back to name.
    group_col = (
        "fg_id" if "fg_id" in combined.columns and combined["fg_id"].notna().all() else "name"
    )

    # Pre-multiply counting stats by normalized weight, then sum per group.
    # Per-stat NaN-aware weight normalization: if a system has NaN for a stat,
    # its weight is excluded from that stat's denominator so other systems'
    # values are not diluted. This correctly handles quality-check exclusions
    # where a bad system's stat column is NaN'd before blending.
    stat_cols = [c for c in counting_cols if c in combined.columns]
    groups = combined[group_col].values
    weights_arr = combined["_weight"].values

    weighted_parts = {}
    for stat in stat_cols:
        vals = combined[stat].values.astype(float)
        nan_mask = np.isnan(vals)
        # Per-row effective weight: 0 where stat is NaN
        eff_w = weights_arr.copy()
        eff_w[nan_mask] = 0.0
        # Normalize within each group based on non-NaN weight sum
        eff_w_series = pd.Series(eff_w, name="_eff_w")
        group_eff_w_sum = eff_w_series.groupby(groups).transform("sum").values
        denom = np.where(group_eff_w_sum > 0, group_eff_w_sum, 1.0)
        nw = np.where(group_eff_w_sum > 0, eff_w / denom, 0.0)
        weighted_parts[stat] = np.where(nan_mask, 0.0, vals) * nw

    weighted = pd.DataFrame(weighted_parts)
    weighted[group_col] = groups
    result = weighted.groupby(group_col, sort=False)[stat_cols].sum()

    # Metadata from highest-weight row per group
    idx_max = combined.groupby(group_col)["_weight"].idxmax()
    meta = combined.loc[idx_max].set_index(group_col)

    result["name"] = meta.index if group_col == "name" else meta["name"]
    result["player_type"] = player_type
    if "team" in combined.columns:
        result["team"] = meta["team"]
    if "fg_id" in combined.columns and group_col != "fg_id":
        result["fg_id"] = combined.groupby(group_col)["fg_id"].first()
    # Carry mlbam_id through as identity metadata (used by post-blend
    # YTD normalization in ros_pipeline). first() picks any non-null
    # value; mlbam_id is invariant per player across systems, so the
    # tiebreak doesn't matter.
    if "mlbam_id" in combined.columns:
        result["mlbam_id"] = combined.groupby(group_col)["mlbam_id"].first()

    # ADP: weighted average of non-null values only
    # Use original group-normalized weights (not per-stat NaN-aware weights)
    if "adp" in combined.columns:
        group_w_sum = pd.Series(weights_arr).groupby(groups).transform("sum").values
        base_nw = np.where(group_w_sum > 0, weights_arr / group_w_sum, 0.0)
        adp_vals = combined["adp"].values.astype(float).copy()
        adp_nw = base_nw.copy()
        mask = np.isnan(adp_vals)
        adp_vals[mask] = 0.0
        adp_nw[mask] = 0.0

        adp_df = pd.DataFrame(
            {
                group_col: combined[group_col].values,
                "_aw": adp_vals * adp_nw,
                "_nw": adp_nw,
            }
        )
        adp_agg = adp_df.groupby(group_col).sum()
        adp_result = adp_agg["_aw"] / adp_agg["_nw"]
        result["adp"] = adp_result.replace([np.inf, -np.inf, np.nan], float("inf"))

    return result.reset_index()


def _blend_hitters(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend hitter projections. Recomputes AVG from blended H and AB."""
    result = _blend_players(dfs, HITTING_COUNTING_COLS, PlayerType.HITTER)
    if result.empty:
        return result
    result["avg"] = np.where(result["ab"] > 0, result["h"] / result["ab"], 0.0)
    return result


def _blend_pitchers(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend pitcher projections. Recomputes ERA and WHIP from components."""
    result = _blend_players(dfs, PITCHING_COUNTING_COLS, PlayerType.PITCHER)
    if result.empty:
        return result
    ip = result["ip"]
    result["era"] = np.where(ip > 0, result["er"] * 9 / ip, 0.0)
    result["whip"] = np.where(ip > 0, (result["bb"] + result["h_allowed"]) / ip, 0.0)
    return result


def _build_full_season_index(
    full_df: pd.DataFrame | None,
) -> tuple[dict[int, dict], dict[str, dict]]:
    """Index a full-season projections frame for O(1) per-player lookup.

    Returns ``(by_mlbam, by_namenorm)`` — record dicts keyed first by
    ``mlbam_id`` (identity, robust to accent/encoding differences)
    and second by ``_name_norm`` (fallback for rows missing an id).
    Either dict may be empty when the corresponding column is missing
    or the frame is None/empty.
    """
    if full_df is None or full_df.empty:
        return {}, {}
    by_mlbam: dict[int, dict] = {}
    if "mlbam_id" in full_df.columns:
        for record in full_df.to_dict(orient="records"):
            mid = record.get("mlbam_id")
            if mid is None or pd.isna(mid):
                continue
            by_mlbam[int(mid)] = record
    by_namenorm: dict[str, dict] = {}
    if "_name_norm" in full_df.columns:
        for record in full_df.to_dict(orient="records"):
            nn = record.get("_name_norm")
            if nn:
                by_namenorm.setdefault(nn, record)
    return by_mlbam, by_namenorm


def _lookup_full_season_record(
    proj_row: pd.Series,
    name_norm: str,
    by_mlbam: dict[int, dict],
    by_namenorm: dict[str, dict],
) -> dict | None:
    """Look up a player's full-season (ROS+YTD) record using prebuilt indices.

    Prefers ``mlbam_id`` (identity, immune to name encoding) and falls
    back to ``_name_norm``. Returns the matched record dict or ``None``.
    """
    mlbam_id = proj_row.get("mlbam_id") if "mlbam_id" in proj_row.index else None
    if mlbam_id is not None and not pd.isna(mlbam_id):
        record = by_mlbam.get(int(mlbam_id))
        if record is not None:
            return record
    return by_namenorm.get(name_norm)


def match_roster_to_projections(
    roster: list[dict],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    *,
    full_hitters_proj: pd.DataFrame | None = None,
    full_pitchers_proj: pd.DataFrame | None = None,
    context: str = "",
) -> list[Player]:
    """Match roster players to blended projections by normalized name.

    Expects ``_name_norm`` column precomputed on both DataFrames
    (call ``df["_name_norm"] = df["name"].apply(normalize_name)`` first).

    Returns a list of :class:`Player` objects with ``.rest_of_season`` populated as
    :class:`HitterStats` or :class:`PitcherStats`. Unmatched players are
    omitted.

    Emits ``WARNING`` logs for three matching anomalies so silent failures
    surface in the refresh log:

    - Unmatched roster player (no projection found in either DataFrame)
    - Ambiguous match (multiple projection rows share a normalized name)
    - Fallback match (positions did not disambiguate hitter vs pitcher)

    The ``context`` kwarg is included in log messages as a ``[context]``
    prefix to identify which call site produced the warning (e.g.
    ``"user"``, ``"opp:Sharks"``, ``"preseason"``, ``"ros"``).

    ``hitters_proj``/``pitchers_proj`` carry ROS-only counting stats
    (per ``cache:ros_projections`` after the ROS-only-decision-projections
    fix). They populate ``Player.rest_of_season`` and drive every
    forward-looking decision (lineup, waivers, trades, transactions).

    ``full_hitters_proj``/``full_pitchers_proj`` are the optional
    ROS+YTD blob (per ``cache:full_season_projections``). When provided,
    each player's full-season counterpart row is looked up by ``mlbam_id``
    (preferred) or ``_name_norm`` and assigned to
    ``Player.full_season_projection`` — used for display and end-of-season
    projected standings. Defaults to ``None``; legacy callers keep
    working with ROS-only frames.
    """
    prefix = f"[{context}] " if context else ""
    matched: list[Player] = []
    full_hitters_by_id, full_hitters_by_name = _build_full_season_index(full_hitters_proj)
    full_pitchers_by_id, full_pitchers_by_name = _build_full_season_index(full_pitchers_proj)
    for player in roster:
        name = player["name"].replace(" (Batter)", "").replace(" (Pitcher)", "")
        name_norm = normalize_name(name)
        positions = player.get("positions", [])

        proj = None
        ptype = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["_name_norm"] == name_norm]
            if not matches.empty:
                if len(matches) > 1:
                    logger.warning(
                        "%sambiguous hitter match for %r — %d candidates, picked first",
                        prefix,
                        name,
                        len(matches),
                    )
                proj = matches.iloc[0]
                ptype = PlayerType.HITTER
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["_name_norm"] == name_norm]
            if not matches.empty:
                if len(matches) > 1:
                    logger.warning(
                        "%sambiguous pitcher match for %r — %d candidates, picked first",
                        prefix,
                        name,
                        len(matches),
                    )
                proj = matches.iloc[0]
                ptype = PlayerType.PITCHER
        if proj is None:
            for df, pt in [(hitters_proj, PlayerType.HITTER), (pitchers_proj, PlayerType.PITCHER)]:
                if df.empty:
                    continue
                matches = df[df["_name_norm"] == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    logger.warning(
                        "%s%r matched via fallback branch — positions=%r did not disambiguate",
                        prefix,
                        name,
                        positions,
                    )
                    break

        if proj is None or ptype is None:
            logger.warning(
                "%sno projection match for %r (positions=%r)",
                prefix,
                name,
                positions,
            )
            continue

        ros: HitterStats | PitcherStats
        if ptype == PlayerType.HITTER:
            ros = HitterStats.from_dict(proj.to_dict())
        else:
            ros = PitcherStats.from_dict(proj.to_dict())

        # Look up matching full-season (ROS+YTD) record if indices were built.
        if ptype == PlayerType.HITTER:
            full_record = _lookup_full_season_record(
                proj, name_norm, full_hitters_by_id, full_hitters_by_name
            )
        else:
            full_record = _lookup_full_season_record(
                proj, name_norm, full_pitchers_by_id, full_pitchers_by_name
            )

        full_season_stats: HitterStats | PitcherStats | None = None
        if full_record is not None:
            if ptype == PlayerType.HITTER:
                full_season_stats = HitterStats.from_dict(full_record)
            else:
                full_season_stats = PitcherStats.from_dict(full_record)

        # Parse positions and selected_position explicitly
        parsed_positions = [p if isinstance(p, Position) else Position.parse(p) for p in positions]
        raw_slot = player.get("selected_position", "")
        if raw_slot is None or raw_slot == "":
            parsed_slot = None
        elif isinstance(raw_slot, Position):
            parsed_slot = raw_slot
        else:
            parsed_slot = Position.parse(raw_slot)

        p = Player(
            name=name,
            player_type=ptype,
            positions=parsed_positions,
            yahoo_id=player.get("player_id", ""),
            selected_position=parsed_slot,
            status=player.get("status", ""),
            rest_of_season=ros,
            full_season_projection=full_season_stats,
        )
        matched.append(p)

    return matched


def hydrate_roster_entries(
    roster: Roster,
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    *,
    full_hitters_proj: pd.DataFrame | None = None,
    full_pitchers_proj: pd.DataFrame | None = None,
    context: str = "",
) -> list[Player]:
    """Convert a :class:`Roster`'s entries into ``list[Player]`` with
    projection stats populated.

    Thin adapter around :func:`match_roster_to_projections`: converts
    each :class:`RosterEntry` into the dict shape the legacy matcher
    expects, then delegates so every edge case (name normalization,
    accent handling, "(Batter)"/"(Pitcher)" suffix stripping, position
    collisions) is preserved for free.

    Unmatched entries are omitted, matching
    :func:`match_roster_to_projections`'s contract.

    The ``full_hitters_proj``/``full_pitchers_proj`` kwargs forward the
    optional full-season (ROS+YTD) frames into
    :func:`match_roster_to_projections` so each Player's
    ``.full_season_projection`` is populated for display + standings.

    The ``context`` kwarg is forwarded for log clarity.
    """
    roster_dicts = [
        {
            "name": entry.name,
            "positions": [p.value for p in entry.positions],
            "selected_position": entry.selected_position.value,
            "status": entry.status,
            "player_id": entry.yahoo_id,
        }
        for entry in roster.entries
    ]
    return match_roster_to_projections(
        roster_dicts,
        hitters_proj,
        pitchers_proj,
        full_hitters_proj=full_hitters_proj,
        full_pitchers_proj=full_pitchers_proj,
        context=context,
    )
