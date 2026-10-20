"""The keeper board: every rostered player's projected next-season SGP and VAR.

Pure and I/O-free (``scripts/keeper_report.py`` loads the inputs and prints the
report). A player's season line is our nets' projected rates times the PT source's
playing time (a projection file: FanGraphs Depth Charts or Steamer once out). SGP is
that line in standings points (``hitter_ros.evaluate.fantasy_value``,
``pitcher_ros.evaluate.fantasy_value``); VAR is SGP over the replacement at his best
position, the first player past the league's starting slots there in the whole
projected pool (``sgp.replacement`` / ``sgp.var``: the 11th C, 41st OF, 91st P, ...).

Rostered players are matched to the pool by normalized name within player type
(rosters carry no MLBAM id). A name two players share is settled by MLB team; one
still ambiguous, or with no projection, is reported rather than guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.evaluate import fantasy_value as hitter_value
from fantasy_baseball.pitcher_ros.evaluate import fantasy_value as pitcher_value
from fantasy_baseball.pt_model.range_specs import STARTER_SHARE
from fantasy_baseball.sgp.replacement import find_replacement_players
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.name_utils import normalize_name

HITTER = "hitter"
PITCHER = "pitcher"
HIT_SLOTS = ("C", "1B", "2B", "3B", "SS", "OF")
# A hitter with no known slot is an outfielder (the draft board's default).
DEFAULT_HIT_SLOT = "OF"
# A reliever's innings are capped here: ZiPS gave some relievers starter innings,
# which times a closer's save rate made 50-save seasons (Varland: 104 IP, 51 SV).
RP_IP_CAP = 70.0
HITTER_STATS = ("r", "hr", "rbi", "sb", "avg")
PITCHER_STATS = ("k", "w", "sv", "era", "whip")


@dataclass(frozen=True)
class Rostered:
    """One rostered player as the league's rosters list him."""

    team: str
    name: str
    player_type: str
    positions: tuple[str, ...]
    mlb_team: str


def hitter_board(
    rates: pd.DataFrame,
    pt: pd.DataFrame,
    positions: Mapping[int, Sequence[str]],
    denoms: Mapping[Category, float],
) -> pd.DataFrame:
    """Hitters with a projected line, by MLBAM id: ``rates`` (r, hr, rbi, sb per PA, avg)
    times ``pt``'s ``pa``/``ab``; ``positions`` their eligible slots (hitting slots
    only, ``DEFAULT_HIT_SLOT`` if none). Columns: kind, positions, pa, the stats as
    season totals (avg as a rate), sgp."""
    ids = rates.index.intersection(pt.index[pt["pa"] > 0])
    r, p = rates.loc[ids], pt.loc[ids]
    line = r[list(HITTER_STATS)].assign(pa=p["pa"].astype(float), ab=p["ab"].astype(float))
    sgp = hitter_value(line, dict(denoms))
    out = pd.DataFrame(
        {
            "kind": HITTER,
            "positions": [
                [s for s in positions.get(i, ()) if s in HIT_SLOTS] or [DEFAULT_HIT_SLOT]
                for i in ids
            ],
            "pa": line["pa"],
            **{c: line[c] * line["pa"] for c in ("r", "hr", "rbi", "sb")},
            "avg": line["avg"],
            "sgp": sgp,
        },
        index=ids,
    )
    return out[out["sgp"].notna()]


def pitcher_board(
    rates: pd.DataFrame, pt: pd.DataFrame, denoms: Mapping[Category, float]
) -> pd.DataFrame:
    """Pitchers with a projected line, by MLBAM id: ``rates`` (k, w, sv per batter
    faced, era, whip) over ``pt``'s ``ip``/``bf``, a reliever's (``gs_share`` under
    ``STARTER_SHARE``) capped at ``RP_IP_CAP`` innings. Columns: kind, role, positions,
    ip, the stats as season totals (era, whip as rates), sgp."""
    ok = (pt["ip"] > 0) & (pt["bf"] > 0) & pt["gs_share"].notna()
    ids = rates.index.intersection(pt.index[ok])
    r, p = rates.loc[ids], pt.loc[ids]
    starter = p["gs_share"] >= STARTER_SHARE
    scale = np.where(~starter & (p["ip"] > RP_IP_CAP), RP_IP_CAP / p["ip"], 1.0)
    line = r[list(PITCHER_STATS)].assign(ip=p["ip"] * scale, bf=p["bf"] * scale)
    sgp = pitcher_value(line, dict(denoms))
    out = pd.DataFrame(
        {
            "kind": PITCHER,
            "role": np.where(starter, "SP", "RP"),
            "positions": [["P"]] * len(ids),
            "ip": line["ip"],
            **{c: line[c] * line["bf"] for c in ("k", "w", "sv")},
            "era": line["era"],
            "whip": line["whip"],
            "sgp": sgp,
        },
        index=ids,
    )
    return out[out["sgp"].notna()]


def replacement(
    board: pd.DataFrame, starters: Mapping[str, int]
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """The replacement SGP per position from this pool (both kinds), and the
    replacement player's row there."""
    repl = find_replacement_players(board.rename(columns={"sgp": "total_sgp"}), dict(starters))
    return {pos: float(row["total_sgp"]) for pos, row in repl.items()}, repl


def var(
    sgp: Iterable[float], positions: Iterable[Sequence[str]], levels: Mapping[str, float]
) -> list[float]:
    """Each player's VAR: SGP over the replacement at his best eligible position."""
    return [
        calculate_var(pd.Series({"total_sgp": v, "positions": list(ps)}), dict(levels))
        for v, ps in zip(sgp, positions, strict=True)
    ]


def match_rostered(
    rostered: Iterable[Rostered],
    board: pd.DataFrame,
    names: Mapping[int, str],
    mlb_teams: Mapping[int, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Each rostered player's board row (by normalized name within player type, a
    shared name settled by ``mlb_teams``), with his ``team`` and roster ``positions``
    (VAR against his own eligible slots). Rows that can't be matched are left out
    and described in the returned problems. ``board`` is keyed by MLBAM id, which a
    two-way player has twice (a hitter row and a pitcher row), so rows are found by
    position."""
    ids = board.index.to_numpy()
    by_name: dict[tuple[str, str], list[int]] = {}
    for k, (pid, kind) in enumerate(zip(ids, board["kind"], strict=True)):
        by_name.setdefault((normalize_name(names.get(pid, "")), kind), []).append(k)
    rows, problems = [], []
    for p in rostered:
        hits = by_name.get((normalize_name(p.name), p.player_type), [])
        if len(hits) > 1:
            same = [k for k in hits if mlb_teams.get(ids[k]) == p.mlb_team]
            if len(same) != 1:
                problems.append(
                    f"{p.name} ({p.team}, {p.mlb_team}): {len(hits)} {p.player_type}s by that "
                    "name, no unique MLB-team match; left out"
                )
                continue
            hits = same
        if not hits:
            problems.append(f"{p.name} ({p.team}, {p.player_type}): no projection; left out")
            continue
        row = board.iloc[hits[0]].to_dict()
        if p.player_type == HITTER:
            own = [s for s in p.positions if s in HIT_SLOTS]
            row["positions"] = own or row["positions"]
        rows.append({"player_id": ids[hits[0]], **row, "team": p.team, "name": p.name})
    return pd.DataFrame(rows), problems
