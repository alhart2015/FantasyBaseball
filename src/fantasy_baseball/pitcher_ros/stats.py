"""The pitcher stats projections are scored on, and the game lines they come from (#465).

The five roto categories as rates:

* K, W, SV per batter faced. A projection's total is its rate x the batters the
  pitcher faces, so a rate can be scored on the batters he actually faced, apart from
  how many that was (volume is its own question).
* ERA = 27 x ER / outs and WHIP = 3 x (H + BB) / outs, the same numbers as
  9 x ER / IP and (H + BB) / IP with IP = outs / 3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The date a pitching line counts on (see :func:`load_games`): a suspended game's
# resume date, else its date. SQL over the store's ``pitching`` view.
GAME_DATE_SQL = "CAST(COALESCE(resumed_date, game_date) AS DATE)"

# Summed per pitcher over any span of games; ``g`` is games pitched.
COUNTS = ("g", "gs", "bf", "outs", "k", "bb", "h", "hr", "er", "w", "sv")
TARGETS = ("k", "w", "sv", "era", "whip")
PER_BF = ("k", "w", "sv")


def rates_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """``TARGETS`` from summed counts (``bf``, ``outs``, ``k``, ``w``, ``sv``, ``er``,
    ``h``, ``bb``). NaN where the denominator is 0: no batters faced, or no outs."""
    bf = counts["bf"].astype(float).where(counts["bf"] > 0)
    outs = counts["outs"].astype(float).where(counts["outs"] > 0)
    out = pd.DataFrame({s: counts[s] / bf for s in PER_BF}, index=counts.index)
    out["era"] = 27 * counts["er"] / outs
    out["whip"] = 3 * (counts["h"] + counts["bb"]) / outs
    return out


def load_games(store: Path) -> pd.DataFrame:
    """Every pitcher-game line in the store's ``pitching`` view: ``player_id``,
    ``team_id``, ``game_date`` (datetime), ``season`` and ``COUNTS``.

    A suspended game is dated to the day it was finished (the store's ``resumed_date``),
    not the day it started (the store's ``game_date``): the box score doesn't say which
    of the two days a pitcher worked, and only from the later one is his whole line
    known. So a projection dated between the two never sees the line as history, and a
    pitcher who worked only the first day counts in the actuals after it.
    """
    # Here, not at the top: the store reaches the Redis client (data.mlb_game_logs), and
    # the draft board imports this module's TARGETS (draft.our_pitchers via final), so
    # the live draft must not (tests/test_draft/test_isolation.py).
    from fantasy_baseball.pitch_data.store import connect

    cols = ", ".join(c for c in COUNTS if c != "g")
    conn = connect(store)
    try:
        games = conn.execute(
            f"""
            SELECT player_id, team_id, {GAME_DATE_SQL} AS game_date, 1 AS g, {cols}
            FROM pitching
            """
        ).df()
    finally:
        conn.close()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = games["game_date"].dt.year
    return games


def totals(games: pd.DataFrame) -> pd.DataFrame:
    """``COUNTS`` summed per ``player_id`` over ``games``."""
    out: pd.DataFrame = games.groupby("player_id")[list(COUNTS)].sum()
    return out
