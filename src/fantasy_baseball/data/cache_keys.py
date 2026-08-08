"""Canonical cache keys and the Redis key-prefix helper."""

from enum import StrEnum


class CacheKey(StrEnum):
    """Canonical names of every cached payload.

    Typos on member access (e.g. ``CacheKey.LEVARAGE``) raise
    ``AttributeError`` the first time the code path runs and are flagged
    statically by mypy/ruff — unlike the bare-string alternative, where a
    typo like ``"levarage"`` silently reads or writes the wrong cache
    entry.
    """

    STANDINGS = "standings"
    ROSTER = "roster"
    PROJECTIONS = "projections"
    LINEUP_OPTIMAL = "lineup_optimal"
    PROBABLE_STARTERS = "probable_starters"
    MONTE_CARLO = "monte_carlo"
    META = "meta"
    RANKINGS = "rankings"
    ROSTER_AUDIT = "roster_audit"
    SPOE = "spoe"
    OPP_ROSTERS = "opp_rosters"
    LEVERAGE = "leverage"
    PENDING_MOVES = "pending_moves"
    TRANSACTION_ANALYZER = "transaction_analyzer"
    TRANSACTIONS = "transactions"
    ROS_PROJECTIONS = "ros_projections"
    FULL_SEASON_PROJECTIONS = "full_season_projections"
    POSITIONS = "positions"
    STANDINGS_BREAKDOWN = "standings_breakdown"
    STREAK_SCORES = "streak_scores"
    STASH = "stash"
    DRAFT_VALUE = "draft_value"
    STANDINGS_SNAPSHOT = "standings_snapshot"
    PACE_DEVIATIONS = "pace_deviations"
    #: Written OFFLINE by scripts/push_trajectory_board.py, never by the refresh
    #: pipeline: the fit needs `data/trajectory/` and `data/cache/keeper_skills`, both
    #: gitignored and so absent on Render. Read-only to the web app.
    TRAJECTORY_BOARD = "trajectory_board"
    #: Per-player career history and comps for the trajectory PLAYER chart -- written
    #: offline by the same script, in the same run, and read-only to the web app like
    #: the board it pairs with.
    #:
    #: SPLIT OUT OF THE BOARD (#344), not additional data. These two fields took the
    #: board blob from 762 KB to 1,861 KB while only `build_player_view` read them, so
    #: the league board and the By-team view -- the two default views -- paid ~1.1 MB of
    #: egress and a JSON parse per request to carry rows they never render. Read ONLY on
    #: the player view; a board or teams request must not touch this key.
    #:
    #: PAIRED BY VINTAGE. It carries the same `generated_at` as the board written beside
    #: it, and the player view refuses to draw extras whose stamp does not match the
    #: board it is rendering -- otherwise a board refreshed at noon draws a career line
    #: from Tuesday under a fresh projection, with both halves looking plausible.
    TRAJECTORY_CHART_DATA = "trajectory_chart_data"


def redis_key(key: CacheKey) -> str:
    """Return the Redis key for a cache entry (``cache:<name>``)."""
    return f"cache:{key}"
