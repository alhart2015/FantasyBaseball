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
    POSITIONS = "positions"


def redis_key(key: CacheKey) -> str:
    """Return the Redis key for a cache entry (``cache:<name>``)."""
    return f"cache:{key}"
