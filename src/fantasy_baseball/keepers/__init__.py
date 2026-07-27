"""Keepers module: raw MLB Stats API + Baseball Savant data pulls.

Fetchers return the upstream response fully raw -- no derivation, rename, or join.
Downstream keeper-value logic (#266) is built on top of these; nothing here computes.
"""

from __future__ import annotations

from fantasy_baseball.keepers.cache import fetch_or_cache

__all__ = ["fetch_or_cache"]
