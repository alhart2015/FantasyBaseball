"""Shared helpers for tests that inspect the cache provenance envelope."""

import json

from fantasy_baseball.web import season_data


def unwrap_cache_value(raw):
    """Parse a raw stored cache string and return the bare payload.

    Delegates the ``{_meta, _data}`` unwrap to the production single decoder
    (``season_data.unwrap_cache_envelope``) so tests that read a stored blob
    directly don't re-encode the envelope shape. Returns ``None`` for a missing key.
    """
    if raw is None:
        return None
    return season_data.unwrap_cache_envelope(json.loads(raw))
