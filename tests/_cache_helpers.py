"""Shared helpers for tests that inspect the cache provenance envelope."""

import json

from fantasy_baseball.web import season_data


def unwrap_cache_value(raw):
    """Parse a raw stored cache string and return the bare payload.

    Mirrors ``season_data.read_cache``'s unwrap by reusing the production
    envelope contract (``_is_envelope`` / ``_ENVELOPE_DATA``), so tests that
    read a stored blob directly don't each re-encode the ``{_meta, _data}``
    shape with literal keys. Returns ``None`` for a missing key.
    """
    if raw is None:
        return None
    obj = json.loads(raw)
    if season_data._is_envelope(obj):
        return obj[season_data._ENVELOPE_DATA]
    return obj
