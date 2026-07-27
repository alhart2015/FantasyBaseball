"""Normalize raw MLB Stats API season frames to the canonical rate/PT schema.

The API returns innings as a baseball-notation STRING ("5.1" = 5 1/3), and ERA/WHIP
as strings, so every numeric field is coerced explicitly. See spec section 6.5.
"""

from __future__ import annotations

_NULLISH = {"", "nan", "none", "-", "-.--", ".---"}


def coerce_numeric(value: object) -> float:
    """Best-effort float for an MLB Stats API scalar; nullish/unparseable -> 0.0."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if text.lower() in _NULLISH:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def innings_to_float(value: object) -> float:
    """Convert baseball-notation innings to decimal innings.

    The fractional digit counts OUTS, not tenths: "5.1" is 5 1/3 innings. Only .0,
    .1 and .2 are legal; anything else means the input was not baseball notation
    and is raised rather than silently mis-scaled.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if text.lower() in _NULLISH:
        return 0.0
    if "." not in text:
        return coerce_numeric(text)
    whole, _, frac = text.partition(".")
    outs_text = frac[:1] or "0"
    if outs_text not in {"0", "1", "2"}:
        raise ValueError(f"not baseball-notation innings: {value!r}")
    return coerce_numeric(whole) + int(outs_text) / 3.0
