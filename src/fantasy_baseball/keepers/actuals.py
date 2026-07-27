"""Normalize raw MLB Stats API season frames to the canonical rate/PT schema.

The API returns innings as a baseball-notation STRING ("5.1" = 5 1/3), and ERA/WHIP
as strings, so every numeric field is coerced explicitly. See spec section 6.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


HITTER_RATES = ("hr_pa", "r_pa", "rbi_pa", "sb_pa", "h_ab", "ab_pa")
PITCHER_RATES = ("k_ip", "w_ip", "er_ip", "bb_ip", "h_ip")
HITTER_PT = "pa"
PITCHER_PT = "ip"


def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Elementwise numer/denom with 0/0 -> NaN (never 0.0).

    NaN is the honest answer for "no observation"; 0.0 would read downstream as a
    real observation of a zero rate. Spec 5.5.
    """
    result: pd.Series = numer.divide(denom.where(denom > 0, other=np.nan))
    return result


def _indexed(raw: pd.DataFrame) -> pd.DataFrame:
    frame: pd.DataFrame = raw.loc[raw["player.id"].notna()].copy()
    frame["mlbam_id"] = frame["player.id"].astype(int)
    indexed: pd.DataFrame = frame.set_index("mlbam_id")
    return indexed


def normalize_hitting(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _indexed(raw)
    num = {
        col: frame[f"stat.{col}"].map(coerce_numeric)
        for col in ("plateAppearances", "atBats", "hits", "runs", "homeRuns", "rbi", "stolenBases")
    }
    pa, ab = num["plateAppearances"], num["atBats"]
    out = pd.DataFrame(
        {
            HITTER_PT: pa,
            "ab_pa": safe_ratio(ab, pa),
            "h_ab": safe_ratio(num["hits"], ab),
            "hr_pa": safe_ratio(num["homeRuns"], pa),
            "r_pa": safe_ratio(num["runs"], pa),
            "rbi_pa": safe_ratio(num["rbi"], pa),
            "sb_pa": safe_ratio(num["stolenBases"], pa),
        },
        index=frame.index,
    )
    return out


def normalize_pitching(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _indexed(raw)
    ip = frame["stat.inningsPitched"].map(innings_to_float)
    num = {
        col: frame[f"stat.{col}"].map(coerce_numeric)
        for col in ("earnedRuns", "baseOnBalls", "hits", "strikeOuts", "wins")
    }
    out = pd.DataFrame(
        {
            PITCHER_PT: ip,
            "k_ip": safe_ratio(num["strikeOuts"], ip),
            "w_ip": safe_ratio(num["wins"], ip),
            "er_ip": safe_ratio(num["earnedRuns"], ip),
            "bb_ip": safe_ratio(num["baseOnBalls"], ip),
            "h_ip": safe_ratio(num["hits"], ip),
        },
        index=frame.index,
    )
    return out
