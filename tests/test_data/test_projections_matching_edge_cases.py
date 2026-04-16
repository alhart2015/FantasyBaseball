"""Regression and observability tests for match_roster_to_projections.

Covers three historically fragile players:
- Julio Rodriguez: accent encoding (NFC vs NFD vs ASCII)
- Mason Miller: cross-type same-name collision (hitter + pitcher)
- Shohei Ohtani: dual roster entries with "(Pitcher)" suffix

And the matcher's observability: WARNING logs on unmatched, ambiguous,
and fallback matches so future regressions surface immediately instead
of silently dropping or mis-matching players.
"""
import logging
import pandas as pd
import pytest

from fantasy_baseball.data.projections import match_roster_to_projections
from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType


# --- Tiny in-memory DataFrame builders ---

def _hitters_df(rows):
    """Build a hitters projection DataFrame with the minimum columns the
    matcher and HitterStats.from_dict require. Each row in ``rows`` is a
    dict with at least ``name`` and ``_name_norm``; missing stat columns
    default to 0.
    """
    defaults = {
        "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0.0,
        "ab": 0, "h": 0, "pa": 0, "player_type": "hitter",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _pitchers_df(rows):
    """Build a pitchers projection DataFrame with the minimum columns the
    matcher and PitcherStats.from_dict require.
    """
    defaults = {
        "w": 0, "k": 0, "sv": 0, "ip": 0, "er": 0, "bb": 0, "h_allowed": 0,
        "era": 0.0, "whip": 0.0, "player_type": "pitcher",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _empty_hitters():
    return pd.DataFrame(columns=["name", "_name_norm", "player_type"])


def _empty_pitchers():
    return pd.DataFrame(columns=["name", "_name_norm", "player_type"])


# --- Julio Rodriguez: accent encoding ---

class TestJulioRodriguezAccentEncoding:
    """Verify normalize_name handles all three Unicode forms of 'í'.

    These tests exercise the matcher end-to-end with realistic encoding
    variants Yahoo and FanGraphs have been observed to send.
    """

    PROJECTION = {"name": "Julio Rodríguez", "_name_norm": "julio rodriguez", "hr": 32}

    def test_roster_nfc_precomposed_matches(self):
        roster = [{"name": "Julio Rodríguez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.PROJECTION]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_roster_nfd_decomposed_matches(self):
        # 'í' as 'i' + combining acute accent (U+0301)
        roster = [{"name": "Julio Rodri\u0301guez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.PROJECTION]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_roster_ascii_matches_accented_projection(self):
        roster = [{"name": "Julio Rodriguez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.PROJECTION]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_accented_roster_matches_ascii_projection(self):
        # Mirror: roster has accents, projection is plain ASCII
        roster = [{"name": "Julio Rodríguez", "positions": ["OF"]}]
        ascii_proj = {"name": "Julio Rodriguez", "_name_norm": "julio rodriguez", "hr": 32}
        result = match_roster_to_projections(
            roster, _hitters_df([ascii_proj]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32
