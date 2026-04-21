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


# --- Mason Miller: cross-type same-name collision ---

class TestMasonMillerCrossTypeCollision:
    """Verify the matcher uses positions to pick the right Mason Miller
    when both hitter and pitcher entries exist in projections.
    """

    HITTER_PROJ = {
        "name": "Mason Miller", "_name_norm": "mason miller",
        "hr": 18, "ab": 480,
    }
    PITCHER_PROJ = {
        "name": "Mason Miller", "_name_norm": "mason miller",
        "k": 95, "sv": 28, "ip": 65,
    }

    def test_hitter_position_picks_hitter_projection(self):
        roster = [{"name": "Mason Miller", "positions": ["3B"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        assert isinstance(result[0].rest_of_season, HitterStats)
        assert result[0].rest_of_season.hr == 18

    def test_pitcher_position_picks_pitcher_projection(self):
        roster = [{"name": "Mason Miller", "positions": ["SP"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.PITCHER
        assert isinstance(result[0].rest_of_season, PitcherStats)
        assert result[0].rest_of_season.sv == 28

    def test_empty_positions_falls_back_to_hitter_first(self, caplog):
        """Empty positions: matcher falls through both branches and uses
        the 'any' fallback, which checks hitters first.
        """
        roster = [{"name": "Mason Miller", "positions": []}]
        with caplog.at_level(logging.WARNING):
            result = match_roster_to_projections(
                roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
            )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        # Fallback warning is asserted in TestMatchObservability — not here.


# --- Shohei Ohtani: dual-entry roster ---

class TestShoheiOhtaniDualEntry:
    """Yahoo returns Ohtani as two roster entries:
    - "Shohei Ohtani" with hitter positions
    - "Shohei Ohtani (Pitcher)" with pitcher positions

    Both must match correctly. The "(Pitcher)" suffix is stripped before
    name normalization so both find the right projection by name.
    """

    HITTER_PROJ = {
        "name": "Shohei Ohtani", "_name_norm": "shohei ohtani",
        "hr": 44, "r": 110,
    }
    PITCHER_PROJ = {
        "name": "Shohei Ohtani", "_name_norm": "shohei ohtani",
        "k": 180, "ip": 140, "w": 12,
    }

    def test_dual_roster_entries_produce_two_player_objects(self, caplog):
        roster = [
            {"name": "Shohei Ohtani", "positions": ["Util"],
             "selected_position": "Util", "player_id": "100", "status": ""},
            {"name": "Shohei Ohtani (Pitcher)", "positions": ["SP"],
             "selected_position": "SP", "player_id": "200", "status": ""},
        ]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(
                roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
            )
        assert len(result) == 2
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 0

        by_yahoo_id = {p.yahoo_id: p for p in result}
        assert set(by_yahoo_id) == {"100", "200"}

        hitter = by_yahoo_id["100"]
        assert hitter.player_type == PlayerType.HITTER
        assert hitter.name == "Shohei Ohtani"
        assert hitter.rest_of_season.hr == 44

        pitcher = by_yahoo_id["200"]
        assert pitcher.player_type == PlayerType.PITCHER
        # Suffix stripped from the stored Player.name as well
        assert pitcher.name == "Shohei Ohtani"
        assert pitcher.rest_of_season.k == 180


# --- Observability ---

class TestMatchObservability:
    """Verify match_roster_to_projections emits WARNING logs for the three
    insidious cases (unmatched, ambiguous, fallback) so future matching
    regressions surface immediately instead of silently dropping or
    mis-matching players.
    """

    def test_unmatched_player_logs_warning(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(
                roster, _empty_hitters(), _empty_pitchers(),
            )
        assert result == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "no projection match" in msg
        assert "Nobody Special" in msg
        assert "OF" in msg

    def test_unmatched_player_with_context_includes_context_in_log(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            match_roster_to_projections(
                roster, _empty_hitters(), _empty_pitchers(), context="opp:Sharks",
            )
        msg = caplog.records[0].getMessage()
        assert "[opp:Sharks]" in msg

    def test_unmatched_player_without_context_omits_brackets(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            match_roster_to_projections(
                roster, _empty_hitters(), _empty_pitchers(),
            )
        msg = caplog.records[0].getMessage()
        assert "[]" not in msg
        assert not msg.startswith("[")

    def test_ambiguous_hitter_match_logs_warning(self, caplog):
        # Two projection rows with the same normalized name and matching positions
        hitters = _hitters_df([
            {"name": "John Smith", "_name_norm": "john smith", "hr": 25},
            {"name": "John Smith", "_name_norm": "john smith", "hr": 12},
        ])
        roster = [{"name": "John Smith", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, hitters, _empty_pitchers())
        assert len(result) == 1
        # First row wins (matches.iloc[0])
        assert result[0].rest_of_season.hr == 25
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "ambiguous" in msg
        assert "hitter" in msg
        assert "John Smith" in msg
        assert "2 candidates" in msg

    def test_ambiguous_pitcher_match_logs_warning(self, caplog):
        pitchers = _pitchers_df([
            {"name": "Joe Pitcher", "_name_norm": "joe pitcher", "k": 200},
            {"name": "Joe Pitcher", "_name_norm": "joe pitcher", "k": 50},
        ])
        roster = [{"name": "Joe Pitcher", "positions": ["SP"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, _empty_hitters(), pitchers)
        assert len(result) == 1
        assert result[0].rest_of_season.k == 200
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "ambiguous" in msg
        assert "pitcher" in msg
        assert "2 candidates" in msg

    def test_fallback_match_logs_warning(self, caplog):
        # Position list doesn't qualify as hitter or pitcher (empty),
        # but name matches a hitter projection via fallback.
        hitters = _hitters_df([
            {"name": "Mystery Player", "_name_norm": "mystery player", "hr": 10},
        ])
        roster = [{"name": "Mystery Player", "positions": []}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, hitters, _empty_pitchers())
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "fallback" in msg
        assert "Mystery Player" in msg
        assert "did not disambiguate" in msg
