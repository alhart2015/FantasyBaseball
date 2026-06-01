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
from typing import ClassVar

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
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "ab": 0,
        "h": 0,
        "pa": 0,
        "player_type": "hitter",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _pitchers_df(rows):
    """Build a pitchers projection DataFrame with the minimum columns the
    matcher and PitcherStats.from_dict require.
    """
    defaults = {
        "w": 0,
        "k": 0,
        "sv": 0,
        "ip": 0,
        "er": 0,
        "bb": 0,
        "h_allowed": 0,
        "era": 0.0,
        "whip": 0.0,
        "player_type": "pitcher",
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

    PROJECTION: ClassVar[dict] = {
        "name": "Julio Rodríguez",
        "_name_norm": "julio rodriguez",
        "hr": 32,
    }

    def test_roster_nfc_precomposed_matches(self):
        roster = [{"name": "Julio Rodríguez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster,
            _hitters_df([self.PROJECTION]),
            _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_roster_nfd_decomposed_matches(self):
        # 'í' as 'i' + combining acute accent (U+0301)
        roster = [{"name": "Julio Rodri\u0301guez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster,
            _hitters_df([self.PROJECTION]),
            _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_roster_ascii_matches_accented_projection(self):
        roster = [{"name": "Julio Rodriguez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster,
            _hitters_df([self.PROJECTION]),
            _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_accented_roster_matches_ascii_projection(self):
        # Mirror: roster has accents, projection is plain ASCII
        roster = [{"name": "Julio Rodríguez", "positions": ["OF"]}]
        ascii_proj = {"name": "Julio Rodriguez", "_name_norm": "julio rodriguez", "hr": 32}
        result = match_roster_to_projections(
            roster,
            _hitters_df([ascii_proj]),
            _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32


# --- Mason Miller: cross-type same-name collision ---


class TestMasonMillerCrossTypeCollision:
    """Verify the matcher uses positions to pick the right Mason Miller
    when both hitter and pitcher entries exist in projections.
    """

    HITTER_PROJ: ClassVar[dict] = {
        "name": "Mason Miller",
        "_name_norm": "mason miller",
        "hr": 18,
        "ab": 480,
    }
    PITCHER_PROJ: ClassVar[dict] = {
        "name": "Mason Miller",
        "_name_norm": "mason miller",
        "k": 95,
        "sv": 28,
        "ip": 65,
    }

    def test_hitter_position_picks_hitter_projection(self):
        roster = [{"name": "Mason Miller", "positions": ["3B"]}]
        result = match_roster_to_projections(
            roster,
            _hitters_df([self.HITTER_PROJ]),
            _pitchers_df([self.PITCHER_PROJ]),
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        assert isinstance(result[0].rest_of_season, HitterStats)
        assert result[0].rest_of_season.hr == 18

    def test_pitcher_position_picks_pitcher_projection(self):
        roster = [{"name": "Mason Miller", "positions": ["SP"]}]
        result = match_roster_to_projections(
            roster,
            _hitters_df([self.HITTER_PROJ]),
            _pitchers_df([self.PITCHER_PROJ]),
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
                roster,
                _hitters_df([self.HITTER_PROJ]),
                _pitchers_df([self.PITCHER_PROJ]),
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

    HITTER_PROJ: ClassVar[dict] = {
        "name": "Shohei Ohtani",
        "_name_norm": "shohei ohtani",
        "hr": 44,
        "r": 110,
    }
    PITCHER_PROJ: ClassVar[dict] = {
        "name": "Shohei Ohtani",
        "_name_norm": "shohei ohtani",
        "k": 180,
        "ip": 140,
        "w": 12,
    }

    def test_dual_roster_entries_produce_two_player_objects(self, caplog):
        roster = [
            {
                "name": "Shohei Ohtani",
                "positions": ["Util"],
                "selected_position": "Util",
                "player_id": "100",
                "status": "",
            },
            {
                "name": "Shohei Ohtani (Pitcher)",
                "positions": ["SP"],
                "selected_position": "SP",
                "player_id": "200",
                "status": "",
            },
        ]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(
                roster,
                _hitters_df([self.HITTER_PROJ]),
                _pitchers_df([self.PITCHER_PROJ]),
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
                roster,
                _empty_hitters(),
                _empty_pitchers(),
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
                roster,
                _empty_hitters(),
                _empty_pitchers(),
                context="opp:Sharks",
            )
        msg = caplog.records[0].getMessage()
        assert "[opp:Sharks]" in msg

    def test_unmatched_player_without_context_omits_brackets(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            match_roster_to_projections(
                roster,
                _empty_hitters(),
                _empty_pitchers(),
            )
        msg = caplog.records[0].getMessage()
        assert "[]" not in msg
        assert not msg.startswith("[")

    def test_ambiguous_hitter_match_logs_warning(self, caplog):
        # Two projection rows with the same normalized name and matching positions
        hitters = _hitters_df(
            [
                {"name": "John Smith", "_name_norm": "john smith", "hr": 25},
                {"name": "John Smith", "_name_norm": "john smith", "hr": 12},
            ]
        )
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
        pitchers = _pitchers_df(
            [
                {"name": "Joe Pitcher", "_name_norm": "joe pitcher", "k": 200},
                {"name": "Joe Pitcher", "_name_norm": "joe pitcher", "k": 50},
            ]
        )
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

    def test_ambiguous_pitcher_match_picks_highest_ip(self, caplog):
        """Two pitchers share a normalized name (the audit's Mason Miller): the
        real rostered arm (54 projected IP) and an obscure namesake (2 IP). The
        matcher must pick the high-volume real player even when the junk row is
        first -- otherwise displacement reads a ~2 IP preseason and the
        slot-share denominator blows up.
        """
        pitchers = _pitchers_df(
            [
                {"name": "Mason Miller", "_name_norm": "mason miller", "ip": 2, "k": 3},
                {"name": "Mason Miller", "_name_norm": "mason miller", "ip": 54, "k": 78},
            ]
        )
        roster = [{"name": "Mason Miller", "positions": ["RP"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, _empty_hitters(), pitchers)
        assert len(result) == 1
        # Highest IP wins -- the real arm, not the 2-IP namesake (which is first).
        assert result[0].rest_of_season.ip == 54
        assert result[0].rest_of_season.k == 78

    def test_ambiguous_hitter_match_picks_highest_pa(self, caplog):
        """Same collision rule for hitters: pick the row with the most plate
        appearances (the established regular), not whichever row is first.
        """
        hitters = _hitters_df(
            [
                {"name": "Will Smith", "_name_norm": "will smith", "pa": 40, "hr": 1},
                {"name": "Will Smith", "_name_norm": "will smith", "pa": 550, "hr": 24},
            ]
        )
        roster = [{"name": "Will Smith", "positions": ["C"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, hitters, _empty_pitchers())
        assert len(result) == 1
        # Highest PA wins -- the 550-PA regular, not the 40-PA namesake (first).
        assert result[0].rest_of_season.hr == 24

    def test_preseason_attach_is_type_correct_on_dual_name(self):
        """A roster carrying the same normalized name as BOTH a hitter and a
        pitcher (Shohei Ohtani) must get type-correct preseason: the hitter a
        HitterStats, the pitcher a PitcherStats. Keying the preseason lookup on
        bare name (no player_type) would cross-assign them -- the pitcher would
        get a HitterStats with no IP and silently fall back to the legacy swap.
        """
        from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType

        hitters = _hitters_df(
            [{"name": "Shohei Ohtani", "_name_norm": "shohei ohtani", "mlbam_id": 660271, "hr": 40}]
        )
        pitchers = _pitchers_df(
            [
                {
                    "name": "Shohei Ohtani",
                    "_name_norm": "shohei ohtani",
                    "mlbam_id": 808888,
                    "ip": 150,
                }
            ]
        )
        pre_hitters = _hitters_df(
            [{"name": "Shohei Ohtani", "_name_norm": "shohei ohtani", "mlbam_id": 660271, "hr": 44}]
        )
        pre_pitchers = _pitchers_df(
            [
                {
                    "name": "Shohei Ohtani",
                    "_name_norm": "shohei ohtani",
                    "mlbam_id": 808888,
                    "ip": 180,
                }
            ]
        )
        roster = [
            {"name": "Shohei Ohtani", "positions": ["Util"], "player_id": "1"},
            {"name": "Shohei Ohtani", "positions": ["SP"], "player_id": "2"},
        ]
        result = match_roster_to_projections(
            roster,
            hitters,
            pitchers,
            preseason_hitters_proj=pre_hitters,
            preseason_pitchers_proj=pre_pitchers,
        )
        by_type = {p.player_type: p for p in result}
        assert isinstance(by_type[PlayerType.HITTER].preseason, HitterStats)
        pitcher_pre = by_type[PlayerType.PITCHER].preseason
        assert isinstance(pitcher_pre, PitcherStats)
        assert pitcher_pre.ip == 180

    def test_preseason_attach_follows_ros_identity_on_same_type_collision(self):
        """Two same-name pitchers (Mason Miller). The ROS match picks the real
        closer by max IP; the preseason line attached must be the SAME player's
        (by mlbam_id), NOT whichever same-name row has the most PRESEASON volume
        -- otherwise .rest_of_season and .preseason come from two people and the
        slot-share denominator is garbage.
        """
        pitchers = _pitchers_df(
            [
                {
                    "name": "Mason Miller",
                    "_name_norm": "mason miller",
                    "mlbam_id": 700,
                    "ip": 60,
                    "k": 90,
                },
                {
                    "name": "Mason Miller",
                    "_name_norm": "mason miller",
                    "mlbam_id": 701,
                    "ip": 5,
                    "k": 4,
                },
            ]
        )
        pre_pitchers = _pitchers_df(
            [
                {"name": "Mason Miller", "_name_norm": "mason miller", "mlbam_id": 700, "ip": 65},
                {"name": "Mason Miller", "_name_norm": "mason miller", "mlbam_id": 701, "ip": 180},
            ]
        )
        roster = [{"name": "Mason Miller", "positions": ["RP"], "player_id": "1"}]
        result = match_roster_to_projections(
            roster,
            _empty_hitters(),
            pitchers,
            preseason_hitters_proj=_empty_hitters(),
            preseason_pitchers_proj=pre_pitchers,
        )
        assert len(result) == 1
        p = result[0]
        assert p.rest_of_season.ip == 60  # ROS = real closer (mlbam 700)
        # preseason must follow mlbam 700 (ip 65), NOT the higher-volume
        # namesake mlbam 701 (ip 180).
        assert p.preseason is not None and p.preseason.ip == 65

    def test_fallback_match_logs_warning(self, caplog):
        # Position list doesn't qualify as hitter or pitcher (empty),
        # but name matches a hitter projection via fallback.
        hitters = _hitters_df(
            [
                {"name": "Mystery Player", "_name_norm": "mystery player", "hr": 10},
            ]
        )
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
