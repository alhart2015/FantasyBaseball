from datetime import date

import pandas as pd


def _hitters_df():
    """Two-row hitters projection DataFrame matching the real columns."""
    df = pd.DataFrame(
        [
            {
                "name": "Juan Soto",
                "fg_id": "22579",
                "mlbam_id": 665742,
                "team": "NYY",
                "player_type": "hitter",
                "pa": 650,
                "ab": 540,
                "h": 155,
                "r": 105,
                "hr": 35,
                "rbi": 100,
                "sb": 8,
                "avg": 0.287,
            },
            {
                "name": "Ivan Herrera",
                "fg_id": "26664",
                "mlbam_id": 672744,
                "team": "STL",
                "player_type": "hitter",
                "pa": 500,
                "ab": 430,
                "h": 116,
                "r": 60,
                "hr": 15,
                "rbi": 55,
                "sb": 5,
                "avg": 0.270,
            },
        ]
    )
    df["_name_norm"] = df["name"].str.lower()
    return df


def _pitchers_df():
    df = pd.DataFrame(
        [
            {
                "name": "Bryan Woo",
                "fg_id": "22300",
                "mlbam_id": 694973,
                "team": "SEA",
                "player_type": "pitcher",
                "ip": 180,
                "w": 12,
                "k": 190,
                "sv": 0,
                "er": 65,
                "bb": 35,
                "h_allowed": 150,
                "era": 3.25,
                "whip": 1.03,
            },
        ]
    )
    df["_name_norm"] = df["name"].str.lower()
    return df


class TestHydrateRosterEntries:
    def _roster(self, *entries):
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import Roster, RosterEntry

        return Roster(
            effective_date=date(2026, 4, 14),
            entries=[
                RosterEntry(
                    name=name,
                    positions=[Position.parse(p) for p in positions],
                    selected_position=Position.parse(slot),
                    status=status,
                    yahoo_id=yid,
                )
                for name, positions, slot, status, yid in entries
            ],
        )

    def test_hydrates_matched_hitter(self):
        from fantasy_baseball.data.projections import hydrate_roster_entries
        from fantasy_baseball.models.player import PlayerType

        roster = self._roster(
            ("Juan Soto", ["OF", "Util"], "OF", "", "10626"),
        )
        result = hydrate_roster_entries(roster, _hitters_df(), _pitchers_df())
        assert len(result) == 1
        player = result[0]
        assert player.name == "Juan Soto"
        assert player.player_type == PlayerType.HITTER
        assert player.rest_of_season is not None
        assert player.rest_of_season.hr == 35

    def test_hydrates_pitcher(self):
        from fantasy_baseball.data.projections import hydrate_roster_entries
        from fantasy_baseball.models.player import PlayerType

        roster = self._roster(
            ("Bryan Woo", ["P", "SP"], "P", "", "60584"),
        )
        result = hydrate_roster_entries(roster, _hitters_df(), _pitchers_df())
        assert len(result) == 1
        player = result[0]
        assert player.player_type == PlayerType.PITCHER
        assert player.rest_of_season.k == 190

    def test_drops_unmatched_player(self):
        """Players not in the projection DataFrames are omitted (same as
        match_roster_to_projections semantics)."""
        from fantasy_baseball.data.projections import hydrate_roster_entries

        roster = self._roster(
            ("Juan Soto", ["OF", "Util"], "OF", "", "10626"),
            ("Nobody Matches", ["OF", "Util"], "BN", "", "99999"),
        )
        result = hydrate_roster_entries(roster, _hitters_df(), _pitchers_df())
        assert [p.name for p in result] == ["Juan Soto"]

    def test_preserves_selected_position_string(self):
        """Downstream code reads player.selected_position as a string.

        Position enum values are StrEnum so they compare equal to the
        canonical string representation.
        """
        from fantasy_baseball.data.projections import hydrate_roster_entries

        roster = self._roster(
            ("Juan Soto", ["OF", "Util"], "OF", "", "10626"),
        )
        result = hydrate_roster_entries(roster, _hitters_df(), _pitchers_df())
        player = result[0]
        assert player.selected_position == "OF"

    def test_empty_roster_returns_empty_list(self):
        from fantasy_baseball.data.projections import hydrate_roster_entries
        from fantasy_baseball.models.roster import Roster

        roster = Roster(effective_date=date(2026, 4, 14), entries=[])
        result = hydrate_roster_entries(roster, _hitters_df(), _pitchers_df())
        assert result == []

    def test_passes_positions_as_strings_to_matcher(self):
        """hydrate_roster_entries converts Position enum values to raw
        strings before delegating to match_roster_to_projections, which
        uses is_hitter/is_pitcher on them."""
        from fantasy_baseball.data.projections import hydrate_roster_entries

        roster = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C", "", "11836"),
        )
        result = hydrate_roster_entries(roster, _hitters_df(), _pitchers_df())
        assert len(result) == 1
        assert result[0].name == "Ivan Herrera"


def test_player_holds_both_ros_and_full_season():
    """match_roster_to_projections must populate Player.rest_of_season
    (ROS-only) and Player.full_season_projection (ROS+YTD) from the
    matching FanGraphs row plus YTD actuals."""
    import pandas as pd

    from fantasy_baseball.data.projections import match_roster_to_projections

    roster = [
        {
            "name": "Test Hitter",
            "positions": ["OF"],
            "selected_position": "OF",
            "player_id": "12345",
            "status": "",
        }
    ]
    ros_hitters = pd.DataFrame(
        [
            {
                "_name_norm": "test hitter",
                "name": "Test Hitter",
                "mlbam_id": 12345,
                "r": 100.0,
                "hr": 25.0,
                "rbi": 75.0,
                "sb": 8.0,
                "h": 110.0,
                "ab": 400.0,
                "pa": 440.0,
                "avg": 0.275,
            }
        ]
    )
    full_hitters = pd.DataFrame(
        [
            {
                "_name_norm": "test hitter",
                "name": "Test Hitter",
                "mlbam_id": 12345,
                "r": 130.0,
                "hr": 30.0,
                "rbi": 95.0,
                "sb": 10.0,
                "h": 140.0,
                "ab": 500.0,
                "pa": 550.0,
                "avg": 0.280,
            }
        ]
    )
    matched = match_roster_to_projections(
        roster,
        hitters_proj=ros_hitters,
        pitchers_proj=pd.DataFrame(),
        full_hitters_proj=full_hitters,
        full_pitchers_proj=pd.DataFrame(),
    )
    p = matched[0]
    assert p.rest_of_season.r == 100.0
    assert p.full_season_projection.r == 130.0
