"""Tests for fantasy_baseball.data.fangraphs_fetch."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fantasy_baseball.data.fangraphs_fetch import (
    _fetch_fangraphs_data,
    _to_csv,
    fetch_ros_projections,
)


def _make_html(players: list[dict]) -> str:
    """Build minimal HTML page containing __NEXT_DATA__ with given player list."""
    blob = {
        "props": {
            "pageProps": {
                "data": {
                    "data": players,
                }
            }
        }
    }
    return f'<script id="__NEXT_DATA__">{json.dumps(blob)}</script>'


class TestFetchFangraphsData:
    def test_fetch_fangraphs_data_parses_next_data(self):
        players = [
            {"PlayerName": "Aaron Judge", "HR": 45, "playerid": "abc123"},
            {"PlayerName": "Juan Soto", "HR": 38, "playerid": "def456"},
        ]
        html = _make_html(players)

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status.return_value = None

        with patch("fantasy_baseball.data.fangraphs_fetch.requests.get", return_value=mock_resp):
            result = _fetch_fangraphs_data("steamerr", "bat")

        assert len(result) == 2
        assert result[0]["PlayerName"] == "Aaron Judge"
        assert result[1]["HR"] == 38

    def test_fetch_fangraphs_data_returns_empty_on_no_data(self):
        """Empty queries list in __NEXT_DATA__ returns empty list."""
        blob = {
            "props": {
                "pageProps": {
                    "data": {
                        "data": [],
                    }
                }
            }
        }
        html = f'<script id="__NEXT_DATA__">{json.dumps(blob)}</script>'

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status.return_value = None

        with patch("fantasy_baseball.data.fangraphs_fetch.requests.get", return_value=mock_resp):
            result = _fetch_fangraphs_data("steamerr", "bat")

        assert result == []

    def test_fetch_fangraphs_data_returns_empty_on_missing_key(self):
        """Missing __NEXT_DATA__ key path returns empty list instead of raising."""
        blob = {"props": {"pageProps": {}}}  # no "data" key
        html = f'<script id="__NEXT_DATA__">{json.dumps(blob)}</script>'

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status.return_value = None

        with patch("fantasy_baseball.data.fangraphs_fetch.requests.get", return_value=mock_resp):
            result = _fetch_fangraphs_data("steamerr", "bat")

        assert result == []

    def test_fetch_fangraphs_data_returns_empty_on_request_error(self):
        """Network errors return empty list instead of propagating."""
        with patch(
            "fantasy_baseball.data.fangraphs_fetch.requests.get",
            side_effect=Exception("connection refused"),
        ):
            result = _fetch_fangraphs_data("steamerr", "bat")

        assert result == []

    def test_fetch_fangraphs_data_returns_empty_on_no_script_tag(self):
        """HTML with no __NEXT_DATA__ script tag returns empty list."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>No data here</body></html>"
        mock_resp.raise_for_status.return_value = None

        with patch("fantasy_baseball.data.fangraphs_fetch.requests.get", return_value=mock_resp):
            result = _fetch_fangraphs_data("steamerr", "bat")

        assert result == []


    def test_fetch_fangraphs_data_parses_dehydrated_state(self):
        """Current FanGraphs structure: dehydratedState.queries[0].state.data."""
        players = [
            {"PlayerName": "Bobby Witt Jr.", "HR": 28, "playerid": "25764"},
        ]
        blob = {
            "props": {
                "pageProps": {
                    "dehydratedState": {
                        "queries": [
                            {"state": {"data": players}}
                        ]
                    }
                }
            }
        }
        html = f'<script id="__NEXT_DATA__">{json.dumps(blob)}</script>'

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status.return_value = None

        with patch("fantasy_baseball.data.fangraphs_fetch.requests.get", return_value=mock_resp):
            result = _fetch_fangraphs_data("steamerr", "bat")

        assert len(result) == 1
        assert result[0]["PlayerName"] == "Bobby Witt Jr."


class TestToCsv:
    def test_to_csv_renames_columns(self, tmp_path):
        players = [
            {"PlayerName": "Aaron Judge", "playerid": "abc123", "xMLBAMID": 592450, "HR": 45},
            {"PlayerName": "Juan Soto", "playerid": "def456", "xMLBAMID": 665742, "HR": 38},
        ]
        out = tmp_path / "test.csv"
        _to_csv(players, out)

        df = pd.read_csv(out)
        assert "Name" in df.columns, "PlayerName should be renamed to Name"
        assert "PlayerId" in df.columns, "playerid should be renamed to PlayerId"
        assert "MLBAMID" in df.columns, "xMLBAMID should be renamed to MLBAMID"
        # Original column names should be gone
        assert "PlayerName" not in df.columns
        assert "playerid" not in df.columns
        assert "xMLBAMID" not in df.columns

    def test_to_csv_creates_parent_dirs(self, tmp_path):
        players = [{"Name": "Test Player", "HR": 10}]
        out = tmp_path / "deep" / "nested" / "dir" / "output.csv"
        _to_csv(players, out)
        assert out.exists()

    def test_to_csv_columns_not_present_are_skipped(self, tmp_path):
        """Columns absent from the data are not added to the rename mapping."""
        players = [{"PlayerName": "Test", "HR": 10}]
        out = tmp_path / "output.csv"
        _to_csv(players, out)
        df = pd.read_csv(out)
        assert "Name" in df.columns
        # playerid and xMLBAMID were absent, so no MLBAMID/PlayerId columns appear
        assert "MLBAMID" not in df.columns
        assert "PlayerId" not in df.columns


class TestFetchRosProjections:
    def _mock_response(self, players: list[dict]):
        mock_resp = MagicMock()
        mock_resp.text = _make_html(players)
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_fetch_ros_projections_saves_csvs(self, tmp_path):
        hitters = [{"PlayerName": "Aaron Judge", "playerid": "1", "HR": 45}]
        pitchers = [{"PlayerName": "Gerrit Cole", "playerid": "2", "IP": 180}]

        responses = [
            self._mock_response(hitters),
            self._mock_response(pitchers),
        ]

        with patch(
            "fantasy_baseball.data.fangraphs_fetch.requests.get",
            side_effect=responses,
        ), patch("fantasy_baseball.data.fangraphs_fetch.time.sleep"):
            results = fetch_ros_projections(
                tmp_path, systems=["steamer"], season_year=2026
            )

        assert results["steamer"] == "ok"

        from datetime import date
        today = date.today().isoformat()
        snapshot_dir = tmp_path / "2026" / "ros" / today
        assert (snapshot_dir / "steamer-hitters.csv").exists()
        assert (snapshot_dir / "steamer-pitchers.csv").exists()

    def test_fetch_ros_projections_handles_unknown_system(self, tmp_path):
        """Unknown system names return an error without crashing."""
        with patch(
            "fantasy_baseball.data.fangraphs_fetch.requests.get",
        ) as mock_get, patch("fantasy_baseball.data.fangraphs_fetch.time.sleep"):
            results = fetch_ros_projections(
                tmp_path, systems=["nonexistent-system"], season_year=2026
            )

        mock_get.assert_not_called()
        assert "nonexistent-system" in results
        assert results["nonexistent-system"].startswith("error:")

    def test_fetch_ros_projections_multiple_systems(self, tmp_path):
        players = [{"PlayerName": "Test Player", "playerid": "1"}]

        with patch(
            "fantasy_baseball.data.fangraphs_fetch.requests.get",
            return_value=self._mock_response(players),
        ), patch("fantasy_baseball.data.fangraphs_fetch.time.sleep"):
            results = fetch_ros_projections(
                tmp_path,
                systems=["steamer", "zips"],
                season_year=2026,
            )

        assert results["steamer"] == "ok"
        assert results["zips"] == "ok"

    def test_fetch_ros_projections_mixed_known_unknown(self, tmp_path):
        """Mix of known + unknown systems: known succeeds, unknown errors."""
        players = [{"PlayerName": "Test", "playerid": "1"}]

        with patch(
            "fantasy_baseball.data.fangraphs_fetch.requests.get",
            return_value=self._mock_response(players),
        ), patch("fantasy_baseball.data.fangraphs_fetch.time.sleep"):
            results = fetch_ros_projections(
                tmp_path,
                systems=["steamer", "bad-system"],
                season_year=2026,
            )

        assert results["steamer"] == "ok"
        assert results["bad-system"].startswith("error:")

    def test_fetch_ros_projections_calls_sleep_between_requests(self, tmp_path):
        """Verifies time.sleep is called between HTTP requests."""
        players = [{"PlayerName": "Test", "playerid": "1"}]

        with patch(
            "fantasy_baseball.data.fangraphs_fetch.requests.get",
            return_value=self._mock_response(players),
        ), patch("fantasy_baseball.data.fangraphs_fetch.time.sleep") as mock_sleep:
            fetch_ros_projections(
                tmp_path, systems=["steamer"], season_year=2026
            )

        # steamer has 2 requests (bat + pit), sleep after 1st but not before
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2)
