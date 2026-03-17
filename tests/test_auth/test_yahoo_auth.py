import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fantasy_baseball.auth.yahoo_auth import (
    get_yahoo_session,
    get_league,
    CONFIG_PATH,
)


def test_config_path_points_to_oauth_json():
    assert CONFIG_PATH.name == "oauth.json"
    assert "config" in CONFIG_PATH.parts


def test_get_yahoo_session_raises_if_no_config(tmp_path):
    with patch("fantasy_baseball.auth.yahoo_auth.CONFIG_PATH", tmp_path / "nope.json"):
        with pytest.raises(FileNotFoundError, match="oauth.json"):
            get_yahoo_session()


def test_get_league_returns_league_object():
    mock_session = MagicMock()
    mock_game = MagicMock()
    mock_league = MagicMock()
    mock_game.to_league.return_value = mock_league
    with patch("fantasy_baseball.auth.yahoo_auth.yfa.Game", return_value=mock_game):
        league = get_league(mock_session, league_id=5652, game_key="mlb")
    mock_game.to_league.assert_called_once()
    assert league == mock_league
