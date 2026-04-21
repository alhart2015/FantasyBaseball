import os
import tempfile
from pathlib import Path

import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2

CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "config" / "oauth.json"

# When running on a hosted platform, oauth.json won't exist on disk.
# The YAHOO_OAUTH_JSON env var holds the full JSON content; we write
# it to a temp file so yahoo_oauth can read/update it normally.
_env_oauth_path: Path | None = None


def _get_oauth_path() -> Path:
    """Resolve the oauth.json path, creating from env var if needed."""
    global _env_oauth_path

    # Prefer the local file if it exists (local development)
    if CONFIG_PATH.exists():
        return CONFIG_PATH

    # Fall back to env var (hosted deployment)
    env_json = os.environ.get("YAHOO_OAUTH_JSON")
    if env_json:
        if _env_oauth_path and _env_oauth_path.exists():
            return _env_oauth_path
        # Write to a persistent temp file (survives for the process lifetime)
        fd, path = tempfile.mkstemp(suffix=".json", prefix="yahoo_oauth_")
        with os.fdopen(fd, "w") as f:
            f.write(env_json)
        _env_oauth_path = Path(path)
        return _env_oauth_path

    raise FileNotFoundError(
        f"oauth.json not found at {CONFIG_PATH} and YAHOO_OAUTH_JSON env var not set. "
        "Create config/oauth.json with your Yahoo app credentials, "
        "or set YAHOO_OAUTH_JSON to the file's JSON content."
    )


def get_yahoo_session(config_path: Path | None = None) -> OAuth2:
    """Create an authenticated Yahoo OAuth2 session.

    Checks config/oauth.json first, then YAHOO_OAUTH_JSON env var.
    Token refreshes are written back to whichever file is in use.

    Args:
        config_path: Path to oauth.json. Defaults to auto-detection.

    Raises:
        FileNotFoundError: If no oauth credentials are available.
    """
    path = config_path or _get_oauth_path()
    return OAuth2(None, None, from_file=str(path))


def get_league(session: OAuth2, league_id: int, game_key: str = "mlb"):
    """Get a Yahoo Fantasy league object.

    Args:
        session: Authenticated OAuth2 session.
        league_id: Yahoo league ID (e.g., 5652).
        game_key: Yahoo game key (default: "mlb" for current season).

    Returns:
        yahoo_fantasy_api League object.
    """
    game = yfa.Game(session, game_key)
    league_ids = game.league_ids()
    # Find the league key matching our league_id
    league_key = None
    for lid in league_ids:
        if lid.split(".l.")[-1] == str(league_id):
            league_key = lid
            break
    if league_key is None:
        league_key = f"{game.game_id()}.l.{league_id}"
    return game.to_league(league_key)
