from pathlib import Path
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "config" / "oauth.json"


def get_yahoo_session(config_path: Path | None = None) -> OAuth2:
    """Create an authenticated Yahoo OAuth2 session.

    On first run, opens a browser for Yahoo login. Token is cached
    in the oauth.json file for subsequent runs.

    Args:
        config_path: Path to oauth.json. Defaults to config/oauth.json.

    Raises:
        FileNotFoundError: If oauth.json doesn't exist.
    """
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"oauth.json not found at {path}. "
            "Create it with your Yahoo app's consumer_key and consumer_secret. "
            "See config/league.yaml.example for setup instructions."
        )
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
