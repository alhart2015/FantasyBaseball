"""Cache management and data assembly for the season dashboard."""

import json
import os
import tempfile
from pathlib import Path

from fantasy_baseball.scoring import score_roto

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

ALL_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE_CATS = {"ERA", "WHIP"}

CACHE_FILES = {
    "standings": "standings.json",
    "roster": "roster.json",
    "projections": "projections.json",
    "lineup_optimal": "lineup_optimal.json",
    "probable_starters": "probable_starters.json",
    "waivers": "waivers.json",
    "trades": "trades.json",
    "monte_carlo": "monte_carlo.json",
    "meta": "meta.json",
}


def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Returns None if missing or corrupt."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # On Windows, must remove target before rename
        if path.exists():
            path.unlink()
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    return read_cache("meta", cache_dir) or {}


def format_standings_for_display(
    standings: list[dict], user_team_name: str
) -> dict:
    """Transform raw standings cache into display-ready structure with roto points and color codes.

    Args:
        standings: List of team dicts from fetch_standings(), each with "name" and "stats" keys.
        user_team_name: The authenticated user's team name for highlighting.

    Returns:
        {"teams": [...]} where each team has roto_points, is_user flag, color_classes, and rank.
    """
    if not standings:
        return {"teams": []}

    all_stats = {t["name"]: t["stats"] for t in standings}
    roto = score_roto(all_stats)

    cat_ranks = _compute_category_ranks(standings)
    num_teams = len(standings)

    teams = []
    for t in standings:
        name = t["name"]
        is_user = name == user_team_name
        roto_pts = roto[name]

        color_classes = {}
        if is_user:
            for cat in ALL_CATEGORIES:
                rank = cat_ranks[cat][name]
                if rank <= 3:
                    color_classes[cat] = "cat-top"
                elif rank > num_teams - 3:
                    color_classes[cat] = "cat-bottom"
                else:
                    color_classes[cat] = ""
        else:
            color_classes = {cat: "" for cat in ALL_CATEGORIES}

        teams.append({
            "name": name,
            "stats": t["stats"],
            "roto_points": roto_pts,
            "is_user": is_user,
            "color_classes": color_classes,
        })

    teams.sort(key=lambda t: t["roto_points"]["total"], reverse=True)

    for i, t in enumerate(teams):
        t["rank"] = i + 1

    return {"teams": teams}


def _compute_category_ranks(standings: list[dict]) -> dict[str, dict[str, int]]:
    """Compute per-category rank for each team (1 = best).

    For inverse categories (ERA, WHIP), lower value = rank 1.
    """
    ranks = {}
    for cat in ALL_CATEGORIES:
        reverse = cat not in INVERSE_CATS
        sorted_teams = sorted(standings, key=lambda t: t["stats"][cat], reverse=reverse)
        ranks[cat] = {t["name"]: i + 1 for i, t in enumerate(sorted_teams)}
    return ranks
