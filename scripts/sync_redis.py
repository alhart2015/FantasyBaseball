#!/usr/bin/env python3
"""Pull all cached data from Upstash Redis to local data/cache/ files.

After running, your local dashboard will have the same data as Render
without needing local Redis — read_cache() checks local JSON files first.

Credentials:
    Create a .env file at the project root (gitignored):

        UPSTASH_REDIS_REST_URL=https://your-instance.upstash.io
        UPSTASH_REDIS_REST_TOKEN=your-token-here

    Find these in the Upstash console under your Redis database > REST API.
    Or export them directly before running the script.

Usage:
    python scripts/sync_redis.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# All known Redis keys and where they map locally
CACHE_KEYS = {
    "cache:standings": "standings.json",
    "cache:roster": "roster.json",
    "cache:projections": "projections.json",
    "cache:lineup_optimal": "lineup_optimal.json",
    "cache:probable_starters": "probable_starters.json",
    "cache:waivers": "waivers.json",
    "cache:monte_carlo": "monte_carlo.json",
    "cache:meta": "meta.json",
    "cache:buy_low": "buy_low.json",
    "cache:rankings": "rankings.json",
    "cache:roster_audit": "roster_audit.json",
    "cache:spoe": "spoe.json",
    "cache:opp_rosters": "opp_rosters.json",
    "cache:leverage": "leverage.json",
}

EXTRA_KEYS = {
    "game_log_totals:hitters": "game_log_totals_hitters.json",
    "game_log_totals:pitchers": "game_log_totals_pitchers.json",
}


def main():
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

    if not url or not token:
        # Try loading from .env file
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            url = os.environ.get("UPSTASH_REDIS_REST_URL")
            token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

    if not url or not token:
        print("Error: UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set")
        print("  Set them as env vars or in .env at project root")
        sys.exit(1)

    from upstash_redis import Redis
    redis = Redis(url=url, token=token)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_keys = {**CACHE_KEYS, **EXTRA_KEYS}
    pulled = 0
    skipped = 0

    for redis_key, filename in all_keys.items():
        try:
            raw = redis.get(redis_key)
        except Exception as e:
            print(f"  FAIL  {redis_key}: {e}")
            continue

        if raw is None:
            print(f"  skip  {redis_key} (not in Redis)")
            skipped += 1
            continue

        # Parse and re-serialize for pretty formatting
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            out = json.dumps(data, indent=2)
        except (json.JSONDecodeError, TypeError):
            out = raw if isinstance(raw, str) else str(raw)

        path = CACHE_DIR / filename
        path.write_text(out, encoding="utf-8")
        size = len(out)
        print(f"  OK    {redis_key} -> {filename} ({size:,} bytes)")
        pulled += 1

    print(f"\nDone: {pulled} pulled, {skipped} empty")


if __name__ == "__main__":
    main()
