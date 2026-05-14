#!/usr/bin/env python
"""Clear cached transaction scores so the next refresh re-computes them.

Use this after changing ``score_transaction`` — most recently, the
migration from wSGP to ΔRoto. The script clears the local disk cache
and (with ``--redis``) the Redis ``cache:transactions`` +
``cache:transaction_analyzer`` keys so the next refresh re-fetches
every transaction from Yahoo and re-scores against today's projected
standings. Historical scores are an approximation in both directions
— ROS projections aren't frozen per-day — but this is the only way
to pick up scoring-logic changes retroactively.

Usage:
    python scripts/rescore_transactions.py           # local cache only
    python scripts/rescore_transactions.py --redis    # also clear Redis cache
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Clear transaction scores for re-computation")
    parser.add_argument(
        "--redis", action="store_true", help="Also clear Redis transaction_analyzer cache"
    )
    args = parser.parse_args()

    # Clear local disk cache — read_cache() reads local first, so stale
    # JSON here would make the refresh see every txn as already-known and
    # skip re-scoring (the whole point of this script).
    project_root = Path(__file__).resolve().parents[1]
    for fname in ("transactions.json", "transaction_analyzer.json"):
        p = project_root / "data" / "cache" / fname
        if p.exists():
            p.unlink()
            print(f"Removed local cache {p.relative_to(project_root)}.")

    if args.redis:
        import os

        # Load .env if available
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if not url or not token:
            print(
                "Redis credentials not found. Set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN."
            )
            sys.exit(1)

        from upstash_redis import Redis

        from fantasy_baseball.data.cache_keys import CacheKey, redis_key

        redis = Redis(url=url, token=token)
        redis.delete(redis_key(CacheKey.TRANSACTION_ANALYZER))
        redis.delete(redis_key(CacheKey.TRANSACTIONS))
        print("Cleared cache:transactions and cache:transaction_analyzer from Redis.")

    print("\nDone. Run a refresh to re-score all transactions.")


if __name__ == "__main__":
    main()
