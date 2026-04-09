#!/usr/bin/env python
"""One-time rescore of all transactions with corrected leverage formula.

Clears the transactions table and Redis cache so the next refresh
re-fetches all transactions from Yahoo and re-scores them using
score_transaction() with the fixed per-category, SGP-normalized leverage.

Usage:
    python scripts/rescore_transactions.py          # local DB only
    python scripts/rescore_transactions.py --redis   # also clear Redis cache
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Clear transaction scores for re-computation")
    parser.add_argument("--redis", action="store_true", help="Also clear Redis transaction_analyzer cache")
    args = parser.parse_args()

    from fantasy_baseball.data.db import get_connection, create_tables

    conn = get_connection()
    create_tables(conn)

    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    if count == 0:
        print("No transactions in local DB — nothing to clear.")
    else:
        conn.execute("DELETE FROM transactions")
        conn.commit()
        print(f"Cleared {count} transactions from local DB.")

    conn.close()

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
            print("Redis credentials not found. Set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN.")
            sys.exit(1)

        from upstash_redis import Redis
        redis = Redis(url=url, token=token)
        redis.delete("cache:transaction_analyzer")
        print("Cleared transaction_analyzer cache from Redis.")

    print("\nDone. Run a refresh to re-score all transactions with corrected leverage.")


if __name__ == "__main__":
    main()
