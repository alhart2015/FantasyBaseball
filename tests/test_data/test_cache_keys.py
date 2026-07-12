from fantasy_baseball.data.cache_keys import CacheKey, redis_key


def test_stash_key_exists_and_namespaced():
    assert CacheKey.STASH == "stash"
    assert redis_key(CacheKey.STASH) == "cache:stash"


def test_standings_snapshot_key():
    assert CacheKey.STANDINGS_SNAPSHOT == "standings_snapshot"
    assert redis_key(CacheKey.STANDINGS_SNAPSHOT) == "cache:standings_snapshot"
