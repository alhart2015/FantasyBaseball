from fantasy_baseball.data.cache_keys import CacheKey, redis_key


def test_stash_key_exists_and_namespaced():
    assert CacheKey.STASH == "stash"
    assert redis_key(CacheKey.STASH) == "cache:stash"
