"""`live_rosters` must not serve PROD Yahoo rosters to a manual-mode process.

`live_rosters` reaches production Upstash on purpose (roster membership is exactly
the state the local mirror gets wrong after a trade). That is right for the Yahoo
caller and wrong for the Yahoo-free manual pipeline, which isolates itself in a
separate KV file: prod rows are Yahoo-vintage and, with Yahoo auth down, up to a
month stale, so splicing them into a page whose other half is fresh hand-transcribed
data produces two plausible-looking vintages in one view.

The two properties pinned here are opposite-facing, and both matter:

  * a manual-mode process gets `ManualStoreRefused` and Upstash is never built;
  * a NON-manual process behaves exactly as it did before -- one Upstash read, no
    refusal. An over-broad guard (keying on `FANTASY_LOCAL_KV_PATH`, say, which every
    pytest run and every ad-hoc local store also sets) would break that half.
"""

import json

import pytest

from fantasy_baseball.data import kv_store, rosters
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.rosters import (
    ManualStoreRefused,
    live_rosters,
    manual_store_active,
)
from fantasy_baseball.manual.seed import PROVENANCE_KEY

MY_TEAM = "Team 01"


@pytest.fixture
def local_kv(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV, standing in for a plain local store."""
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "store.db"))
    monkeypatch.delenv("RENDER", raising=False)
    kv_store._reset_singleton()
    yield kv_store.get_kv()
    kv_store._reset_singleton()


class _FakeUpstash:
    """Minimal stand-in for the prod client `live_rosters` builds."""

    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload
        self.calls = 0

    def mget(self, *keys: str) -> list[str | None]:
        self.calls += 1
        return [self.payload.get(k) for k in keys]


@pytest.fixture
def fake_upstash(monkeypatch):
    """Replace the prod-client builder and record whether it was ever called."""
    client = _FakeUpstash(
        {
            redis_key(CacheKey.ROSTER): json.dumps(
                [{"name": "Bryan Woo", "player_type": "pitcher", "player_id": "11", "status": ""}]
            ),
            redis_key(CacheKey.OPP_ROSTERS): json.dumps(
                {
                    "Team 02": [
                        {
                            "name": "Juan Soto",
                            "player_type": "hitter",
                            "player_id": "22",
                            "status": "",
                        }
                    ]
                }
            ),
        }
    )
    built: list[int] = []

    def _build():
        built.append(1)
        return client

    monkeypatch.setattr(kv_store, "build_explicit_upstash_kv", _build)
    return client, built


def _stamp_manual(client) -> None:
    """Write the store-level breadcrumb a seeded manual store carries.

    The same key `manual.seed.seed_manual_kv` sets, imported rather than retyped.
    """
    client.set(PROVENANCE_KEY, json.dumps({"source": "manual-transcription", "yahoo": False}))


class TestNonManualIsUnchanged:
    """Revert tests. If one fails, the guard has leaked into the Yahoo path."""

    def test_plain_local_store_is_not_manual(self, local_kv):
        assert manual_store_active() is False, (
            "an ordinary local store carries no manual breadcrumb; FANTASY_LOCAL_KV_PATH "
            "alone must NOT read as manual mode"
        )

    def test_live_rosters_still_reads_upstash(self, local_kv, fake_upstash):
        client, built = fake_upstash
        spots = live_rosters(MY_TEAM)

        assert built, "the Yahoo caller must still build the prod Upstash client"
        assert client.calls == 1, f"expected exactly one mget round trip; got {client.calls}"
        assert {(s.name, s.team) for s in spots} == {
            ("Bryan Woo", MY_TEAM),
            ("Juan Soto", "Team 02"),
        }

    def test_remote_process_never_probes_the_store(self, monkeypatch):
        """On Render `get_kv()` IS prod Upstash -- the mode check must not pay for it."""
        monkeypatch.setenv("RENDER", "true")

        def _boom():  # pragma: no cover - the point is that it never runs
            raise AssertionError("manual_store_active must not call get_kv() on Render")

        monkeypatch.setattr(kv_store, "get_kv", _boom)
        assert manual_store_active() is False


class TestManualModeIsRefused:
    def test_seeded_manual_store_reads_as_manual(self, local_kv):
        _stamp_manual(local_kv)
        assert manual_store_active() is True

    def test_live_rosters_refuses_and_never_touches_prod(self, local_kv, fake_upstash):
        _stamp_manual(local_kv)
        client, built = fake_upstash

        with pytest.raises(ManualStoreRefused) as excinfo:
            live_rosters(MY_TEAM)

        assert not built, "manual mode must not build a prod Upstash client at all"
        assert client.calls == 0
        message = str(excinfo.value)
        assert "manual" in message.lower() and "stale" in message.lower(), (
            f"the refusal has to say WHY, not just refuse: {message!r}"
        )

    def test_refusal_is_caught_by_the_existing_callers(self):
        """Both call sites already wrap the read in `except Exception` and degrade to
        'no roster data' (season_routes' trajectory route renders the board unmarked;
        scripts/trajectory_board.py re-raises only for the team views). A refusal that
        was not an ordinary exception subclass would take those pages down instead."""
        assert issubclass(ManualStoreRefused, RuntimeError)
        assert issubclass(ManualStoreRefused, Exception)

    def test_a_failed_probe_does_not_refuse(self, local_kv, monkeypatch, caplog):
        """Fail OPEN: a KV read that raises is not evidence of manual mode."""

        class _Broken:
            def get(self, key):
                raise OSError("store unreadable")

        monkeypatch.setattr(kv_store, "get_kv", _Broken)
        with caplog.at_level("WARNING", logger=rosters.__name__):
            assert manual_store_active() is False
        assert any("manual-store check failed" in r.message for r in caplog.records), (
            "a swallowed probe failure must still be audible in the log"
        )
