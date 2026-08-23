"""Tests for the manual-mode seeder.

Two things are being pinned here:

1. **The seams.** The seeder must land the transcription in exactly the
   places the existing stale-data path reads -- ``weekly_rosters_history``,
   ``standings_history`` and the enveloped ``cache:standings`` blob -- in a
   shape ``League.from_redis`` and ``read_cache_with_meta`` accept unchanged.
2. **The refusal.** Manual mode is isolated by writing to a whole separate
   SQLite file. A seed that reached ``data/local.db`` would overwrite real
   Yahoo history with hand-typed rows, so the refusal is tested from every
   direction it could be reached.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB, SqliteKVStore, _reset_singleton
from fantasy_baseball.data.redis_store import (
    STANDINGS_HISTORY_KEY,
    WEEKLY_ROSTERS_HISTORY_KEY,
)
from fantasy_baseball.manual.seed import (
    MANUAL_JOB_LABEL,
    MANUAL_SOURCE,
    PROVENANCE_KEY,
    ManualSeedRefused,
    SeedStats,
    assert_isolated_store,
    describe_kv_target,
    read_team_keys,
    resolve_kv_path,
    seed_manual_kv,
)
from fantasy_baseball.manual.transcripts import ManualRosterSnapshot, load_manual_rosters
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats, Standings, StandingsEntry

SEASON = 2026
SNAPSHOT = date(SEASON, 8, 22)

TEAM_A = "Hart of the Order"
TEAM_B = "Hello Peanuts!"

# Yahoo team keys are LOOKED UP from the store's existing cache:standings in
# every real code path. The two literals below are fixture inputs that the
# tests seed into that blob first and then read back -- never hand-typed ids
# standing in for real ones.
FIXTURE_KEY_A = "431.l.000000.t.1"
FIXTURE_KEY_B = "431.l.000000.t.2"


# --------------------------------------------------------------- fixtures


@pytest.fixture
def store(tmp_path):
    """An isolated SQLite KV store that is NOT named local.db."""
    kv = SqliteKVStore(tmp_path / "manual.db")
    try:
        yield kv
    finally:
        # SqliteKVStore has no public close(); Windows will not let pytest
        # remove tmp_path while the connection is open.
        kv._conn.close()


def _row(name: str, slot: str, positions: str, status: str = "") -> dict[str, str]:
    """A weekly_rosters_history row in the exact shape transcripts.py emits."""
    return {
        "slot": Position.parse(slot).value,
        "player_name": name,
        "positions": positions,
        "status": status,
        "yahoo_id": "",
    }


@pytest.fixture
def rosters() -> ManualRosterSnapshot:
    return ManualRosterSnapshot(
        snapshot_date=SNAPSHOT,
        rows_by_team={
            TEAM_A: [
                _row("Ceddanne Rafaela", "UTIL", "2B, SS, OF, Util"),
                _row("Juan Soto", "IL", "OF", status="IL10"),
                _row("Logan Webb", "P", "P"),
            ],
            TEAM_B: [
                _row("Elly De La Cruz", "SS", "SS"),
                _row("Jacob deGrom", "P", "P"),
            ],
        },
    )


def _standings(*, team_keys: dict[str, str] | None = None, eff: date = SNAPSHOT) -> Standings:
    keys = team_keys or {}
    return Standings(
        effective_date=eff,
        entries=[
            StandingsEntry(
                team_name=TEAM_A,
                team_key=keys.get(TEAM_A, ""),
                rank=1,
                stats=CategoryStats(),
                yahoo_points_for=79.0,
            ),
            StandingsEntry(
                team_name=TEAM_B,
                team_key=keys.get(TEAM_B, ""),
                rank=2,
                stats=CategoryStats(),
                yahoo_points_for=71.0,
            ),
        ],
    )


@pytest.fixture
def standings() -> Standings:
    return _standings()


class _RecordingClient:
    """KVStore stand-in that records writes instead of performing them.

    Used for the refusals that must not open a real database file (in
    particular the one aimed at the repo's own ``data/local.db``).
    """

    def __init__(self, path=None):
        if path is not None:
            self._path = path
        self.writes: list[tuple] = []

    def get(self, key):
        return None

    def set(self, key, value, *, ex=None):
        self.writes.append(("set", key))

    def hget(self, name, field):
        return None

    def hset(self, name, field, value):
        self.writes.append(("hset", name, field))


class _OrderedClient:
    """Delegating wrapper that appends every write to a shared event log."""

    def __init__(self, inner, events: list[str]):
        self._inner = inner
        self._path = inner._path
        self._events = events

    def get(self, key):
        return self._inner.get(key)

    def set(self, key, value, *, ex=None):
        self._events.append(f"WRITE set {key}")
        self._inner.set(key, value, ex=ex)

    def hget(self, name, field):
        return self._inner.hget(name, field)

    def hset(self, name, field, value):
        self._events.append(f"WRITE hset {name}")
        self._inner.hset(name, field, value)


def _rows_for(store, team: str, snapshot: str = SNAPSHOT.isoformat()) -> list[dict]:
    day = json.loads(store.hget(WEEKLY_ROSTERS_HISTORY_KEY, snapshot))
    return [r for r in day if r["team"] == team]


def _use_store_as_kv(monkeypatch, tmp_path_db) -> None:
    """Point the process-global get_kv() at a specific SQLite file."""
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path_db))
    _reset_singleton()


# ------------------------------------------------------- refusal / safety


def test_refuses_a_store_named_local_db(tmp_path, standings, rosters):
    """The Yahoo baseline is identified by file name, wherever it lives."""
    baseline = SqliteKVStore(tmp_path / "local.db")
    try:
        with pytest.raises(ManualSeedRefused) as exc:
            seed_manual_kv(baseline, standings, rosters, echo=lambda _msg: None)
        assert "local.db" in str(exc.value)

        # A refusal must leave the store exactly as it was.
        assert baseline.hgetall(WEEKLY_ROSTERS_HISTORY_KEY) == {}
        assert baseline.hgetall(STANDINGS_HISTORY_KEY) == {}
        assert baseline.get(redis_key(CacheKey.STANDINGS)) is None
        assert baseline.get(PROVENANCE_KEY) is None
    finally:
        baseline._conn.close()


def test_refuses_a_store_named_local_db_case_insensitively(tmp_path, standings, rosters):
    upper = SqliteKVStore(tmp_path / "LOCAL.DB")
    try:
        with pytest.raises(ManualSeedRefused):
            seed_manual_kv(upper, standings, rosters, echo=lambda _msg: None)
    finally:
        upper._conn.close()


def test_refuses_the_repo_baseline_path(standings, rosters):
    """Pinned against kv_store's own constant, and without opening the file.

    Constructing a real SqliteKVStore here would create WAL sidecars next to
    the production store, so the refusal is exercised through a stand-in that
    merely claims that path.
    """
    client = _RecordingClient(path=_DEFAULT_LOCAL_DB)
    with pytest.raises(ManualSeedRefused):
        seed_manual_kv(client, standings, rosters, echo=lambda _msg: None)
    assert client.writes == []


def test_refuses_a_client_with_no_local_file(standings, rosters):
    """An Upstash client has no path -- hand-typed rows must never reach prod."""
    client = _RecordingClient()
    with pytest.raises(ManualSeedRefused) as exc:
        seed_manual_kv(client, standings, rosters, echo=lambda _msg: None)
    assert "_RecordingClient" in str(exc.value)
    assert client.writes == []


@pytest.mark.parametrize("value", ["true", "1", "false"])
def test_refuses_when_render_is_set(monkeypatch, store, standings, rosters, value):
    """Stricter than is_remote(): ANY non-empty RENDER refuses."""
    monkeypatch.setenv("RENDER", value)
    with pytest.raises(ManualSeedRefused) as exc:
        seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    assert "RENDER" in str(exc.value)
    assert store.hgetall(WEEKLY_ROSTERS_HISTORY_KEY) == {}


def test_empty_render_is_not_a_refusal(monkeypatch, store, standings, rosters):
    monkeypatch.setenv("RENDER", "")
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    assert store.hget(WEEKLY_ROSTERS_HISTORY_KEY, SNAPSHOT.isoformat()) is not None


def test_assert_isolated_store_returns_the_resolved_path(store, tmp_path):
    assert assert_isolated_store(store) == (tmp_path / "manual.db").resolve()


def test_resolve_kv_path_and_describe_handle_a_pathless_client():
    client = _RecordingClient()
    assert resolve_kv_path(client) is None
    assert "no local file" in describe_kv_target(client)


def test_echoes_the_absolute_kv_path_before_the_first_write(store, standings, rosters):
    events: list[str] = []
    client = _OrderedClient(store, events)
    seed_manual_kv(client, standings, rosters, echo=lambda msg: events.append(f"ECHO {msg}"))

    assert events, "seeder produced neither echo nor write"
    first = events[0]
    assert first.startswith("ECHO ")
    assert str((store._path).resolve()) in first
    # And the path is echoed strictly before anything is written.
    first_write = next(i for i, e in enumerate(events) if e.startswith("WRITE "))
    assert first_write > 0


def test_default_echo_prints_the_path(capsys, store, standings, rosters):
    seed_manual_kv(store, standings, rosters)
    out = capsys.readouterr().out
    assert str(store._path.resolve()) in out
    assert "MANUAL" in out


def test_mismatched_vintages_are_announced(store, rosters):
    msgs: list[str] = []
    older = _standings(eff=date(SEASON, 8, 15))
    seed_manual_kv(store, older, rosters, echo=msgs.append)
    assert any("WARNING" in m and "vintages" in m for m in msgs)


def test_empty_transcription_raises_before_writing(store, standings):
    empty = ManualRosterSnapshot(snapshot_date=SNAPSHOT, rows_by_team={})
    with pytest.raises(ValueError, match="Nothing to seed"):
        seed_manual_kv(store, standings, empty, echo=lambda _msg: None)
    assert store.hgetall(WEEKLY_ROSTERS_HISTORY_KEY) == {}
    assert store.get(PROVENANCE_KEY) is None


# ------------------------------------------------------------- the seams


def test_seed_writes_weekly_rosters_history_and_standings_history(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)

    day = json.loads(store.hget(WEEKLY_ROSTERS_HISTORY_KEY, SNAPSHOT.isoformat()))
    assert {r["team"] for r in day} == {TEAM_A, TEAM_B}
    assert len(day) == 5

    hist = json.loads(store.hget(STANDINGS_HISTORY_KEY, SNAPSHOT.isoformat()))
    assert Standings.from_json(hist).entries[0].team_name == TEAM_A


def test_roster_rows_keep_the_yahoo_field_names(store, standings, rosters):
    """League.from_redis indexes these keys directly; renaming one breaks it."""
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    row = _rows_for(store, TEAM_A)[0]
    assert {"slot", "player_name", "positions", "status", "yahoo_id", "team"} <= set(row)


def test_seed_writes_cache_standings_envelope(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)

    envelope = json.loads(store.get(redis_key(CacheKey.STANDINGS)))
    assert set(envelope) == {"_meta", "_data"}
    meta = envelope["_meta"]
    # The three fields serialize_cache_payload always stamps ...
    assert {"_written_at", "_sha", "_job"} <= set(meta)
    assert meta["_job"] == MANUAL_JOB_LABEL
    # ... plus the manual provenance, which is the point.
    assert meta["_source"] == MANUAL_SOURCE
    assert meta["_manual"] is True
    assert meta["_yahoo"] is False
    assert meta["_manual_roster_snapshot"] == SNAPSHOT.isoformat()

    reparsed = Standings.from_json(envelope["_data"])
    assert reparsed.effective_date == SNAPSHOT
    assert [e.team_name for e in reparsed.entries] == [TEAM_A, TEAM_B]


def test_cache_standings_reads_back_through_read_cache_with_meta(
    monkeypatch, tmp_path, standings, rosters
):
    """The dashboard reader must see a well-formed envelope, not a bare blob."""
    from fantasy_baseball.web.season_data import read_cache_with_meta

    db = tmp_path / "manual.db"
    kv = SqliteKVStore(db)
    try:
        seed_manual_kv(kv, standings, rosters, echo=lambda _msg: None)
    finally:
        kv._conn.close()

    _use_store_as_kv(monkeypatch, db)
    payload, meta = read_cache_with_meta(CacheKey.STANDINGS)

    assert isinstance(payload, dict)
    assert Standings.from_json(payload).effective_date == SNAPSHOT
    assert meta["_source"] == MANUAL_SOURCE
    assert meta["_job"] == MANUAL_JOB_LABEL


def test_job_label_does_not_leak_after_the_seed(store, standings, rosters):
    """set_cache_job is reset in a finally, so later writes are not labelled manual."""
    from fantasy_baseball.web.season_data import serialize_cache_payload

    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    after = json.loads(serialize_cache_payload({"x": 1}))
    assert after["_meta"]["_job"] != MANUAL_JOB_LABEL


def test_seed_writes_only_to_the_injected_client(monkeypatch, tmp_path, standings, rosters):
    """write_cache_to takes an explicit client -- no process-global get_kv()."""
    from fantasy_baseball.data.kv_store import get_kv

    other = tmp_path / "elsewhere.db"
    _use_store_as_kv(monkeypatch, other)
    ambient = get_kv()

    target = SqliteKVStore(tmp_path / "manual.db")
    try:
        seed_manual_kv(target, standings, rosters, echo=lambda _msg: None)
        assert target.get(redis_key(CacheKey.STANDINGS)) is not None
    finally:
        target._conn.close()

    assert ambient.get(redis_key(CacheKey.STANDINGS)) is None
    assert ambient.hgetall(WEEKLY_ROSTERS_HISTORY_KEY) == {}


def test_reseeding_same_date_replaces_rather_than_appends(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)

    fixed = ManualRosterSnapshot(
        snapshot_date=SNAPSHOT,
        rows_by_team={
            TEAM_A: [_row("Logan Webb", "P", "P")],
            TEAM_B: rosters.rows_by_team[TEAM_B],
        },
    )
    stats = seed_manual_kv(store, standings, fixed, echo=lambda _msg: None)

    assert [r["player_name"] for r in _rows_for(store, TEAM_A)] == ["Logan Webb"]
    assert len(_rows_for(store, TEAM_B)) == 2
    assert stats.players == 3

    hist = store.hgetall(STANDINGS_HISTORY_KEY)
    assert list(hist) == [SNAPSHOT.isoformat()]


def test_reseeding_leaves_other_snapshot_dates_alone(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)

    later = date(SEASON, 8, 29)
    seed_manual_kv(
        store,
        _standings(eff=later),
        ManualRosterSnapshot(snapshot_date=later, rows_by_team=rosters.rows_by_team),
        echo=lambda _msg: None,
    )

    assert set(store.hgetall(WEEKLY_ROSTERS_HISTORY_KEY)) == {
        SNAPSHOT.isoformat(),
        later.isoformat(),
    }
    assert len(_rows_for(store, TEAM_A, SNAPSHOT.isoformat())) == 3


# ------------------------------------------------------------ provenance


def test_every_seeded_roster_row_is_stamped(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    day = json.loads(store.hget(WEEKLY_ROSTERS_HISTORY_KEY, SNAPSHOT.isoformat()))
    assert day and all(r["source"] == MANUAL_SOURCE for r in day)


def test_stamping_does_not_mutate_the_caller_rows(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    assert all("source" not in r for r in rosters.rows_by_team[TEAM_A])


def test_store_carries_a_provenance_breadcrumb(store, standings, rosters):
    seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    blob = json.loads(store.get(PROVENANCE_KEY))
    assert blob["source"] == MANUAL_SOURCE
    assert blob["yahoo"] is False
    assert blob["roster_snapshot_date"] == SNAPSHOT.isoformat()
    assert blob["teams"] == 2
    assert blob["players"] == 5
    assert str(store._path.resolve()) == blob["kv_path"]
    assert "NOT from the Yahoo API" in blob["note"]


def test_seed_stats_describes_the_write(store, standings, rosters):
    stats = seed_manual_kv(store, standings, rosters, echo=lambda _msg: None)
    assert stats == SeedStats(
        teams=2,
        players=5,
        snapshot_date=SNAPSHOT.isoformat(),
        standings_date=SNAPSHOT.isoformat(),
        kv_path=str(store._path.resolve()),
    )


# --------------------------------------------------------- read_team_keys


def _seed_yahoo_standings_blob(store, *, enveloped: bool) -> None:
    """Write a Yahoo-shaped cache:standings, as the bootstrap copy would carry."""
    data = _standings(team_keys={TEAM_A: FIXTURE_KEY_A, TEAM_B: FIXTURE_KEY_B}).to_json()
    if enveloped:
        from fantasy_baseball.web.season_data import serialize_cache_payload

        store.set(redis_key(CacheKey.STANDINGS), serialize_cache_payload(data))
    else:
        store.set(redis_key(CacheKey.STANDINGS), json.dumps(data))


def test_read_team_keys_from_the_enveloped_blob(store):
    _seed_yahoo_standings_blob(store, enveloped=True)
    assert read_team_keys(store) == {TEAM_A: FIXTURE_KEY_A, TEAM_B: FIXTURE_KEY_B}


def test_read_team_keys_from_a_bare_legacy_blob(store):
    _seed_yahoo_standings_blob(store, enveloped=False)
    assert read_team_keys(store) == {TEAM_A: FIXTURE_KEY_A, TEAM_B: FIXTURE_KEY_B}


def test_read_team_keys_missing_blob_is_empty_not_an_error(store):
    assert read_team_keys(store) == {}


def test_read_team_keys_survives_corrupt_or_odd_payloads(store):
    store.set(redis_key(CacheKey.STANDINGS), "{not json")
    assert read_team_keys(store) == {}
    store.set(redis_key(CacheKey.STANDINGS), json.dumps([1, 2, 3]))
    assert read_team_keys(store) == {}
    store.set(redis_key(CacheKey.STANDINGS), json.dumps({"teams": "nope"}))
    assert read_team_keys(store) == {}


def test_read_team_keys_omits_blank_keys(store):
    store.set(
        redis_key(CacheKey.STANDINGS),
        json.dumps({"teams": [{"name": TEAM_A, "team_key": ""}, {"name": TEAM_B}]}),
    )
    assert read_team_keys(store) == {}


def test_seeded_standings_carry_the_looked_up_team_keys(store, rosters):
    """End to end: keys come out of the store, not out of the YAML."""
    _seed_yahoo_standings_blob(store, enveloped=True)
    looked_up = read_team_keys(store)

    seed_manual_kv(store, _standings(team_keys=looked_up), rosters, echo=lambda _msg: None)

    envelope = json.loads(store.get(redis_key(CacheKey.STANDINGS)))
    reparsed = Standings.from_json(envelope["_data"])
    assert {e.team_name: e.team_key for e in reparsed.entries} == {
        TEAM_A: FIXTURE_KEY_A,
        TEAM_B: FIXTURE_KEY_B,
    }
    # And re-reading the now-manual blob still returns the same real keys, so
    # a second seed does not degrade them.
    assert read_team_keys(store) == looked_up


# ------------------------------------------------- League.from_redis seam


def test_league_from_redis_round_trips_the_seeded_snapshot(
    monkeypatch, tmp_path, standings, rosters
):
    from fantasy_baseball.models.league import League

    db = tmp_path / "manual.db"
    kv = SqliteKVStore(db)
    try:
        seed_manual_kv(kv, standings, rosters, echo=lambda _msg: None)
    finally:
        kv._conn.close()

    _use_store_as_kv(monkeypatch, db)
    league = League.from_redis(SEASON)

    assert {t.name for t in league.teams} == {TEAM_A, TEAM_B}
    roster = league.team_by_name(TEAM_A).latest_roster()
    assert roster.effective_date == SNAPSHOT
    assert len(roster.entries) == 3

    by_name = {e.name: e for e in roster.entries}

    # The IL player keeps his badge -- several teams park an IL'd player in an
    # active slot, and status is the only way to see it.
    soto = by_name["Juan Soto"]
    assert soto.status == "IL10"
    assert soto.selected_position == Position.IL

    # Multi-position eligibility survives the string round-trip.
    rafaela = by_name["Ceddanne Rafaela"]
    assert rafaela.positions == Position.parse_list("2B, SS, OF, Util")
    assert rafaela.selected_position == Position.UTIL
    assert rafaela.status == ""

    assert league.latest_standings().effective_date == SNAPSHOT


def test_league_from_redis_ignores_the_provenance_marker(monkeypatch, tmp_path, standings, rosters):
    """The extra ``source`` field must be inert to every existing reader."""
    from fantasy_baseball.models.league import League

    db = tmp_path / "manual.db"
    kv = SqliteKVStore(db)
    try:
        seed_manual_kv(kv, standings, rosters, echo=lambda _msg: None)
    finally:
        kv._conn.close()

    _use_store_as_kv(monkeypatch, db)
    entry = League.from_redis(SEASON).team_by_name(TEAM_B).latest_roster().entries[0]
    assert not hasattr(entry, "source")
    assert entry.yahoo_id == ""


def test_rows_from_the_real_loader_survive_the_round_trip(monkeypatch, tmp_path):
    """transcripts.load_manual_rosters output goes in unmodified and comes back."""
    from fantasy_baseball.models.league import League

    yaml_path = tmp_path / "rosters.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f'snapshot_date: "{SNAPSHOT.isoformat()}"',
                "teams:",
                f'  - name: "{TEAM_A}"',
                "    players:",
                '      - name: "Ceddanne Rafaela"',
                '        slot: "UTIL"',
                '        positions: "2B, SS, OF, Util"',
                '      - name: "Juan Soto"',
                '        slot: "IL"',
                '        positions: "OF"',
                '        status: "IL10"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_manual_rosters(yaml_path)

    db = tmp_path / "manual.db"
    kv = SqliteKVStore(db)
    try:
        stats = seed_manual_kv(kv, _standings(), loaded, echo=lambda _msg: None)
    finally:
        kv._conn.close()
    assert stats.players == 2

    _use_store_as_kv(monkeypatch, db)
    entries = League.from_redis(SEASON).team_by_name(TEAM_A).latest_roster().entries
    assert {e.name for e in entries} == {"Ceddanne Rafaela", "Juan Soto"}
    assert {e.name: e.status for e in entries}["Juan Soto"] == "IL10"
