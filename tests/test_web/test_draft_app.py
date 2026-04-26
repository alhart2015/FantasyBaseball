import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers: []\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    # Stub the board rebuild — these tests don't need real SQLite data.
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)
    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    return a


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


def test_new_draft_seeds_state(client):
    r = client.post("/api/new-draft")
    assert r.status_code == 200
    body = r.get_json()
    assert body["on_the_clock"] == "Hart of the Order"
    assert body["picks"] == []


def test_pick_then_undo_round_trips(client):
    client.post("/api/new-draft")
    r = client.post(
        "/api/pick",
        json={
            "player_id": "P1::hitter",
            "player_name": "Player One",
            "position": "OF",
            "team": "Hart of the Order",
        },
    )
    assert r.status_code == 200
    assert r.get_json()["on_the_clock"] == "Opp"

    r = client.post("/api/undo")
    assert r.status_code == 200
    assert r.get_json()["on_the_clock"] == "Hart of the Order"


def test_pick_wrong_team_rejected(client):
    client.post("/api/new-draft")
    r = client.post(
        "/api/pick",
        json={
            "player_id": "P1::hitter",
            "player_name": "Player One",
            "position": "OF",
            "team": "Opp",
        },
    )
    assert r.status_code == 409


def test_recs_endpoint_exists(client):
    client.post("/api/new-draft")
    r = client.get("/api/recs?team=Hart of the Order")
    # With no board file on disk, the real-data path can't run — the
    # endpoint signals 503 so the frontend can render a placeholder.
    assert r.status_code == 503


def test_meta_returns_teams_and_user_team(client):
    """/api/meta is fetched once on page load to populate the team-picker
    dropdown and pick the default team."""
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.get_json()
    assert body["teams"] == ["Hart of the Order", "Opp"]
    assert body["user_team"] == "Hart of the Order"


def test_pick_missing_fields_returns_400(client):
    client.post("/api/new-draft")
    r = client.post("/api/pick", json={"player_id": "P1::hitter"})  # missing most fields
    assert r.status_code == 400
    body = r.get_json()
    assert "missing" in body["error"].lower()


def test_on_the_clock_missing_team_returns_400(client):
    client.post("/api/new-draft")
    r = client.post("/api/on-the-clock", json={})
    assert r.status_code == 400


def test_recs_returns_ranked_rows(client, monkeypatch):
    """/api/recs returns a list of rec rows sorted by immediate_delta desc."""
    from fantasy_baseball.draft import eroto_recs
    from fantasy_baseball.draft.eroto_recs import RecRow

    def fake_rank(**_kwargs):
        return [
            RecRow("p1::hitter", "Player One", ["OF"], 3.2, 1.1, {"HR": 1.2}),
            RecRow("p2::hitter", "Player Two", ["SS"], 2.1, 0.4, {"SB": 0.9}),
        ]

    monkeypatch.setattr(eroto_recs, "rank_candidates", fake_rank)

    from fantasy_baseball.draft.recs_integration import RecInputs
    from fantasy_baseball.web import app as web_app

    fake_inputs = RecInputs(
        candidates=[],
        replacements={},
        projected_standings=None,  # type: ignore[arg-type]
        team_sds={},
        adp_table=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(web_app, "_build_rec_inputs", lambda *_a, **_kw: fake_inputs)
    monkeypatch.setattr(web_app, "_load_board_cached", lambda _app: None)
    monkeypatch.setattr(web_app, "_picks_until_next_turn", lambda state, team, league_yaml: 3)

    client.post("/api/new-draft")
    r = client.get("/api/recs?team=Hart of the Order")
    assert r.status_code == 200
    body = r.get_json()
    assert body[0]["name"] == "Player One"
    assert body[0]["immediate_delta"] == 3.2
    assert body[0]["per_category"]["HR"] == 1.2


def test_roster_endpoint_returns_empty_for_fresh_draft(client):
    client.post("/api/new-draft")
    r = client.get("/api/roster?team=Hart of the Order")
    assert r.status_code == 200
    body = r.get_json()
    # Empty league.yaml has no roster_slots -> no replacement padding.
    assert body == []


def test_roster_endpoint_reflects_a_pick(client):
    client.post("/api/new-draft")
    client.post(
        "/api/pick",
        json={
            "player_id": "P1::hitter",
            "player_name": "Player One",
            "position": "OF",
            "team": "Hart of the Order",
        },
    )
    r = client.get("/api/roster?team=Hart of the Order")
    assert r.status_code == 200
    body = r.get_json()
    assert any(row["name"] == "Player One" for row in body)


def test_roster_missing_team_returns_400(client):
    client.post("/api/new-draft")
    r = client.get("/api/roster")
    assert r.status_code == 400


def test_standings_endpoint_returns_empty_list_with_empty_cache(client):
    client.post("/api/new-draft")
    r = client.get("/api/standings")
    assert r.status_code == 200
    assert r.get_json() == []


def test_resolve_keeper_finds_pitcher_by_normalized_name(tmp_path, monkeypatch):
    """Real keeper resolver must honor player_type and best_position
    from the board — not hardcode hitter/OF."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    # Seed league.yaml with one pitcher keeper.
    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Tarik Skubal, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    # Seed a tiny board.
    board_path = tmp_path / "draft_state_board.json"
    write_board(
        [
            {
                "name": "Tarik Skubal",
                "name_normalized": "tarik skubal",
                "player_id": "12345::pitcher",
                "player_type": "pitcher",
                "positions": ["SP"],
                "best_position": "SP",
                "var": 8.5,
            },
        ],
        board_path,
    )

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 200
        state = r.get_json()

    assert len(state["keepers"]) == 1
    keeper = state["keepers"][0]
    assert keeper["player_id"] == "12345::pitcher"  # real ID, not {name}::hitter
    assert keeper["position"] == "SP"  # real position, not "OF"
    assert keeper["player_name"] == "Tarik Skubal"


def test_resolve_keeper_tie_breaks_by_var(tmp_path, monkeypatch):
    """When two board rows share a normalized name, pick the one with
    higher VAR (real player vs namesake)."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Jose Ramirez, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    board_path = tmp_path / "draft_state_board.json"
    # Two Jose Ramirezes: the real one (high VAR) and a namesake (low VAR).
    write_board(
        [
            {
                "name": "Jose Ramirez",
                "name_normalized": "jose ramirez",
                "player_id": "111::hitter",
                "player_type": "hitter",
                "positions": ["3B"],
                "best_position": "3B",
                "var": 9.2,
            },
            {
                "name": "Jose Ramirez",
                "name_normalized": "jose ramirez",
                "player_id": "222::pitcher",
                "player_type": "pitcher",
                "positions": ["SP"],
                "best_position": "SP",
                "var": 0.5,
            },
        ],
        board_path,
    )

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 200
        state = r.get_json()

    assert state["keepers"][0]["player_id"] == "111::hitter"


def test_resolve_keeper_missing_player_returns_400(tmp_path, monkeypatch):
    """Keeper not on board → UnresolvedKeeperError → HTTP 400."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Nobody Important, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)
    write_board([], tmp_path / "draft_state_board.json")

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 400
        assert "Nobody Important" in r.get_json()["error"]


# Shared 4-player board fixture for integration tests that need real
# stats: 2 hitters, 2 pitchers, enough to exercise project_team_stats.
_INTEGRATION_BOARD_ROWS = [
    {
        "name": "Slugger",
        "name_normalized": "slugger",
        "player_id": "1::hitter",
        "player_type": "hitter",
        "positions": ["OF"],
        "best_position": "OF",
        "var": 9.0,
        "total_sgp": 9.0,
        "r": 100.0,
        "hr": 40.0,
        "rbi": 110.0,
        "sb": 5.0,
        "avg": 0.280,
        "h": 160.0,
        "ab": 580.0,
    },
    {
        "name": "Slapper",
        "name_normalized": "slapper",
        "player_id": "2::hitter",
        "player_type": "hitter",
        "positions": ["OF"],
        "best_position": "OF",
        "var": 5.0,
        "total_sgp": 5.0,
        "r": 85.0,
        "hr": 10.0,
        "rbi": 55.0,
        "sb": 12.0,
        "avg": 0.300,
        "h": 170.0,
        "ab": 570.0,
    },
    {
        "name": "Ace",
        "name_normalized": "ace",
        "player_id": "3::pitcher",
        "player_type": "pitcher",
        "positions": ["SP"],
        "best_position": "SP",
        "var": 8.0,
        "total_sgp": 8.0,
        "w": 15.0,
        "k": 220.0,
        "sv": 0.0,
        "era": 3.10,
        "whip": 1.05,
        "ip": 195.0,
        "h_allowed": 160.0,
        "er": 67.0,
        "bb": 50.0,
    },
    {
        "name": "Closer",
        "name_normalized": "closer",
        "player_id": "4::pitcher",
        "player_type": "pitcher",
        "positions": ["RP"],
        "best_position": "RP",
        "var": 6.0,
        "total_sgp": 6.0,
        "w": 3.0,
        "k": 90.0,
        "sv": 35.0,
        "era": 2.50,
        "whip": 0.95,
        "ip": 65.0,
        "h_allowed": 50.0,
        "er": 18.0,
        "bb": 22.0,
    },
]


_INTEGRATION_LEAGUE_YAML = (
    "league:\n  team_name: Hart of the Order\n"
    "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
    "keepers: []\n"
    "roster_slots:\n  OF: 1\n  SP: 1\n  RP: 1\n  BN: 0\n  IL: 0\n"
)


def test_recs_returns_real_rows_with_board_and_picks(tmp_path, monkeypatch):
    """Integration: with a real board file on disk, /api/recs returns
    RecRow dicts with non-trivial immediate_delta values."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    board_path = tmp_path / "draft_state_board.json"
    write_board(_INTEGRATION_BOARD_ROWS, board_path)

    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        r = c.get("/api/recs?team=Hart of the Order")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert isinstance(body, list)
    assert len(body) >= 1
    # Plausibility: at least one candidate has a non-zero immediate_delta.
    assert any(abs(row["immediate_delta"]) > 1e-6 for row in body), body


def test_standings_endpoint_returns_real_rows_after_pick(tmp_path, monkeypatch):
    """Integration: with a real board on disk, /api/pick refreshes
    projected_standings_cache and /api/standings renders it."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    board_path = tmp_path / "draft_state_board.json"
    write_board(_INTEGRATION_BOARD_ROWS, board_path)

    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        c.post(
            "/api/pick",
            json={
                "player_id": "1::hitter",
                "player_name": "Slugger",
                "position": "OF",
                "team": "Hart of the Order",
            },
        )
        r = c.get("/api/standings")
    assert r.status_code == 200
    rows = r.get_json()
    assert len(rows) == 2  # two teams
    assert all(isinstance(row["total"], int | float) for row in rows)
    assert rows[0]["total"] >= rows[1]["total"]  # sorted desc


def test_roster_endpoint_organizes_by_slot_with_replacements(tmp_path, monkeypatch):
    """Slot-aware /api/roster. With OF=1, SP=1, RP=1 and one OF pick, the
    response is one row per slot capacity: filled OF + replacement SP/RP.
    """
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    write_board(_INTEGRATION_BOARD_ROWS, tmp_path / "draft_state_board.json")
    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        c.post(
            "/api/pick",
            json={
                "player_id": "1::hitter",
                "player_name": "Slugger",
                "position": "OF",
                "team": "Hart of the Order",
            },
        )
        r = c.get("/api/roster?team=Hart of the Order")

    assert r.status_code == 200
    rows = r.get_json()
    slots = {row["slot"]: row for row in rows}
    # OF was the pick → assigned to OF slot, not "Replacement".
    assert slots["OF"]["name"] == "Slugger"
    assert slots["OF"]["replacement"] is False
    # SP and RP are unfilled — they show "Replacement", not the OF pick.
    assert slots["SP"]["name"] == "Replacement"
    assert slots["SP"]["replacement"] is True
    assert slots["RP"]["replacement"] is True


def test_rec_inputs_cached_across_attach_standings_and_recs(tmp_path, monkeypatch):
    """Regression: a single pick should compute_rec_inputs only ONCE.

    _attach_standings_cache used to call compute_rec_inputs and discard
    the result; /api/recs would then call it again moments later. Now
    both go through _build_rec_inputs which caches on the keepers+picks
    fingerprint.
    """
    from fantasy_baseball.draft import recs_integration
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    board_path = tmp_path / "draft_state_board.json"
    write_board(_INTEGRATION_BOARD_ROWS, board_path)

    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    call_count = {"n": 0}
    real_compute = recs_integration.compute_rec_inputs

    def counting_compute(*args, **kwargs):
        call_count["n"] += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(recs_integration, "compute_rec_inputs", counting_compute)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        baseline = call_count["n"]
        c.post(
            "/api/pick",
            json={
                "player_id": "1::hitter",
                "player_name": "Slugger",
                "position": "OF",
                "team": "Hart of the Order",
            },
        )
        # /api/pick triggered _attach_standings_cache (1 compute). The
        # subsequent /api/recs call must hit the cache.
        after_pick = call_count["n"]
        c.get("/api/recs?team=Opp")
        after_recs = call_count["n"]

    assert after_pick - baseline == 1, "pick should compute inputs exactly once"
    assert after_recs == after_pick, "/api/recs should reuse cached inputs from the pick"


def test_recs_excludes_drafted_players_from_candidates(tmp_path, monkeypatch):
    """Drafted players (keepers + picks) must not appear in /api/recs."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    board_path = tmp_path / "draft_state_board.json"
    write_board(_INTEGRATION_BOARD_ROWS, board_path)

    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    monkeypatch.setattr(web_app, "rebuild_board", lambda *a, **kw: 0)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        # Pick the highest-VAR hitter ("Slugger" in the integration fixture).
        c.post(
            "/api/pick",
            json={
                "player_id": "1::hitter",
                "player_name": "Slugger",
                "position": "OF",
                "team": "Hart of the Order",
            },
        )
        r = c.get("/api/recs?team=Opp")
    assert r.status_code == 200
    body = r.get_json()
    drafted_ids = {row["player_id"] for row in body}
    assert "1::hitter" not in drafted_ids


def test_new_draft_triggers_board_rebuild(tmp_path, monkeypatch):
    """POSTing to /api/new-draft must rebuild the board JSON before
    seeding draft state. Otherwise the board on disk could be stale
    (e.g., regenerated before backfill_blending was removed) and the
    fresh-start draft would silently use wrong projections."""
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers: []\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    rebuild_calls: list[tuple] = []

    def _fake_rebuild(config_path, board_path):
        rebuild_calls.append((config_path, board_path))
        return 42

    monkeypatch.setattr(web_app, "rebuild_board", _fake_rebuild)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as client:
        r = client.post("/api/new-draft")

    assert r.status_code == 200
    assert len(rebuild_calls) == 1, "rebuild_board should be called exactly once per /api/new-draft"
    cfg_path, board_path = rebuild_calls[0]
    assert str(cfg_path) == str(league_path)
    assert board_path == a.config["BOARD_PATH"]


def test_new_draft_returns_500_when_rebuild_fails(tmp_path, monkeypatch):
    """If the board rebuild raises, the route must return an error
    rather than silently proceeding with a stale board."""
    from fantasy_baseball.web import app as web_app
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers: []\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    def _fail(*_a, **_kw):
        raise RuntimeError("simulated SQLite outage")

    monkeypatch.setattr(web_app, "rebuild_board", _fail)

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as client:
        r = client.post("/api/new-draft")

    assert r.status_code == 500
    assert "board rebuild failed" in r.get_json()["error"]
