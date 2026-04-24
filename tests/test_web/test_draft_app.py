import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers: []\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
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


def test_reset_deletes_state(client, tmp_path):
    client.post("/api/new-draft")
    r = client.post("/api/reset", json={"confirm": "RESET"})
    assert r.status_code == 200
    assert not (tmp_path / "draft_state.json").exists()


def test_reset_without_confirm_rejected(client):
    client.post("/api/new-draft")
    r = client.post("/api/reset", json={})
    assert r.status_code == 400


def test_recs_endpoint_exists(client):
    client.post("/api/new-draft")
    r = client.get("/api/recs?team=Hart of the Order")
    # Phase 4 wires the real recommender; for now the endpoint exists but
    # is marked 501 so the frontend knows to render a placeholder.
    assert r.status_code == 501


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
            RecRow("p1::hitter", "Player One", ["OF"], 3.2, 0.5, 1.1, {"HR": 1.2}),
            RecRow("p2::hitter", "Player Two", ["SS"], 2.1, 0.3, 0.4, {"SB": 0.9}),
        ]

    monkeypatch.setattr(eroto_recs, "rank_candidates", fake_rank)

    from fantasy_baseball.web import app as web_app

    monkeypatch.setattr(
        web_app, "_build_rec_inputs", lambda *_a, **_kw: (None, None, None, None, None)
    )
    monkeypatch.setattr(web_app, "_load_board_cached", lambda: None)
    monkeypatch.setattr(web_app, "_picks_until_next_turn", lambda state, team: 3)

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
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Jose Ramirez, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

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
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Nobody Important, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    write_board([], tmp_path / "draft_state_board.json")

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 400
        assert "Nobody Important" in r.get_json()["error"]
