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
