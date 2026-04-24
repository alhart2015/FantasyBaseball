"""Hard isolation guarantees for the draft pipeline.

The live-draft code path MUST NOT:
- Import kv_store / redis_store / cache_keys.
- Write outside data/draft_state*.json.
- Call anything that writes to Redis.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

BANNED_MODULES = {
    "fantasy_baseball.data.kv_store",
    "fantasy_baseball.data.kv_sync",
    "fantasy_baseball.data.redis_store",
    "fantasy_baseball.data.cache_keys",
}

DRAFT_LIVE_MODULES = [
    "fantasy_baseball.draft.draft_controller",
    "fantasy_baseball.draft.eroto_recs",
    "fantasy_baseball.draft.adp",
    "fantasy_baseball.draft.state",
    "fantasy_baseball.draft.roster_state",
    "fantasy_baseball.web.app",
]


def test_live_draft_modules_do_not_import_redis_or_kv():
    """Assert no banned module is imported (directly or transitively) by any
    live-draft module. Covers both ``import X`` and ``from X import Y`` forms.

    The direct ``sys.modules`` intersection is run in a subprocess so the
    baseline is clean: other tests in this session (Redis/KV coverage) load
    the banned modules themselves, and an in-process check would always fail.
    The attribute-walk form runs in-process as a secondary defence against
    ``from X import Y`` where ``Y.__module__`` points back at ``X``.
    """
    probe = (
        "import importlib, sys\n"
        f"DRAFT_LIVE = {DRAFT_LIVE_MODULES!r}\n"
        f"BANNED = {sorted(BANNED_MODULES)!r}\n"
        "for m in DRAFT_LIVE:\n"
        "    importlib.import_module(m)\n"
        "leaked = [m for m in BANNED if m in sys.modules]\n"
        "if leaked:\n"
        "    print('LEAKED:' + ','.join(leaked))\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Live-draft modules transitively imported banned modules.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Attribute-walk check: seeds the import graph in-process, then walks
    # ``__module__`` references from module attributes. Catches
    # ``from X import Y`` where Y.__module__ traces back to a banned module.
    # (``import X as X`` style references are caught by the subprocess check
    # above, since module objects expose ``__module__ is None``.)
    for mod_name in DRAFT_LIVE_MODULES:
        importlib.import_module(mod_name)

    visited: set[str] = set()
    stack = list(DRAFT_LIVE_MODULES)
    while stack:
        mod_name = stack.pop()
        if mod_name in visited:
            continue
        visited.add(mod_name)
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            attr_mod = getattr(attr, "__module__", None)
            if attr_mod and attr_mod.startswith("fantasy_baseball.") and attr_mod not in visited:
                stack.append(attr_mod)

    leaked_walk = visited & BANNED_MODULES
    assert not leaked_walk, f"Attribute walk found banned module reference: {sorted(leaked_walk)}"


def test_draft_flow_touches_only_draft_state_files(tmp_path: Path, monkeypatch):
    """Run a synthetic draft sequence and verify no other files change."""
    from fantasy_baseball.web.app import create_app

    # Seed a minimal league.yaml stub in tmp_path.
    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers: []\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    app = create_app(state_path=tmp_path / "draft_state.json")
    app.config["TESTING"] = True

    snapshot_before = {p.name for p in tmp_path.iterdir()}

    with app.test_client() as client:
        r = client.post("/api/new-draft")
        assert r.status_code == 200
        r = client.post(
            "/api/pick",
            json={
                "player_id": "Juan Soto::hitter",
                "player_name": "Juan Soto",
                "position": "OF",
                "team": "Hart of the Order",
            },
        )
        assert r.status_code == 200

    snapshot_after = {p.name for p in tmp_path.iterdir()}
    new_files = snapshot_after - snapshot_before
    allowed = {"draft_state.json", "draft_state_board.json", "draft_state_delta.json"}
    unexpected = new_files - allowed
    assert not unexpected, f"draft flow created unexpected files: {unexpected}"
