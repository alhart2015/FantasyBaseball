"""The slot normalization is this module's whole reason for existing, and
`calculate_var` silently SKIPS a slot it does not recognize -- so a casing miss
charges a hitter the wrong replacement level instead of failing."""

import json
from pathlib import Path

from fantasy_baseball.keepers import positions as module


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "player_positions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _no_live(monkeypatch):
    monkeypatch.setattr(module, "_live_positions", dict)


def test_yahoo_lowercase_util_is_canonicalized(tmp_path, monkeypatch):
    """Yahoo writes "Util"; `sgp.replacement` keys on "UTIL". Left alone the slot
    matches nothing and the player silently nets against a different floor."""
    _no_live(monkeypatch)
    path = _write(tmp_path, {"Freddie Freeman": ["1B", "Util"]})
    assert module.load_positions(path)["freddie freeman"] == ["1B", "UTIL"]


def test_a_suffixed_outfield_slot_collapses_to_of(tmp_path, monkeypatch):
    _no_live(monkeypatch)
    path = _write(tmp_path, {"Byron Buxton": ["OF2", "Util"]})
    assert module.load_positions(path)["byron buxton"] == ["OF", "UTIL"]


def test_an_unrecognized_slot_is_dropped_not_passed_through(tmp_path, monkeypatch):
    _no_live(monkeypatch)
    path = _write(tmp_path, {"Someone": ["SS", "ZZZ"]})
    assert module.load_positions(path)["someone"] == ["SS"]


def test_a_player_with_no_usable_slot_is_omitted(tmp_path, monkeypatch):
    """Better absent than present-with-an-empty-list: the caller's FALLBACK_POS
    then charges him the deepest floor rather than skipping the netting."""
    _no_live(monkeypatch)
    path = _write(tmp_path, {"Nobody": ["ZZZ"], "Real": ["C"]})
    assert module.load_positions(path) == {"real": ["C"]}


def test_names_are_keyed_normalized_so_accents_join(tmp_path, monkeypatch):
    _no_live(monkeypatch)
    path = _write(tmp_path, {"José Ramírez": ["3B"]})
    assert "jose ramirez" in module.load_positions(path)


def test_the_live_blob_wins_where_both_know_a_player(tmp_path, monkeypatch):
    """Yahoo eligibility moves during a season; the committed file is a snapshot."""
    monkeypatch.setattr(module, "_live_positions", lambda: {"ivan herrera": ["C", "1B"]})
    path = _write(tmp_path, {"Ivan Herrera": ["C"]})
    assert module.load_positions(path)["ivan herrera"] == ["C", "1B"]


def test_the_local_file_still_contributes_names_the_blob_lacks(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_live_positions", lambda: {"ivan herrera": ["C"]})
    path = _write(tmp_path, {"Ivan Herrera": ["C"], "Otto Lopez": ["2B"]})
    merged = module.load_positions(path)
    assert set(merged) == {"ivan herrera", "otto lopez"}


def test_a_missing_local_file_is_not_fatal(tmp_path, monkeypatch):
    """The blob alone is enough; `load_positions_cache` would raise on its own."""
    monkeypatch.setattr(module, "_live_positions", lambda: {"otto lopez": ["2B"]})
    assert module.load_positions(tmp_path / "absent.json") == {"otto lopez": ["2B"]}


def test_an_unreachable_blob_degrades_to_the_local_file(tmp_path, monkeypatch):
    """The main caller is an offline script with no credentials. Patches the real
    import target rather than a module attribute -- `_live_positions` imports it
    lazily, so patching a name on this module would silently hit the network."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("fantasy_baseball.web.season_data.read_cache_dict", boom)
    path = _write(tmp_path, {"Otto Lopez": ["2B"]})
    assert module.load_positions(path) == {"otto lopez": ["2B"]}


def test_a_non_list_slot_value_is_ignored(tmp_path, monkeypatch):
    _no_live(monkeypatch)
    path = _write(tmp_path, {"Broken": "C", "Fine": ["C"]})
    assert module.load_positions(path) == {"fine": ["C"]}
