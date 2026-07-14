# IL Return Planner Two-Scenario View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-14-il-return-two-scenario-design.md`

**Goal:** Show the IL Return Planner's drop/reshuffle decision under both the injury-reduced ROS projection and a healthy-remainder projection, side by side, so the manager sees whether the call hinges on the returning player's health.

**Architecture:** A pure `healthy_rest_of_season` transform inflates a returning player's ROS counting stats to a healthy remaining volume (derived from his own preseason pace prorated by `fraction_remaining`), preserving rates. A `plan_il_returns_scenarios` wrapper runs the existing, unmodified `plan_il_returns` twice — once as-is, once on a roster (and activating list) with the healthy swap applied — and reports both results plus a `tops_differ` flag. The route serializes both; the client-side JS renders two ranked lists with a robust/differs headline, falling back to the single list when no returnee is volume-suppressed.

**Tech Stack:** Python 3.12, dataclasses, Flask (season dashboard), Jinja + vanilla JS template, pytest.

## Global Constraints

- **ASCII-only** in all code, strings, log messages, and format strings (Windows cp1252 stdout). Use `->` not arrows, straight quotes, `--` not em-dash.
- **Player identity is `player_key` (`name::player_type`)** — never key returnees or drops on bare name in new logic.
- **No `x or default` for numeric defaults** — use explicit `is None` checks (a `0.0` volume must not be silently treated as missing).
- **`plan_il_returns` must not be modified** — the healthy scenario is produced by feeding it a roster variant, preserving its existing tests.
- **Every commit message ends with the trailer:**
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **End-of-effort verification** (run at repo root, fix every failure): `pytest -v` for the touched suites (`tests/test_lineup/test_il_return_planner.py`, `tests/test_web/test_season_routes.py`), `ruff check .`, `ruff format --check .`, `vulture` (no new findings), and `mypy` only if a touched file is listed under `[tool.mypy].files` in `pyproject.toml` (check the current list; `il_return_planner.py` / `season_routes.py` are not currently listed — state this in the final report).

---

## File Structure

- `src/fantasy_baseball/lineup/il_return_planner.py` — add `healthy_rest_of_season`, `IlReturnScenarios`, `_tops_differ`, `plan_il_returns_scenarios`. `plan_il_returns` and everything above it untouched.
- `src/fantasy_baseball/web/season_routes.py` — `api_il_return_plan` calls the new wrapper and serializes its `to_dict()`.
- `src/fantasy_baseball/web/templates/season/roster_audit.html` — client-side `render()` JS renders two lists + headline, single-list fallback.
- `tests/test_lineup/test_il_return_planner.py` — unit tests for the transform, `_tops_differ`, and the wrapper.
- `tests/test_web/test_season_routes.py` — route envelope test + roster-audit page smoke test.

---

### Task 1: `healthy_rest_of_season` transform

**Files:**
- Modify: `src/fantasy_baseball/lineup/il_return_planner.py` (add function near the top-level helpers, after imports)
- Test: `tests/test_lineup/test_il_return_planner.py` (new `TestHealthyRestOfSeason` class)

**Interfaces:**
- Consumes: `Player`, `PlayerType`, `HitterStats`, `PitcherStats` (already imported in the module / test file), `dataclasses` (already imported in the module).
- Produces: `healthy_rest_of_season(player: Player, fraction_remaining: float) -> Player | None` — a copy of `player` with `rest_of_season` scaled up to a healthy remaining volume, or `None` when no adjustment applies (no preseason, zero current volume, or `healthy_vol <= current_vol`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_lineup/test_il_return_planner.py`. The module already imports `HitterStats, PitcherStats, Player, PlayerType` and `Position`; add `healthy_rest_of_season` to the `from ...il_return_planner import (...)` block and `import pytest` if not already present.

```python
class TestHealthyRestOfSeason:
    def _cruz(self):
        # Injury-reduced ROS (175 PA) + healthy preseason (543 PA).
        ros = HitterStats(pa=175.0, ab=154.0, h=37.7, r=25.0, hr=8.0,
                          rbi=23.0, sb=10.0, g=40.0, avg=0.245, sgp=4.52)
        pre = HitterStats(pa=543.0, ab=478.0, h=114.0, r=74.0, hr=23.0,
                          rbi=68.0, sb=28.0, g=127.0, avg=0.239)
        return Player(name="Cruz", player_type=PlayerType.HITTER,
                      positions=[Position.OF], rest_of_season=ros, preseason=pre)

    def test_hitter_scales_volume_preserves_rate_and_clears_sgp(self):
        p = self._cruz()
        out = healthy_rest_of_season(p, fraction_remaining=0.41)
        assert out is not None
        scale = (543.0 * 0.41) / 175.0
        assert out.rest_of_season.pa == pytest.approx(543.0 * 0.41)
        assert out.rest_of_season.hr == pytest.approx(8.0 * scale)
        assert out.rest_of_season.sb == pytest.approx(10.0 * scale)
        assert out.rest_of_season.g == pytest.approx(40.0 * scale)
        assert out.rest_of_season.avg == pytest.approx(0.245)  # rate preserved
        assert out.rest_of_season.sgp is None                  # cached SGP cleared
        # Original object untouched (transform returns a copy).
        assert p.rest_of_season.pa == 175.0
        assert p.rest_of_season.sgp == 4.52

    def test_pitcher_scales_ip_and_gs_preserves_rate(self):
        ros = PitcherStats(ip=43.0, w=3.0, k=53.0, sv=0.0, er=17.0, bb=15.0,
                           h_allowed=40.0, g=9.0, gs=9.0, era=3.49, whip=1.22, sgp=3.48)
        pre = PitcherStats(ip=150.0, w=10.0, k=180.0, sv=0.0, er=60.0, bb=45.0,
                           h_allowed=130.0, g=28.0, gs=28.0, era=3.60, whip=1.17)
        p = Player(name="Snell", player_type=PlayerType.PITCHER,
                   positions=[Position.P], rest_of_season=ros, preseason=pre)
        out = healthy_rest_of_season(p, fraction_remaining=0.41)
        assert out is not None
        scale = (150.0 * 0.41) / 43.0
        assert out.rest_of_season.ip == pytest.approx(150.0 * 0.41)
        assert out.rest_of_season.k == pytest.approx(53.0 * scale)
        assert out.rest_of_season.gs == pytest.approx(9.0 * scale)
        assert out.rest_of_season.era == pytest.approx(3.49)   # rate preserved
        assert out.rest_of_season.whip == pytest.approx(1.22)
        assert out.rest_of_season.sgp is None

    def test_none_when_no_preseason(self):
        p = self._cruz()
        p.preseason = None
        assert healthy_rest_of_season(p, 0.41) is None

    def test_none_when_current_volume_zero(self):
        p = self._cruz()
        p.rest_of_season = HitterStats(pa=0.0, ab=0.0, h=0.0, r=0.0, hr=0.0,
                                       rbi=0.0, sb=0.0, g=0.0, avg=0.0)
        assert healthy_rest_of_season(p, 0.41) is None

    def test_none_when_not_volume_suppressed(self):
        # preseason.pa * fr = 500 * 0.41 = 205 <= current 300 -> no adjustment.
        p = self._cruz()
        p.rest_of_season = HitterStats(pa=300.0, ab=270.0, h=75.0, r=45.0,
                                       hr=12.0, rbi=40.0, sb=6.0, g=70.0, avg=0.278)
        p.preseason = HitterStats(pa=500.0, ab=450.0, h=125.0, r=70.0, hr=20.0,
                                  rbi=65.0, sb=10.0, g=150.0, avg=0.278)
        assert healthy_rest_of_season(p, 0.41) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestHealthyRestOfSeason -v`
Expected: FAIL / ERROR — `ImportError: cannot import name 'healthy_rest_of_season'`.

- [ ] **Step 3: Implement the transform**

Add to `src/fantasy_baseball/lineup/il_return_planner.py` (after the module imports, before `@dataclass class Move`). `dataclasses`, `Player`, and `PlayerType` are already imported.

```python
def healthy_rest_of_season(player: Player, fraction_remaining: float) -> Player | None:
    """Return a copy of ``player`` with ROS counting stats scaled up to a
    healthy remaining volume, or ``None`` when no adjustment applies.

    Healthy remaining volume = the player's pre-injury (``preseason``)
    full-season pace prorated to the games left: ``preseason.pa * fraction_
    remaining`` (hitters) / ``preseason.ip * fraction_remaining`` (pitchers).
    The current ROS counting stats are multiplied by ``healthy_vol /
    current_vol``, preserving rates (avg/era/whip are left untouched, so they
    stay correct since their components scale together). Games (``g``, and
    pitcher ``gs``) scale with volume. The cached ``sgp`` is cleared so any
    downstream read recomputes from the scaled line.

    Returns ``None`` (no healthy/limited difference to show) when preseason is
    absent, the current volume is zero, or the healthy volume would not exceed
    the current volume (the player is not volume-suppressed). The transform
    only ever inflates.
    """
    ros = player.rest_of_season
    pre = player.preseason
    if ros is None or pre is None:
        return None

    if player.player_type == PlayerType.PITCHER:
        current_vol = ros.ip
        healthy_vol = pre.ip * fraction_remaining
    else:
        current_vol = ros.pa
        healthy_vol = pre.pa * fraction_remaining

    if current_vol <= 0 or healthy_vol <= current_vol:
        return None

    scale = healthy_vol / current_vol
    if player.player_type == PlayerType.PITCHER:
        new_ros = dataclasses.replace(
            ros,
            ip=ros.ip * scale, w=ros.w * scale, k=ros.k * scale, sv=ros.sv * scale,
            er=ros.er * scale, bb=ros.bb * scale, h_allowed=ros.h_allowed * scale,
            g=ros.g * scale, gs=ros.gs * scale, sgp=None,
        )
    else:
        new_ros = dataclasses.replace(
            ros,
            pa=ros.pa * scale, ab=ros.ab * scale, h=ros.h * scale, r=ros.r * scale,
            hr=ros.hr * scale, rbi=ros.rbi * scale, sb=ros.sb * scale, g=ros.g * scale,
            sgp=None,
        )
    return dataclasses.replace(player, rest_of_season=new_ros)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestHealthyRestOfSeason -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(il-planner): healthy_rest_of_season volume-restore transform

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `plan_il_returns_scenarios` wrapper + `IlReturnScenarios`

**Files:**
- Modify: `src/fantasy_baseball/lineup/il_return_planner.py` (add `IlReturnScenarios`, `_tops_differ`, `plan_il_returns_scenarios` at the end of the module)
- Test: `tests/test_lineup/test_il_return_planner.py` (new `TestPlanIlReturnsScenarios` class)

**Interfaces:**
- Consumes: `plan_il_returns` and `IlReturnPlanResult` (same module), `healthy_rest_of_season` (Task 1), `Any` (already imported).
- Produces:
  - `IlReturnScenarios` dataclass: `as_projected: IlReturnPlanResult`, `if_healthy: IlReturnPlanResult | None`, `adjusted: list[dict[str, Any]]`, `tops_differ: bool`, with `to_dict() -> dict[str, Any]`.
  - `plan_il_returns_scenarios(roster, activating_il, roster_slots, *, projected_standings, team_name, fraction_remaining, team_sds=None, max_plans=5, sgp_overrides=None) -> IlReturnScenarios`.
  - `_tops_differ(a: IlReturnPlanResult, b: IlReturnPlanResult) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add `plan_il_returns_scenarios`, `IlReturnScenarios`, `_tops_differ` to the import block. This class reuses the existing `_full_hitters`, `_good_pitcher`, `_pitcher`, `_hitter`, `_standings`, and `SMALL_SLOTS` helpers already in the file.

```python
class TestPlanIlReturnsScenarios:
    def _roster_with_il_hitter(self):
        """9 regular hitters (fill the 9 active hitter slots) + a BN hitter +
        3 pitchers + an IL-slot returnee with a healthy preseason. Counted
        bodies = 9 + 1 BN + 3 P = 13 = SMALL_SLOTS capacity, so activating the
        IL-slot returnee brings the pool to 14 and forces exactly one drop
        (overflow 1)."""
        hitters = _full_hitters()  # 9 counted, fill the 9 hitter active slots
        bn1 = _hitter("BN1", ["OF"], slot="BN", r=40, hr=6, rbi=35, sb=2,
                      avg=0.240, ab=300, h=72)  # 13th counted body
        sp1 = _good_pitcher("SP1", k=160, era=3.4, whip=1.15)
        sp2 = _good_pitcher("SP2", k=155, era=3.5, whip=1.18)
        sp1.selected_position = Position.P
        sp2.selected_position = Position.P
        scrub = _pitcher("Scrub", slot="P")  # weak counted pitcher
        # IL-slot returnee: injury ROS (~177 PA, ab*1.15), healthy preseason (543 PA).
        cruz = _hitter("Cruz", ["OF"], slot="IL", r=25, hr=8, rbi=23, sb=10,
                       avg=0.245, ab=154, h=37)
        cruz.status = "IL10"
        cruz.preseason = HitterStats(pa=543.0, ab=478.0, h=114.0, r=74.0,
                                     hr=23.0, rbi=68.0, sb=28.0, g=127.0, avg=0.239)
        return [*hitters, bn1, sp1, sp2, scrub, cruz]

    def _call(self, roster, activating):
        return plan_il_returns_scenarios(
            roster, activating, SMALL_SLOTS,
            projected_standings=_standings(), team_name=TEAM_NAME,
            fraction_remaining=0.41, team_sds=None,
        )

    def test_suppressed_il_slot_returnee_produces_healthy_scenario(self):
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        res = self._call(roster, [cruz])

        assert res.as_projected.overflow == 1  # IL-slot returnee forces one drop
        assert res.if_healthy is not None
        assert len(res.adjusted) == 1
        adj = res.adjusted[0]
        assert adj["name"] == "Cruz"
        assert adj["vol_unit"] == "PA"
        cruz_pa = cruz.rest_of_season.pa  # _hitter sets pa = int(ab * 1.15) = 177
        assert adj["vol_projected"] == pytest.approx(round(cruz_pa, 1))
        assert adj["vol_healthy"] > adj["vol_projected"]
        # Healthy volume is preseason.pa prorated: 543 * 0.41 ~= 222.6 PA.
        assert adj["vol_healthy"] == pytest.approx(round(543.0 * 0.41, 1))

    def test_healthy_swap_reaches_the_activating_list_not_just_roster(self):
        # Regression: an IL-slot returnee enters _build_pool via the passed
        # activating_il list, so the healthy swap must reach it too. If it only
        # reached the roster copy, if_healthy would equal as_projected exactly.
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        res = self._call(roster, [cruz])
        assert res.if_healthy is not None
        assert res.if_healthy.to_dict() != res.as_projected.to_dict()

    def test_as_projected_reproduces_plan_il_returns_exactly(self):
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        res = self._call(roster, [cruz])
        direct = plan_il_returns(
            roster, [cruz], SMALL_SLOTS,
            projected_standings=_standings(), team_name=TEAM_NAME,
            fraction_remaining=0.41, team_sds=None,
        )
        assert res.as_projected.to_dict() == direct.to_dict()

    def test_no_adjustment_yields_null_if_healthy(self):
        # Returnee without preseason -> no healthy scenario, single-list fallback.
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        cruz.preseason = None
        res = self._call(roster, [cruz])
        assert res.if_healthy is None
        assert res.adjusted == []
        assert res.tops_differ is False

    def test_tops_differ_compares_top_drop_sets_order_independent(self):
        a = IlReturnPlanResult(activating=["Cruz"], capacity=13, overflow=1,
            plans=[MovePlan(drops=["Cruz"], moves=[], delta_roto=0.3, band={})])
        b = IlReturnPlanResult(activating=["Cruz"], capacity=13, overflow=1,
            plans=[MovePlan(drops=["Scrub"], moves=[], delta_roto=0.1, band={})])
        same = IlReturnPlanResult(activating=["Cruz"], capacity=13, overflow=1,
            plans=[MovePlan(drops=["Cruz"], moves=[], delta_roto=0.1, band={})])
        empty = IlReturnPlanResult(activating=["Cruz"], capacity=13, overflow=1, plans=[])
        assert _tops_differ(a, b) is True
        assert _tops_differ(a, same) is False   # same top drop set
        assert _tops_differ(a, empty) is False  # no top plan to compare
```

`MovePlan` and `IlReturnPlanResult` are already imported in the test file's import block; if not, add them.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestPlanIlReturnsScenarios -v`
Expected: FAIL / ERROR — `ImportError: cannot import name 'plan_il_returns_scenarios'` (and `IlReturnScenarios`, `_tops_differ`).

- [ ] **Step 3: Implement the wrapper**

Append to `src/fantasy_baseball/lineup/il_return_planner.py`. `asdict`, `dataclass`, `field`, `Any`, `Mapping`, `Player`, `PlayerType`, `Category`, `SgpOverrides`, `ProjectedStandings` are already imported at the top.

```python
@dataclass
class IlReturnScenarios:
    """The IL-return decision under both ROS volume assumptions.

    ``as_projected`` is the plan set on the roster as-is (injury-reduced
    volume). ``if_healthy`` is the plan set with each volume-suppressed
    returnee restored to a healthy remaining volume, or ``None`` when no
    returnee was adjusted (the caller then shows a single list). ``adjusted``
    lists the returnees that were restored with their projected-vs-healthy
    volume. ``tops_differ`` is True when both scenarios have a top plan and
    those top plans drop different players.
    """

    as_projected: IlReturnPlanResult
    if_healthy: IlReturnPlanResult | None
    adjusted: list[dict[str, Any]] = field(default_factory=list)
    tops_differ: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_projected": self.as_projected.to_dict(),
            "if_healthy": self.if_healthy.to_dict() if self.if_healthy is not None else None,
            "adjusted": list(self.adjusted),
            "tops_differ": self.tops_differ,
        }


def _tops_differ(a: IlReturnPlanResult, b: IlReturnPlanResult) -> bool:
    """True when both scenarios have a top plan (plans[0]) and their drop sets
    differ, compared order-independently (drops are name-sorted)."""
    if not a.plans or not b.plans:
        return False
    return set(a.plans[0].drops) != set(b.plans[0].drops)


def plan_il_returns_scenarios(
    roster: list[Player],
    activating_il: list[Player],
    roster_slots: dict[str, int],
    *,
    projected_standings: ProjectedStandings,
    team_name: str,
    fraction_remaining: float,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    max_plans: int = 5,
    sgp_overrides: SgpOverrides | None = None,
) -> IlReturnScenarios:
    """Plan IL returns under both the projected (injury) volume and a healthy
    remaining volume, for the ``adjusted`` returnees.

    Runs the unmodified :func:`plan_il_returns` twice. The healthy run applies
    :func:`healthy_rest_of_season` to each suppressed returnee in BOTH the
    roster copy and the ``activating_il`` list -- an IL-slot returnee enters
    ``_build_pool`` via the activating list (``extra`` path), so a roster-only
    swap would be a silent no-op. Returns ``if_healthy=None`` when no returnee
    is volume-suppressed.
    """
    as_projected = plan_il_returns(
        roster, activating_il, roster_slots,
        projected_standings=projected_standings, team_name=team_name,
        fraction_remaining=fraction_remaining, team_sds=team_sds,
        max_plans=max_plans, sgp_overrides=sgp_overrides,
    )

    healthy_by_key: dict[str, Player] = {}
    adjusted: list[dict[str, Any]] = []
    for p in activating_il:
        healthy = healthy_rest_of_season(p, fraction_remaining)
        if healthy is None:
            continue
        healthy_by_key[p.player_key] = healthy
        is_pitcher = p.player_type == PlayerType.PITCHER
        cur = p.rest_of_season
        new = healthy.rest_of_season
        adjusted.append(
            {
                "name": p.name,
                "player_type": p.player_type.value,
                "vol_unit": "IP" if is_pitcher else "PA",
                "vol_projected": round(cur.ip if is_pitcher else cur.pa, 1),
                "vol_healthy": round(new.ip if is_pitcher else new.pa, 1),
            }
        )

    if not healthy_by_key:
        return IlReturnScenarios(as_projected=as_projected, if_healthy=None)

    # Swap the healthy ROS into both the roster copy and the activating list,
    # keyed on player_key so an IL-slot returnee (sourced from activating_il in
    # _build_pool) carries the healthy line.
    healthy_roster = [healthy_by_key.get(p.player_key, p) for p in roster]
    activating_keys = {p.player_key for p in activating_il}
    healthy_activating = [p for p in healthy_roster if p.player_key in activating_keys]

    if_healthy = plan_il_returns(
        healthy_roster, healthy_activating, roster_slots,
        projected_standings=projected_standings, team_name=team_name,
        fraction_remaining=fraction_remaining, team_sds=team_sds,
        max_plans=max_plans, sgp_overrides=sgp_overrides,
    )

    return IlReturnScenarios(
        as_projected=as_projected,
        if_healthy=if_healthy,
        adjusted=adjusted,
        tops_differ=_tops_differ(as_projected, if_healthy),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestPlanIlReturnsScenarios -v`
Expected: PASS (5 tests). If `test_healthy_swap_reaches_the_activating_list_not_just_roster` fails with the two dicts equal, the swap is not reaching `activating_il` — recheck that `healthy_activating` is derived from `healthy_roster`, not the original `activating_il`.

- [ ] **Step 5: Run the full IL-planner suite (no regressions in `plan_il_returns`)**

Run: `pytest tests/test_lineup/test_il_return_planner.py -v`
Expected: PASS (all pre-existing tests + the new classes).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(il-planner): plan_il_returns_scenarios dual-scenario wrapper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Route wiring + envelope test

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py:791-830` (the `api_il_return_plan` route)
- Test: `tests/test_web/test_season_routes.py` (new tests)

**Interfaces:**
- Consumes: `plan_il_returns_scenarios` (Task 2) — imported inside the route function.
- Produces: `/api/il-return-plan` returns `jsonify(scenarios.to_dict())` — the envelope `{as_projected, if_healthy, adjusted, tops_differ}`. The 404 guards (no roster / no projected standings) and the `activate` id parsing are unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web/test_season_routes.py`. The file already has the `client` fixture and uses `patch(...)` on `fantasy_baseball.web.season_routes.*`.

```python
def test_il_return_plan_returns_scenario_envelope(client):
    from unittest.mock import patch
    from fantasy_baseball.lineup.il_return_planner import (
        IlReturnPlanResult, IlReturnScenarios,
    )

    scenarios = IlReturnScenarios(
        as_projected=IlReturnPlanResult(activating=["Cruz"], capacity=23, overflow=1, plans=[]),
        if_healthy=IlReturnPlanResult(activating=["Cruz"], capacity=23, overflow=1, plans=[]),
        adjusted=[{"name": "Cruz", "player_type": "hitter",
                   "vol_unit": "PA", "vol_projected": 175.0, "vol_healthy": 223.0}],
        tops_differ=True,
    )
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_list",
              return_value=[{"name": "Cruz", "player_type": "hitter",
                             "player_id": 11370, "status": "IL10",
                             "selected_position": "IL", "positions": ["OF"]}]),
        patch("fantasy_baseball.web.season_routes.read_cache_dict",
              return_value={"projected_standings": {"teams": []}, "team_sds": None,
                            "fraction_remaining": 0.41}),
        patch("fantasy_baseball.web.season_routes._projected_from_cache", return_value=object()),
        patch("fantasy_baseball.web.season_routes._team_sds_from_cache", return_value=None),
        patch("fantasy_baseball.lineup.il_return_planner.plan_il_returns_scenarios",
              return_value=scenarios),
    ):
        resp = client.get("/api/il-return-plan?activate=11370")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"as_projected", "if_healthy", "adjusted", "tops_differ"}
    assert body["tops_differ"] is True
    assert body["adjusted"][0]["vol_unit"] == "PA"


def test_il_return_plan_if_healthy_null_when_no_adjustment(client):
    from unittest.mock import patch
    from fantasy_baseball.lineup.il_return_planner import (
        IlReturnPlanResult, IlReturnScenarios,
    )

    scenarios = IlReturnScenarios(
        as_projected=IlReturnPlanResult(activating=["Buxton"], capacity=23, overflow=0, plans=[]),
        if_healthy=None,
    )
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_list",
              return_value=[{"name": "Buxton", "player_type": "hitter",
                             "player_id": 9590, "status": "IL10",
                             "selected_position": "BN", "positions": ["OF"]}]),
        patch("fantasy_baseball.web.season_routes.read_cache_dict",
              return_value={"projected_standings": {"teams": []}, "team_sds": None,
                            "fraction_remaining": 0.41}),
        patch("fantasy_baseball.web.season_routes._projected_from_cache", return_value=object()),
        patch("fantasy_baseball.web.season_routes._team_sds_from_cache", return_value=None),
        patch("fantasy_baseball.lineup.il_return_planner.plan_il_returns_scenarios",
              return_value=scenarios),
    ):
        resp = client.get("/api/il-return-plan?activate=9590")
    assert resp.status_code == 200
    assert resp.get_json()["if_healthy"] is None
```

Note: these patch `plan_il_returns_scenarios` at its definition module (`fantasy_baseball.lineup.il_return_planner`) because the route imports it inside the function body at call time.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_web/test_season_routes.py -k il_return_plan -v`
Expected: FAIL — the current route imports/calls `plan_il_returns` and returns its `IlReturnPlanResult.to_dict()` (keys `activating/capacity/overflow/plans/warning`), so `set(body)` will not equal the scenario envelope keys.

- [ ] **Step 3: Update the route**

In `src/fantasy_baseball/web/season_routes.py`, replace the import line and the final `result = plan_il_returns(...)` / `return jsonify(result.to_dict())` in `api_il_return_plan` (lines ~793 and ~820-830). Keep everything else (guards, `activate_ids` parsing, `il_players`/`activating` selection) unchanged.

```python
    @app.route("/api/il-return-plan")
    def api_il_return_plan():
        from fantasy_baseball.lineup.il_return_planner import plan_il_returns_scenarios
        from fantasy_baseball.models.player import Player

        activate_param = request.args.get("activate", "")
        activate_ids = {a for a in activate_param.split(",") if a}

        roster_raw = read_cache_list(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404
        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        ps_raw = proj_cache.get("projected_standings")
        if not ps_raw:
            return jsonify({"error": "No projected standings. Run a refresh first."}), 404

        config = _load_config()
        roster = [Player.from_dict(p) for p in roster_raw]
        projected = _projected_from_cache(ps_raw)
        team_sds = _team_sds_from_cache(proj_cache.get("team_sds"))
        fr = proj_cache.get("fraction_remaining")
        fr = 1.0 if fr is None else float(fr)

        il_players = [p for p in roster if p.is_on_il()]
        if activate_ids:
            activating = [p for p in il_players if (p.yahoo_id or p.name) in activate_ids]
        else:
            activating = il_players

        scenarios = plan_il_returns_scenarios(
            roster,
            activating,
            config.roster_slots,
            projected_standings=projected,
            team_name=config.team_name,
            fraction_remaining=fr,
            team_sds=team_sds,
            sgp_overrides=config.sgp_overrides,
        )
        return jsonify(scenarios.to_dict())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_web/test_season_routes.py -k il_return_plan -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py
git commit -m "feat(il-planner): serve dual-scenario IL return envelope from the route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Template two-list rendering + headline + fallback

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/roster_audit.html` (the `render(res)` JS function and helpers, ~lines 198-222)
- Test: `tests/test_web/test_season_routes.py` (roster-audit page smoke test)

**Interfaces:**
- Consumes: the JSON envelope from Task 3 (`res.as_projected`, `res.if_healthy`, `res.adjusted`, `res.tops_differ`), each `*_plan` result carrying `capacity`, `overflow`, `plans[]`, `warning`.
- Produces: client-side rendering — one list when `res.if_healthy` is null (today's behavior), two labeled lists + a robust/differs headline when it is non-null.

Note on verification: the plan results are rendered by client-side JS, not Jinja, so there is no server-side render test for the two-list output. Task 3's envelope test covers the data contract; this task adds a page smoke test (the section + JS load) and relies on manual/`verify`-skill visual confirmation during execution (drive `/roster-audit`, check a returnee's two lists).

- [ ] **Step 1: Write the failing smoke test**

Add to `tests/test_web/test_season_routes.py`.

```python
def test_roster_audit_page_renders_il_returns_scenario_js(client):
    from unittest.mock import patch
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/roster-audit")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # The dual-scenario client logic is present (static JS is served
    # unconditionally, outside the {% if audit %} guard).
    assert "if_healthy" in html
    assert "renderPlans" in html
    # Both column labels and both headline branches are present, so a deletion
    # of the labels or either headline branch fails CI (the runtime rendering is
    # JS and is verified manually in Step 5).
    assert "As projected" in html
    assert "If healthy" in html
    assert "depends on" in html   # tops_differ == true branch
    assert "Robust:" in html      # tops_differ == false branch
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_roster_audit_page_renders_il_returns_scenario_js -v`
Expected: FAIL — current template has neither `if_healthy` nor a `renderPlans` helper (the plan-rendering is inline in `render`).

- [ ] **Step 3: Update the template JS**

In `src/fantasy_baseball/web/templates/season/roster_audit.html`, replace the `render(res)` function (the block from `function render(res) {` through its closing `}` before `})();`) with a version that extracts single-result rendering into `renderPlans` and adds the two-list + headline logic. `gapClass` is unchanged and stays above.

```javascript
  function renderPlans(result) {
    if (result.warning) { return '<p>' + result.warning + '</p>'; }
    if (!result.plans || !result.plans.length) {
      return '<p>No legal plans found.</p>';
    }
    var html = '<p style="color:var(--text-secondary)">Roster cap ' + result.capacity +
      '; must drop ' + result.overflow + '.</p>';
    result.plans.forEach(function (p) {
      var b = p.band;
      var sign = b.mean >= 0 ? '+' : '';
      html += '<div class="il-plan" style="border:1px solid var(--border);border-radius:6px;padding:0.75rem;margin-bottom:0.75rem">';
      html += '<div><span class="gap-badge ' + gapClass(b.verdict) + '">' +
        sign + b.mean.toFixed(1) + '</span> ' +
        'Std dev ' + b.sd.toFixed(2) + ' &middot; P(helps) ' +
        Math.round(b.p_positive * 100) + '%</div>';
      html += '<ul style="margin:0.5rem 0 0;padding-left:1.25rem">';
      p.moves.forEach(function (m) {
        html += '<li>' + m.name + ': ' + m.from_slot + ' -&gt; ' + m.to_slot + '</li>';
      });
      html += '</ul></div>';
    });
    return html;
  }
  function adjustedNames(res) {
    return (res.adjusted || []).map(function (a) { return a.name; }).join(', ');
  }
  function volLabel(res) {
    return (res.adjusted || []).map(function (a) {
      return a.name + ' ' + Math.round(a.vol_projected) + ' -> ' +
        Math.round(a.vol_healthy) + ' ' + a.vol_unit;
    }).join('; ');
  }
  function render(res) {
    var out = document.getElementById('il-plan-results');
    if (res.error) { out.innerHTML = '<p>' + res.error + '</p>'; return; }
    // Single-list fallback: no volume-suppressed returnee to bracket.
    if (!res.if_healthy) {
      out.innerHTML = renderPlans(res.as_projected);
      return;
    }
    var names = adjustedNames(res);
    var headline = res.tops_differ
      ? '<p><strong>The call depends on ' + names + "'s return.</strong> " +
        'Compare the two columns below.</p>'
      : '<p><strong>Robust:</strong> same top plan whether ' + names +
        ' return healthy or stay limited.</p>';
    var html = headline +
      '<p style="color:var(--text-secondary)">Volume: ' + volLabel(res) + '</p>' +
      '<div style="display:flex;gap:1rem;flex-wrap:wrap">' +
      '<div style="flex:1;min-width:16rem">' +
        '<h3>As projected (limited)</h3>' + renderPlans(res.as_projected) + '</div>' +
      '<div style="flex:1;min-width:16rem">' +
        '<h3>If healthy</h3>' + renderPlans(res.if_healthy) + '</div>' +
      '</div>';
    out.innerHTML = html;
  }
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pytest tests/test_web/test_season_routes.py::test_roster_audit_page_renders_il_returns_scenario_js -v`
Expected: PASS.

- [ ] **Step 5: Manual visual verification**

Drive the running dashboard (via the `verify` or `run` skill): open `/roster-audit`, check a volume-suppressed returnee (e.g. Cruz), and confirm two labeled lists render with the headline; check a near-full-time returnee (e.g. Buxton) or none and confirm the single list still renders. Capture the result in the execution notes.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/roster_audit.html tests/test_web/test_season_routes.py
git commit -m "feat(il-planner): render two-scenario IL return lists with headline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: End-of-effort verification

**Files:** none (verification only).

- [ ] **Step 1: Full touched-suite tests**

Run: `pytest tests/test_lineup/test_il_return_planner.py tests/test_web/test_season_routes.py tests/test_models/test_player.py -v`
Expected: all PASS.

- [ ] **Step 2: Lint + format + dead-code**

Run:
```bash
python -m ruff check .
python -m ruff format --check .
python -m vulture
```
Expected: `ruff check` clean; `ruff format --check` reports no drift (run `python -m ruff format .` if it does, then re-check); `vulture` shows no NEW findings from the touched files (`healthy_rest_of_season`, `plan_il_returns_scenarios`, `IlReturnScenarios`, `_tops_differ` are all referenced — the wrapper by the route, the helpers by the wrapper/tests).

- [ ] **Step 3: mypy (only if a touched file is covered)**

Check `[tool.mypy].files` in `pyproject.toml`. `il_return_planner.py` and `season_routes.py` are not currently listed; if that is still true, mypy is N/A for this change — state so in the report. If either is now listed, run `python -m mypy <file>` and fix any error.

- [ ] **Step 4: Report**

State each command run and its result in the final message (do not just claim "checks pass"). Note the pre-existing, unrelated `ModuleNotFoundError: No module named 'resend'` failures in `tests/test_summary` / `tests/test_scripts/test_send_daily_summary.py` if a broader run surfaces them.

---

## Self-Review

- **Spec coverage:** Healthy-volume transform with full field set incl. `g`/`gs` and `sgp` clear, only-inflate + None conditions (Task 1); two-run wrapper with the load-bearing healthy-swap-into-`activating_il` + `tops_differ` set semantics + `adjusted` unit-generic volumes (Task 2); route envelope (Task 3); two-list render + headline + single-list fallback (Task 4); edge cases (no preseason, not suppressed, zero volume) tested in Task 1/2; IL-slot regression guard + as_projected reproduction guard in Task 2. Non-goals honored: `plan_il_returns` unmodified; no optimizer/audit/standings/MC changes; activating player still droppable.
- **Placeholder scan:** No TBD/TODO; every code step shows complete code; every test step shows real assertions.
- **Type consistency:** `plan_il_returns_scenarios` signature, `IlReturnScenarios` fields, and `_tops_differ` are used with the same names/types across Tasks 2-3; the route calls the wrapper with the same kwargs `plan_il_returns` already accepts; the JSON keys (`as_projected/if_healthy/adjusted/tops_differ`, `vol_unit/vol_projected/vol_healthy`) match between Tasks 2, 3, and 4.
- **Known interpretation:** the two-list output is client-side JS, so Task 4's automated guard is a page smoke test; visual correctness is verified manually via the running app during execution (recorded per the spec's "rendering/smoke test" verification intent).
