# ROS Deviation Indicators

Replace the stat significance checkmark feature with SGP-based projection shift indicators that show whether a player's updated full-season outlook (actual + ROS) has meaningfully deviated from their preseason projection.

## Motivation

The current green checkmark indicates when a stat has reached a sample-size stabilization threshold. This isn't actionable. What matters is whether the projection systems have revised their view of a player — and by how much. A player whose actual+ROS RBI projection is 2 SGP above preseason is gaining real standings value; one whose ERA has drifted 1 SGP worse is a concern. SGP is the natural unit because it directly maps to standings points.

## Design

### Data: `pace.py`

Add a `ros_deviation_sgp` field to each stat entry returned by `compute_player_pace()`. The function already receives `projected_stats` (preseason) — it will also accept the player's ROS projection stats and an SGP denominators dict.

**Counting stats** (R, HR, RBI, SB, W, K, SV):
```
deviation = (ros_value - preseason_value) / sgp_denominator
```

**Rate stats** (AVG, ERA, WHIP):
```
deviation = (ros_value - preseason_value) / sgp_denominator
```
For ERA and WHIP (inverse stats where lower is better), the sign is flipped so that positive = good and negative = bad, consistent with counting stats.

The result is a float. The template truncates toward zero to get the indicator count: `int(ros_deviation_sgp)`. So +1.8 SGP = 1 green plus, -2.3 SGP = 2 red X's, anything between -1.0 and +1.0 = no indicator.

New signature:
```python
def compute_player_pace(
    actual_stats: dict,
    projected_stats: dict,   # preseason (existing)
    player_type: str,
    ros_stats: dict | None = None,      # new: ROS projection
    sgp_denoms: dict | None = None,     # new: SGP denominators
) -> dict:
```

Each stat entry gains:
```python
"ros_deviation_sgp": <float>  # signed, positive = good
```

### Cleanup: remove significance feature

**Delete:**
- `STABILIZATION_THRESHOLDS` dict from `constants.py`
- `is_significant()` methods from `HitterStats` and `PitcherStats` in `player.py`
- `significant` and `below_threshold` fields from all pace result dicts in `pace.py`
- The `stats_cls` / `actual_obj` construction in `compute_player_pace()` (only used for `is_significant`)

### Caller: `season_data.py`

Two call sites build pace data — the roster cache builder (~line 1083) and the enrichment function (~line 452). Both must pass the new `ros_stats` and `sgp_denoms` arguments.

The ROS stats are already available on `player.ros` and the SGP denominators come from `config.sgp_denominators` (loaded from `league.yaml`).

### Template: `lineup.html`

Replace the `stat-significant` class with data-driven indicator rendering. For each stat cell:

```jinja2
<td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}
    {%- set dev = st.get('ros_deviation_sgp', 0)|int -%}
    {%- if dev > 0 %} stat-ros-up stat-ros-{{ dev }}{% elif dev < 0 %} stat-ros-down stat-ros-{{ dev|abs }}{% endif -%}
">
```

The CSS `::after` pseudo-element renders the indicators using the `stat-ros-N` class to control content repetition.

### CSS: `season.css`

Remove `.stat-significant` and its `::after` rule. Add:

```css
.stat-ros-up, .stat-ros-down { position: relative; }

.stat-ros-up::after,
.stat-ros-down::after {
    position: absolute;
    top: 1px;
    right: 2px;
    font-size: 8px;
    opacity: 0.7;
    line-height: 1;
}

.stat-ros-up::after { color: #22c55e; }
.stat-ros-down::after { color: #ef4444; }

.stat-ros-1::after { content: attr(data-ros-indicator); }
.stat-ros-2::after { content: attr(data-ros-indicator); }
.stat-ros-3::after { content: attr(data-ros-indicator); }
```

Using a `data-ros-indicator` attribute on the `<td>` set from the template (e.g., `data-ros-indicator="+++"` or `data-ros-indicator="xxx"`) keeps the CSS simple — one rule per direction, content driven by the attribute.

Alternatively, hard-code content per class:
```css
.stat-ros-up.stat-ros-1::after { content: "+"; }
.stat-ros-up.stat-ros-2::after { content: "++"; }
.stat-ros-up.stat-ros-3::after { content: "+++"; }
.stat-ros-down.stat-ros-1::after { content: "\00d7"; }
.stat-ros-down.stat-ros-2::after { content: "\00d7\00d7"; }
.stat-ros-down.stat-ros-3::after { content: "\00d7\00d7\00d7"; }
```

Cap at 3 indicators for now. Values beyond 3 SGP still show 3 indicators. This can be revisited if the UI gets cluttered.

## Files changed

| File | Change |
|------|--------|
| `src/fantasy_baseball/analysis/pace.py` | Add `ros_deviation_sgp` field, remove `significant`/`below_threshold`, accept new params |
| `src/fantasy_baseball/web/season_data.py` | Pass ROS stats and SGP denoms to `compute_player_pace()` at both call sites |
| `src/fantasy_baseball/models/player.py` | Delete `is_significant()` from `HitterStats` and `PitcherStats` |
| `src/fantasy_baseball/utils/constants.py` | Delete `STABILIZATION_THRESHOLDS` |
| `src/fantasy_baseball/web/templates/season/lineup.html` | Replace `stat-significant` with `stat-ros-up`/`stat-ros-down` rendering |
| `src/fantasy_baseball/web/static/season.css` | Replace `.stat-significant` with `.stat-ros-*` styles |
| `tests/` | Update any tests that assert on `significant` or `below_threshold` fields |
