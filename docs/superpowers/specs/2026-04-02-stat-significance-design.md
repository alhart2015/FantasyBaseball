# Stat Significance Indicators

## Problem

The season dashboard displays player stats (projections, actuals) across multiple pages, but there's no visual indication of whether a stat has accumulated enough sample size to be trustworthy. Early-season numbers are noisy — a pitcher's 1.50 ERA after 10 IP means little, but the same ERA after 100 IP is meaningful.

## Approach

Add `is_significant(cat)` methods to `HitterStats` and `PitcherStats` that check season-to-date sample size against empirically-derived stabilization thresholds from sabermetric research (Carleton / FanGraphs). Display a small green checkmark on stat cells that have reached significance.

## Stabilization Thresholds

Based on FanGraphs Sabermetrics Library and Russell Carleton's split-half reliability research:

### Hitters (measured in PA)

| Category | Threshold | Notes |
|----------|-----------|-------|
| R | Always significant | Counting stat, no canonical stabilization point |
| HR | 170 PA | HR rate stabilizes at ~170 PA |
| RBI | Always significant | Context-dependent counting stat |
| SB | Always significant | No published stabilization point |
| AVG | Always significant | Stabilizes at ~910 AB (over a full season), so threshold is impractical within a single season |

### Pitchers (measured in BF)

| Category | Threshold | Notes |
|----------|-----------|-------|
| W | Always significant | Role/team-dependent |
| K | 70 BF | K rate is the fastest-stabilizing pitching stat |
| SV | Always significant | Entirely role-dependent |
| ERA | 630 BF | Compound stat estimate from component stabilization |
| WHIP | 570 BF | Midpoint of component stat stabilization range (540-670 BF) |

BF (batters faced) is computed from available fields: `ip * 3 + h_allowed + bb`. This omits HBP (~1% of BF), which is acceptable precision for threshold comparison.

## Data Layer

### `HitterStats.is_significant(cat: str) -> bool`

Checks `self.pa` against the threshold for `cat`. Only HR has a real threshold (170 PA). All other categories return `True`.

### `PitcherStats.is_significant(cat: str) -> bool`

Computes BF as `self.ip * 3 + self.h_allowed + self.bb`, then checks against thresholds. K: 70, ERA: 630, WHIP: 570. All other categories return `True`.

### Serialization

Both `to_dict()` methods gain a `"significant"` key containing a dict of category → bool for the 5 relevant roto categories:

```python
# HitterStats example
{"pa": 150, "r": 20, "hr": 8, ..., "significant": {"R": True, "HR": False, "RBI": True, "SB": True, "AVG": True}}

# PitcherStats example  
{"ip": 30, "k": 35, ..., "significant": {"W": True, "K": True, "SV": True, "ERA": False, "WHIP": False}}
```

## Display Layer

Every page that renders individual player stats shows a small green checkmark in the corner of stat cells where the stat is significant. This applies to:

- Player browse table (`/players`)
- Lineup page player stats (`/lineup`)
- Waivers & trades page (`/waivers-trades`)
- Player search card view (`/api/players/search` consumers)

### CSS

A `.stat-significant` class adds a green checkmark via CSS `::after` pseudo-element, positioned in the top-right corner of the cell. Small and unobtrusive — the checkmark should not displace the stat value.

## Testing

- Unit tests on `HitterStats.is_significant()` and `PitcherStats.is_significant()` covering threshold boundaries (just below, at, just above).
- Verify `to_dict()` includes the `significant` key with correct values.
- Stats that are always significant return `True` regardless of sample size.
