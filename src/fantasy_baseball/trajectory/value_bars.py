"""What a top-10 / top-30 / top-100 season actually PRODUCED, measured from the panel.

The trajectory board projects a VAR total. Turning that into "how likely is he to earn a
keeper slot" needs a bar to clear, and the bar has to be a REALIZED quantity: the
calibrated distribution `calibration.exceedance` inverts is a distribution over realized
outcomes, so comparing it against a quantile of the PROJECTED pool answers a different
question entirely.

That mistake is worth stating plainly because it was made here first. Projections regress
to the mean and outcomes do not, so the projected pool is squeezed at both ends. Measured
on held-out 3-year totals, SD(actual) is 1.36x SD(predicted), and the gap grows with the
quantile (+6.0 SGP at the 90th, +8.7 at the 99th). The consequence for the bars:

    rank    projected 2027-29    realized 2022-24 / 2023-25
    #10                  11.8          19.9 / 21.1
    #30                   4.6          14.3 / 14.2
    #100                 -2.2           5.0 /  4.9

The projected #30 bar is roughly the realized #100 bar, so a probability computed against
it overstates by about two whole tiers. Only 76 of 1,277 projected players clear
replacement over three years; 157-170 actually did.

METHOD, matching how the board builds its own multi-year VAR so the two are comparable:

  * Era-normalized, like every other trajectory number, so seasons across the window are
    on one run environment.
  * The FLOOR IS FIXED AT THE STARTING SEASON and applied to every year of the window --
    `SweptPlayer.points("var")` nets the same slot floor off all its horizons, and a bar
    computed against a moving floor would not be the same quantity the board projects.
  * A season the player did not appear in is a real 0 SGP, hence `-floor` in VAR. Out of
    the league is an outcome, not missing data (see `model.played`).

WINDOW COVERAGE IS THE BINDING CONSTRAINT. Eligibility is cached only from 2022
(`data/cache/keeper_skills/mlb_fielding_*.csv`), and a k-year window needs its start year
eligible and all k seasons complete. Against a panel complete through 2025 that gives 4
windows at k=1, 3 at k=2, 2 at k=3, 1 at k=4 and NONE at k=5. Spans with no window carry
no bar and `bar()` returns None; the board refuses to headline probabilities it cannot
compute rather than extrapolating one. Bars do not scale linearly in k -- the k=3 #30 bar
is 2.2x the k=1 bar, not 3x, because holding a rank for three years is harder than
holding it for one -- so there is no honest way to invent the missing spans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import MAX_HORIZON, PROJECT_ROOT, panel_vintage_of

#: Where the fitted bars ship, beside the panel and the band calibration they belong with.
VALUE_BARS_PATH = Path("data") / "trajectory" / "value_bars.json"

#: The rank that means "outside the useful pool". Not derived from the league the way the
#: other two are -- a 10-team league rosters ~240 players, so a roster-derived bar would
#: sit far below replacement and call almost everyone useful. 100 is a judgement, and it
#: lands close to where realized VAR crosses zero (157-170 players clear replacement over
#: a 3-year window), which is the property that makes it a sensible floor.
BUST_RANK = 100

#: Cached eligibility is what bounds the windows. Named so a future season's fielding pull
#: makes the limit move on its own rather than needing this file edited.
ELIGIBILITY_GLOB = "mlb_fielding_*.csv"


@dataclass(frozen=True)
class ValueBars:
    """Realized VAR at each headline rank, per span, plus how it was measured."""

    panel_vintage: str
    #: span key ("s3") -> rank ("30") -> realized VAR, averaged over the windows.
    bars: dict[str, dict[str, float]]
    #: span key -> the start seasons averaged. One window is thin and the caller should be
    #: able to say so; zero means the span has no bar at all.
    windows: dict[str, list[int]]
    #: The ranks these bars were cut at, so a reader is never guessing which is which.
    ranks: dict[str, int]

    def bar(self, span: str, name: str) -> float | None:
        """The realized VAR for one bar, or None when it is not measured here.

        `.get(name)` rather than `[name]`: `ranks` comes out of a JSON artifact, so an
        older or hand-edited one can be missing a bar the code asks for, and a KeyError
        raised from a board render takes out the whole page over one absent threshold.
        None is already the "no bar for this span" answer every caller handles, and
        `bar_probabilities` turns it into a blank row rather than a fabricated number.
        """
        rank = self.ranks.get(name)
        if rank is None:
            return None
        value = self.bars.get(span, {}).get(str(rank))
        return None if value is None else float(value)

    def spans(self) -> tuple[str, ...]:
        """Spans that have at least one window, so a caller can offer only those."""
        return tuple(s for s, w in self.windows.items() if w)

    def to_json(self) -> str:
        return json.dumps(
            {
                "panel_vintage": self.panel_vintage,
                "bars": self.bars,
                "windows": self.windows,
                "ranks": self.ranks,
            },
            indent=2,
            sort_keys=True,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, panel_vintage: str | None = None) -> ValueBars:
        raw = json.loads(path.read_text(encoding="utf-8"))
        bars = cls(
            panel_vintage=raw["panel_vintage"],
            bars=raw["bars"],
            windows=raw["windows"],
            ranks=raw["ranks"],
        )
        if panel_vintage is not None and bars.panel_vintage != panel_vintage:
            raise ValueError(
                f"value bars were measured on panel {bars.panel_vintage!r} but the panel "
                f"in use is {panel_vintage!r}. Regenerate with "
                f"`python scripts/build_value_bars.py`."
            )
        return bars


@lru_cache(maxsize=4)
def load_shipped_bars(panel_dir: Path | None = None) -> ValueBars | None:
    """The artifact under `data/trajectory/`, or None when it has not been built."""
    path = PROJECT_ROOT / VALUE_BARS_PATH
    if not path.exists():
        return None
    # `missing_ok`: the deployed dashboard ships these bars and NOT the panel CSVs they
    # were measured on -- see `panel_vintage_of`.
    return ValueBars.load(path, panel_vintage=panel_vintage_of(panel_dir, missing_ok=True))


def eligible_seasons(cache_dir: Path) -> list[int]:
    """Seasons with cached fielding data, which is what a window's start year needs."""
    return sorted(int(p.stem.rsplit("_", 1)[-1]) for p in cache_dir.glob(ELIGIBILITY_GLOB))


def windows_for(span: int, cache_dir: Path, last_complete: int) -> list[int]:
    """Start seasons of every complete `span`-year window."""
    return [s for s in eligible_seasons(cache_dir) if s + span - 1 <= last_complete]


def realized_var(
    panels: dict[str, pd.DataFrame],
    floors: dict[str, dict[int, float]],
    start: int,
    span: int,
) -> pd.Series:
    """Every player's realized VAR over `start .. start+span-1`, strongest first.

    `panels` are era-normalized and collapsed; `floors` is pool -> mlbam id -> the floor
    from the STARTING season, which is applied to all `span` years (see the module note).
    """
    totals: list[float] = []
    for pool, by_id in floors.items():
        live = panels[pool]
        seasons = {
            year: live[live["season"] == year].set_index("mlbam_id")["sgp"].astype(float)
            for year in range(start, start + span)
        }
        for pid, floor in by_id.items():
            # `.get(pid, 0.0)` is the out-of-league convention: a real 0 SGP, so VAR is
            # minus the floor for that year rather than a skipped term.
            totals.append(sum(float(s.get(pid, 0.0)) - floor for s in seasons.values()))
    return pd.Series(totals).sort_values(ascending=False).reset_index(drop=True)


def build_bars(
    panels: dict[str, pd.DataFrame],
    floors_by_season: dict[int, dict[str, dict[int, float]]],
    *,
    panel_vintage: str,
    ranks: dict[str, int],
    cache_dir: Path,
    last_complete: int,
    max_span: int = MAX_HORIZON,
) -> ValueBars:
    """Measure every rank at every span that has a complete window.

    Averaged across windows rather than taking the newest: the single-season bars move by
    ~1.4 VAR between 2022 and 2025 with no trend, so one window is noise around a stable
    number and the mean of what exists is the better estimate of it.
    """
    bars: dict[str, dict[str, float]] = {}
    windows: dict[str, list[int]] = {}
    for span in range(1, max_span + 1):
        key = f"s{span}"
        starts = windows_for(span, cache_dir, last_complete)
        windows[key] = starts
        if not starts:
            bars[key] = {}
            continue
        per_rank: dict[str, list[float]] = {str(r): [] for r in ranks.values()}
        for start in starts:
            ordered = realized_var(panels, floors_by_season[start], start, span)
            for rank in ranks.values():
                if rank <= len(ordered):
                    per_rank[str(rank)].append(float(ordered.iloc[rank - 1]))
        bars[key] = {r: float(np.mean(v)) for r, v in per_rank.items() if v}
    return ValueBars(panel_vintage=panel_vintage, bars=bars, windows=windows, ranks=ranks)


def longest_calibrated_span(available: int, bars: ValueBars | None) -> int:
    """The longest 1..k span at or below `available` that has measured bars, else 0.

    ONE definition, because two surfaces need it and they must not answer differently:
    the single-player CLI projects five years by default and the web chart draws the
    board's full range, while no five-year window is measurable yet. Both fall back to
    the longest span that IS measured and label it; a fallback computed twice is a
    fallback that eventually disagrees with itself about which span is on screen.
    """
    if bars is None:
        return 0
    return next((k for k in range(available, 0, -1) if bars.bars.get(f"s{k}")), 0)


def league_ranks(num_teams: int, keepers_per_team: int) -> dict[str, int]:
    """The three headline ranks, two of them DERIVED from the league's own rules.

    `elite` is one per team and `keeper` is every keeper slot in the league, so a rule
    change in `league.yaml` moves the bars rather than leaving a stale constant behind.
    `bust` is `BUST_RANK`, which is a judgement rather than a derivation -- see there.
    """
    return {
        "elite": num_teams,
        "keeper": num_teams * keepers_per_team,
        "bust": BUST_RANK,
    }
