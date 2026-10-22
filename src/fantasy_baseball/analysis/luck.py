"""Luck: how much each team's players beat or missed their projections, with the
team's decisions held fixed.

Each player is judged against his projection ON THE DAY THE TEAM GOT HIM, held for
the whole stint (a stint is his unbroken run of weekly roster snapshots on that
team; dropped and re-added starts a new one; drafted and kept players use the
preseason projection). Later projection moves don't change it: a player who keeps
beating an early projection keeps counting as luck. The projection on a day is the
newest snapshot dated on or before it (``data/projections/<season>/`` for preseason,
then ``rest_of_season/<date>/``), blended like the draft board (the league's systems
on file, at the league's weights).

Two pieces, per stint:

* **Performance luck**: only games while he sat in an ACTIVE lineup slot (lineups lock
  weekly, so each snapshot is exactly who scored until the next). Expected = his
  projected rates x the playing time he actually got (PA and AB; IP for pitchers),
  so lineup choices and playing time drop out. Luck = actual - expected.
* **Playing-time luck**: every rostered day (active, bench, IL), his PA or IP against
  his projected playing time for those days (projected PA or IP per remaining day at
  that snapshot), valued at his projected rates. Injuries and lost roles count; who
  was started doesn't. Counting categories only: playing time doesn't move a rate.

Team luck in standings points re-scores the whole league on expected stats at once
(each team's real stats minus its performance luck), so a league-wide miss cancels
and the points sum to zero. Per-stint luck is also in SGP (the league's
denominators; AVG/ERA/WHIP as marginal hits / runs / baserunners over a team
season) so hitters and pitchers compare on one scale.

Replaces SPoE, which compared every rostered player's preseason projection
(bench and IL included) with standings and so mixed lineup decisions and
playing time into "luck".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from fantasy_baseball.models.league import League
from fantasy_baseball.models.positions import BENCH_SLOTS, PITCHER_ELIGIBLE
from fantasy_baseball.models.standings import Standings
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, DEFAULT_TEAM_IP, Category
from fantasy_baseball.utils.name_utils import normalize_name

HIT_STATS = ("r", "hr", "rbi", "sb", "h")  # h per AB; the rest per PA
PIT_STATS = ("w", "k", "sv", "er", "bbh")  # per IP; bbh = walks + hits allowed
# A stint's projection is "stale" when it is this many days older than the day he
# was acquired (snapshots have gaps; the page marks these quietly).
STALE_DAYS = 14
CATS = ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP")
_TWO_WAY_SUFFIXES = (" (Batter)", " (Pitcher)")


@dataclass(frozen=True)
class Projection:
    """One player's rates in one snapshot, and his playing time per remaining day.
    Hitters: R/HR/RBI/SB per PA, H per AB; pitchers: per IP."""

    rates: Mapping[str, float]
    per_day: float


@dataclass(frozen=True)
class Snapshot:
    as_of: date
    hitters: Mapping[int, Projection]
    pitchers: Mapping[int, Projection]
    # (normalized name, MLB team) -> MLBAM id, for :class:`IdResolver`.
    ids: Mapping[tuple[str, str], int] = field(default_factory=dict)

    def get(self, kind: str) -> Mapping[int, Projection]:
        return self.hitters if kind == "hitter" else self.pitchers


def _num(v: Any) -> float:
    return float(v) if v is not None and v == v else 0.0


def snapshot_from_blend(as_of: date, season_end: date, hitters, pitchers) -> Snapshot:
    """A :class:`Snapshot` from one blend (``data.projections.blend_projections``
    frames with ``mlbam_id``). Playing time per day spreads the projected PA / IP over
    the days left in the season from ``as_of``."""
    days = max((season_end - as_of).days + 1, 1)
    hit: dict[int, Projection] = {}
    for r in hitters.itertuples(index=False):
        mid, pa, ab = getattr(r, "mlbam_id", None), _num(r.pa), _num(r.ab)
        if mid is None or mid != mid or pa <= 0 or ab <= 0 or int(mid) in hit:
            continue
        rates = {s: _num(getattr(r, s)) / pa for s in ("r", "hr", "rbi", "sb")}
        rates["h"] = _num(r.h) / ab
        hit[int(mid)] = Projection(rates=rates, per_day=pa / days)
    pit: dict[int, Projection] = {}
    for r in pitchers.itertuples(index=False):
        mid, ip = getattr(r, "mlbam_id", None), _num(r.ip)
        if mid is None or mid != mid or ip <= 0 or int(mid) in pit:
            continue
        rates = {s: _num(getattr(r, s)) / ip for s in ("w", "k", "sv", "er")}
        rates["bbh"] = (_num(r.bb) + _num(r.h_allowed)) / ip
        pit[int(mid)] = Projection(rates=rates, per_day=ip / days)
    ids: dict[tuple[str, str], int] = {}
    for frame in (hitters, pitchers):
        for r in frame.itertuples(index=False):
            mid = getattr(r, "mlbam_id", None)
            if mid is not None and mid == mid:
                ids[(normalize_name(r.name), str(getattr(r, "team", "") or ""))] = int(mid)
    return Snapshot(as_of=as_of, hitters=hit, pitchers=pit, ids=ids)


# Blended snapshots by (folder, its CSVs' mtimes, systems, weights, season end): a
# committed snapshot doesn't change, so each refresh re-blends only new or edited ones.
_SNAPSHOT_CACHE: dict[tuple[Any, ...], Snapshot] = {}


def _blend_snapshot(
    folder: Path,
    as_of: date,
    season_end: date,
    systems: list[str],
    weights: dict[str, float],
) -> Snapshot:
    from fantasy_baseball.draft.our_players import board_blend

    files = tuple(sorted((f.name, f.stat().st_mtime_ns) for f in folder.glob("*.csv")))
    key = (str(folder.resolve()), files, tuple(systems), tuple(sorted(weights.items())), season_end)
    if key not in _SNAPSHOT_CACHE:
        hitters, pitchers, _ = board_blend(folder, systems, weights)
        _SNAPSHOT_CACHE[key] = snapshot_from_blend(as_of, season_end, hitters, pitchers)
    return _SNAPSHOT_CACHE[key]


def load_snapshots(
    projections_dir: Path,
    season: int,
    season_start: date,
    season_end: date,
    systems: list[str],
    weights: dict[str, float],
) -> list[Snapshot]:
    """The preseason blend (as of ``season_start``) and every dated rest-of-season
    snapshot, oldest first."""
    from fantasy_baseball.data.ros_pipeline import parse_snapshot_date, ros_snapshot_root

    folders = [(season_start, projections_dir / str(season))]
    root = ros_snapshot_root(projections_dir, season)
    if root.is_dir():
        for d in root.iterdir():
            as_of = parse_snapshot_date(d.name) if d.is_dir() else None
            if as_of is not None:
                folders.append((as_of, d))
    return [
        _blend_snapshot(folder, as_of, season_end, systems, weights)
        for as_of, folder in sorted(folders, key=lambda x: x[0])
    ]


def projection_on(
    snapshots: list[Snapshot], mid: int, kind: str, day: date
) -> tuple[Projection, date] | None:
    """His projection on ``day`` and its snapshot date: the newest snapshot dated on
    or before ``day`` that projects him; else (a call-up projected only later) the
    first that does."""
    for snap in reversed(snapshots):
        if snap.as_of <= day and mid in snap.get(kind):
            return snap.get(kind)[mid], snap.as_of
    for snap in snapshots:
        if mid in snap.get(kind):
            return snap.get(kind)[mid], snap.as_of
    return None


@dataclass
class IdResolver:
    """Rostered name -> MLBAM id, never guessed.

    With the live roster's MLB team, the projection file's (name, team) id settles it
    (whether or not he has a game log). Otherwise a name unique among players with
    game logs -- unless the projection file puts that id on a different MLB team than
    the roster does (a same-name player who has played, while the rostered one
    hasn't). A name with no game log and no roster team: a unique projection name."""

    logged: Mapping[str, Mapping[str, set[int]]]  # kind -> normalized name -> ids
    roster_mlb_team: Mapping[tuple[str, str], str]  # (fantasy team, name) -> MLB team
    projected: Mapping[tuple[str, str], int]  # (normalized name, MLB team) -> id

    def __call__(self, team: str, name: str, kind: str) -> int | None:
        cand = self.logged.get(kind, {}).get(name, set())
        mlb = self.roster_mlb_team.get((team, name))
        if mlb and (name, mlb) in self.projected:
            return self.projected[(name, mlb)]
        if len(cand) == 1:
            mid = next(iter(cand))
            elsewhere = {t for (n, t), m in self.projected.items() if n == name and m == mid}
            if mlb and elsewhere and mlb not in elsewhere:
                return None
            return mid
        if not cand and not mlb:
            ids = {m for (n, _t), m in self.projected.items() if n == name}
            if len(ids) == 1:
                return next(iter(ids))
        return None


@dataclass
class StintLuck:
    """One player's stint on one team."""

    team: str
    name: str
    kind: str
    mid: int
    acquired: date
    proj_as_of: date
    proj: Projection  # the projection on the day he was acquired, held all stint
    actual: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    expected: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    volume: float = 0.0  # active PA (hitters) or IP (pitchers)
    ab: float = 0.0
    pt_got: float = 0.0  # rostered PA / IP
    pt_proj: float = 0.0
    pt_stats: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def stale(self) -> bool:
        return (self.acquired - self.proj_as_of).days > STALE_DAYS


def _kind(entry) -> str:
    slot = entry.selected_position
    if slot in PITCHER_ELIGIBLE:
        return "pitcher"
    if slot in BENCH_SLOTS:
        return "pitcher" if set(entry.positions) & PITCHER_ELIGIBLE else "hitter"
    return "hitter"


def player_name(raw: str) -> str:
    """A rostered name as the join key: two-way suffixes off, normalized."""
    for suffix in _TWO_WAY_SUFFIXES:
        raw = raw.replace(suffix, "")
    return normalize_name(raw)


def stint_luck(
    league: League,
    snapshots: list[Snapshot],
    games: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    resolve: Callable[[str, str, str], int | None],
    season_start: date,
    season_end: date,
    today: date,
) -> tuple[list[StintLuck], list[tuple[str, str]]]:
    """Every stint's luck pieces, and the rostered players that couldn't be resolved
    (team, "name (kind)"); they're left out of the team's luck.
    ``games``: kind -> MLBAM id -> his game rows (``date`` plus counts)."""
    stints: list[StintLuck] = []
    unresolved: list[tuple[str, str]] = []
    for team in league.teams:
        open_stint: dict[tuple[int, str], StintLuck] = {}
        last_end: dict[tuple[int, str], date] = {}
        for entry, p_start, p_end in team.ownership_periods(season_start, season_end, today):
            kind = _kind(entry)
            mid = resolve(team.name, player_name(entry.name), kind)
            if mid is None:
                unresolved.append((team.name, f"{entry.name} ({kind})"))
                continue
            key = (mid, kind)
            if last_end.get(key) != p_start or key not in open_stint:
                found = projection_on(snapshots, mid, kind, p_start)
                if found is None:
                    unresolved.append((team.name, f"{entry.name} ({kind}, no projection)"))
                    last_end[key] = p_end
                    continue
                st = StintLuck(team.name, entry.name, kind, mid, p_start, found[1], found[0])
                open_stint[key] = st
                stints.append(st)
            st = open_stint[key]
            last_end[key] = p_end
            proj = st.proj
            played = [
                g
                for g in games.get(kind, {}).get(mid, ())
                if p_start <= date.fromisoformat(g["date"]) < p_end
            ]
            _add_playing_time(st, proj, played, (p_end - p_start).days)
            if entry.selected_position not in BENCH_SLOTS:
                _add_performance(st, proj, played)
    return stints, unresolved


def _add_performance(st: StintLuck, proj: Projection, played: Iterable[dict[str, Any]]) -> None:
    for g in played:
        if st.kind == "hitter":
            pa, ab = _num(g.get("pa")), _num(g.get("ab"))
            st.volume += pa
            st.ab += ab
            for s in HIT_STATS:
                st.actual[s] += _num(g.get(s))
                st.expected[s] += proj.rates[s] * (ab if s == "h" else pa)
        else:
            ip = _num(g.get("ip"))
            st.volume += ip
            for s in ("w", "k", "sv", "er"):
                st.actual[s] += _num(g.get(s))
            st.actual["bbh"] += _num(g.get("bb")) + _num(g.get("h_allowed"))
            for s in PIT_STATS:
                st.expected[s] += proj.rates[s] * ip


def _add_playing_time(
    st: StintLuck, proj: Projection, played: list[dict[str, Any]], days: int
) -> None:
    unit = "pa" if st.kind == "hitter" else "ip"
    got = sum(_num(g.get(unit)) for g in played)
    projected = proj.per_day * days
    st.pt_got += got
    st.pt_proj += projected
    for s in ("r", "hr", "rbi", "sb") if st.kind == "hitter" else ("w", "k", "sv"):
        st.pt_stats[s] += proj.rates[s] * (got - projected)


def stint_sgp(st: StintLuck, denoms: Mapping[Category, float]) -> tuple[float, float]:
    """(performance luck, playing-time luck) in SGP."""
    a, e, pt = st.actual, st.expected, st.pt_stats
    if st.kind == "hitter":
        perf = sum((a[s] - e[s]) / denoms[c] for s, c in _HIT_CATS) + (a["h"] - e["h"]) / (
            denoms[Category.AVG] * DEFAULT_TEAM_AB
        )
        ptl = sum(pt[s] / denoms[c] for s, c in _HIT_CATS)
    else:
        perf = (
            sum((a[s] - e[s]) / denoms[c] for s, c in _PIT_CATS)
            + (e["er"] - a["er"]) / (denoms[Category.ERA] * DEFAULT_TEAM_IP / 9)
            + (e["bbh"] - a["bbh"]) / (denoms[Category.WHIP] * DEFAULT_TEAM_IP)
        )
        ptl = sum(pt[s] / denoms[c] for s, c in _PIT_CATS)
    return perf, ptl


_HIT_CATS = (("r", Category.R), ("hr", Category.HR), ("rbi", Category.RBI), ("sb", Category.SB))
_PIT_CATS = (("w", Category.W), ("k", Category.K), ("sv", Category.SV))


def team_stat_luck(stints: Iterable[StintLuck]) -> dict[str, dict[str, float]]:
    """Per team, actual minus expected over its active games: the counting
    categories, and the rate categories' numerators (``h``, ``er``, ``bbh``), with the
    AB and IP they came from (``ab``, ``ip``)."""
    tot: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for st in stints:
        t = tot[st.team]
        for s, v in st.actual.items():
            t[s] += v
        for s, v in st.expected.items():
            t[s] -= v
        if st.kind == "hitter":
            t["ab"] += st.ab
        else:
            t["ip"] += st.volume
    return {team: dict(t) for team, t in tot.items()}


def expected_stats(
    stats: Mapping[str, float], luck: Mapping[str, float], entry
) -> dict[str, float]:
    """A team's real standings stats minus its performance luck. Rates shift by the
    luck in their numerator over the TEAM's real AB / IP (``entry.ytd_components()``),
    not just the resolved players'; the resolved AB / IP stand in only when the
    standings carry none."""
    comp = entry.ytd_components()
    ab = comp.ab if comp.ab > 0 else luck.get("ab", 0.0)
    ip = comp.ip if comp.ip > 0 else luck.get("ip", 0.0)
    out = {c: stats[c] - luck.get(c.lower(), 0.0) for c in ("R", "HR", "RBI", "SB", "W", "K", "SV")}
    out["AVG"] = stats["AVG"] - (luck.get("h", 0.0) / ab if ab else 0.0)
    out["ERA"] = stats["ERA"] - (9 * luck.get("er", 0.0) / ip if ip else 0.0)
    out["WHIP"] = stats["WHIP"] - (luck.get("bbh", 0.0) / ip if ip else 0.0)
    return out


def compute_luck(
    league: League,
    standings: Standings,
    snapshots: list[Snapshot],
    games: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    resolve: Callable[[str, str, str], int | None],
    denoms: Mapping[Category, float],
    season_start: date,
    season_end: date,
    today: date,
) -> dict[str, Any]:
    """The luck page's payload: teams (standings points, points if everyone hit
    projection, luck by category, playing time, players left out) and every stint.
    ``standings`` should carry each team's AB and IP in ``extras`` (the refresh's
    team-YTD standings) so rate luck is spread over the team's real volume."""
    stints, unresolved = stint_luck(
        league, snapshots, games, resolve, season_start, season_end, today
    )
    luck = team_stat_luck(stints)
    real = {e.team_name: e.stats.to_dict() for e in standings.entries}
    expected = {
        e.team_name: expected_stats(real[e.team_name], luck.get(e.team_name, {}), e)
        for e in standings.entries
    }
    real_pts, exp_pts = score_roto_dict(real), score_roto_dict(expected)
    sgp = {id(st): stint_sgp(st, denoms) for st in stints}
    teams: list[dict[str, Any]] = []
    for t in real:
        mine = [st for st in stints if st.team == t]
        teams.append(
            {
                "team": t,
                "points": real_pts[t]["total"],
                "expected_points": exp_pts[t]["total"],
                "luck_points": real_pts[t]["total"] - exp_pts[t]["total"],
                "categories": {c: real_pts[t][f"{c}_pts"] - exp_pts[t][f"{c}_pts"] for c in CATS},
                "pa_vs_proj": sum(st.pt_got - st.pt_proj for st in mine if st.kind == "hitter"),
                "ip_vs_proj": sum(st.pt_got - st.pt_proj for st in mine if st.kind == "pitcher"),
                "perf_sgp": sum(sgp[id(st)][0] for st in mine),
                "pt_sgp": sum(sgp[id(st)][1] for st in mine),
                "unresolved": sorted({who for team, who in unresolved if team == t}),
            }
        )
    teams.sort(key=lambda x: -x["points"])
    players: list[dict[str, Any]] = []
    for st in stints:
        perf, ptl = sgp[id(st)]
        players.append(
            {
                "team": st.team,
                "name": st.name,
                "kind": st.kind,
                "acquired": st.acquired.isoformat(),
                "proj_as_of": st.proj_as_of.isoformat(),
                "stale": st.stale,
                "perf_sgp": perf,
                "pt_sgp": ptl,
                "total_sgp": perf + ptl,
                "volume": st.volume,
                "pt_got": st.pt_got,
                "pt_proj": st.pt_proj,
                "actual": _line(st, st.actual),
                "expected": _line(st, st.expected),
            }
        )
    players.sort(key=lambda p: -p["total_sgp"])
    return {
        "as_of": today.isoformat(),
        "standings_date": standings.effective_date.isoformat(),
        "teams": teams,
        "players": players,
        "unresolved": sorted({f"{team}: {who}" for team, who in unresolved}),
    }


def _line(st: StintLuck, s: Mapping[str, float]) -> dict[str, float]:
    """A stint's line in display units (AVG, ERA, WHIP as rates)."""
    if st.kind == "hitter":
        return {
            "R": s["r"],
            "HR": s["hr"],
            "RBI": s["rbi"],
            "SB": s["sb"],
            "AVG": s["h"] / st.ab if st.ab else 0.0,
        }
    ip = st.volume
    return {
        "W": s["w"],
        "K": s["k"],
        "SV": s["sv"],
        "ERA": 9 * s["er"] / ip if ip else 0.0,
        "WHIP": s["bbh"] / ip if ip else 0.0,
    }


def rostered_ids(
    league: League,
    resolve: Callable[[str, str, str], int | None],
    season_start: date,
    season_end: date,
    today: date,
) -> dict[str, set[int]]:
    """kind -> the MLBAM ids of every resolvable player any team rostered."""
    out: dict[str, set[int]] = {"hitter": set(), "pitcher": set()}
    for team in league.teams:
        for entry, _s, _e in team.ownership_periods(season_start, season_end, today):
            kind = _kind(entry)
            mid = resolve(team.name, player_name(entry.name), kind)
            if mid is not None:
                out[kind].add(mid)
    return out


def run_luck(
    kv,
    league: League,
    standings: Standings,
    config,
    projections_dir: Path,
    rosters: Mapping[str, Iterable[Any]],
    today: date,
) -> dict[str, Any]:
    """Assemble the inputs and :func:`compute_luck`: the projection snapshots on disk,
    the rostered players' game logs from ``kv`` (names from the game-log rollups), and
    ``rosters`` (fantasy team -> live roster players with ``.name`` and ``.team``)
    for the MLB team that settles a shared name."""
    from fantasy_baseball.data.redis_store import build_ytd_game_logs, get_game_log_totals
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    start, end = date.fromisoformat(config.season_start), date.fromisoformat(config.season_end)
    today = min(today, end)
    snapshots = load_snapshots(
        projections_dir,
        config.season_year,
        start,
        end,
        list(config.projection_systems),
        dict(config.projection_weights),
    )
    logged: dict[str, dict[str, set[int]]] = {}
    for kind, rollup_kind in (("hitter", "hitters"), ("pitcher", "pitchers")):
        names: dict[str, set[int]] = defaultdict(set)
        for mid, rec in get_game_log_totals(kv, rollup_kind).items():
            # MLBAM ids are integers; a key that isn't one can't be joined, so it's skipped.
            if str(mid).isdigit():
                names[normalize_name(rec.get("name", ""))].add(int(mid))
        logged[kind] = names
    roster_team = {
        (team, player_name(p.name)): p.team or "" for team, ps in rosters.items() for p in ps
    }
    resolve = IdResolver(logged, roster_team, snapshots[-1].ids)
    wanted = rostered_ids(league, resolve, start, end, today)
    games: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for kind, ids in wanted.items():
        logs = build_ytd_game_logs(kv, config.season_year, kind, ids={str(m) for m in ids})
        games[kind] = {int(mid): rec["games"] for mid, rec in logs.items()}
    return compute_luck(
        league,
        standings,
        snapshots,
        games,
        resolve,
        get_sgp_denominators(config.sgp_overrides),
        start,
        end,
        today,
    )
