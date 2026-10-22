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
from fantasy_baseball.models.standings import Standings
from fantasy_baseball.scoring import score_roto_dict
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, DEFAULT_TEAM_IP, Category
from fantasy_baseball.utils.name_utils import normalize_name

# Slots whose games don't count toward a team's stats.
INACTIVE_SLOTS = frozenset({"BN", "IL", "IL+", "NA"})
PITCHER_POSITIONS = frozenset({"P", "SP", "RP"})
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
    Hitters: R/HR/RBI/SB per PA, H per AB, ``ab_pa``; pitchers: per IP."""

    rates: Mapping[str, float]
    per_day: float
    ab_pa: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    as_of: date
    hitters: Mapping[int, Projection]
    pitchers: Mapping[int, Projection]

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
        hit[int(mid)] = Projection(rates=rates, per_day=pa / days, ab_pa=ab / pa)
    pit: dict[int, Projection] = {}
    for r in pitchers.itertuples(index=False):
        mid, ip = getattr(r, "mlbam_id", None), _num(r.ip)
        if mid is None or mid != mid or ip <= 0 or int(mid) in pit:
            continue
        rates = {s: _num(getattr(r, s)) / ip for s in ("w", "k", "sv", "er")}
        rates["bbh"] = (_num(r.bb) + _num(r.h_allowed)) / ip
        pit[int(mid)] = Projection(rates=rates, per_day=ip / days)
    return Snapshot(as_of=as_of, hitters=hit, pitchers=pit)


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
    from fantasy_baseball.draft.our_players import board_blend

    folders = [(season_start, projections_dir / str(season))]
    root = ros_snapshot_root(projections_dir, season)
    if root.is_dir():
        for d in root.iterdir():
            as_of = parse_snapshot_date(d.name) if d.is_dir() else None
            if as_of is not None:
                folders.append((as_of, d))
    out = []
    for as_of, folder in sorted(folders, key=lambda x: x[0]):
        hitters, pitchers, _ = board_blend(folder, systems, weights)
        out.append(snapshot_from_blend(as_of, season_end, hitters, pitchers))
    return out


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
    """Rostered name -> MLBAM id, never guessed. Unique name among players with game
    logs; a shared name settled by the live roster's MLB team and the projection
    file's (name, team) id; a name with no game log at all (out all season) by a
    unique projection name."""

    logged: Mapping[str, Mapping[str, set[int]]]  # kind -> normalized name -> ids
    roster_mlb_team: Mapping[tuple[str, str], str]  # (fantasy team, name) -> MLB team
    projected: Mapping[tuple[str, str], int]  # (normalized name, MLB team) -> id

    def __call__(self, team: str, name: str, kind: str) -> int | None:
        cand = self.logged.get(kind, {}).get(name, set())
        if len(cand) == 1:
            return next(iter(cand))
        mlb = self.roster_mlb_team.get((team, name))
        mid = self.projected.get((name, mlb)) if mlb else None
        if mid in cand:
            return mid
        if not cand:
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


def _slot(entry) -> str:
    return str(getattr(entry.selected_position, "value", entry.selected_position))


def _kind(entry, slot: str) -> str:
    if slot == "P":
        return "pitcher"
    if slot in INACTIVE_SLOTS:
        positions = {str(getattr(p, "value", p)) for p in entry.positions}
        return "pitcher" if positions & PITCHER_POSITIONS else "hitter"
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
) -> tuple[list[StintLuck], list[str]]:
    """Every stint's luck pieces, and the rostered names that couldn't be resolved.
    ``games``: kind -> MLBAM id -> his game rows (``date`` plus counts)."""
    stints: list[StintLuck] = []
    unresolved: list[str] = []
    for team in league.teams:
        open_stint: dict[tuple[int, str], StintLuck] = {}
        last_end: dict[tuple[int, str], date] = {}
        for entry, p_start, p_end in team.ownership_periods(season_start, season_end, today):
            slot = _slot(entry)
            kind = _kind(entry, slot)
            mid = resolve(team.name, player_name(entry.name), kind)
            if mid is None:
                unresolved.append(f"{team.name}: {entry.name} ({kind})")
                continue
            key = (mid, kind)
            if last_end.get(key) != p_start or key not in open_stint:
                found = projection_on(snapshots, mid, kind, p_start)
                if found is None:
                    unresolved.append(f"{team.name}: {entry.name} ({kind}, no projection)")
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
            if slot not in INACTIVE_SLOTS:
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
    """Per team and category, actual minus expected over its active games (rate
    categories from their components)."""
    tot: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for st in stints:
        t = tot[st.team]
        for s, v in st.actual.items():
            t[f"a_{s}"] += v
        for s, v in st.expected.items():
            t[f"e_{s}"] += v
        t["ab" if st.kind == "hitter" else "ip"] += st.ab if st.kind == "hitter" else st.volume
    out = {}
    for team, t in tot.items():
        ab, ip = t["ab"], t["ip"]
        out[team] = {
            "R": t["a_r"] - t["e_r"],
            "HR": t["a_hr"] - t["e_hr"],
            "RBI": t["a_rbi"] - t["e_rbi"],
            "SB": t["a_sb"] - t["e_sb"],
            "AVG": (t["a_h"] - t["e_h"]) / ab if ab else 0.0,
            "W": t["a_w"] - t["e_w"],
            "K": t["a_k"] - t["e_k"],
            "SV": t["a_sv"] - t["e_sv"],
            "ERA": 9 * (t["a_er"] - t["e_er"]) / ip if ip else 0.0,
            "WHIP": (t["a_bbh"] - t["e_bbh"]) / ip if ip else 0.0,
        }
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
    projection, luck by category, playing time) and every stint."""
    stints, unresolved = stint_luck(
        league, snapshots, games, resolve, season_start, season_end, today
    )
    luck = team_stat_luck(stints)
    real = {e.team_name: e.stats.to_dict() for e in standings.entries}
    expected = {
        t: {c: stats[c] - luck.get(t, {}).get(c, 0.0) for c in CATS} for t, stats in real.items()
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
        "unresolved": sorted(set(unresolved)),
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


def run_luck(
    kv,
    league: League,
    standings: Standings,
    config,
    projections_dir: Path,
    roster_raw: list[dict[str, Any]] | None,
    opp_rosters_raw: Mapping[str, list[dict[str, Any]]] | None,
    today: date,
) -> dict[str, Any]:
    """Assemble the inputs (projection snapshots on disk, every player's game logs and
    the live rosters' MLB teams from ``kv``) and :func:`compute_luck`."""
    from fantasy_baseball.data.redis_store import build_ytd_game_logs
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    start, end = date.fromisoformat(config.season_start), date.fromisoformat(config.season_end)
    weights = dict(config.projection_weights)
    snapshots = load_snapshots(
        projections_dir, config.season_year, start, end, list(config.projection_systems), weights
    )
    games: dict[str, dict[int, list[dict[str, Any]]]] = {}
    logged: dict[str, dict[str, set[int]]] = {}
    for kind in ("hitter", "pitcher"):
        # MLBAM ids are integers; a key that isn't one can't be joined, so it's skipped.
        logs = {
            int(mid): rec
            for mid, rec in build_ytd_game_logs(kv, config.season_year, kind).items()
            if str(mid).isdigit()
        }
        games[kind] = {mid: rec["games"] for mid, rec in logs.items()}
        names: dict[str, set[int]] = defaultdict(set)
        for mid, rec in logs.items():
            names[normalize_name(rec.get("name", ""))].add(mid)
        logged[kind] = names
    roster_team: dict[tuple[str, str], str] = {}
    for team, players in (opp_rosters_raw or {}).items():
        for p in players:
            roster_team[(team, player_name(p.get("name", "")))] = p.get("team") or ""
    for p in roster_raw or []:
        roster_team[(config.team_name, player_name(p.get("name", "")))] = p.get("team") or ""
    resolve = IdResolver(logged, roster_team, _projected_ids(projections_dir, config, weights))
    return compute_luck(
        league,
        standings,
        snapshots,
        games,
        resolve,
        get_sgp_denominators(config.sgp_overrides),
        start,
        end,
        min(today, end),
    )


def _projected_ids(
    projections_dir: Path, config, weights: dict[str, float]
) -> dict[tuple[str, str], int]:
    """(normalized name, MLB team) -> MLBAM id from the newest projection snapshot."""
    from fantasy_baseball.data.ros_pipeline import latest_ros_snapshot
    from fantasy_baseball.draft.our_players import board_blend

    latest = latest_ros_snapshot(projections_dir, config.season_year)
    folder = latest[0] if latest else projections_dir / str(config.season_year)
    hitters, pitchers, _ = board_blend(folder, list(config.projection_systems), weights)
    out: dict[tuple[str, str], int] = {}
    for frame in (hitters, pitchers):
        for r in frame.itertuples(index=False):
            mid = getattr(r, "mlbam_id", None)
            if mid is not None and mid == mid:
                out[(normalize_name(r.name), str(getattr(r, "team", "") or ""))] = int(mid)
    return out
