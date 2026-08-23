"""Run the season refresh WITHOUT Yahoo, from hand-transcribed inputs.

Yahoo's API is unavailable, so the two things only Yahoo can supply -- league
standings and the ten rosters -- are typed by hand into
``data/manual/standings.yaml`` and ``data/manual/rosters.yaml``, and the
free-agent pool is synthesized from the rest-of-season projections instead of
fetched. Everything else (MLB game logs, the ROS blend, projected standings,
the lineup optimizer, deltaRoto, the roster audit) is the SAME code the Yahoo
pipeline runs. This script is a data-acquisition adapter, not a second
pipeline.

    python scripts/run_manual_refresh.py --dry-run     # validate only, no KV
    python scripts/run_manual_refresh.py               # full run + report
    python scripts/run_manual_refresh.py --skip-blend  # reuse the cached ROS blob

ISOLATION IS BY WHOLE KV FILE. A manual run writes hand-typed data into
``cache:standings``, ``weekly_rosters_history``, ``standings_history`` and the
whole ``cache:*`` family. Landing that in ``data/local.db`` would silently
corrupt the Yahoo baseline, so this script points ``FANTASY_LOCAL_KV_PATH`` at
``data/manual.db`` (create it with ``scripts/bootstrap_manual_kv.py``) and
REFUSES TO START, before touching anything, unless the resolved store passes
the allowlist in :func:`_kv_path_rejection` -- it must be an existing KV store
(exactly the ``kv``/``hash_kv`` shape) or a path that does not exist yet, and
never a protected application database -- or when ``RENDER`` is set. The
resolved absolute path is printed first, every time, so a terminal's mode is
never ambiguous.

WHY THE IMPORT ORDER IN ``main`` IS LOAD-BEARING -- DO NOT "TIDY" IT.
``kv_store.get_kv()`` is a process-wide singleton that captures
``FANTASY_LOCAL_KV_PATH`` on its FIRST call (``kv_store._build_sqlite_kv``).
Any ``fantasy_baseball`` import that reaches ``get_kv()`` at import time before
this script sets the env var would bind the singleton to ``data/local.db`` for
the rest of the process, and every subsequent write would land in the Yahoo
baseline. So:

  * there is NO module-level ``fantasy_baseball`` import in this file -- every
    one of them lives inside a function body, after
    :func:`_activate_manual_environment` has run.
    ``tests/test_scripts/test_run_manual_refresh.py`` walks this file's AST via
    :func:`module_level_fantasy_imports` and fails if one is ever added;
  * :func:`_activate_manual_environment` calls ``kv_store._reset_singleton()``
    (the same move ``scripts/refresh_remote.py`` makes when it flips
    ``RENDER``), so even an already-built singleton is discarded;
  * :func:`_verify_kv_target` then asserts the OUTCOME -- that the live
    ``get_kv()`` really is backed by the requested file -- and aborts rc 2
    before the first write if it is not. Checking the outcome rather than the
    mechanism is what makes the guarantee survive a future refactor.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import logging
import os
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from fantasy_baseball.config import LeagueConfig
    from fantasy_baseball.lineup.waivers import FreeAgentSource
    from fantasy_baseball.manual.transcripts import ManualRosterSnapshot
    from fantasy_baseball.models.standings import Standings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KV_PATH = PROJECT_ROOT / "data" / "manual.db"
DEFAULT_ROSTERS = PROJECT_ROOT / "data" / "manual" / "rosters.yaml"
DEFAULT_STANDINGS = PROJECT_ROOT / "data" / "manual" / "standings.yaml"
DEFAULT_EXCLUSIONS = PROJECT_ROOT / "data" / "manual" / "fa_exclusions.yaml"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "manual"
CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"

#: The Yahoo baseline store. A manual run must never open it.
BASELINE_DB_NAME = "local.db"

#: The application database -- projections, draft results, weekly rosters, game
#: logs (``fantasy_baseball.data.db.DB_PATH``). It is not a KV store at all, and
#: on a fresh clone it does not exist yet, so a shape check alone would not stop
#: a run from creating one there.
APP_DB_NAME = "fantasy.db"

#: Databases refused by bare filename, wherever on disk they sit, with the
#: reason printed to the operator. ``resolve_path`` resolves symlinks before the
#: check, so the repo's own ``data/local.db`` and ``data/fantasy.db`` are caught
#: by this rule however they are spelled on the command line.
#:
#: Spelled as literals, not imported from ``fantasy_baseball.data.db``, because
#: this guard runs BEFORE the first ``fantasy_baseball`` import -- see the module
#: docstring. ``tests/test_scripts/test_run_manual_refresh_guard.py`` pins each
#: literal against the constant it mirrors, so a rename cannot drift silently.
PROTECTED_DBS = {
    BASELINE_DB_NAME: "the Yahoo baseline KV store",
    APP_DB_NAME: "the application database (fantasy_baseball.data.db.DB_PATH)",
}

#: A manual KV store must be named like a SQLite file. Cheap, but it turns a
#: mistyped ``--kv-path data/manual/rosters.yaml`` into a refusal instead of a
#: new store nobody meant to create.
KV_STORE_SUFFIXES = (".db", ".sqlite", ".sqlite3")

#: The EXACT shape ``kv_store.SqliteKVStore`` creates. An existing file has to
#: match this to be adopted as a manual store; see :func:`_kv_path_rejection`.
KV_STORE_SCHEMA: dict[str, frozenset[str]] = {
    "kv": frozenset({"key", "value", "expires_at"}),
    "hash_kv": frozenset({"hash_name", "field", "value"}),
}

#: First 16 bytes of every SQLite database file. Checked before handing the path
#: to ``sqlite3`` so a YAML file, a text report or an empty ``touch``ed file is
#: reported as "not a SQLite database" rather than as an empty one.
SQLITE_HEADER = b"SQLite format 3\x00"

#: Exit codes, distinct on purpose: 2 means "refused, nothing happened", 1
#: means "started, then failed". A wrapper can tell the two apart.
RC_OK = 0
RC_FAILED = 1
RC_REFUSED = 2

#: Substrings that mean a transcription file was shipped as a template and
#: never filled in. Matched case-insensitively against the raw file text.
TEMPLATE_MARKERS = ("REPLACE-ME", "REPLACE ME", "TODO:", "<team name>", "<player name>")

#: Placeholder used in the dry run's report-path preview: the real stamp is the
#: roster snapshot date, which is only known once the transcription loads.
_STAMP_PLACEHOLDER = "<roster-snapshot-date>"


# --------------------------------------------------------------------------
# Startup guards. Everything here runs BEFORE any fantasy_baseball import.
# --------------------------------------------------------------------------


def resolve_path(raw: str | Path) -> Path:
    """Absolute path for ``raw``, anchoring relative paths at the repo root.

    Same rule as ``scripts/bootstrap_manual_kv.py``: the same command means the
    same thing from any working directory, so ``--kv-path data/manual.db``
    cannot silently create a second store beside wherever the shell happens to
    be.
    """
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _refuse(message: str) -> int:
    print("")
    print(f"REFUSING TO RUN: {message}")
    print("Nothing was written, and no KV data was read.")
    return RC_REFUSED


def _read_sqlite_schema(path: Path, *, immutable: bool) -> dict[str, frozenset[str]] | None:
    """``{table: columns}`` for the SQLite database at ``path``, read-only.

    ``None`` means "sqlite could not read this as a database" -- corrupt,
    encrypted, or not one at all.

    ``immutable=1`` is not decoration: a plain ``mode=ro`` open of a WAL
    database CREATES ``-shm``/``-wal`` sidecars beside it and, being read-only,
    cannot remove them again on close. This function's whole job is to inspect a
    file the caller may be about to refuse, so the default probe must leave no
    trace. The cost is that ``immutable=1`` reads only the main database image
    and ignores an uncheckpointed WAL, which is why :func:`_kv_store_shape`
    retries without it in the one case where that can hide a real store.
    """
    flags = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    try:
        conn = sqlite3.connect("file:" + path.as_posix() + flags, uri=True)
    except sqlite3.Error:  # pragma: no cover - defensive; the open rarely fails
        return None
    try:
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            table: frozenset(str(row[1]) for row in conn.execute(f"PRAGMA table_info('{table}')"))
            for table in tables
        }
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _kv_store_shape(path: Path) -> dict[str, frozenset[str]] | None:
    """``{table: columns}`` for an existing file, or ``None`` if not a database.

    Two-step because of WAL. The zero-side-effect probe (``immutable=1``) cannot
    see tables that live in an uncheckpointed ``-wal``, and a KV store whose
    schema was created moments ago by a still-running process looks EMPTY
    through it. So when the immutable read does not produce the KV shape and a
    ``-wal`` sidecar exists, re-read through the WAL. That second open can leave
    sidecars behind, which is exactly why it is reached only for a file that
    already has one.
    """
    with path.open("rb") as handle:
        if handle.read(len(SQLITE_HEADER)) != SQLITE_HEADER:
            return None
    shape = _read_sqlite_schema(path, immutable=True)
    if shape != KV_STORE_SCHEMA and Path(str(path) + "-wal").exists():
        shape = _read_sqlite_schema(path, immutable=False)
    return shape


def _kv_path_rejection(kv_path: Path) -> str | None:
    """Why ``kv_path`` may not be used as the manual KV store, or ``None``.

    An ALLOWLIST, deliberately. The first version of this guard refused exactly
    one filename (``local.db``) and accepted everything else -- including
    ``data/fantasy.db``, 25 MB of real projections, draft results and game logs
    (``fantasy_baseball.data.db.DB_PATH``), which a manual run would have
    injected ``kv``/``hash_kv`` tables into. "Anything except the one file I
    remembered" is not a safety rule. The rule below is instead that a target
    must be recognisably a manual KV store, or not exist yet.

    Safe means all four of:

    1. not a protected application database, by bare filename (see
       :data:`PROTECTED_DBS`);
    2. named like a SQLite store, one of :data:`KV_STORE_SUFFIXES`;
    3. not a directory or other non-file;
    4. either it does not exist -- a brand-new store the operator names is fine,
       ``--dry-run`` legitimately runs before bootstrap and :func:`main` refuses
       a missing store on the live path with the bootstrap instructions -- or it
       exists and is ALREADY a KV store: exactly the ``kv``/``hash_kv`` tables
       and columns of :data:`KV_STORE_SCHEMA`, nothing more, nothing less.

    Rule 4 is what makes this an allowlist rather than a longer denylist: an
    unknown existing database has to PROVE it is a KV store to be written to,
    instead of being adopted because nobody thought to name it. Rules 1 and 4
    overlap on purpose -- ``local.db`` IS a KV store by shape, and
    ``fantasy.db`` may not exist yet on a fresh clone, so neither rule alone
    covers both.

    Inspection is strictly read-only and reads no KV VALUE: at most the file
    header and the ``sqlite_master`` schema, through a handle that cannot write
    (see :func:`_read_sqlite_schema`).
    """
    protected = PROTECTED_DBS.get(kv_path.name.lower())
    if protected is not None:
        return (
            f"the requested KV store is {protected}: {kv_path}. A manual run writes "
            "hand-typed standings and rosters into cache:standings, "
            "weekly_rosters_history and standings_history; landing those there would "
            "corrupt it. Use --kv-path data/manual.db (create it with: python "
            "scripts/bootstrap_manual_kv.py)."
        )
    if kv_path.suffix.lower() not in KV_STORE_SUFFIXES:
        return (
            f"{kv_path} is not named like a SQLite store (expected one of "
            f"{', '.join(KV_STORE_SUFFIXES)}). Use --kv-path data/manual.db (create it "
            "with: python scripts/bootstrap_manual_kv.py)."
        )
    if kv_path.is_dir():
        return (
            f"{kv_path} is a directory, not a KV store file. Use --kv-path "
            "data/manual.db (create it with: python scripts/bootstrap_manual_kv.py)."
        )
    if not kv_path.exists():
        # A path the operator named that is not there yet. Allowed: --dry-run
        # opens no store at all, and a live run refuses further down with the
        # bootstrap instructions rather than running against an empty store.
        return None
    if not kv_path.is_file():
        return (
            f"{kv_path} exists but is not a regular file. Use --kv-path data/manual.db "
            "(create it with: python scripts/bootstrap_manual_kv.py)."
        )

    shape = _kv_store_shape(kv_path)
    if shape == KV_STORE_SCHEMA:
        return None
    if shape is None:
        found = "it is not a SQLite database"
    elif not shape:
        found = "it is an empty SQLite database with no tables"
    else:
        found = "its tables are " + ", ".join(sorted(shape))
    return (
        f"{kv_path} already exists and is not a KV store -- {found}. A KV store has "
        "exactly the kv and hash_kv tables that kv_store.SqliteKVStore creates; writing "
        "this run's cache:* blobs into any other database would corrupt it. Point "
        "--kv-path at a new file, or create the standard store with: python "
        "scripts/bootstrap_manual_kv.py."
    )


def _guard_environment(kv_path: Path) -> int:
    """Return ``RC_OK`` when a manual run may proceed, else ``RC_REFUSED``.

    Refuses when ``RENDER`` is set to anything non-empty -- stricter than
    ``kv_store.is_remote()``, which only treats ``RENDER=true`` as remote. This
    is a local-only tool, so a half-set flag is a mistake, not a mode. Then
    applies :func:`_kv_path_rejection`, the allowlist over the target store.

    Deliberate deviation from the build plan, which spelled this ``-> None``
    with an in-place ``sys.exit``: returning the code keeps a single exit point
    in :func:`main`, which is what the tests call, and a ``SystemExit`` raised
    three layers down is far easier to swallow by accident than a returned int
    is to ignore.
    """
    render = os.environ.get("RENDER")
    if render:
        return _refuse(
            f"RENDER is set ({render!r}). On Render the KV store is production "
            "Upstash; a manual run must never write hand-transcribed data there. "
            "Unset RENDER and re-run."
        )
    rejection = _kv_path_rejection(kv_path)
    if rejection is not None:
        return _refuse(rejection)
    return RC_OK


def _activate_manual_environment(kv_path: Path) -> None:
    """Point this process at the manual store and disable every Yahoo step.

    Must run BEFORE the first ``fantasy_baseball`` import -- see the module
    docstring. ``sys.path`` is extended here too so the repo's ``src/`` layout
    works without the editable install, matching every other script in
    ``scripts/``.
    """
    os.environ["FANTASY_LOCAL_KV_PATH"] = str(kv_path)
    os.environ["FB_SKIP_YAHOO"] = "1"

    src = str(PROJECT_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    # Belt and braces: if anything in this process already built the KV
    # singleton (an ambient import, an earlier run in the same interpreter, a
    # test harness), discard it so the next get_kv() rebinds to the env var
    # just set. Same move as scripts/refresh_remote.py.
    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()


def _verify_kv_target(kv_path: Path) -> int:
    """Assert the live ``get_kv()`` really is backed by ``kv_path``.

    This is the guarantee that matters: not "we set the env var in the right
    order" but "the store this process will actually write to is the isolated
    one". Runs before the first write.
    """
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.manual.seed import describe_kv_target, resolve_kv_path

    client = get_kv()
    actual = resolve_kv_path(client)
    if actual != kv_path:
        return _refuse(
            "the KV singleton did not bind to the manual store. Requested "
            f"{kv_path}, got {describe_kv_target(client)}. Something imported "
            "fantasy_baseball and built the singleton before this script set "
            "FANTASY_LOCAL_KV_PATH."
        )
    return RC_OK


# --------------------------------------------------------------------------
# Transcription loading and validation -- step 1, and all of --dry-run.
# --------------------------------------------------------------------------


def template_hints(paths: dict[str, Path]) -> list[str]:
    """Actionable lines for any transcription still shipped as a template.

    The shipped ``rosters.yaml`` is a two-team worked example under a
    REPLACE-ME header. Run against the real ten-team standings it fails
    validation with a team-set mismatch, which is accurate and completely
    unhelpful. This turns that into "you have not filled the file in yet, here
    is what to do".
    """
    hints: list[str] = []
    for label, path in paths.items():
        if not path.is_file():
            hints.append(
                f"{label} does not exist: {path}. Copy the template from data/manual/ "
                "and transcribe the Yahoo pages into it."
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hit = next((m for m in TEMPLATE_MARKERS if m.lower() in text), None)
        if hit is not None:
            hints.append(
                f"{label} still contains the template marker {hit!r} ({path}). It has "
                "not been transcribed yet -- replace the whole worked-example block "
                "with the real Yahoo data before running."
            )
    return hints


def _print_errors(title: str, errors: list[str], hints: list[str]) -> None:
    print("")
    print(title)
    print("-" * max(len(title), 40))
    for err in errors:
        print(f"  * {err}")
    if hints:
        print("")
        print("What to do:")
        for hint in hints:
            print(f"  -> {hint}")
    print("")
    print("Nothing was written. Fix the transcription and re-run.")


@contextlib.contextmanager
def _quiet_missing_team_keys() -> Iterator[None]:
    """Silence ``transcripts``' per-team "no team_key" warning for one call.

    That warning is correct and worth keeping in the live path, where an empty
    ``team_key`` means the store had no Yahoo standings to look one up from.
    Here the keys are withheld ON PURPOSE -- validation must not open the KV
    store -- so the ten resulting lines are noise the reader cannot act on, and
    noise that teaches them to skim warnings. Scoped to this one logger and
    restored in a ``finally``.
    """
    logger = logging.getLogger("fantasy_baseball.manual.transcripts")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _load_transcriptions(
    args: argparse.Namespace, config: LeagueConfig
) -> tuple[Standings, ManualRosterSnapshot, frozenset[str]] | int:
    """Load and cross-validate both transcriptions, or return an exit code.

    ``team_keys={}`` on purpose: resolving the real Yahoo team keys needs the
    KV store, and ``--dry-run`` must not open it. Validation looks only at
    names and stats, and the live path re-loads the standings with the
    looked-up keys in :func:`_seed` before anything is written.
    """
    from fantasy_baseball.manual.transcripts import (
        ManualTranscriptError,
        load_fa_exclusions,
        load_manual_rosters,
        load_manual_standings,
        validate_transcripts,
    )

    rosters_path = resolve_path(args.rosters)
    standings_path = resolve_path(args.standings)
    exclusions_path = resolve_path(args.exclusions)
    labelled = {"rosters.yaml": rosters_path, "standings.yaml": standings_path}

    try:
        rosters = load_manual_rosters(rosters_path)
        with _quiet_missing_team_keys():
            standings = load_manual_standings(standings_path, team_keys={})
    except ManualTranscriptError as exc:
        _print_errors(
            f"TRANSCRIPTION ERRORS in {exc.path}", list(exc.errors), template_hints(labelled)
        )
        return RC_FAILED
    except FileNotFoundError as exc:
        _print_errors("TRANSCRIPTION FILE MISSING", [str(exc)], template_hints(labelled))
        return RC_FAILED

    errors = validate_transcripts(
        standings,
        rosters,
        team_name=config.team_name,
        roster_slots=config.roster_slots,
    )
    if errors:
        _print_errors("TRANSCRIPTIONS DISAGREE", errors, template_hints(labelled))
        return RC_FAILED

    exclusions = load_fa_exclusions(exclusions_path if exclusions_path.is_file() else None)
    return standings, rosters, exclusions


def _summarize_transcriptions(
    standings: Standings,
    rosters: ManualRosterSnapshot,
    exclusions: frozenset[str],
    config: LeagueConfig,
) -> None:
    """Print what was transcribed, per team, with the roster-size sanity check.

    Counted as active + IL rather than as one number, because the two behave
    differently: the IL slots are genuinely optional (a healthy team fills
    none), while a short ACTIVE count means either a real empty roster spot or
    -- far more likely -- a row that was skipped while reading the screenshot.
    Printed for every team, not only when it trips, so the reader can eyeball
    the shape of all ten at once.
    """
    active_slots = sum(int(v) for k, v in config.roster_slots.items() if k.upper() != "IL")
    il_slots = int(config.roster_slots.get("IL", 0))
    slot_summary = ", ".join(f"{k}:{v}" for k, v in config.roster_slots.items())
    print("")
    print(
        f"Transcriptions OK. Roster snapshot {rosters.snapshot_date.isoformat()}, "
        f"standings effective {standings.effective_date.isoformat()}."
    )
    print(f"  configured slots: {active_slots} active + {il_slots} IL ({slot_summary})")
    total = 0
    for team, rows in rosters.rows_by_team.items():
        total += len(rows)
        il = sum(1 for row in rows if row["slot"].upper() == "IL")
        active = len(rows) - il
        flag = "" if active == active_slots else f"   <-- {active_slots} active slots; re-check"
        mark = "*" if team == config.team_name else " "
        print(f"  {mark} {team:<32} {active:>3} active + {il} IL{flag}")
    print(
        f"  {total} players across {len(rosters.rows_by_team)} teams "
        f"(* = your team, {config.team_name})"
    )
    if exclusions:
        print(f"  free-agent exclusions: {len(exclusions)} name(s)")
    else:
        print("  free-agent exclusions: none (data/manual/fa_exclusions.yaml is empty)")
        print("    NOTE: synthesized free agents carry no injury status. An FA who")
        print("    went on the IL after the projection snapshot still shows as a")
        print("    full-value upgrade. Confirm availability before making a move.")


def _describe_ros_snapshot(season: int) -> tuple[str, bool]:
    """Return ``(description, fresh_enough_to_blend)`` for the latest ROS dir.

    Reuses ``ros_pipeline``'s own selection and staleness helpers rather than
    re-deriving either, so this preview cannot disagree with the guard that
    actually refuses the blend.
    """
    from fantasy_baseball.data.ros_pipeline import (
        ROS_SNAPSHOT_STALE_DAYS,
        latest_ros_snapshot,
        ros_snapshot_days_stale,
    )

    found = latest_ros_snapshot(PROJECT_ROOT / "data" / "projections", season)
    if found is None:
        return f"no dated snapshot under data/projections/{season}/rest_of_season", False
    path, snap = found
    stale = ros_snapshot_days_stale(snap)
    fresh = stale <= ROS_SNAPSHOT_STALE_DAYS
    verdict = "fresh" if fresh else f"STALE (> {ROS_SNAPSHOT_STALE_DAYS} days; the blend refuses)"
    return f"{path.name} -- {stale} day(s) old, {verdict}", fresh


def _report_path(args: argparse.Namespace, stamp: str) -> Path:
    """Where the report is written. ``stamp`` is the roster snapshot date."""
    if args.report_out is not None:
        return resolve_path(args.report_out)
    return DEFAULT_REPORT_DIR / f"audit-{stamp}.txt"


def _print_dry_run_plan(args: argparse.Namespace, kv_path: Path, season: int) -> None:
    ros_desc, ros_fresh = _describe_ros_snapshot(season)
    report_out = _report_path(args, _STAMP_PLACEHOLDER)
    missing = "" if kv_path.exists() else "   <-- MISSING; run scripts/bootstrap_manual_kv.py"
    print("")
    # "Read" here means KV data. The startup guard does open the target
    # read-only to check that it has the kv/hash_kv shape; it reads no value
    # and cannot write. See _kv_path_rejection.
    print("DRY RUN -- validation only. Nothing was read from or written to the store.")
    print("A real run would then:")
    print(f"  1. write to {kv_path}{missing}")
    if args.skip_game_logs:
        print("  2. SKIP the MLB game-log sync (--skip-game-logs)")
    else:
        print(f"  2. sync MLB Stats API game logs for {season} into that store")
    if args.skip_blend:
        print("  3. SKIP the ROS blend (--skip-blend); reuse the store's cache:ros_projections")
        print(f"     latest on-disk snapshot: {ros_desc}")
    else:
        print(f"  3. blend ROS projections from the latest snapshot: {ros_desc}")
        if not ros_fresh:
            print(
                "     ACTION: stage a fresh FanGraphs export "
                "(python scripts/ingest_ros_export.py --no-push), or pass --skip-blend."
            )
    print("  4. seed cache:standings, standings_history and weekly_rosters_history")
    print("  5. run RefreshRun(skip_yahoo=True, free_agent_source=manual, job_label='manual')")
    print(f"  6. print the audit report and write it to {report_out}")
    if args.with_keeper_board:
        span = _panel_span()
        panel_desc = span[0].name if span else "NO PANEL FOUND"
        if args.skip_panel_rebuild:
            print(f"  7. rebuild the keeper board, REUSING the panel {panel_desc}")
        else:
            print(f"  7. rebuild the keeper board, refreshing {panel_desc} first if stale")


# --------------------------------------------------------------------------
# Live-run steps.
# --------------------------------------------------------------------------


def _sync_game_logs(season: int) -> int:
    """Incremental MLB Stats API sync into the manual store.

    The rollups are asserted non-empty afterwards because an empty one fails
    SILENTLY downstream: ``derive_full_season`` would add nothing to the ROS
    frame, full-season would equal ROS, and ``scoring._full_season_volume``
    would then read ROS volume as though it were a season's worth -- inflating
    every team SD without a single error line.
    """
    from fantasy_baseball.data.mlb_game_logs import fetch_game_log_totals

    print("")
    print(f"[2/6] Syncing MLB game logs for {season} (MLB Stats API, no auth needed)...")
    hitters, pitchers, games_elapsed = fetch_game_log_totals(season, progress_cb=_indent)
    if not hitters or not pitchers:
        print("")
        print(
            "ERROR: the game-log rollup came back empty "
            f"(hitters={len(hitters)}, pitchers={len(pitchers)})."
        )
        print("  Full-season projections would silently collapse to ROS-only and every")
        print("  team SD would be inflated. Refusing to continue.")
        return RC_FAILED
    print(
        f"  game logs: {len(hitters)} hitters, {len(pitchers)} pitchers, "
        f"{games_elapsed} team-games elapsed"
    )
    return RC_OK


def _blend_ros(season: int, config: LeagueConfig) -> int:
    """Blend the latest ROS CSV snapshot into the manual store.

    Calls ``blend_and_cache_ros`` DIRECTLY. Never route this through
    ``scripts/ingest_ros_export._push_to_prod`` (it force-sets ``RENDER=true``,
    which would push hand-transcribed context to production Upstash) or through
    ``scripts/refresh_remote`` (its tail wipes and re-syncs whatever
    ``get_kv()`` resolves to -- here, the manual store).

    ``roster_names=None`` deliberately: it only feeds
    ``check_projection_quality``'s coverage log line and changes no output
    number.
    """
    from fantasy_baseball.data.ros_pipeline import StaleROSSnapshotError, blend_and_cache_ros

    systems = list(config.projection_systems)
    weights = {s: config.projection_weights[s] for s in systems if s in config.projection_weights}

    print("")
    print(f"[3/6] Blending ROS projections ({', '.join(systems)})...")
    try:
        ros_h, ros_p = blend_and_cache_ros(
            PROJECT_ROOT / "data" / "projections",
            systems,
            weights,
            None,
            season,
            progress_cb=_indent,
        )
    except StaleROSSnapshotError as exc:
        print("")
        print(f"ERROR: {exc}")
        print("  Stage a fresh export (python scripts/ingest_ros_export.py --no-push),")
        print("  or pass --skip-blend to reuse the ROS blob already in the manual store.")
        return RC_FAILED
    except FileNotFoundError as exc:
        print("")
        print(f"ERROR: {exc}")
        return RC_FAILED
    print(f"  blended {len(ros_h)} ROS hitters + {len(ros_p)} ROS pitchers")
    return RC_OK


def _seed(args: argparse.Namespace, rosters: ManualRosterSnapshot) -> int:
    """Re-load the standings with looked-up team keys, then seed the store.

    The standings are loaded a SECOND time here rather than reusing the
    validation copy: ``team_key`` is a Yahoo identifier and this project's rule
    is that identifiers are looked up, never recalled. The keys come from the
    store's own ``cache:standings`` blob via
    :func:`manual.seed.read_team_keys`, so a re-seed preserves the real keys
    the last Yahoo run wrote.
    """
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.manual.seed import ManualSeedRefused, read_team_keys, seed_manual_kv
    from fantasy_baseball.manual.transcripts import ManualTranscriptError, load_manual_standings

    print("")
    print("[4/6] Seeding the transcriptions into the manual store...")
    client = get_kv()
    team_keys = read_team_keys(client)
    print(f"  looked up {len(team_keys)} Yahoo team_key(s) from the store's cache:standings")

    try:
        standings = load_manual_standings(resolve_path(args.standings), team_keys=team_keys)
        stats = seed_manual_kv(client, standings, rosters, echo=_indent)
    except ManualTranscriptError as exc:
        _print_errors(f"TRANSCRIPTION ERRORS in {exc.path}", list(exc.errors), [])
        return RC_FAILED
    except ManualSeedRefused as exc:
        return _refuse(str(exc))
    print(f"  seeded {stats.players} players across {stats.teams} teams")
    return RC_OK


def _build_free_agent_source(
    args: argparse.Namespace, exclusions: frozenset[str]
) -> FreeAgentSource:
    """``functools.partial`` over ``build_manual_free_agents`` -- a FreeAgentSource.

    ``positions_by_name`` comes from ``keepers.positions.load_positions()``:
    the committed ``data/player_positions.json`` merged under the frozen
    ``cache:positions`` blob. Coverage of that map is the binding constraint on
    the pool -- a hitter with no position data cannot be offered -- so the count
    is printed. Run ``python scripts/fetch_positions_mlb.py`` to backfill it
    before a real run.
    """
    from functools import partial

    from fantasy_baseball.keepers.positions import load_positions
    from fantasy_baseball.manual.free_agents import (
        DEFAULT_PER_POSITION_CAP,
        build_manual_free_agents,
    )

    cap = args.per_position_cap if args.per_position_cap is not None else DEFAULT_PER_POSITION_CAP
    positions_by_name = load_positions()
    print(
        f"  position eligibility map: {len(positions_by_name)} players; "
        f"per-position cap {cap}; {len(exclusions)} exclusion(s)"
    )
    return partial(
        build_manual_free_agents,
        positions_by_name=positions_by_name,
        excluded_names=exclusions,
        per_position_cap=cap,
    )


def _run_pipeline(args: argparse.Namespace, exclusions: frozenset[str]) -> int:
    """Run the real pipeline with Yahoo disabled and the synthesized FA pool.

    ``job_label="manual"`` stamps every ``cache:*`` blob's ``_meta._job`` and
    routes the job log to ``job_log:manual:*``, so a manual run's output is
    distinguishable from a Yahoo run's after the fact.
    """
    from fantasy_baseball.web.refresh_pipeline import ManualRosterUnmatched, RefreshRun

    print("")
    print("[5/6] Running the refresh pipeline (Yahoo steps disabled)...")
    source = _build_free_agent_source(args, exclusions)
    try:
        RefreshRun(skip_yahoo=True, free_agent_source=source, job_label="manual").run()
    except ManualRosterUnmatched as exc:
        # A transcription error the operator can fix, not a crash. Surfaced the
        # same way the transcription-validation errors are, because it IS one --
        # it is simply not detectable until the projection frames are loaded.
        _print_errors(
            "ROSTER PLAYERS DID NOT MATCH A PROJECTION ROW",
            [str(exc)],
            [
                "Check the spelling against the projection frames -- the matcher "
                "strips accents and suffixes, so the mismatch is usually a middle "
                "initial, a Jr./Sr. difference, or a player absent from the ROS export.",
                "Re-run after editing data/manual/rosters.yaml.",
            ],
        )
        return RC_FAILED
    return RC_OK


def _current_roto_standings(standings: Standings) -> list[tuple[str, float]]:
    """``(team, roto_total)`` in Yahoo's own rank order -- ordered, not re-ranked.

    Prefers ``yahoo_points_for`` (Yahoo's authoritative total, transcribed from
    the standings page) over a locally scored total whenever every team has
    one, mirroring ``season_data.build_standings_view``. Local scoring differs
    by up to +/-0.5 per display tie in the rounded rate categories, and the
    transcription has several such ties -- so re-deriving here would print
    numbers that disagree with the page the user is reading.
    """
    from fantasy_baseball.scoring import score_roto

    entries = sorted(standings.entries, key=lambda e: e.rank)

    # Spelled as an explicit `is None` break rather than `e.yahoo_points_for or
    # 0.0`: 0.0 is a legal roto total (a team last in all ten categories scores
    # 10, but the arithmetic must not depend on that), and a falsy default
    # would silently substitute for a missing one. The `else` runs only when no
    # entry was missing a total.
    yahoo_totals: list[tuple[str, float]] = []
    for entry in entries:
        total = entry.yahoo_points_for
        if total is None:
            break
        yahoo_totals.append((entry.team_name, float(total)))
    else:
        return yahoo_totals

    # Standings is structurally a TeamStatsTable (team_name/stats per entry);
    # mypy cannot see the protocol variance through list[StandingsEntry] vs
    # Sequence[TeamStatsRow]. Same cast, same reason, as season_data.
    roto = score_roto(cast("Any", standings))
    return [(e.team_name, float(roto[e.team_name].total)) for e in entries]


def _render_report(
    args: argparse.Namespace,
    config: LeagueConfig,
    standings: Standings,
    rosters: ManualRosterSnapshot,
    kv_path: Path,
) -> int:
    """Read the audit back out of the manual store, render it, print and save it."""
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.lineup.roster_audit import AuditEntry
    from fantasy_baseball.manual.report import render_audit_report
    from fantasy_baseball.models.standings import ProjectedStandings
    from fantasy_baseball.web.season_data import (
        read_cache_dict,
        read_cache_list,
        read_cache_with_meta,
    )

    print("")
    print("[6/6] Rendering the audit report...")
    raw = read_cache_list(CacheKey.ROSTER_AUDIT) or []
    entries = [AuditEntry(**row) for row in raw if isinstance(row, dict)]

    projections = read_cache_dict(CacheKey.PROJECTIONS) or {}
    fraction_remaining = projections.get("fraction_remaining")
    raw_projected = projections.get("projected_standings")
    projected_standings = (
        ProjectedStandings.from_json(raw_projected) if isinstance(raw_projected, dict) else None
    )

    _, ros_meta = read_cache_with_meta(CacheKey.ROS_PROJECTIONS)
    ros_snapshot_date = str(ros_meta.get("_ros_snapshot_date", "unknown"))

    transcribed = len(rosters.rows_by_team.get(config.team_name, []))
    if len(entries) != transcribed:
        print(
            f"  NOTE: {transcribed} players transcribed for {config.team_name} but the "
            f"audit returned {len(entries)} -- the difference did not match a "
            "projection row. Check the spelling of the missing name(s)."
        )

    # The free start/bench swaps. They are the highest-value moves available --
    # they cost no waiver claim and cannot be sniped -- and the add/drop deltas
    # are computed against this lineup, so the report must show both or a reader
    # will act on the add/drop half alone.
    lineup_optimal = read_cache_dict(CacheKey.LINEUP_OPTIMAL) or {}
    lineup_moves = lineup_optimal.get("moves")

    report = render_audit_report(
        entries,
        team_name=config.team_name,
        effective_date=rosters.snapshot_date,
        fraction_remaining=fraction_remaining if fraction_remaining is not None else 0.0,
        ros_snapshot_date=ros_snapshot_date,
        kv_path=str(kv_path),
        projected_standings=projected_standings,
        roto_standings=_current_roto_standings(standings),
        lineup_moves=lineup_moves,
    )

    out_path = _report_path(args, rosters.snapshot_date.isoformat())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")

    print("")
    print(report)
    print("")
    print(f"Report written to {out_path}")
    return RC_OK


# --------------------------------------------------------------------------
# Wiring.
# --------------------------------------------------------------------------


#: How far the panel's own elapsed-season reading may drift from the pipeline's before
#: the keeper board is treated as built on stale actuals. The two measure elapsed season
#: DIFFERENTLY -- the panel takes the busiest hitter's games over a full schedule, the
#: pipeline takes calendar days -- so they never agree exactly even when both are current;
#: on 2026-08-22 they read 0.809 and 0.800. 0.05 is wide enough to ignore that definitional
#: gap and narrow enough to catch a genuinely stale panel: the 2026-08-08 panel still in
#: place on 2026-08-22 read 0.698 against a true 0.800, a drift of 0.10.
PANEL_DRIFT_TOLERANCE = 0.05


def _panel_span() -> tuple[Path, int, int] | None:
    """The hitter panel ``panel_path`` will actually read, and its ``(start, end)`` years.

    The span is parsed back off the filename so a rebuild OVERWRITES the file in use.
    ``scripts/build_pt_panel.py`` defaults to ``--start 2010``, which writes
    ``hitter_pt_panel_2010_2026.csv`` -- a DIFFERENT file from the ``_2000_2026`` one
    ``panel_path`` resolves (it picks the widest span, not the newest mtime). Rebuilding
    with the default therefore leaves the stale panel in place and silently in use, which
    is exactly how a board came to blend fresh projections with three-week-old actuals.
    """
    from fantasy_baseball.trajectory.panel import panel_path

    try:
        path = panel_path("hitter")
    except FileNotFoundError:
        return None
    try:
        start, end = (int(x) for x in path.stem.rsplit("_", 2)[-2:])
    except ValueError:
        return None
    return path, start, end


def _panel_drift(season: int) -> tuple[float, float] | None:
    """``(panel_elapsed, pipeline_elapsed)``, or None when either is unavailable."""
    import pandas as pd

    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.trajectory.panel import season_elapsed_fraction
    from fantasy_baseball.web.season_data import read_cache_dict

    span = _panel_span()
    if span is None:
        return None
    projections = read_cache_dict(CacheKey.PROJECTIONS) or {}
    remaining = projections.get("fraction_remaining")
    if remaining is None:
        return None
    try:
        panel_elapsed = season_elapsed_fraction(pd.read_csv(span[0]), season)
    except (ValueError, OSError):
        return None
    return panel_elapsed, 1.0 - float(remaining)


def _run_child(label: str, argv: list[str]) -> int:
    """Run a sibling script in a child process, inheriting the manual environment.

    ``FANTASY_LOCAL_KV_PATH`` and ``FB_SKIP_YAHOO`` are already in ``os.environ`` by the
    time any step runs (see ``_activate_manual_environment``), and a child inherits them,
    so ``push_trajectory_board.py --local`` resolves to the SAME isolated store this run
    is writing -- never ``data/local.db``.
    """
    import subprocess

    print(f"  {label}")
    completed = subprocess.run([sys.executable, *argv], check=False)
    if completed.returncode != RC_OK:
        print(f"  FAILED: {' '.join(argv)} exited {completed.returncode}")
    return completed.returncode


def _keeper_board(args: argparse.Namespace, season: int) -> int:
    """Rebuild the trajectory (keeper value) board into the manual store.

    Runs AFTER the audit because it is an independent product: the board is a
    multi-year keeper valuation, not an input to any in-season recommendation, and a
    failure here must not cost the operator the audit they came for. That is why the
    caller treats a non-zero return as a warning rather than a hard failure.

    The board is NOT part of the refresh pipeline and cannot be -- the fit reads
    ``data/trajectory/*.csv`` and ``data/cache/keeper_skills``, both gitignored and both
    absent on Render. It therefore does not move when the pipeline runs, which is the
    whole reason it is wired here.
    """
    print("")
    print("[keeper board] Rebuilding the trajectory board...")

    drift = _panel_drift(season)
    span = _panel_span()
    if span is None:
        print("  no hitter panel found; build one with scripts/build_pt_panel.py")
        return RC_FAILED

    path, start, end = span
    if drift is None:
        print(f"  panel {path.name}: could not read its elapsed-season fraction")
    else:
        panel_elapsed, pipeline_elapsed = drift
        gap = abs(panel_elapsed - pipeline_elapsed)
        print(
            f"  panel {path.name}: {panel_elapsed:.1%} of season played; "
            f"pipeline says {pipeline_elapsed:.1%} (drift {gap:.1%})"
        )
        if gap > PANEL_DRIFT_TOLERANCE and args.skip_panel_rebuild:
            print(
                "  WARNING: the panel is stale and --skip-panel-rebuild was passed. The "
                "board's season-to-date half will be older than its projections."
            )
        elif gap > PANEL_DRIFT_TOLERANCE:
            rc = _run_child(
                f"panel is stale -- rebuilding {path.name} ({start}-{end})",
                [
                    str(PROJECT_ROOT / "scripts" / "build_pt_panel.py"),
                    "--start",
                    str(start),
                    "--end",
                    str(end),
                    "--refresh",
                ],
            )
            if rc != RC_OK:
                return rc

    return _run_child(
        "sweeping the player pool and writing the board",
        [str(PROJECT_ROOT / "scripts" / "push_trajectory_board.py"), "--local"],
    )


def _indent(message: str) -> None:
    """Progress callback: indent a pipeline line under its step heading."""
    print(f"  {message}")


def _reconfigure_stdout() -> None:
    """UTF-8 stdout with replacement, so an accented name cannot kill the run.

    The report is ASCII by construction but the DATA is not -- this league
    rosters players whose names carry accents, and a Windows cp1252 console
    raises ``UnicodeEncodeError`` on the first one. Guarded because pytest's
    capture object is not a real ``TextIOWrapper``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):  # pragma: no cover - defensive
            reconfigure(encoding="utf-8", errors="replace")


def module_level_fantasy_imports(source: str) -> list[str]:
    """Names of ``fantasy_baseball`` modules imported at ``source``'s TOP LEVEL.

    Exposed, and used by the test suite, because "no top-level
    ``fantasy_baseball`` import in this script" is the invariant that keeps
    ``get_kv()`` from binding to ``data/local.db`` before :func:`main` has set
    ``FANTASY_LOCAL_KV_PATH``. A comment cannot enforce that; a test over this
    function can. ``if TYPE_CHECKING:`` blocks are exempt -- they never
    execute.
    """
    tree = ast.parse(source)
    found: list[str] = []

    def _is_type_checking(node: ast.stmt) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"

    for node in tree.body:
        if _is_type_checking(node):
            continue
        # Walk compound top-level statements (if/try/with) too: an import
        # nested in one of those still runs at import time.
        nodes = ast.walk(node) if isinstance(node, ast.If | ast.Try | ast.With) else [node]
        for sub in nodes:
            if isinstance(sub, ast.Import):
                found += [a.name for a in sub.names if a.name.split(".")[0] == "fantasy_baseball"]
            elif isinstance(sub, ast.ImportFrom) and sub.level == 0:
                module = sub.module or ""
                if module.split(".")[0] == "fantasy_baseball":
                    found.append(module)
    return found


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the season refresh from hand-transcribed inputs, without Yahoo."
    )
    parser.add_argument(
        "--kv-path",
        default=str(DEFAULT_KV_PATH),
        help="Isolated KV store to read and write (default: data/manual.db). Never "
        "data/local.db -- that is the Yahoo baseline.",
    )
    parser.add_argument(
        "--rosters", default=str(DEFAULT_ROSTERS), help="Roster transcription YAML."
    )
    parser.add_argument(
        "--standings", default=str(DEFAULT_STANDINGS), help="Standings transcription YAML."
    )
    parser.add_argument(
        "--exclusions",
        default=str(DEFAULT_EXCLUSIONS),
        help="Names to drop from the synthesized free-agent pool.",
    )
    parser.add_argument(
        "--season", type=int, default=None, help="Season year (default: config season_year)."
    )
    parser.add_argument(
        "--per-position-cap",
        type=int,
        default=None,
        help="Free agents kept per hitter position bucket (default: manual.free_agents default).",
    )
    parser.add_argument(
        "--skip-game-logs", action="store_true", help="Reuse the store's existing MLB game logs."
    )
    parser.add_argument(
        "--skip-blend", action="store_true", help="Reuse the store's existing ROS projection blob."
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Where to write the report (default: data/manual/audit-<snapshot-date>.txt).",
    )
    parser.add_argument(
        "--with-keeper-board",
        action="store_true",
        help="Also rebuild the trajectory (keeper value) board into the manual store. "
        "Adds several minutes: it rebuilds the playing-time panel when stale, then "
        "sweeps the whole player pool.",
    )
    parser.add_argument(
        "--skip-panel-rebuild",
        action="store_true",
        help="With --with-keeper-board, reuse the existing playing-time panel even when "
        "it is stale. Warns rather than refuses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the transcriptions and print the plan. Opens no KV store.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. The step order below is the safety contract -- see the module docstring."""
    _reconfigure_stdout()
    args = _build_parser().parse_args(argv)
    kv_path = resolve_path(args.kv_path)
    mode = "DRY RUN (no KV access)" if args.dry_run else "LIVE (writes the store above)"

    print("=" * 72)
    print("MANUAL PIPELINE -- hand-transcribed inputs, Yahoo DISABLED")
    print(f"  KV store : {kv_path}")
    print(f"  mode     : {mode}")
    print("=" * 72)

    guard_rc = _guard_environment(kv_path)
    if guard_rc != RC_OK:
        return guard_rc

    # ---- NOTHING ABOVE THIS LINE MAY IMPORT fantasy_baseball. ----
    _activate_manual_environment(kv_path)

    from fantasy_baseball.config import load_config

    config = load_config(CONFIG_PATH)
    season = args.season if args.season is not None else config.season_year

    loaded = _load_transcriptions(args, config)
    if isinstance(loaded, int):
        return loaded
    standings, rosters, exclusions = loaded
    _summarize_transcriptions(standings, rosters, exclusions, config)

    if args.dry_run:
        _print_dry_run_plan(args, kv_path, season)
        return RC_OK

    if not kv_path.exists():
        return _refuse(
            f"the manual KV store does not exist: {kv_path}. Create it first with "
            "'python scripts/bootstrap_manual_kv.py' -- an empty store has no blended "
            "projections and the pipeline would fail partway through."
        )

    verify_rc = _verify_kv_target(kv_path)
    if verify_rc != RC_OK:
        return verify_rc

    if args.skip_game_logs:
        print("")
        print("[2/6] Skipping the MLB game-log sync (--skip-game-logs).")
    else:
        rc = _sync_game_logs(season)
        if rc != RC_OK:
            return rc

    if args.skip_blend:
        print("")
        print("[3/6] Skipping the ROS blend (--skip-blend).")
    else:
        rc = _blend_ros(season, config)
        if rc != RC_OK:
            return rc

    rc = _seed(args, rosters)
    if rc != RC_OK:
        return rc

    rc = _run_pipeline(args, exclusions)
    if rc != RC_OK:
        return rc

    report_rc = _render_report(args, config, standings, rosters, kv_path)
    if report_rc != RC_OK or not args.with_keeper_board:
        return report_rc

    # A board failure does not invalidate the audit that already printed and saved, so
    # it degrades to a warning. Returning RC_FAILED here would tell a caller the whole
    # run failed when the thing they asked for succeeded.
    if _keeper_board(args, season) != RC_OK:
        print("")
        print("  WARNING: the keeper board did not rebuild. The audit above is unaffected;")
        print("  the board in the store is whatever the last successful push wrote.")
    return RC_OK


if __name__ == "__main__":
    raise SystemExit(main())
