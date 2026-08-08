# loop-review Rebuild (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-08-08-loop-review-rebuild-design.md`

**Goal:** Replace `/loop-review`'s finding-production engine with three model/effort-tiered dimension agents, compute gates once in the parent, and add weighted per-agent cost instrumentation so a later tier-down is data-driven.

**Architecture:** The skill keeps its existing ledger, arbitration rules, churn/drift discriminators, and one-fix-at-a-time protocol. New Python modules (`cost.py`, `gates.py`, `goal.py`) sit beside `ledger.py` and are imported by it or invoked as CLIs; three new agent definitions in `~/.claude/agents/` carry per-dimension `model:` and `effort:` frontmatter; `SKILL.md` is rewritten to drive them via the Agent tool's `subagent_type`.

**Tech Stack:** Python 3.12 standard library only (no third-party imports — these scripts run under both PowerShell and bash on a Windows box and must not require an install). `git` and `gh` CLIs. `rg` (ripgrep) for blast-radius search.

## Global Constraints

- **ASCII-only** in every source file, log message, format string, and report renderer. This dev box is Windows; stdout defaults to cp1252 and a non-ASCII glyph raises `UnicodeEncodeError` on the first print. Use `-` not en/em dash, `sigma` not the glyph, straight quotes, `->` not the arrow.
- **Standard library only.** No `requests`, no `pytest`, no third-party anything. `selftest.py` is the test harness.
- **Hermetic tests.** Every test uses `tempfile.mkdtemp()` and never touches a real ledger, a real git repo, or the real `~/.claude/projects` tree.
- **Never `x or default` for numeric defaults.** `0`, `0.0`, and `""` are falsy; use `d["k"] if d.get("k") is not None else default`. Load-bearing throughout cost math, where a genuine zero token count is meaningful.
- **Append-only ledger.** New state goes in via `ledger._append` to `ledger.jsonl` or `passes.jsonl`. Never rewrite a log in place.
- **Paths normalize** through `ledger.norm_path()` to repo-relative POSIX before anything compares them.
- **Skill directory:** `C:/Users/HartAlden/.claude/skills/loop-review` (spelled out; `~` does not expand reliably across both shells here).
- **Files live outside this repo.** The repo's `pytest` does not cover them. `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py` is the gate for every task.
- **Commit each task in the FantasyBaseball repo** on branch `feat/loop-review-rebuild`. The skill files are outside the repo and are not version-controlled by it; each task's commit records the plan progress and any spec/doc changes. State this in the commit body when the task touched only external files.

---

### Task 1: Price table and weighted cost math

**Files:**
- Create: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/prices.json`
- Create: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py` (modify)

**Interfaces:**
- Consumes: nothing.
- Produces: `cost.load_prices(path=None) -> dict`; `cost.resolve_model(alias: str) -> str`; `cost.weighted_cost(usage: dict, model: str, prices: dict) -> float`; `cost.price_staleness(prices: dict, today: str) -> int | None` (days since verified, `None` if unparseable); `cost.STALE_DAYS = 90`.

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_cost_math():
    prices = cost.load_prices()
    check("prices has opus", "claude-opus-5" in prices["models"])
    check("alias opus", cost.resolve_model("opus") == "claude-opus-5")
    check("alias passthrough", cost.resolve_model("claude-sonnet-5") == "claude-sonnet-5")

    # 1M input, 1M output on opus-5 at $5/$25 = $30.00
    plain = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    got = cost.weighted_cost(plain, "claude-opus-5", prices)
    check("plain cost", abs(got - 30.0) < 1e-9, f"got {got}")

    # Cache-read-dominated: the R8.3 case. 345145 cache reads + 281 output.
    # reads: 345145 * 5.0 * 0.1 / 1e6 = 0.1725725
    # output: 281 * 25.0 / 1e6      = 0.007025
    cached = {
        "input_tokens": 0,
        "output_tokens": 281,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 345_145,
    }
    got = cost.weighted_cost(cached, "claude-opus-5", prices)
    check("cache-read cost", abs(got - 0.1795975) < 1e-9, f"got {got}")

    # Raw-token counting would rank this agent above the plain one; weighted must not.
    check("weighting reorders", got < 30.0)

    # Cache writes bill at 1.25x input.
    written = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 0,
    }
    got = cost.weighted_cost(written, "claude-opus-5", prices)
    check("cache-write cost", abs(got - 6.25) < 1e-9, f"got {got}")

    # A genuine zero must not be treated as missing.
    zeroed = {"input_tokens": 0, "output_tokens": 0,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    check("zero usage is zero", cost.weighted_cost(zeroed, "claude-opus-5", prices) == 0.0)

    # Unknown model must fail loudly, not silently price at zero.
    try:
        cost.weighted_cost(plain, "claude-nonexistent-9", prices)
        check("unknown model raises", False, "no exception")
    except KeyError:
        check("unknown model raises", True)

    check("fresh table", cost.price_staleness(prices, "2026-08-08") == 0)
    check("stale table", cost.price_staleness(prices, "2026-12-01") > cost.STALE_DAYS)
```

Register it in `main()` by adding `test_cost_math()` after `test_harvest_classify()`, and add `import cost` beside `import harvest`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'cost'`

- [ ] **Step 3: Write minimal implementation**

`prices.json` — standard rates, not the Sonnet 5 introductory rate. Introductory pricing expires 2026-08-31 and using it would understate cost from September onward, which biases the table toward recommending a downgrade:

```json
{
  "verified": "2026-08-08",
  "source": "claude-api skill model table, verified 2026-08-08",
  "note": "Standard rates. Sonnet 5 has introductory $2.00/$10.00 through 2026-08-31; standard is used here so the table never understates cost.",
  "cache_read_multiplier": 0.1,
  "cache_write_multiplier": 1.25,
  "models": {
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0}
  }
}
```

`cost.py`:

```python
"""Weighted cost attribution for loop-review's finder agents.

Raw token totals misrank agents by more than an order of magnitude: cache reads
bill at 0.1x input and cache writes at 1.25x, so a subagent with 345k cache
reads and 281 output tokens costs about 18 cents, not "345k tokens of work".
Everything here prices against prices.json rather than counting tokens.

ASCII-only output: this runs on Windows terminals (cp1252 stdout).
"""

from __future__ import annotations

import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES_PATH = os.path.join(HERE, "prices.json")
STALE_DAYS = 90

# Agent frontmatter uses short aliases; transcripts and the price table use ids.
MODEL_ALIASES = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5",
}


def load_prices(path: str | None = None) -> dict:
    with open(path or PRICES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_model(alias: str) -> str:
    """Map a frontmatter alias to a price-table model id; pass ids through."""
    return MODEL_ALIASES.get(alias, alias)


def weighted_cost(usage: dict, model: str, prices: dict) -> float:
    """USD for one usage block. Raises KeyError on an unpriced model."""
    rates = prices["models"][resolve_model(model)]
    read_mult = prices["cache_read_multiplier"]
    write_mult = prices["cache_write_multiplier"]

    def tok(key: str) -> int:
        # Not `usage.get(key) or 0`: a real 0 must stay 0 and a missing key must
        # also be 0, but the two must not be conflated with a falsy non-zero.
        value = usage.get(key)
        return 0 if value is None else int(value)

    dollars = (
        tok("input_tokens") * rates["input"]
        + tok("cache_creation_input_tokens") * rates["input"] * write_mult
        + tok("cache_read_input_tokens") * rates["input"] * read_mult
        + tok("output_tokens") * rates["output"]
    )
    return dollars / 1_000_000


def price_staleness(prices: dict, today: str) -> int | None:
    """Days between the table's verification date and `today` (YYYY-MM-DD)."""
    try:
        verified = datetime.date.fromisoformat(prices["verified"])
        now = datetime.date.fromisoformat(today)
    except (KeyError, ValueError):
        return None
    return (now - verified).days
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, with `ok  prices has opus` through `ok  stale table` in the output.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git add docs/superpowers/plans/2026-08-08-loop-review-rebuild.md
git commit -m "plan: loop-review rebuild task 1 - price table and weighted cost math

Skill files live outside this repo (~/.claude/skills/loop-review); this commit
records plan progress only."
```

---

### Task 2: Transcript discovery and usage summing

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `cost.weighted_cost`, `cost.resolve_model` from Task 1.
- Produces: `cost.marker(pass_no: int, dimension: str) -> str`; `cost.find_transcripts(project_dir: str, marker: str) -> list[str]`; `cost.agent_usage(path: str) -> dict` returning keys `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `wall_ms`, `model`.

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def _write_transcript(directory, name, marker, messages):
    """Write a fake subagent transcript. `messages` is a list of usage dicts."""
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": marker + "\n\nreview this diff"},
            "timestamp": "2026-08-08T10:00:00.000Z",
        }) + "\n")
        for i, usage in enumerate(messages):
            fh.write(json.dumps({
                "type": "assistant",
                "message": {"role": "assistant", "model": "claude-opus-5", "usage": usage},
                "timestamp": f"2026-08-08T10:0{i + 1}:00.000Z",
            }) + "\n")
    return path


def test_transcript_scanning():
    directory = tempfile.mkdtemp()
    try:
        sub = os.path.join(directory, "sess-a", "subagents")
        os.makedirs(sub)
        m = cost.marker(3, "correctness")
        check("marker form", m == "LR-PASS-3-correctness", m)

        _write_transcript(sub, "agent-aaa.jsonl", m, [
            {"input_tokens": 10, "output_tokens": 5,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 100},
            {"input_tokens": 2, "output_tokens": 7,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 200},
        ])
        # A different pass's agent must not match.
        _write_transcript(sub, "agent-bbb.jsonl", cost.marker(4, "correctness"), [
            {"input_tokens": 1, "output_tokens": 1,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ])
        # A second session directory: cost.py must scan all of them.
        sub2 = os.path.join(directory, "sess-b", "subagents")
        os.makedirs(sub2)
        _write_transcript(sub2, "agent-ccc.jsonl", m, [
            {"input_tokens": 3, "output_tokens": 3,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ])

        found = cost.find_transcripts(directory, m)
        check("finds both matching", len(found) == 2, f"got {len(found)}")
        check("skips non-matching",
              all("agent-bbb" not in p for p in found))

        u = cost.agent_usage(os.path.join(sub, "agent-aaa.jsonl"))
        check("sums input", u["input_tokens"] == 12, str(u))
        check("sums output", u["output_tokens"] == 12, str(u))
        check("sums cache reads", u["cache_read_input_tokens"] == 300, str(u))
        check("reads model", u["model"] == "claude-opus-5", str(u))
        # 10:00 -> 10:02 across the user message and two assistant messages.
        check("wall_ms", u["wall_ms"] == 120_000, str(u))

        # No usable timestamps -> None, never 0 (0 would read as an instant agent).
        flat = os.path.join(sub, "agent-flat.jsonl")
        with open(flat, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "user", "message": {"content": m}}) + "\n")
        check("wall_ms null when untimed", cost.agent_usage(flat)["wall_ms"] is None)

        # A torn final line must not lose the whole transcript.
        torn = os.path.join(sub, "agent-torn.jsonl")
        with open(torn, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "type": "assistant",
                "message": {"usage": {"input_tokens": 4, "output_tokens": 0,
                                      "cache_creation_input_tokens": 0,
                                      "cache_read_input_tokens": 0}},
            }) + "\n")
            fh.write('{"type": "assist')
        check("torn line tolerated", cost.agent_usage(torn)["input_tokens"] == 4)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_transcript_scanning()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'cost' has no attribute 'marker'`

- [ ] **Step 3: Write minimal implementation**

Append to `cost.py`:

```python
import datetime as _dt
import glob


def marker(pass_no: int, dimension: str) -> str:
    """The join key embedded in every finder prompt and matched in transcripts."""
    return f"LR-PASS-{pass_no}-{dimension}"


def _iter_records(path: str):
    """Yield parsed JSONL records, skipping a torn final line."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _first_user_text(path: str) -> str:
    for rec in _iter_records(path):
        if rec.get("type") != "user":
            continue
        content = (rec.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return ""
    return ""


def find_transcripts(project_dir: str, wanted: str) -> list[str]:
    """Every subagent transcript whose first user message carries `wanted`.

    Scans all session directories, not only the current one: a resumed or
    compacted session writes to a different directory mid-loop.
    """
    pattern = os.path.join(project_dir, "*", "subagents", "agent-*.jsonl")
    return sorted(p for p in glob.glob(pattern) if wanted in _first_user_text(p))


def _parse_ts(value: str):
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def agent_usage(path: str) -> dict:
    """Sum one agent's usage blocks and measure its wall clock."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    model = None
    stamps = []
    for rec in _iter_records(path):
        stamp = _parse_ts(rec.get("timestamp", ""))
        if stamp is not None:
            stamps.append(stamp)
        message = rec.get("message") or {}
        if message.get("model"):
            model = message["model"]
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if value is not None:
                totals[key] += int(value)
    # None, not 0: an untimed transcript is unknown duration, not an instant one.
    totals["wall_ms"] = (
        int((max(stamps) - min(stamps)).total_seconds() * 1000) if len(stamps) >= 2 else None
    )
    totals["model"] = model
    return totals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  wall_ms null when untimed` and `ok  torn line tolerated`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 2 - transcript discovery and usage summing

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 3: `cost.py collect` writes per-agent records

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `cost.find_transcripts`, `cost.agent_usage`, `cost.weighted_cost`, `cost.load_prices`, `ledger._append`, `ledger._read`, `ledger.ledger_dir`.
- Produces: `cost.collect(directory: str, project_dir: str, pass_no: int, dimensions: list[str], prices: dict) -> list[dict]`, appending one `{"pass": N, "event": "agent", ...}` record per agent to `passes.jsonl`; `cost.agent_records(directory: str) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
def test_cost_collect():
    directory = tempfile.mkdtemp()
    proj = tempfile.mkdtemp()
    try:
        sub = os.path.join(proj, "sess-a", "subagents")
        os.makedirs(sub)
        _write_transcript(sub, "agent-1.jsonl", cost.marker(1, "correctness"), [
            {"input_tokens": 1_000_000, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ])
        # Two transcripts for one dimension: sum them, do not pick one.
        _write_transcript(sub, "agent-2.jsonl", cost.marker(1, "quality"), [
            {"input_tokens": 500_000, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ])
        _write_transcript(sub, "agent-3.jsonl", cost.marker(1, "quality"), [
            {"input_tokens": 500_000, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ])
        prices = cost.load_prices()

        recs = cost.collect(directory, proj, 1,
                            ["correctness", "quality", "intent"], prices)
        by_dim = {r["dimension"]: r for r in recs}

        check("correctness cost", abs(by_dim["correctness"]["cost_usd"] - 5.0) < 1e-9)
        check("quality summed", abs(by_dim["quality"]["cost_usd"] - 5.0) < 1e-9,
              str(by_dim["quality"]))
        check("quality agent count", by_dim["quality"]["agents"] == 2)
        # A dimension that never ran must be unknown, never zero: zero would make
        # the cost-per-finding table read as if it were free.
        check("absent dimension unknown", by_dim["intent"]["cost_usd"] is None)
        check("absent dimension flagged", by_dim["intent"]["status"] == "unknown")

        # Idempotent: re-collecting the same pass must not double-count.
        cost.collect(directory, proj, 1, ["correctness", "quality", "intent"], prices)
        stored = [r for r in cost.agent_records(directory) if r["pass"] == 1]
        check("collect idempotent", len(stored) == 3, f"got {len(stored)}")
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(proj, ignore_errors=True)
```

Register `test_cost_collect()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'cost' has no attribute 'collect'`

- [ ] **Step 3: Write minimal implementation**

Append to `cost.py`:

```python
import sys

sys.path.insert(0, HERE)

import ledger  # noqa: E402  (path is set immediately above)


def agent_records(directory: str) -> list[dict]:
    path = os.path.join(directory, "passes.jsonl")
    return [r for r in ledger._read(path) if r.get("event") == "agent"]


def collect(directory: str, project_dir: str, pass_no: int,
            dimensions: list[str], prices: dict) -> list[dict]:
    """Attribute this pass's subagent spend, one record per dimension.

    Idempotent: a dimension already recorded for this pass is left alone, so a
    re-run after a partial flush does not double-count.
    """
    already = {
        r["dimension"] for r in agent_records(directory) if r.get("pass") == pass_no
    }
    out = []
    for dimension in dimensions:
        if dimension in already:
            continue
        paths = find_transcripts(project_dir, marker(pass_no, dimension))
        if not paths:
            # Unknown, not zero. A missing transcript means we do not know what
            # this agent cost; recording 0 would make the tier look free.
            record = {
                "pass": pass_no, "event": "agent", "dimension": dimension,
                "agents": 0, "status": "unknown", "cost_usd": None, "wall_ms": None,
                "input_tokens": None, "output_tokens": None,
                "cache_creation_input_tokens": None, "cache_read_input_tokens": None,
                "model": None,
            }
            ledger._append(os.path.join(directory, "passes.jsonl"), record)
            out.append(record)
            continue

        totals = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        walls, model = [], None
        for path in paths:
            usage = agent_usage(path)
            for key in totals:
                totals[key] += usage[key]
            if usage["wall_ms"] is not None:
                walls.append(usage["wall_ms"])
            model = model or usage["model"]

        record = {
            "pass": pass_no, "event": "agent", "dimension": dimension,
            "agents": len(paths), "status": "ok",
            "model": model,
            "cost_usd": weighted_cost(totals, model or "claude-opus-5", prices),
            "wall_ms": max(walls) if walls else None,
            **totals,
        }
        ledger._append(os.path.join(directory, "passes.jsonl"), record)
        out.append(record)
    return out
```

Then add the CLI at the bottom of `cost.py`:

```python
def _project_dir(repo_root: str) -> str:
    """Claude Code slugifies the project path by replacing separators with '-'."""
    projects = os.path.join(os.path.expanduser("~"), ".claude", "projects")
    slug = repo_root.replace("\\", "/").replace(":", "-").replace("/", "-")
    candidate = os.path.join(projects, slug)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(projects):
        want = slug.lower()
        for name in os.listdir(projects):
            if name.lower() == want:
                return os.path.join(projects, name)
    raise SystemExit(f"cannot locate Claude project telemetry for {repo_root}")


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="cost.py")
    p.add_argument("--branch")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="attribute this pass's subagent spend")
    c.add_argument("--pass", dest="pass_no", type=int, required=True)
    c.add_argument("--dimensions", nargs="+",
                   default=["correctness", "quality", "intent"])

    args = p.parse_args(argv)
    directory = ledger.ledger_dir(args.branch)
    prices = load_prices()
    stale = price_staleness(prices, _dt.date.today().isoformat())
    if stale is not None and stale > STALE_DAYS:
        print(f"WARNING: price table is {stale} days old (verified {prices['verified']});"
              f" re-verify before trusting cost comparisons")
    if args.cmd == "collect":
        repo_root = ledger._run_git(["rev-parse", "--show-toplevel"]).strip()
        for rec in collect(directory, _project_dir(repo_root), args.pass_no,
                           args.dimensions, prices):
            print(f"  {rec['dimension']:12s} {rec['status']:8s} "
                  f"agents={rec['agents']} cost={rec['cost_usd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  absent dimension unknown` and `ok  collect idempotent`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 3 - cost.py collect

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 4: Pass metadata in the ledger

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py` (`cmd_pass` at :524, `build_parser` at :609)
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger._append`, `ledger._read`.
- Produces: `ledger.pass_meta(directory: str) -> dict[int, dict]` keyed by pass number, each with `scope` (list of repo-relative files), `dimensions` (list that ran), `fix_files` (list modified by that pass's fixes), `gate_file` (str), `sha` (str). The `pass` subcommand gains `--scope`, `--dimensions`, `--fix-files`, `--gate-file`, `--sha`.

Every later task in this plan reads `pass_meta`. It is the join between passes and the escaped/gating/trigger computations.

- [ ] **Step 1: Write the failing test**

```python
def test_pass_meta():
    directory = tempfile.mkdtemp()
    try:
        path = os.path.join(directory, "passes.jsonl")
        ledger._append(path, {
            "pass": 1, "event": "start",
            "scope": ["src/a.py", "src/b.py"],
            "dimensions": ["correctness", "quality", "intent"],
            "gate_file": "gates-pass-0.txt", "sha": "abc123",
        })
        ledger._append(path, {"pass": 1, "event": "end", "fix_files": ["src/a.py"]})
        ledger._append(path, {
            "pass": 2, "event": "start",
            "scope": ["src/a.py"], "dimensions": ["correctness"],
            "gate_file": "gates-pass-1.txt", "sha": "def456",
        })

        meta = ledger.pass_meta(directory)
        check("scope recorded", meta[1]["scope"] == ["src/a.py", "src/b.py"])
        check("dimensions recorded", meta[1]["dimensions"] ==
              ["correctness", "quality", "intent"])
        # start and end events for one pass must merge, not overwrite.
        check("fix_files merged from end", meta[1]["fix_files"] == ["src/a.py"])
        check("gate file recorded", meta[1]["gate_file"] == "gates-pass-0.txt")
        check("pass 2 gating", meta[2]["dimensions"] == ["correctness"])
        # A pass with no metadata must yield empty containers, not KeyError.
        ledger._append(path, {"pass": 3, "event": "start"})
        check("absent meta defaults", ledger.pass_meta(directory)[3]["scope"] == [])
        check("absent skipped defaults", ledger.pass_meta(directory)[3]["skipped"] == {})
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_pass_meta()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'ledger' has no attribute 'pass_meta'`

- [ ] **Step 3: Write minimal implementation**

Add to `ledger.py` after `pass_counts` (:336):

```python
def pass_meta(directory: str) -> dict:
    """Per-pass scope, dimensions, fix files, gate file and sha.

    Start and end events for one pass are merged, so a field set on either lands
    in the same record. Missing list fields default to [] rather than raising --
    a pass recorded before this field existed must still be readable.
    """
    per: dict[int, dict] = {}
    for rec in _read(os.path.join(directory, "passes.jsonl")):
        number = rec.get("pass")
        if number is None or rec.get("event") not in ("start", "end"):
            continue
        slot = per.setdefault(
            number,
            {"scope": [], "dimensions": [], "fix_files": [],
             "skipped": {}, "gate_file": None, "sha": None},
        )
        for key in ("scope", "dimensions", "fix_files"):
            value = rec.get(key)
            if value:
                slot[key] = [norm_path(v) if key != "dimensions" else v for v in value]
        # Skip reasons merge rather than replace: a dimension recorded as skipped
        # on the start event must survive whatever the end event carries.
        if rec.get("skipped"):
            slot["skipped"].update(rec["skipped"])
        for key in ("gate_file", "sha"):
            if rec.get(key) is not None:
                slot[key] = rec[key]
    return per
```

Then extend the `pass` subparser in `build_parser` (after :667):

```python
    pa.add_argument("--scope", nargs="+", help="files this pass reviewed")
    pa.add_argument("--dimensions", nargs="+", help="dimensions that ran this pass")
    pa.add_argument("--fix-files", dest="fix_files", nargs="+",
                    help="files this pass's fixes modified")
    pa.add_argument("--gate-file", dest="gate_file", help="gate output this pass read")
    pa.add_argument("--sha", help="HEAD sha this pass reviewed")
```

And extend `cmd_pass` (:524) to carry them through:

```python
def cmd_pass(args) -> int:
    directory = ledger_dir(args.branch)
    _append(
        os.path.join(directory, "passes.jsonl"),
        {
            "pass": args.pass_no,
            "event": args.event,
            "fingerprint": args.fingerprint,
            "effort": args.effort,
            "note": args.note,
            "scope": args.scope,
            "dimensions": args.dimensions,
            "fix_files": args.fix_files,
            "gate_file": args.gate_file,
            "sha": args.sha,
        },
    )
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  fix_files merged from end` and `ok  absent meta defaults`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 4 - pass metadata in the ledger

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 5: `found_by` and fractional credit apportionment

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py` (`fold` at :209, `cmd_add` at :401, `build_parser` at :616)
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger.fold`.
- Produces: each folded record gains `found_by: list[str]` (deduped, insertion-ordered); `ledger.dimension_credit(records: dict) -> dict[str, dict]` mapping dimension to `{"fractional": float, "raw": int, "refuted_fractional": float, "refuted_raw": int}`. `add` gains `--found-by`.

Implements spec R8.5 and R8.5.1. A finding observed by K dimensions counts 1/K to each fractionally and 1 to each raw. Whole credit to each would make per-dimension counts sum above the number of real findings and understate every dimension's cost-per-finding by exactly the overlap rate.

- [ ] **Step 1: Write the failing test**

```python
def test_dimension_credit():
    directory = tempfile.mkdtemp()
    try:
        # One finding seen by two dimensions.
        observe(directory, "f1", file="src/a.py", summary="shared defect",
                found_by="correctness", **{"pass": 1})
        observe(directory, "f1", file="src/a.py", summary="shared defect",
                found_by="quality", **{"pass": 1})
        # One finding seen by one dimension.
        observe(directory, "f2", file="src/b.py", summary="solo defect",
                found_by="correctness", **{"pass": 1})
        # One refuted, seen by one dimension.
        observe(directory, "f3", file="src/c.py", summary="bad premise",
                found_by="quality", **{"pass": 1})
        op(directory, "resolve", "f1", status="fixed")
        op(directory, "resolve", "f2", status="fixed")
        op(directory, "resolve", "f3", status="refuted")

        records = ledger.fold(directory)
        check("found_by accumulates",
              records["f1"]["found_by"] == ["correctness", "quality"],
              str(records["f1"].get("found_by")))
        check("found_by dedupes", records["f2"]["found_by"] == ["correctness"])

        credit = ledger.dimension_credit(records)
        # correctness: 0.5 (shared f1) + 1.0 (f2) = 1.5 confirmed
        check("correctness fractional",
              abs(credit["correctness"]["fractional"] - 1.5) < 1e-9,
              str(credit["correctness"]))
        check("correctness raw", credit["correctness"]["raw"] == 2)
        # quality: 0.5 (shared f1) confirmed; f3 is refuted, counted separately.
        check("quality fractional",
              abs(credit["quality"]["fractional"] - 0.5) < 1e-9,
              str(credit["quality"]))
        check("quality refuted", abs(credit["quality"]["refuted_fractional"] - 1.0) < 1e-9)
        # The invariant that makes the table honest.
        total = sum(v["fractional"] for v in credit.values())
        check("fractional sums to distinct confirmed", abs(total - 2.0) < 1e-9, str(total))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_dimension_credit()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'ledger' has no attribute 'dimension_credit'`

- [ ] **Step 3: Write minimal implementation**

In `fold` (:209), inside the `if kind == "observe":` branch where the record is first built, add `"found_by": []` to the initial dict. Then, still in that branch and after the record exists (so it runs for both first and repeat observations), add:

```python
            source = op.get("found_by")
            if source and source not in rec["found_by"]:
                rec["found_by"].append(source)
```

Add after `open_items` (:313):

```python
def dimension_credit(records: dict) -> dict:
    """Per-dimension confirmed/refuted counts, fractional and raw.

    A finding observed by K dimensions counts 1/K to each fractionally: whole
    credit to each would make the per-dimension counts sum above the number of
    distinct findings, and cost-per-finding would understate every dimension's
    cost by exactly the overlap rate. The raw count is reported alongside because
    "how many findings did this dimension touch" is a different question from
    "how much of the yield is attributable to it".
    """
    out: dict[str, dict] = {}
    for rec in records.values():
        finders = rec.get("found_by") or []
        if not finders:
            continue
        share = 1.0 / len(finders)
        refuted = rec["status"] == "refuted"
        for dimension in finders:
            slot = out.setdefault(
                dimension,
                {"fractional": 0.0, "raw": 0, "refuted_fractional": 0.0, "refuted_raw": 0},
            )
            if refuted:
                slot["refuted_fractional"] += share
                slot["refuted_raw"] += 1
            elif rec["status"] == "fixed":
                slot["fractional"] += share
                slot["raw"] += 1
    return out
```

Add `a.add_argument("--found-by", dest="found_by", help="dimension that observed this")` to the `add` subparser (after :624), and add `"found_by": args.found_by,` to the dict `cmd_add` appends (:401).

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  fractional sums to distinct confirmed`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 5 - found_by and fractional credit

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 6: Escaped findings

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger.fold`, `ledger.pass_meta`, `ledger.norm_path`.
- Produces: `ledger.escaped_findings(directory: str, records: dict) -> dict[str, int]` mapping dimension to the number of escapes charged to it.

Implements R9.1, R9.1.1, R9.1.2. Three conditions must all hold for an escape: the finding was first observed at pass N; its file was in pass N-1's recorded scope; and that file was **not** modified by pass N-1's fixes. The escape is charged whole to every dimension that ran in pass N-1, and to none that were gated off.

- [ ] **Step 1: Write the failing test**

```python
def test_escaped_findings():
    directory = tempfile.mkdtemp()
    try:
        path = os.path.join(directory, "passes.jsonl")
        ledger._append(path, {
            "pass": 1, "event": "start",
            "scope": ["src/a.py", "src/b.py", "src/c.py"],
            "dimensions": ["correctness", "quality"],
        })
        ledger._append(path, {"pass": 1, "event": "end", "fix_files": ["src/b.py"]})
        ledger._append(path, {
            "pass": 2, "event": "start",
            "scope": ["src/a.py", "src/b.py", "src/c.py"],
            "dimensions": ["correctness", "quality", "intent"],
        })

        # In pass 1's scope, untouched by its fixes -> a genuine escape.
        observe(directory, "e1", file="src/a.py", summary="missed bug",
                found_by="correctness", **{"pass": 2})
        # In pass 1's scope but its fix modified that file -> fix-induced, not escaped.
        observe(directory, "e2", file="src/b.py", summary="fix-induced bug",
                found_by="correctness", **{"pass": 2})
        # Not in pass 1's scope at all -> not escaped.
        observe(directory, "e3", file="src/z.py", summary="new file bug",
                found_by="correctness", **{"pass": 2})

        records = ledger.fold(directory)
        escaped = ledger.escaped_findings(directory, records)

        # Both dimensions that ran in pass 1 are charged in full for e1.
        check("correctness charged", escaped.get("correctness") == 1, str(escaped))
        check("quality charged in full", escaped.get("quality") == 1, str(escaped))
        # intent did not run in pass 1, so it had no chance to miss anything.
        check("gated-off dimension not charged", "intent" not in escaped, str(escaped))
        check("fix-induced excluded", escaped.get("correctness") != 2, str(escaped))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_escaped_findings()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'ledger' has no attribute 'escaped_findings'`

- [ ] **Step 3: Write minimal implementation**

Add to `ledger.py` after `dimension_credit`:

```python
def first_seen_pass(directory: str) -> dict:
    """Finding id -> the earliest pass that observed it."""
    out: dict[str, int] = {}
    for op in _read(os.path.join(directory, "ledger.jsonl")):
        if op.get("op") != "observe" or op.get("pass") is None:
            continue
        fid = op["id"]
        if fid not in out or op["pass"] < out[fid]:
            out[fid] = op["pass"]
    return out


def escaped_findings(directory: str, records: dict) -> dict:
    """Escapes charged per dimension.

    An escape is a finding first seen at pass N in a file that pass N-1 had in
    scope and did not modify. It is charged whole to every dimension that ran in
    pass N-1 and to none that were gated off: a dimension that never looked at
    the code had no opportunity to miss it, and charging it would make the
    under-spend signal punish exactly the economy the gating exists to achieve.
    """
    meta = pass_meta(directory)
    aliases = _alias_map(_read(os.path.join(directory, "ledger.jsonl")))
    seen = first_seen_pass(directory)
    out: dict[str, int] = {}
    for raw_id, number in seen.items():
        fid = aliases.get(raw_id, raw_id)
        rec = records.get(fid)
        if rec is None or number < 2:
            continue
        prior = meta.get(number - 1)
        if not prior:
            continue
        target = norm_path(rec["file"])
        if target not in prior["scope"]:
            continue
        if target in prior["fix_files"]:
            continue  # fix-induced, not escaped
        for dimension in prior["dimensions"]:
            out[dimension] = out.get(dimension, 0) + 1
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  gated-off dimension not charged`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 6 - escaped findings

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 7: Subsystem escalation trigger

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger.fold`, `ledger.severity_rank`, `ledger.OPEN_STATUSES`.
- Produces: `ledger.subsystem_candidates(records: dict, threshold: str = "medium") -> list[dict]`, each `{"subsystem": str, "kind": "file"|"theme", "ids": [str, ...]}`.

Implements R7.2/R7.2.1/R7.2.2. Fires on two or more open, triaged, at-or-above-threshold findings sharing a file or a theme — evaluated **before** any of them is fixed, because firing after two fixes means band-aiding two facets first, which is the behavior the escalation exists to prevent. The caller (the arbitrator, per `reference/arbitration.md`) premise-checks each candidate and proceeds only if at least two survive; this function reports candidates, it does not decide.

- [ ] **Step 1: Write the failing test**

```python
def test_subsystem_trigger():
    directory = tempfile.mkdtemp()
    try:
        # Two open, triaged, medium+ findings in one file -> candidate.
        observe(directory, "s1", file="src/pool.py", summary="empty pool crash",
                found_by="correctness", **{"pass": 1})
        observe(directory, "s2", file="src/pool.py", summary="all-NaN pool silent drop",
                found_by="correctness", **{"pass": 1})
        op(directory, "triage", "s1", severity="high")
        op(directory, "triage", "s2", severity="high")

        # One low-severity finding elsewhere -> below threshold, not a candidate.
        observe(directory, "s3", file="src/other.py", summary="unused import",
                found_by="quality", **{"pass": 1})
        op(directory, "triage", "s3", severity="low")

        # Two findings sharing a theme across different files -> candidate.
        observe(directory, "t1", file="src/x.py", summary="guard missing on x",
                found_by="correctness", **{"pass": 1})
        observe(directory, "t2", file="src/y.py", summary="guard missing on y",
                found_by="correctness", **{"pass": 1})
        op(directory, "triage", "t1", severity="high", theme="empty-pool")
        op(directory, "triage", "t2", severity="high", theme="empty-pool")

        records = ledger.fold(directory)
        cands = ledger.subsystem_candidates(records, "medium")
        keys = {c["subsystem"] for c in cands}
        check("file candidate", "src/pool.py" in keys, str(keys))
        check("theme candidate", "empty-pool" in keys, str(keys))
        check("below threshold excluded", "src/other.py" not in keys, str(keys))

        # Fires before either is fixed: both are still open here.
        pool = [c for c in cands if c["subsystem"] == "src/pool.py"][0]
        check("both ids reported", sorted(pool["ids"]) == ["s1", "s2"], str(pool))

        # An untriaged pair is not yet a candidate -- severity is undecided.
        observe(directory, "u1", file="src/u.py", summary="one", found_by="correctness",
                **{"pass": 1})
        observe(directory, "u2", file="src/u.py", summary="two", found_by="correctness",
                **{"pass": 1})
        records = ledger.fold(directory)
        keys = {c["subsystem"] for c in ledger.subsystem_candidates(records, "medium")}
        check("untriaged excluded", "src/u.py" not in keys, str(keys))

        # Refuting one drops the pair below two survivors.
        op(directory, "resolve", "s1", status="refuted")
        records = ledger.fold(directory)
        keys = {c["subsystem"] for c in ledger.subsystem_candidates(records, "medium")}
        check("refuted drops candidate", "src/pool.py" not in keys, str(keys))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_subsystem_trigger()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'ledger' has no attribute 'subsystem_candidates'`

- [ ] **Step 3: Write minimal implementation**

Add to `ledger.py` after `escaped_findings`:

```python
def subsystem_candidates(records: dict, threshold: str = "medium") -> list[dict]:
    """Subsystems holding 2+ open, triaged, at-or-above-threshold findings.

    Evaluated before any of the group is fixed. Firing after two fixes would
    mean band-aiding two facets first, which is exactly the failure this
    escalation exists to prevent (see the PR #280 case study: seven passes went
    to what should have been one design).

    Untriaged findings are excluded: an unrated finding has no severity yet, so
    including it would fire the most expensive action in the loop on an
    undecided premise. The caller premise-checks each candidate and proceeds
    only if at least two survive.
    """
    floor = SEVERITY_RANK.get(threshold, 2)
    by_file: dict[str, list[str]] = {}
    by_theme: dict[str, list[str]] = {}
    for rec in records.values():
        if rec["status"] not in OPEN_STATUSES:
            continue
        if not rec.get("severity"):
            continue  # untriaged: severity undecided, not yet a candidate
        if SEVERITY_RANK.get(rec["severity"], 0) < floor:
            continue
        by_file.setdefault(rec["file"], []).append(rec["id"])
        if rec.get("theme"):
            by_theme.setdefault(rec["theme"], []).append(rec["id"])

    out = []
    for key, ids in sorted(by_file.items()):
        if len(ids) >= 2:
            out.append({"subsystem": key, "kind": "file", "ids": sorted(ids)})
    for key, ids in sorted(by_theme.items()):
        if len(ids) >= 2:
            out.append({"subsystem": key, "kind": "theme", "ids": sorted(ids)})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  refuted drops candidate` and `ok  untriaged excluded`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 7 - subsystem escalation trigger

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 8: Dimension gating decision

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger.fold`, `ledger.dimension_credit`.
- Produces: `ledger.dimensions_for_pass(records: dict, pass_no: int, scope_has_new_file: bool, scope_added_lines: int) -> dict[str, str]` mapping each of `correctness`, `quality`, `intent` to a reason string; a dimension absent from the returned dict is gated off.

Implements R5.1 and R5.2. `correctness` always runs. `quality` and `intent` run on pass 1, or later if they have a confirmed finding on this branch, or if the scope contains a file new to the branch or adds 50+ lines.

- [ ] **Step 1: Write the failing test**

```python
def test_dimension_gating():
    directory = tempfile.mkdtemp()
    try:
        # quality produced a confirmed finding; intent produced only a refuted one.
        observe(directory, "g1", file="src/a.py", summary="dup logic",
                found_by="quality", **{"pass": 1})
        op(directory, "resolve", "g1", status="fixed")
        observe(directory, "g2", file="src/b.py", summary="goal drift",
                found_by="intent", **{"pass": 1})
        op(directory, "resolve", "g2", status="refuted")
        records = ledger.fold(directory)

        first = ledger.dimensions_for_pass(records, 1, False, 0)
        check("pass 1 runs all three", set(first) ==
              {"correctness", "quality", "intent"}, str(first))

        later = ledger.dimensions_for_pass(records, 3, False, 10)
        check("correctness always runs", "correctness" in later, str(later))
        check("quality runs on prior yield", "quality" in later, str(later))
        check("intent gated on refuted-only", "intent" not in later, str(later))

        # R5.2(b): a new file re-enables everything regardless of prior yield.
        newfile = ledger.dimensions_for_pass(records, 3, True, 0)
        check("new file re-enables intent", "intent" in newfile, str(newfile))
        check("reason names the trigger",
              "new file" in newfile["intent"], newfile["intent"])

        # R5.2(b): so does a 50+ line addition. 49 must not.
        big = ledger.dimensions_for_pass(records, 3, False, 50)
        check("50 lines re-enables intent", "intent" in big, str(big))
        small = ledger.dimensions_for_pass(records, 3, False, 49)
        check("49 lines does not", "intent" not in small, str(small))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_dimension_gating()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'ledger' has no attribute 'dimensions_for_pass'`

- [ ] **Step 3: Write minimal implementation**

Add to `ledger.py` after `subsystem_candidates`:

```python
GATED_DIMENSIONS = ("quality", "intent")
NEW_CODE_LINE_THRESHOLD = 50


def dimensions_for_pass(records: dict, pass_no: int, scope_has_new_file: bool,
                        scope_added_lines: int) -> dict:
    """Which dimensions run this pass, and why. Absent means gated off.

    correctness always runs. quality and intent run on pass 1, or later when
    they have earned it (a confirmed finding on this branch) or when the scope
    holds code they have never seen. Gating purely on prior yield would let a
    dimension miss the only code it was ever going to have an opinion about.
    """
    out = {"correctness": "always runs"}
    if pass_no <= 1:
        for dimension in GATED_DIMENSIONS:
            out[dimension] = "pass 1: full breadth"
        return out

    credit = dimension_credit(records)
    new_code = scope_has_new_file or scope_added_lines >= NEW_CODE_LINE_THRESHOLD
    for dimension in GATED_DIMENSIONS:
        if credit.get(dimension, {}).get("raw", 0) > 0:
            out[dimension] = "prior confirmed finding on this branch"
        elif scope_has_new_file:
            out[dimension] = "scope contains a new file"
        elif new_code:
            out[dimension] = f"scope adds {scope_added_lines} lines"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  intent gated on refuted-only` and `ok  49 lines does not`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 8 - dimension gating

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 9: Gate command derivation

**Files:**
- Create: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/gates.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `gates.derive(repo_root: str, config_path: str) -> dict` returning `{"commands": [str, ...], "rule": str}`; `gates.pin(config_path: str, resolved: dict) -> None`.

Implements R2.2/R2.2.1/R2.2.2. Resolution order: pinned `config.json` -> a fenced block under a verification-ish heading in `CLAUDE.md` -> ecosystem default by manifest -> none.

- [ ] **Step 1: Write the failing test**

```python
def test_gate_derivation():
    root = tempfile.mkdtemp()
    cfgdir = tempfile.mkdtemp()
    try:
        cfg = os.path.join(cfgdir, "config.json")

        # (4) nothing at all -> no gates, recorded as such, never an empty success.
        got = gates.derive(root, cfg)
        check("no gates rule", got["rule"] == "none", str(got))
        check("no gates commands", got["commands"] == [], str(got))

        # (3) ecosystem default by manifest.
        with open(os.path.join(root, "pyproject.toml"), "w", encoding="utf-8") as fh:
            fh.write("[project]\nname='x'\n")
        got = gates.derive(root, cfg)
        check("ecosystem rule", got["rule"] == "ecosystem:python", str(got))
        check("ecosystem commands", got["commands"] == ["pytest -q"], str(got))

        # (2) CLAUDE.md verification block wins over the ecosystem default.
        with open(os.path.join(root, "CLAUDE.md"), "w", encoding="utf-8") as fh:
            fh.write("# Rules\n\n## End-of-effort verification\n\n"
                     "```bash\npytest -v\nruff check .\n```\n")
        got = gates.derive(root, cfg)
        check("claude.md rule", got["rule"] == "claude-md", str(got))
        check("claude.md commands",
              got["commands"] == ["pytest -v", "ruff check ."], str(got))

        # (1) a pinned config wins over everything.
        gates.pin(cfg, {"commands": ["make check"], "rule": "config"})
        got = gates.derive(root, cfg)
        check("config wins", got["commands"] == ["make check"], str(got))
        check("config rule", got["rule"] == "config", str(got))
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(cfgdir, ignore_errors=True)
```

Register `test_gate_derivation()` in `main()` and add `import gates` at the top of `selftest.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gates'`

- [ ] **Step 3: Write minimal implementation**

`gates.py`:

```python
"""Derive the project's gate commands so the parent runs them once per pass.

Before this, every review agent independently re-ran the full suite -- on the
#274 branch that meant paying pytest's 153 seconds five-plus times per pass for
one identical answer. The parent computes it once and hands every finder the
output path.

ASCII-only output: this runs on Windows terminals (cp1252 stdout).
"""

from __future__ import annotations

import json
import os
import re

HEADING = re.compile(r"^#{1,6}\s*.*(verification|gates|checks|end-of-effort)",
                     re.IGNORECASE)
FENCE = re.compile(r"^```")

ECOSYSTEMS = [
    (("pyproject.toml", "setup.py"), "python", ["pytest -q"]),
    (("package.json",), "node", ["npm test", "npm run lint"]),
    (("go.mod",), "go", ["go test ./...", "go vet ./..."]),
    (("Cargo.toml",), "rust", ["cargo test", "cargo clippy"]),
]


def _from_config(config_path: str) -> dict | None:
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    commands = data.get("gates")
    if not commands:
        return None
    return {"commands": list(commands), "rule": "config"}


def _from_claude_md(repo_root: str) -> dict | None:
    path = os.path.join(repo_root, "CLAUDE.md")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    commands: list[str] = []
    in_section = False
    in_fence = False
    for line in lines:
        if HEADING.match(line):
            in_section = True
            continue
        if in_section and line.startswith("#") and not HEADING.match(line):
            break  # next unrelated heading ends the section
        if not in_section:
            continue
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            stripped = line.strip()
            # Drop comments and prose; keep command lines.
            if stripped and not stripped.startswith("#"):
                commands.append(stripped)
    if not commands:
        return None
    return {"commands": commands, "rule": "claude-md"}


def _from_ecosystem(repo_root: str) -> dict | None:
    for manifests, name, commands in ECOSYSTEMS:
        if any(os.path.isfile(os.path.join(repo_root, m)) for m in manifests):
            return {"commands": list(commands), "rule": f"ecosystem:{name}"}
    return None


def derive(repo_root: str, config_path: str) -> dict:
    """First hit wins: pinned config -> CLAUDE.md -> ecosystem -> none."""
    for resolver in (
        lambda: _from_config(config_path),
        lambda: _from_claude_md(repo_root),
        lambda: _from_ecosystem(repo_root),
    ):
        got = resolver()
        if got:
            return got
    # Explicitly "none", never an empty list that a reader could mistake for a
    # clean suite.
    return {"commands": [], "rule": "none"}


def pin(config_path: str, resolved: dict) -> None:
    """Record the resolved list so a later pass cannot silently re-derive."""
    data = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            data = {}
    data["gates"] = resolved["commands"]
    data["gates_rule"] = resolved["rule"]
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  no gates rule` and `ok  config wins`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 9 - gate command derivation

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 10: Goal resolution

**Files:**
- Create: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/goal.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `gh` CLI (at runtime only; the test covers the pure parser).
- Produces: `goal.issue_number_from_branch(branch: str) -> int | None`; `goal.issue_number_from_body(body: str) -> int | None`; `goal.resolve(branch: str, runner=None) -> dict` returning `{"goal": str | None, "source": str, "issue": int | None}`.

Implements R1.1-R1.3. `runner` is injected so the test never shells out to `gh`.

- [ ] **Step 1: Write the failing test**

```python
def test_goal_resolution():
    check("branch number", goal.issue_number_from_branch("feat/277-decompose-luck") == 277)
    check("branch underscore", goal.issue_number_from_branch("fix_301_bad_flag") == 301)
    check("no number", goal.issue_number_from_branch("feat/loop-review-rebuild") is None)
    # A bare word containing digits is not a delimited number run.
    check("embedded digits ignored",
          goal.issue_number_from_branch("feat/utf8-encoding") is None)
    check("six digits rejected",
          goal.issue_number_from_branch("feat/1234567-thing") is None)
    check("body reference", goal.issue_number_from_body("closes #284 and #290") == 284)
    check("body none", goal.issue_number_from_body("no refs here") is None)

    # Issue found and plausible -> used.
    def ok_runner(args):
        if args[:2] == ["issue", "view"]:
            return json.dumps({"title": "decompose luck", "body": "split the model"})
        raise AssertionError("unexpected call " + str(args))

    got = goal.resolve("feat/277-decompose-luck", runner=ok_runner)
    check("issue source", got["source"] == "issue", str(got))
    check("issue number kept", got["issue"] == 277, str(got))
    check("goal text", "decompose luck" in got["goal"], str(got))

    # gh unavailable -> fall through to the PR, not a hard failure.
    calls = []

    def pr_runner(args):
        calls.append(args[0])
        if args[0] == "issue":
            raise RuntimeError("gh: not authenticated")
        return json.dumps({"title": "PR title", "body": "PR body"})

    got = goal.resolve("feat/277-decompose-luck", runner=pr_runner)
    check("falls through to pr", got["source"] == "pr", str(got))
    check("tried issue first", calls[0] == "issue", str(calls))

    # Nothing available -> "ask", so the caller prompts rather than guessing.
    def dead_runner(args):
        raise RuntimeError("gh missing")

    got = goal.resolve("feat/no-number", runner=dead_runner)
    check("asks when nothing found", got["source"] == "ask", str(got))
    check("no fabricated goal", got["goal"] is None, str(got))

    # A number that parses but belongs to an unrelated issue must NOT be adopted.
    # feat/2026-cleanup parses to 2026; if issue 2026 is about something else,
    # accepting it hands the intent reviewer the wrong contract entirely.
    check("plausible overlap", goal.plausible("keeper board rework",
                                              "feat/277-keeper-board") is True)
    check("implausible rejected", goal.plausible("Upgrade CI runners to 22.04",
                                                 "feat/2026-cleanup") is False)
    check("short branch words ignored", goal.plausible("x", "feat/2026-a") is False)

    def wrong_issue_runner(args):
        if args[:2] == ["issue", "view"]:
            return json.dumps({"title": "Upgrade CI runners", "body": "bump ubuntu"})
        return json.dumps({"title": "cleanup pass", "body": "tidy the cleanup module"})

    got = goal.resolve("feat/2026-cleanup", runner=wrong_issue_runner)
    check("implausible issue falls through", got["source"] == "pr", str(got))
    check("implausible issue not recorded", got["issue"] is None, str(got))
```

Register `test_goal_resolution()` in `main()` and add `import goal` to `selftest.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'goal'`

- [ ] **Step 3: Write minimal implementation**

`goal.py`:

```python
"""Resolve the branch's stated goal, for the intent reviewer.

The issue is the source of truth; the PR description is the fallback. Nothing
here guesses intent from the diff -- a reviewer that infers the goal from the
code it is reviewing is just a second correctness reviewer.

ASCII-only output: this runs on Windows terminals (cp1252 stdout).
"""

from __future__ import annotations

import json
import re
import subprocess

# A delimited run of 1-5 digits: feat/277-x -> 277, feat/utf8-x -> no match
# (the digits are not delimited), feat/1234567-x -> no match (too long to be an
# issue number, and matching it would fetch an unrelated issue).
_BRANCH_NUM = re.compile(r"(?:^|[/_-])(\d{1,5})(?:$|[/_-])")
_BODY_NUM = re.compile(r"#(\d{1,5})\b")


def issue_number_from_branch(branch: str) -> int | None:
    match = _BRANCH_NUM.search(branch)
    return int(match.group(1)) if match else None


def issue_number_from_body(body: str) -> int | None:
    match = _BODY_NUM.search(body or "")
    return int(match.group(1)) if match else None


_STOPWORDS = {"feat", "fix", "chore", "docs", "refactor", "test", "the", "and", "a"}


def _words(text: str) -> set[str]:
    """Content words of 3+ characters, lowercased."""
    return {
        w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(w) >= 3 and w not in _STOPWORDS and not w.isdigit()
    }


def plausible(issue_title: str, branch: str) -> bool:
    """Does this issue look like it belongs to this branch?

    A branch name yields a number by pattern, and that number resolves to a REAL
    issue whether or not it is the right one -- feat/2026-cleanup parses to 2026,
    and issue 2026 exists. Adopting it would hand the intent reviewer a contract
    from an unrelated piece of work, and every finding it produced would be
    judged against the wrong goal. One shared content word is a deliberately low
    bar: the check exists to catch the year-number and version-number accidents,
    not to second-guess a genuine issue whose title is worded differently from
    its branch.
    """
    return bool(_words(issue_title) & _words(branch))


def _gh(args: list[str]) -> str:
    return subprocess.check_output(["gh", *args], text=True, stderr=subprocess.DEVNULL)


def _as_goal(payload: str) -> str | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    if not title and not body:
        return None
    return f"{title}\n\n{body}".strip()


def resolve(branch: str, runner=None) -> dict:
    """Issue, then PR, then ask. Never fails the loop."""
    run = runner or _gh

    number = issue_number_from_branch(branch)
    pr_payload = None
    if number is None:
        try:
            pr_payload = run(["pr", "view", "--json", "title,body"])
            number = issue_number_from_body(json.loads(pr_payload).get("body", ""))
        except Exception:
            pr_payload = None

    if number is not None:
        try:
            payload = run(["issue", "view", str(number), "--json", "title,body"])
            text = _as_goal(payload)
            title = json.loads(payload).get("title") or ""
            # The number parsed; that does not make it the right issue. A bare
            # id trusted without confirmation resolves to a real row belonging
            # to someone else, and the result looks plausible enough to stop
            # checking -- so confirm before adopting it.
            if text and plausible(title, branch):
                return {"goal": text, "source": "issue", "issue": number}
            number = None  # do not carry an implausible number into the PR result
        except Exception:
            pass  # unauthenticated, offline, or the number does not exist

    try:
        if pr_payload is None:
            pr_payload = run(["pr", "view", "--json", "title,body"])
        text = _as_goal(pr_payload)
        if text:
            return {"goal": text, "source": "pr", "issue": number}
    except Exception:
        pass

    # No fabricated goal: the caller asks the user.
    return {"goal": None, "source": "ask", "issue": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  no fabricated goal` and `ok  six digits rejected`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 10 - goal resolution

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 11: Budget accounting with unknown costs

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `cost.agent_records`.
- Produces: `cost.DEFAULT_BUDGET_USD = 15.0`; `cost.budget_status(directory: str, budget: float | None) -> dict` with keys `known`, `estimated_agents`, `total`, `over`, `evaluable`.

Implements R10.1/R10.1.1/R10.3. Unknown-cost agents are estimated at the running mean, never counted as zero — a loop with unreadable transcripts would otherwise run past the threshold with the safeguard silent.

- [ ] **Step 1: Write the failing test**

```python
def test_budget_status():
    directory = tempfile.mkdtemp()
    try:
        path = os.path.join(directory, "passes.jsonl")
        # No cost data at all -> not evaluable, and it must say so.
        got = cost.budget_status(directory, 15.0)
        check("not evaluable when empty", got["evaluable"] is False, str(got))
        check("does not fire when empty", got["over"] is False, str(got))

        ledger._append(path, {"pass": 1, "event": "agent", "dimension": "correctness",
                              "status": "ok", "cost_usd": 4.0})
        ledger._append(path, {"pass": 1, "event": "agent", "dimension": "quality",
                              "status": "ok", "cost_usd": 2.0})
        # Unknown agent: estimated at the mean of 4.0 and 2.0 = 3.0, not 0.
        ledger._append(path, {"pass": 1, "event": "agent", "dimension": "intent",
                              "status": "unknown", "cost_usd": None})

        got = cost.budget_status(directory, 15.0)
        check("known sum", abs(got["known"] - 6.0) < 1e-9, str(got))
        check("estimated total", abs(got["total"] - 9.0) < 1e-9, str(got))
        check("estimated count", got["estimated_agents"] == 1, str(got))
        check("under budget", got["over"] is False, str(got))

        got = cost.budget_status(directory, 8.0)
        check("estimate can trip budget", got["over"] is True, str(got))

        check("disabled budget", cost.budget_status(directory, None)["over"] is False)
        check("default is 15", cost.DEFAULT_BUDGET_USD == 15.0)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_budget_status()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'cost' has no attribute 'budget_status'`

- [ ] **Step 3: Write minimal implementation**

Add to `cost.py` before `main`:

```python
DEFAULT_BUDGET_USD = 15.0


def budget_status(directory: str, budget: float | None) -> dict:
    """Weighted spend so far, estimating agents whose cost is unknown.

    Unknown agents are charged the running mean rather than zero. Counting them
    as zero would let a loop with unreadable transcripts run straight past the
    threshold with the safeguard silent -- the failure mode where the check is
    quietest is exactly the one where it matters.
    """
    records = agent_records(directory)
    known = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    unknown = sum(1 for r in records if r.get("cost_usd") is None)

    if not known:
        return {"known": 0.0, "estimated_agents": unknown, "total": 0.0,
                "over": False, "evaluable": False}

    mean = sum(known) / len(known)
    total = sum(known) + mean * unknown
    over = budget is not None and total > budget
    return {"known": sum(known), "estimated_agents": unknown, "total": total,
            "over": over, "evaluable": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  estimate can trip budget`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 11 - budget accounting

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 12: `cost.py report`

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `cost.agent_records`, `ledger.fold`, `ledger.dimension_credit`, `ledger.escaped_findings`, `ledger.pass_meta`.
- Produces: `cost.report_rows(directory: str) -> list[dict]`, each with `dimension`, `cost_usd`, `confirmed_fractional`, `confirmed_raw`, `refuted_fractional`, `escaped`, `passes_ran`, `escaped_rate`, `refuted_rate`, `cost_per_confirmed`; plus a `report` subcommand rendering them.

Implements R8.6 and R9.2, including the denominators each rate is over.

- [ ] **Step 1: Write the failing test**

```python
def test_report_rows():
    directory = tempfile.mkdtemp()
    try:
        path = os.path.join(directory, "passes.jsonl")
        ledger._append(path, {"pass": 1, "event": "start",
                              "scope": ["src/a.py"],
                              "dimensions": ["correctness", "quality"]})
        ledger._append(path, {"pass": 1, "event": "end", "fix_files": []})
        ledger._append(path, {"pass": 2, "event": "start",
                              "scope": ["src/a.py"], "dimensions": ["correctness"]})
        ledger._append(path, {"pass": 1, "event": "agent", "dimension": "correctness",
                              "status": "ok", "cost_usd": 3.0})
        ledger._append(path, {"pass": 1, "event": "agent", "dimension": "quality",
                              "status": "ok", "cost_usd": 1.0})
        ledger._append(path, {"pass": 2, "event": "agent", "dimension": "correctness",
                              "status": "ok", "cost_usd": 2.0})

        observe(directory, "r1", file="src/a.py", summary="confirmed one",
                found_by="correctness", **{"pass": 1})
        op(directory, "resolve", "r1", status="fixed")
        # First seen at pass 2, in pass 1's scope, untouched by its fixes -> escaped.
        observe(directory, "r2", file="src/a.py", summary="late find",
                found_by="correctness", **{"pass": 2})
        op(directory, "resolve", "r2", status="fixed")

        rows = {r["dimension"]: r for r in cost.report_rows(directory)}
        corr = rows["correctness"]
        check("cost summed across passes", abs(corr["cost_usd"] - 5.0) < 1e-9, str(corr))
        check("confirmed counted", abs(corr["confirmed_fractional"] - 2.0) < 1e-9, str(corr))
        check("cost per confirmed", abs(corr["cost_per_confirmed"] - 2.5) < 1e-9, str(corr))
        check("escaped counted", corr["escaped"] == 1, str(corr))
        check("passes ran", corr["passes_ran"] == 2, str(corr))
        check("escaped rate over passes ran",
              abs(corr["escaped_rate"] - 0.5) < 1e-9, str(corr))

        # A dimension with no confirmed findings must report None, not divide by zero.
        qual = rows["quality"]
        check("no divide by zero", qual["cost_per_confirmed"] is None, str(qual))
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register `test_report_rows()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'cost' has no attribute 'report_rows'`

- [ ] **Step 3: Write minimal implementation**

Add to `cost.py` before `main`:

```python
def report_rows(directory: str) -> list[dict]:
    """Per-dimension cost, yield, and the two under-spend signals."""
    records = ledger.fold(directory)
    credit = ledger.dimension_credit(records)
    escaped = ledger.escaped_findings(directory, records)
    meta = ledger.pass_meta(directory)

    ran: dict[str, int] = {}
    for slot in meta.values():
        for dimension in slot["dimensions"]:
            ran[dimension] = ran.get(dimension, 0) + 1

    spend: dict[str, float] = {}
    for rec in agent_records(directory):
        if rec.get("cost_usd") is None:
            continue
        spend[rec["dimension"]] = spend.get(rec["dimension"], 0.0) + rec["cost_usd"]

    out = []
    for dimension in sorted(set(spend) | set(credit) | set(ran) | set(escaped)):
        marks = credit.get(dimension, {})
        confirmed = marks.get("fractional", 0.0)
        refuted = marks.get("refuted_fractional", 0.0)
        dollars = spend.get(dimension, 0.0)
        passes = ran.get(dimension, 0)
        reported = confirmed + refuted
        out.append({
            "dimension": dimension,
            "cost_usd": dollars,
            "confirmed_fractional": confirmed,
            "confirmed_raw": marks.get("raw", 0),
            "refuted_fractional": refuted,
            "escaped": escaped.get(dimension, 0),
            "passes_ran": passes,
            # Each rate carries the denominator it is a rate over.
            "escaped_rate": (escaped.get(dimension, 0) / passes) if passes else None,
            "refuted_rate": (refuted / reported) if reported else None,
            "cost_per_confirmed": (dollars / confirmed) if confirmed else None,
        })
    return out
```

Add the subcommand to `main`:

```python
    sub.add_parser("report", help="per-dimension cost and yield")
```

and in the dispatch, after the `collect` branch:

```python
    if args.cmd == "report":
        header = (f"{'dimension':14s} {'cost':>8s} {'conf':>6s} {'raw':>5s} "
                  f"{'ref':>6s} {'esc':>4s} {'runs':>5s} {'$/conf':>9s}")
        print(header)
        print("-" * len(header))
        for row in report_rows(directory):
            per = row["cost_per_confirmed"]
            print(f"{row['dimension']:14s} {row['cost_usd']:8.2f} "
                  f"{row['confirmed_fractional']:6.1f} {row['confirmed_raw']:5d} "
                  f"{row['refuted_fractional']:6.1f} {row['escaped']:4d} "
                  f"{row['passes_ran']:5d} "
                  f"{('n/a' if per is None else format(per, '.2f')):>9s}")
        status = budget_status(directory, None)
        if status["estimated_agents"]:
            print(f"\nNOTE: {status['estimated_agents']} agent(s) had unreadable "
                  f"transcripts and are estimated at the running mean.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  no divide by zero` and `ok  escaped rate over passes ran`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 12 - cost.py report

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 13: The three dimension agents

**Files:**
- Create: `C:/Users/HartAlden/.claude/agents/lr-correctness.md`
- Create: `C:/Users/HartAlden/.claude/agents/lr-quality.md`
- Create: `C:/Users/HartAlden/.claude/agents/lr-intent.md`

**Interfaces:**
- Consumes: nothing at runtime; invoked by `SKILL.md` (Task 15) via the Agent tool's `subagent_type`.
- Produces: three `subagent_type` values — `lr-correctness`, `lr-quality`, `lr-intent`.

These carry no verification instruction and no severity filter, per R4.3/R4.4. That is the substantive change, and the reason for it is written into each file so a future editor does not "helpfully" add one back.

- [ ] **Step 1: Verify the frontmatter contract before writing**

Run: `grep -l "^effort:" C:/Users/HartAlden/.claude/agents/*.md C:/Users/HartAlden/.claude/agents/*/*.md 2>/dev/null | head -3`
Expected: at least one existing agent file listed, confirming `effort:` is honored in frontmatter alongside `model:`. If none is found, stop and report — the whole tiering design rests on this.

- [ ] **Step 2: Write `lr-correctness.md`**

```markdown
---
name: lr-correctness
description: loop-review's correctness dimension. Finds bugs, edge cases, silent failures, guards that do not guard, and invariant violations in a scoped diff. Dispatched by the loop-review skill; not for direct invocation.
model: opus
effort: high
tools: Read, Glob, Grep, Bash
---

You review a scoped diff for defects. Report everything you see.

## What you are looking for

Bugs and wrong behavior on reachable paths. Silent failures. Guards that do not
actually guard the state they name. Violated invariants. Edge cases the code does
not handle: empty, absent, all-null, partially joined, half-populated. Data that
can be zero where zero is falsy. Identifiers used without being resolved.

## Ground rules

**Report everything, including findings you are unsure about.** Do not filter by
severity, importance, or confidence. Mark each finding's confidence and let the
arbitrator decide. A finding that gets filtered out downstream costs one merge
decision; a finding you silently dropped ships.

**Do not verify before reporting.** Do not reproduce the failure, do not write a
script to confirm it, do not re-derive the numbers. The arbitrator checks the
premise of anything it is about to fix, once, at the point of decision. Verifying
every candidate up front is how earlier versions of this loop spent 15 minutes
and 110k tokens per pass to surface a handful of findings.

**Do not run the test suite, the linter, the type checker, or any project gate.**
They were run for you. Your prompt names the file holding their output; read it.
Every agent independently re-running them is the single largest waste this design
removes.

## Output

For each finding: the file and line, one sentence stating the defect, a concrete
failure scenario (inputs or state -> wrong output or crash), and your confidence
(high / medium / low). Order most severe first. No preamble.
```

- [ ] **Step 3: Write `lr-quality.md`**

```markdown
---
name: lr-quality
description: loop-review's quality dimension. Finds duplication, missed reuse, needless complexity, dead code, and altitude problems in a scoped diff. Dispatched by the loop-review skill; not for direct invocation.
model: opus
effort: medium
tools: Read, Glob, Grep, Bash
---

You review a scoped diff for quality problems. Report everything you see.

## What you are looking for

Logic duplicated from something that already exists in the codebase -- search for
it before assuming it does not. Code that could be meaningfully simpler. Dead
code, unreachable branches, unused imports and parameters. Altitude problems:
logic sitting at the wrong layer, or a function doing two jobs.

## Ground rules

**Report everything.** Do not filter by severity or confidence; mark confidence
and move on.

**Do not verify, do not run gates.** Both are handled outside you -- your prompt
names the gate output file.

**Relocating code across a module boundary requires evidence, not intuition.**
One such recommendation moved a guard into a shared helper and crashed two
diagnostics that intentionally build empty inputs. If you propose a move, name
the callers you checked.

**A guard is not duplication.** Do not recommend deleting a check on the grounds
that the state it guards cannot happen, unless you can show it cannot -- with
evidence, not assertion. That reasoning is how an empty-pool crash reaches
production.

## Output

For each finding: the file and line, one sentence stating the problem, what you
would do instead, and your confidence (high / medium / low). No preamble.
```

- [ ] **Step 4: Write `lr-intent.md`**

```markdown
---
name: lr-intent
description: loop-review's intent dimension. Judges whether a scoped diff achieves the goal stated in its issue, what it changed beyond that goal, and what else it touches. Dispatched by the loop-review skill; not for direct invocation.
model: opus
effort: medium
tools: Read, Glob, Grep, Bash
---

You are given a stated goal and a diff. Judge the diff against the goal.

## The three questions

1. **Does it achieve the stated goal?** Name anything the goal asks for that the
   diff does not deliver, and anything delivered only partially -- a code path
   added but never called, a flag accepted but never read, a case handled for one
   input shape but not the others the goal implies.
2. **What did it change that the goal did not ask for?** Unrequested scope is a
   finding. It may be justified, but it should be visible and deliberate rather
   than arriving unannounced.
3. **What else does this touch?** Callers, downstream consumers, cached or
   persisted formats, and anything reading a field whose meaning changed. A
   format change that ships before its reader does is a real failure mode here.

## Ground rules

**The goal in your prompt is the contract.** Do not infer a different goal from
the code -- inferring intent from the diff you are reviewing makes you a second
correctness reviewer, which the loop already has. If the goal genuinely does not
match what the branch is doing, that mismatch is itself your most valuable
finding: report it and say which reading you think is current.

**Report everything, filter nothing.** Mark confidence.

**Do not verify and do not run gates.** Your prompt names the gate output file.

## Output

Open with one sentence: does this diff achieve its stated goal, yes or no. Then
the findings, each with file and line where one applies, and your confidence.
No preamble.
```

- [ ] **Step 5: Verify the agents are registered and commit**

Run: `ls C:/Users/HartAlden/.claude/agents/lr-*.md`
Expected: all three files listed.

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 13 - three dimension agents

Agent files live in ~/.claude/agents/ outside this repo; this commit records
plan progress only."
```

---

### Task 14: `cost.py bakeoff`

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger.fold`, `ledger.pass_meta`, `ledger.fingerprint`.
- Produces: `cost.bakeoff_diff(baseline: list[dict], alternate: list[dict]) -> dict` with keys `baseline_only`, `alternate_only`, `both`, each a list of `{"file", "summary"}`; a `bakeoff` subcommand that prints the scope and the alternate agent's dispatch instructions, then diffs once the alternate's findings are supplied.

Per R8.7 the comparison is set-difference over findings, and it never writes to the ledger's finding table — a bakeoff is a measurement, not a review.

- [ ] **Step 1: Write the failing test**

```python
def test_bakeoff_diff():
    baseline = [
        {"file": "src/a.py", "summary": "off by one in the window bound"},
        {"file": "src/b.py", "summary": "empty pool is silently dropped"},
    ]
    alternate = [
        # Same defect, different wording -> must land in `both`, not in either
        # `only` bucket. A wording-sensitive diff would report a cheap tier as
        # missing findings it actually found.
        {"file": "src/a.py", "summary": "window bound is off by one"},
        {"file": "src/c.py", "summary": "unused import"},
    ]
    got = cost.bakeoff_diff(baseline, alternate)
    both = {(f["file"], ) for f in got["both"]}
    check("shared defect matched", ("src/a.py",) in both, str(got["both"]))
    check("baseline only", [f["file"] for f in got["baseline_only"]] == ["src/b.py"],
          str(got["baseline_only"]))
    check("alternate only", [f["file"] for f in got["alternate_only"]] == ["src/c.py"],
          str(got["alternate_only"]))

    empty = cost.bakeoff_diff([], [])
    check("empty bakeoff", empty["both"] == [] and empty["baseline_only"] == [])
```

Register `test_bakeoff_diff()` in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'cost' has no attribute 'bakeoff_diff'`

- [ ] **Step 3: Write minimal implementation**

Add to `cost.py` before `main`:

```python
BAKEOFF_MATCH = 0.6


def bakeoff_diff(baseline: list[dict], alternate: list[dict]) -> dict:
    """Set-difference two finding sets by same-file summary similarity.

    Matching is by content, not by wording: the same defect described two ways
    must count as found by both tiers. A wording-sensitive diff would report a
    cheap tier as having missed findings it actually surfaced, which is the
    exact conclusion a bakeoff exists to get right.
    """
    remaining = list(alternate)
    both, baseline_only = [], []
    for item in baseline:
        tokens = ledger.summary_tokens(item["summary"])
        match = None
        for candidate in remaining:
            if ledger.norm_path(candidate["file"]) != ledger.norm_path(item["file"]):
                continue
            if ledger.jaccard(tokens, ledger.summary_tokens(candidate["summary"])) >= BAKEOFF_MATCH:
                match = candidate
                break
        if match is None:
            baseline_only.append(item)
        else:
            remaining.remove(match)
            both.append(item)
    return {"baseline_only": baseline_only, "alternate_only": remaining, "both": both}
```

Add the subcommand to `main`:

```python
    b = sub.add_parser("bakeoff", help="compare one dimension at another tier")
    b.add_argument("--pass", dest="pass_no", type=int, required=True)
    b.add_argument("--dimension", required=True)
    b.add_argument("--tier", required=True, help="model/effort, e.g. sonnet/low")
    b.add_argument("--alternate", help="path to the alternate run's findings JSON")
```

and in the dispatch:

```python
    if args.cmd == "bakeoff":
        meta = ledger.pass_meta(directory).get(args.pass_no)
        if not meta:
            raise SystemExit(f"no recorded metadata for pass {args.pass_no}")
        if not args.alternate:
            model, _, effort = args.tier.partition("/")
            print(f"Re-run lr-{args.dimension} at model={model} effort={effort} over:")
            for path in meta["scope"]:
                print(f"  {path}")
            print("\nThen re-run with --alternate <findings.json> to diff the sets.")
            return 0
        with open(args.alternate, encoding="utf-8") as fh:
            alternate = json.load(fh)
        records = ledger.fold(directory)
        baseline = [
            {"file": r["file"], "summary": r["summary"]}
            for r in records.values()
            if args.dimension in (r.get("found_by") or [])
        ]
        result = bakeoff_diff(baseline, alternate)
        for label in ("both", "baseline_only", "alternate_only"):
            print(f"\n{label} ({len(result[label])}):")
            for item in result[label]:
                print(f"  {item['file']}: {item['summary']}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  shared defect matched`.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 14 - cost.py bakeoff

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 15: CLI surfaces and gate-file selection

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py` (`build_parser` at :609)
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/gates.py`
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/goal.py`
- Test: `C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`

**Interfaces:**
- Consumes: `ledger.dimensions_for_pass` (Task 8), `ledger.subsystem_candidates` (Task 7), `gates.derive`/`gates.pin` (Task 9), `goal.resolve` (Task 10), `ledger.pass_meta` (Task 4).
- Produces: `ledger.py dimensions` (with `--record` persisting skip reasons), `ledger.py subsystems`, `cost.py budget --budget <usd|none>`, `gates.py --pin`, `gates.py --run`, `goal.py` as a bare CLI; and `ledger.gate_file_for_pass(pass_no: int, directory: str) -> str`.

The functions from Tasks 4-10 have no command surface yet. `SKILL.md` (Task 16) invokes all of them from the shell, so without this task it would document commands that do not exist.

- [ ] **Step 1: Write the failing test**

```python
def test_gate_file_selection():
    directory = tempfile.mkdtemp()
    try:
        # Pass 1 reads the baseline written before any pass ran.
        check("pass 1 reads baseline",
              ledger.gate_file_for_pass(1, directory) == "gates-pass-0.txt")
        # Pass N reads pass N-1's output: the gates describing the tree it reviews.
        open(os.path.join(directory, "gates-pass-1.txt"), "w").close()
        check("pass 2 reads pass 1",
              ledger.gate_file_for_pass(2, directory) == "gates-pass-1.txt")
        # Pass 3 when pass 2 applied no fixes and wrote no gates: fall back to the
        # most recent file that exists, rather than naming one that does not.
        check("falls back to most recent",
              ledger.gate_file_for_pass(3, directory) == "gates-pass-1.txt")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_cli_surfaces():
    directory = tempfile.mkdtemp()
    try:
        # dimensions: exercised through the parser, not just the function.
        parser = ledger.build_parser()
        args = parser.parse_args(["dimensions", "--pass", "1",
                                  "--new-file", "false", "--added-lines", "0"])
        check("dimensions parses", args.pass_no == 1 and args.new_file == "false")
        args = parser.parse_args(["subsystems", "--threshold", "high"])
        check("subsystems parses", args.threshold == "high")
        args = parser.parse_args(["pass", "--pass", "2", "--event", "start",
                                  "--scope", "a.py", "b.py",
                                  "--dimensions", "correctness",
                                  "--gate-file", "gates-pass-1.txt", "--sha", "abc"])
        check("pass flags parse", args.scope == ["a.py", "b.py"] and args.sha == "abc")

        # R5.3: a skipped dimension and its reason must be readable by a later
        # session, not merely printed to a terminal nobody kept.
        ledger._append(os.path.join(directory, "passes.jsonl"), {
            "pass": 2, "event": "start", "dimensions": ["correctness"],
            "skipped": {"quality": "no prior yield and no new code in scope",
                        "intent": "no prior yield and no new code in scope"},
        })
        meta = ledger.pass_meta(directory)
        check("skip reasons persisted",
              meta[2]["skipped"]["quality"].startswith("no prior yield"),
              str(meta[2].get("skipped")))
        check("skipped names every gated dimension",
              set(meta[2]["skipped"]) == {"quality", "intent"},
              str(meta[2]["skipped"]))

        # cost.py budget flag parses both a number and the disable form.
        check("budget disable", cost.parse_budget("none") is None)
        check("budget value", cost.parse_budget("12.5") == 12.5)
        check("budget default", cost.parse_budget(None) == cost.DEFAULT_BUDGET_USD)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
```

Register both in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: FAIL with `AttributeError: module 'ledger' has no attribute 'gate_file_for_pass'`

- [ ] **Step 3: Write minimal implementation**

Add to `ledger.py` after `dimensions_for_pass`:

```python
def gate_file_for_pass(pass_no: int, directory: str) -> str:
    """The gate output pass N's finders read.

    Gates run at the END of a pass, after its fixes land, so pass N reads pass
    N-1's file -- the one describing exactly the tree it is reviewing. Pass 1
    reads the baseline written before the loop started. A pass that applied no
    fixes writes no gates, so this walks back to the most recent file that
    exists rather than naming one that does not.
    """
    wanted = pass_no - 1
    while wanted > 0:
        name = f"gates-pass-{wanted}.txt"
        if os.path.isfile(os.path.join(directory, name)):
            return name
        wanted -= 1
    return "gates-pass-0.txt"
```

Add the two command handlers to `ledger.py` before `build_parser`:

```python
SKIP_REASON = "no prior yield and no new code in scope"


def cmd_dimensions(args) -> int:
    directory = ledger_dir(args.branch)
    records = fold(directory)
    chosen = dimensions_for_pass(
        records, args.pass_no,
        args.new_file.lower() == "true",
        args.added_lines,
    )
    skipped = {}
    for dimension in ("correctness", "quality", "intent"):
        if dimension in chosen:
            print(f"RUN   {dimension:12s} {chosen[dimension]}")
        else:
            skipped[dimension] = SKIP_REASON
            print(f"SKIP  {dimension:12s} {SKIP_REASON}")
    if args.record:
        # Persist the decision, not just print it: R5.3 exists so a later
        # session can tell coverage it had from coverage it merely looks like
        # it had. A terminal nobody kept is not a record.
        _append(
            os.path.join(directory, "passes.jsonl"),
            {"pass": args.pass_no, "event": "start",
             "dimensions": sorted(chosen), "skipped": skipped},
        )
    _emit({"run": sorted(chosen), "reasons": chosen, "skipped": skipped,
           "gate_file": gate_file_for_pass(args.pass_no, directory)})
    return 0


def cmd_subsystems(args) -> int:
    directory = ledger_dir(args.branch)
    candidates = subsystem_candidates(fold(directory), args.threshold)
    if not candidates:
        print("no subsystem holds 2+ open findings at or above threshold")
        return 0
    print("ESCALATE: stop fixing facets, enumerate the invariant.")
    for item in candidates:
        print(f"  {item['kind']:6s} {item['subsystem']}: {', '.join(item['ids'])}")
    return 0
```

And register them in `build_parser`, after the `pass` subparser block:

```python
    dm = sub.add_parser("dimensions", help="which finders run this pass")
    dm.add_argument("--pass", dest="pass_no", type=int, required=True)
    dm.add_argument("--new-file", dest="new_file", default="false",
                    choices=["true", "false"])
    dm.add_argument("--added-lines", dest="added_lines", type=int, default=0)
    dm.add_argument("--record", action="store_true",
                    help="persist the run/skip decision to the pass record")
    dm.set_defaults(func=cmd_dimensions)

    sy = sub.add_parser("subsystems", help="escalation candidates (2+ open findings)")
    sy.add_argument("--threshold", default="medium", choices=sorted(SEVERITY_RANK))
    sy.set_defaults(func=cmd_subsystems)
```

Add a CLI to the bottom of `gates.py`:

```python
def _config_path() -> str:
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ledger

    return os.path.join(ledger.ledger_dir(), "config.json")


def main(argv=None) -> int:
    import argparse
    import subprocess

    p = argparse.ArgumentParser(prog="gates.py")
    p.add_argument("--pin", action="store_true", help="resolve and record the gate list")
    p.add_argument("--run", action="store_true", help="run the gates, print combined output")
    args = p.parse_args(argv)

    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True).strip()
    config = _config_path()
    resolved = derive(root, config)

    if args.pin:
        pin(config, resolved)
        print(f"gates ({resolved['rule']}): "
              + ("; ".join(resolved["commands"]) or "none derived"))
        return 0

    if args.run:
        # The header is what a finder reads to know what was actually run. A run
        # that derived nothing says so, rather than emitting an empty file that
        # would read as a clean suite.
        print(f"# gate rule: {resolved['rule']}")
        if not resolved["commands"]:
            print("# no gates derived for this repository -- none were run")
            return 0
        for command in resolved["commands"]:
            print(f"\n$ {command}")
            done = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(done.stdout, end="")
            print(done.stderr, end="")
            print(f"# exit {done.returncode}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
```

Add `parse_budget` and the `budget` subcommand to `cost.py`. Insert the helper above `main`:

```python
def parse_budget(raw: str | None) -> float | None:
    """`None` -> the default; the string 'none' -> disabled; else a dollar figure."""
    if raw is None:
        return DEFAULT_BUDGET_USD
    if raw.strip().lower() == "none":
        return None
    return float(raw)
```

Add the subcommand in `main`:

```python
    bg = sub.add_parser("budget", help="spend so far against the threshold")
    bg.add_argument("--budget", dest="budget",
                    help=f"dollars, or 'none' to disable (default {DEFAULT_BUDGET_USD})")
```

and the dispatch branch:

```python
    if args.cmd == "budget":
        limit = parse_budget(args.budget)
        status = budget_status(directory, limit)
        if not status["evaluable"]:
            print("no cost data recorded yet -- budget cannot be evaluated")
            return 0
        print(f"known ${status['known']:.2f}, total ${status['total']:.2f}"
              + (f", limit ${limit:.2f}" if limit is not None else ", no limit"))
        if status["estimated_agents"]:
            print(f"  {status['estimated_agents']} agent(s) estimated at the running mean")
        if status["over"]:
            # Warn and hand the decision back. Every fix is already committed, so
            # stopping is safe -- but whether to stop is the user's call, and an
            # automatic abort would leave a half-hardened branch with no decision.
            print("\nOVER BUDGET. Ask the user whether to continue before the next pass.")
    return 0
```

Add a CLI to the bottom of `goal.py`:

```python
def main(argv=None) -> int:
    import os
    import subprocess
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ledger

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    resolved = resolve(branch)
    if resolved["source"] == "ask":
        print("Could not resolve a goal from the issue or the PR.")
        print("Ask the user for one line, then record it with:")
        print('  python ledger.py pass --pass 1 --event start --note "goal: <text>"')
        return 1
    print(f"source: {resolved['source']}"
          + (f" (#{resolved['issue']})" if resolved["issue"] else ""))
    print(resolved["goal"])
    ledger._append(
        os.path.join(ledger.ledger_dir(), "passes.jsonl"),
        {"event": "goal", "goal": resolved["goal"],
         "source": resolved["source"], "issue": resolved["issue"]},
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, including `ok  falls back to most recent` and `ok  dimensions parses`.

Then smoke-test the three CLIs from inside the FantasyBaseball repo:

```bash
cd C:/Users/HartAlden/FantasyBaseball
python C:/Users/HartAlden/.claude/skills/loop-review/scripts/ledger.py subsystems
python C:/Users/HartAlden/.claude/skills/loop-review/scripts/gates.py --pin
```
Expected: `subsystems` prints the no-candidates line; `--pin` prints `gates (claude-md): pytest -v; ruff check .; ...`. A different rule than `claude-md` here means the CLAUDE.md parser missed this repo's verification section — fix it before Task 16, since every later gate file depends on it.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 15 - CLI surfaces and gate-file selection

Skill files live outside this repo; this commit records plan progress only."
```

---

### Task 16: Rewrite SKILL.md and update arbitration.md

**Files:**
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/SKILL.md`
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/reference/arbitration.md`
- Modify: `C:/Users/HartAlden/.claude/skills/loop-review/reference/ledger.md`

**Interfaces:**
- Consumes: everything built in Tasks 1-15.
- Produces: the operator-facing loop.

This is the last task because it names every script, CLI subcommand, and agent the earlier tasks create. Writing it first would guarantee drift between the documented commands and the real ones.

- [ ] **Step 1: Confirm every referenced artifact exists**

Run:
```bash
ls C:/Users/HartAlden/.claude/skills/loop-review/scripts/{cost,gates,goal,ledger,harvest,selftest}.py \
   C:/Users/HartAlden/.claude/skills/loop-review/scripts/prices.json \
   C:/Users/HartAlden/.claude/agents/lr-{correctness,quality,intent}.md
```
Expected: all ten paths listed. Any missing path means an earlier task is incomplete — stop and finish it rather than writing SKILL.md around a gap.

- [ ] **Step 2: Rewrite the "One pass" section of `SKILL.md`**

Replace the existing "One pass" section (currently lines 42-111, from `## One pass` through the `ledger.py pass --event end` block) with:

````markdown
## One pass

Set `LR` to this skill's directory once
(`C:/Users/HartAlden/.claude/skills/loop-review` on this machine -- spell it out
rather than relying on `~` expanding, since these run under both PowerShell and
bash).

**0. Preflight (first pass only).** Resolve the goal and pin the gate list:

```bash
python $LR/scripts/goal.py            # issue -> PR -> ask you
python $LR/scripts/gates.py --pin     # resolve and record the gate commands
```

If goal resolution returns `ask`, ask for one line and record it. If it returns
an issue, say which issue you took and let the user correct you before the
intent reviewer runs on the wrong contract.

**1. Scope.**

- Pass 1: the whole `main...HEAD` diff plus uncommitted changes.
- Later passes: `git diff <last-reviewed-sha>..HEAD`, plus the blast radius --
  every file containing a literal hit for a symbol the previous pass's fixes
  added, renamed, or removed (`rg --fixed-strings`). This over-includes on
  purpose: an extra file in scope is cheap, an unreviewed caller is not.
- Churn fired: the theme-enumeration pass instead (see `reference/arbitration.md`).

**2. Dispatch the finders concurrently**, in one message, via the Agent tool.
`ledger.py` decides which run:

```bash
python $LR/scripts/ledger.py dimensions --pass N \
  --new-file <true|false> --added-lines <n> --record
```

`--record` writes the run/skip decision and its reason to the pass record. Use
it every pass: a later session reading the ledger needs to tell coverage the
loop had from coverage it merely looks like it had.

`correctness` runs every pass. `quality` and `intent` run on pass 1, when they
have earned it with a confirmed finding on this branch, or when the scope holds
code they have never seen.

Every finder prompt carries, verbatim:

- the marker `LR-PASS-<N>-<dimension>` on its own first line -- this is how
  `cost.py` attributes spend, and a prompt without it is an untracked agent;
- the scoped diff;
- the path to the gate output (`<ledger-dir>/gates-pass-<N-1>.txt`, or the
  baseline on pass 1) and the instruction not to re-run the gates;
- the resolved goal, for `lr-intent`.

**Do not add a verification instruction and do not ask for a severity filter.**
The agent definitions already say so, and both are load-bearing: telling an Opus
5 reviewer to verify before reporting is what produced 15-minute, 110k-token
passes, and telling it to report only high-severity findings makes it report
less while investigating just as hard.

**3. Load the findings.** Record each with the dimension that saw it:

```bash
python $LR/scripts/ledger.py add --pass N --source lr-correctness \
  --found-by correctness --file src/x.py --line 40 --summary "..."
```

**4. Arbitrate.** See `reference/arbitration.md`. Check
`ledger.py subsystems` before choosing what to fix: two or more open,
triaged, at-or-above-threshold findings sharing a file or theme means stop
fixing facets and enumerate the invariant instead.

**5. Fix.** One finding at a time: reproduce, failing test, fix, commit,
`ledger.py resolve`. See `reference/fix-protocol.md`.

**6. Close the pass.** Run the gates once, in the parent, and record everything
the next pass needs:

```bash
python $LR/scripts/gates.py --run > "$(python $LR/scripts/ledger.py path)/gates-pass-N.txt"
python $LR/scripts/ledger.py pass --pass N --event end \
  --fix-files <files the fixes touched> --sha "$(git rev-parse HEAD)"
python $LR/scripts/cost.py collect --pass N
python $LR/scripts/cost.py budget
```

Then check the stop rule. If `budget` reports OVER BUDGET, stop and ask before
opening the next pass.
````

- [ ] **Step 3: Add a Cost section to `SKILL.md`** immediately before `## The stop rule`

````markdown
## Cost

Every finder's spend is attributed from its own transcript and priced against
`scripts/prices.json`. Weighted, not raw: cache reads bill at a tenth of input
and cache writes at 1.25x, so token totals misrank agents by more than an order
of magnitude.

```bash
python $LR/scripts/cost.py report     # cost per confirmed finding, by dimension
```

Two columns answer the question tiering actually turns on. **Escaped** counts
findings a later pass caught in code an earlier pass had in scope and did not
change -- charged only to dimensions that actually ran, since one that was gated
off had no chance to miss anything. **Refuted rate** counts noise, which costs
the arbitrator real work. A cheaper tier that raises either is not cheaper.

Do not tier any dimension down from one run. Three to five branches, then
`cost.py bakeoff` to confirm before committing to it.

```bash
python $LR/scripts/cost.py budget --budget 25    # or --budget none to disable
```

The budget warns at USD 15.00 by default. It warns and asks; it never aborts on
its own. Every fix is already committed, so
stopping is safe at any point -- but whether to stop is yours, not the loop's.
Agents whose transcripts could not be read are charged the running mean rather
than zero, and the report says how many.
````

- [ ] **Step 4: Update `reference/arbitration.md`**

In section 2 ("Check the premise before promoting anything"), append:

```markdown
Premise checking is **on demand**: check a finding when you are about to fix it,
not when it arrives. Verifying every candidate up front is what made passes cost
15 minutes each, and most candidates never reach a fix.

The one exception is the subsystem trigger below. A theme-enumeration pass is
the most expensive action this loop can take, so before firing it, premise-check
each candidate in the group and proceed only if at least two survive.
```

Replace section 5's opening paragraph ("`status` reports it: two consecutive passes...") with:

```markdown
Two signals, and the earlier one matters more.

**The subsystem trigger fires first.** `ledger.py subsystems` reports any file
or theme holding two or more open, triaged, at-or-above-threshold findings. That
is the moment to stop asking what the next issue is -- before you have fixed
either one. Waiting for churn means you have already band-aided two facets, and
each narrow fix shifts the problem to the adjacent one.

**Churn is the backstop**, for when the trigger was missed: two consecutive
passes whose flagged files overlap and whose finding count did not drop. This
discriminator was validated against real runs -- it fires on the ones that burned
to the cap and stays silent on the ones whose findings decayed to zero.
```

- [ ] **Step 5: Update `reference/ledger.md` Commands block**

Add to the command list:

```bash
$L dimensions --pass 3 --new-file false --added-lines 12 --record
$L subsystems --threshold medium                            # escalation candidates
python <skill>/scripts/cost.py collect --pass 3             # attribute spend
python <skill>/scripts/cost.py report                       # cost per confirmed finding
python <skill>/scripts/cost.py budget                       # spend vs threshold
python <skill>/scripts/gates.py --pin                       # resolve gate commands
python <skill>/scripts/goal.py                              # resolve the branch goal
```

- [ ] **Step 6: Run the full self-test and a smoke check**

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py`
Expected: PASS, all checks, ending with the total count.

Run: `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/cost.py report`
Expected: the header row and either a table or an empty body — an exception here means a CLI wiring bug the self-test does not cover.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/HartAlden/FantasyBaseball
git commit --allow-empty -m "plan: loop-review rebuild task 16 - SKILL.md and reference rewrite

Skill files live outside this repo; this commit records plan progress only."
```

---

## Post-implementation validation (spec acceptance A1-A5)

Not a task — the acceptance gate. Run after Task 16.

- [ ] **A1:** `python C:/Users/HartAlden/.claude/skills/loop-review/scripts/selftest.py` passes, including all fifteen new test functions: `test_cost_math`, `test_transcript_scanning`, `test_cost_collect`, `test_pass_meta`, `test_dimension_credit`, `test_escaped_findings`, `test_subsystem_trigger`, `test_dimension_gating`, `test_gate_derivation`, `test_goal_resolution`, `test_budget_status`, `test_report_rows`, `test_bakeoff_diff`, `test_gate_file_selection`, `test_cli_surfaces`.
- [ ] **A2:** Run the rebuilt loop on one real branch to convergence or a named escalation. `python .../ledger.py open --threshold medium` exits 0, or the escalation is reported with the open list.
- [ ] **A3:** `cost.py report` emits a per-dimension table with non-null cost, confirmed, refuted, escaped, and cost-per-confirmed for every dimension that ran.
- [ ] **A4:** No completed agent is recorded `cost: unknown`. If any is, fix the attribution before trusting the table.
- [ ] **A5:** Report that loop's total weighted cost and per-agent wall clock beside the pre-rebuild figures (#272: 14-17 min and 110-118k tokens per review agent; #280: ~1.2M subagent tokens per pass, 10 passes without convergence). State the comparison in tokens and wall clock, since the recorded pre-rebuild figures are in those units, not dollars. A run that costs more than expected is a valid outcome provided the number is measured and reported — Phase 1 delivers the instrument, not a predetermined reading.
- [ ] Record the result in `docs/superpowers/` as a dated note, and open a follow-up issue for Phase 2 (tiering) citing the table.
