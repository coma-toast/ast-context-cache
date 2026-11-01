#!/usr/bin/env python3
"""SWE-bench Lite benchmark runner for ast-context-cache comparison.

Runs opencode against SWE-bench Lite tasks in two conditions:

  control    ast-context-cache MCP server disabled, no ast-context-cache
             AGENTS.md block
  treatment  ast-context-cache MCP server enabled (localhost:7821/mcp),
             ast-context-cache AGENTS.md block present

Measures tokens, wall time, tool calls and a success-rate proxy per task.

Each condition gets its own XDG_CONFIG_HOME so that the two runs are isolated
from the host agent's global opencode config (which normally enables the
ast-context-cache MCP server, ships an ast-context-cache AGENTS.md and loads
the ast-context-cache search nudge plugin).

Usage:
  python3 bench_runner.py --fetch-dataset
  python3 bench_runner.py --task-count 5 --model litellm/free --output-dir benchmarks/runs/pilot
  python3 bench_runner.py --tasks sqlfluff__sqlfluff-1625 --model litellm/free --timeout 600
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "data" / "swe_bench_lite_dev.jsonl"
LEGACY_DATASET = Path("/tmp/swe_bench_lite/dev/dev.jsonl")
DATASET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
    "resolve/main/data/dev-00000-of-00001.parquet"
)

AST_MCP_NAME = "ast-context-cache"
AST_MCP_URL = "http://127.0.0.1:7821/mcp"
# opencode exposes MCP tools as "<server>_<tool>".
AST_TOOL_PREFIXES = (
    "ast-context-cache_",
    "ast_context_cache_",
    "ast-cache-",
    "ast_cache_",
)

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://ai.jasondale.me/v1")

CONDITIONS = ("control", "treatment")

# Variables the host agent session sets that would otherwise leak into the
# benchmark subprocess and change opencode's behaviour (project config off,
# inherited session identity).
LEAKY_ENV_VARS = ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE", "OPENCODE_PID")

# Files the runner writes into the repo checkout. They must never be part of
# the agent's output patch.
INJECTED_PATHSPECS = ("AGENTS.md", "opencode.json", "opencode.jsonc", ".opencode")

BASE_AGENTS_MD = """<!-- repo-work-rules:begin -->
## Repository work

- Work directly in this checkout; do not touch files outside it.
- Do not run git commands that rewrite history, and do not commit credentials
  or secrets.
<!-- repo-work-rules:end -->
"""

AST_AGENTS_BLOCK = """
<!-- ast-context-cache:begin -->
## ast-context-cache

ast-context-cache is your code and context memory. Use its MCP tools on every
task, before your own file tools. Fall back to grep, find, cat or whole-file
reads only for a single known string or file, or when the server is not running.

### Start of work

1. `recall_memory` for rules and prefs, then `list_context` to find notes from
   earlier runs and `fetch_context` the ones you need.
2. New repo or space: `index_status`; if it is not indexed, `index_files` on the
   repo path.

### Explore before grep

3. Explore with `get_project_map`, `get_context_capsule`, `search_semantic`,
   `retrieve` and `get_file_context` BEFORE grep, find, cat or whole-file reads.
   Call `check_symbol_exists` before assuming a name exists.
4. Before changing an exported function or type: `get_impact_graph`.

<!-- ast-context-cache:end -->
"""


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> list[dict]:
    """Load SWE-bench Lite JSONL dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_tasks(records: list[dict], count: int | None,
                 task_ids: list[str] | None) -> list[dict]:
    """Return the tasks to run, preserving the order requested by the caller."""
    if task_ids:
        by_id = {r.get("instance_id"): r for r in records}
        selected = [by_id[t] for t in task_ids if t in by_id]
        missing = [t for t in task_ids if t not in by_id]
        if missing:
            print(f"WARNING: {len(missing)} task(s) not found in dataset: {missing}")
        return selected
    return records[:count] if count else records


def fetch_dataset(dest: Path) -> Path:
    """Download the SWE-bench Lite dev split and convert it to JSONL."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "pyarrow is required to convert the dataset: pip install pyarrow"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = Path(tmp) / "dev.parquet"
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, parquet_path)
        table = pq.read_table(parquet_path)
        columns = table.column_names
        rows = table.to_pylist()
        with open(dest, "w", encoding="utf-8") as f:
            for row in rows:
                record = {c: row[c] for c in columns}
                f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(rows)} instances to {dest}")
    return dest


def resolve_dataset(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path)
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    if LEGACY_DATASET.exists():
        return LEGACY_DATASET
    return DEFAULT_DATASET


# ---------------------------------------------------------------------------
# Condition setup
# ---------------------------------------------------------------------------
def agents_md_for(condition: str) -> str:
    if condition == "treatment":
        return BASE_AGENTS_MD + AST_AGENTS_BLOCK
    return BASE_AGENTS_MD


def split_model(model: str) -> tuple[str, str]:
    if "/" in model:
        provider, name = model.split("/", 1)
        return provider, name
    return "litellm", model


def build_opencode_config(condition: str, model: str,
                          context_limit: int = 131072) -> dict:
    """Build the per-condition opencode config.

    Only the ast-context-cache MCP server is declared, and only in the
    treatment condition. Other MCP servers from the host's global config are
    deliberately absent from both conditions so the comparison isolates one
    variable.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")

    provider, model_name = split_model(model)
    if provider != "litellm":
        raise SystemExit(
            f"unsupported provider {provider!r}; this runner configures litellm only"
        )

    return {
        "$schema": "https://opencode.ai/config.json",
        "permission": "allow",
        "provider": {
            "litellm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "LiteLLM Proxy",
                "options": {
                    "baseURL": LITELLM_BASE_URL,
                    "apiKey": "{env:LITELLM_API_KEY}",
                },
                "models": {
                    model_name: {
                        "name": model_name,
                        "limit": {"context": context_limit, "output": 8192},
                    }
                },
            }
        },
        "mcp": {
            AST_MCP_NAME: {
                "type": "remote",
                "url": AST_MCP_URL,
                "enabled": condition == "treatment",
            }
        },
    }


def clone_repo(repo_name: str, base_commit: str, repo_dir: Path) -> None:
    """Clone repo_name pinned at base_commit (detached HEAD)."""
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"  Reusing existing clone at {repo_dir}")
        return

    clone_url = f"https://github.com/{repo_name}.git"
    if repo_dir.exists():
        # Leftover partial clone from an interrupted run.
        subprocess.run(["rm", "-rf", str(repo_dir)], check=True)

    repo_dir.mkdir(parents=True)
    print(f"  Cloning {clone_url} @{base_commit[:8]} ...")
    subprocess.run(["git", "-C", str(repo_dir), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "add", "origin", clone_url],
        check=True,
    )
    fetched = subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "-q", "--depth", "1",
         "origin", base_commit],
        capture_output=True, text=True, timeout=300,
    )
    if fetched.returncode != 0:
        print("  Direct fetch of base_commit failed; doing a full fetch ...")
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "-q", "--tags", "origin"],
            check=True, timeout=900,
        )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "-q", "--detach", base_commit],
        check=True, timeout=120,
    )


def setup_condition(task_dir: Path, instance: dict, condition: str,
                    model: str, context_limit: int) -> dict:
    """Create one condition directory: config home + pinned repo clone."""
    cond_dir = task_dir / condition
    config_dir = cond_dir / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = build_opencode_config(condition, model, context_limit)
    with open(config_dir / "opencode.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(config_dir / "AGENTS.md", "w", encoding="utf-8") as f:
        f.write(agents_md_for(condition))

    repo_dir = cond_dir / instance["repo"]
    clone_repo(instance["repo"], instance["base_commit"], repo_dir)

    return {
        "cond_dir": cond_dir,
        "config_home": cond_dir / ".config",
        "repo_dir": repo_dir,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Opencode execution
# ---------------------------------------------------------------------------
def _clean_env(config_home: Path) -> dict:
    """Environment for the benchmark subprocess with agent-session leakage removed."""
    env = dict(os.environ)
    for var in LEAKY_ENV_VARS:
        env.pop(var, None)
    env["XDG_CONFIG_HOME"] = str(config_home)
    return env


def run_opencode(repo_dir: Path, config_home: Path, instance: dict,
                 model: str, timeout: int) -> dict | None:
    """Run opencode in a condition and return parsed metrics, or None on failure."""
    problem_statement = instance.get("problem_statement", "")
    prompt = (
        "You are fixing a bug in this repository. Make the change directly in "
        "the working tree; do not open a pull request.\n\n"
        f"Issue:\n{problem_statement}"
    )
    cmd = [
        "opencode", "run",
        "--model", model,
        "--format", "json",
        "--auto",
        prompt,
    ]
    print(f"  Running opencode (timeout {timeout}s) in {repo_dir}")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=_clean_env(config_home),
        )
    except subprocess.TimeoutExpired as exc:
        print(f"  TIMEOUT after {timeout}s for {instance['instance_id']}")
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return _extract_metrics(_parse_events(partial), time.time() - start,
                                returncode=None, timed_out=True)
    except Exception as exc:  # pragma: no cover - subprocess guard
        print(f"  ERROR: {exc}")
        return None

    wall_time = time.time() - start
    events = _parse_events(proc.stdout)
    if proc.returncode != 0 and not events:
        print(f"  opencode exited {proc.returncode}: {proc.stderr[:500]}")

    return _extract_metrics(events, wall_time, proc.returncode)


def _parse_events(stdout: str) -> list[dict]:
    """Parse opencode --format json JSONL events."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def is_ast_tool(name: str) -> bool:
    if not name:
        return False
    return name.startswith(AST_TOOL_PREFIXES) or "ast-context-cache" in name


def _extract_metrics(events: list[dict], wall_time: float,
                     returncode: int | None, timed_out: bool = False) -> dict:
    """Extract benchmark metrics from opencode JSON events.

    opencode emits events shaped like::

        {"type": "step_finish", "timestamp": 1793507614827,
         "sessionID": "ses_...",
         "part": {"type": "step-finish",
                  "tokens": {"total": 22850, "input": 22738, "output": 0,
                             "reasoning": 131, "cache": {"write": 0, "read": 0}},
                  "cost": 0}}

        {"type": "tool_use", "part": {"type": "tool", "tool": "read", ...}}
    """
    tokens_input = tokens_output = tokens_total = tokens_reasoning = 0
    cache_read = cache_write = 0
    tool_calls_total = 0
    ast_mcp_calls = 0
    tool_names: dict[str, int] = {}
    session_id = None
    errors: list[str] = []
    first_ts = last_ts = None

    for evt in events:
        ts = evt.get("timestamp")
        if isinstance(ts, (int, float)):
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)

        part = evt.get("part") if isinstance(evt.get("part"), dict) else {}
        sid = evt.get("sessionID") or part.get("sessionID")
        if sid and not session_id:
            session_id = sid

        etype = evt.get("type")

        if etype == "step_finish":
            tokens = part.get("tokens") or {}
            tokens_input += int(tokens.get("input") or 0)
            tokens_output += int(tokens.get("output") or 0)
            tokens_reasoning += int(tokens.get("reasoning") or 0)
            tokens_total += int(tokens.get("total") or 0)
            cache = tokens.get("cache") or {}
            cache_read += int(cache.get("read") or 0)
            cache_write += int(cache.get("write") or 0)

        elif etype == "tool_use":
            name = part.get("tool") or ""
            if name:
                tool_calls_total += 1
                tool_names[name] = tool_names.get(name, 0) + 1
                if is_ast_tool(name):
                    ast_mcp_calls += 1

        elif etype == "error":
            err = evt.get("error") or {}
            data = err.get("data") if isinstance(err, dict) else None
            message = (data or {}).get("message") if isinstance(data, dict) else None
            errors.append(str(message or err)[:300])

    if tokens_total == 0:
        tokens_total = tokens_input + tokens_output + tokens_reasoning

    event_span = None
    if first_ts is not None and last_ts is not None:
        event_span = round((last_ts - first_ts) / 1000.0, 2)

    return {
        "wall_time_sec": round(wall_time, 2),
        "event_span_sec": event_span,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_reasoning": tokens_reasoning,
        "tokens_total": tokens_total,
        "tokens_cache_read": cache_read,
        "tokens_cache_write": cache_write,
        "tool_calls_total": tool_calls_total,
        "ast_mcp_calls": ast_mcp_calls,
        "tool_calls_by_name": tool_names,
        "returncode": returncode,
        "timed_out": timed_out,
        "session_id": session_id,
        "errors": errors,
        "events_count": len(events),
    }


# ---------------------------------------------------------------------------
# Output patch + success-rate proxy
# ---------------------------------------------------------------------------
def collect_patch(repo_dir: Path) -> str:
    """Return the agent's working-tree diff, excluding runner-injected files."""
    exclude = [f":(exclude){p}" for p in INJECTED_PATHSPECS]
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", "-A", "--", ".", *exclude],
        check=False, capture_output=True, text=True, timeout=120,
    )
    diff = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "--cached", "--binary"],
        capture_output=True, text=True, timeout=120,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "reset", "-q"],
        check=False, capture_output=True, text=True, timeout=60,
    )
    return diff.stdout if diff.returncode == 0 else ""


def patch_applies(repo_dir: Path, patch: str) -> bool:
    """Check the patch applies cleanly to the base commit.

    The worktree is base_commit + patch, so a clean reverse-apply proves the
    forward patch is well-formed against the base tree.
    """
    if not patch.strip():
        return False
    fd, path = tempfile.mkstemp(suffix=".patch")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patch)
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "apply", "--check", "--reverse", path],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    finally:
        os.unlink(path)


def evaluate_success_proxy(instance: dict, metrics: dict) -> str:
    """Evaluate the success-rate proxy against the gold patch.

    confirmed   patch applies cleanly and touches at least one file the gold
                patch touches
    proxy_pass  patch applies cleanly but touches no gold-patch file
    proxy_fail  no patch, or the patch does not apply
    """
    patch = (metrics or {}).get("output_patch") or ""
    if not patch.strip():
        return "proxy_fail"
    if not metrics.get("patch_applies"):
        return "proxy_fail"

    gold_files = extract_patch_files(instance.get("patch", ""))
    output_files = extract_patch_files(patch)
    if gold_files & output_files:
        return "confirmed"
    return "proxy_pass"


def extract_patch_files(patch: str) -> set[str]:
    """Extract file paths from a unified diff."""
    files: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.add(parts[3].removeprefix("b/"))
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line[4:].split("\t")[0].strip()
            path = path.removeprefix("a/").removeprefix("b/")
            if path and path != "/dev/null":
                files.add(path)
    return files


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


REPORT_METRICS = (
    "tokens_total",
    "tokens_input",
    "tokens_output",
    "wall_time_sec",
    "tool_calls_total",
    "ast_mcp_calls",
)


def _success_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    good = sum(1 for r in records
               if r["success_proxy"] in ("confirmed", "proxy_pass"))
    return good / len(records)


def generate_report(results: list[dict], output_dir: Path) -> Path:
    """Generate the aggregate markdown report. Returns its path."""
    report_path = output_dir / "benchmark_report.md"

    by_condition: dict[str, list[dict]] = {}
    for record in results:
        by_condition.setdefault(record["condition"], []).append(record)

    def metrics_of(condition: str) -> list[dict]:
        return [r["metrics"] for r in by_condition.get(condition, [])
                if r.get("metrics")]

    lines = [
        "# Benchmark Report: ast-context-cache vs Control",
        "",
        f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z",
        f"Results: {len(results)} runs across "
        f"{len({r['instance_id'] for r in results})} tasks",
        "",
        "## Per-Task Metrics",
        "",
        "| instance_id | condition | tokens_total | tokens_in | tokens_out | "
        "wall_time_sec | tool_calls | ast_mcp_calls | patch_applies | success_proxy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for record in results:
        metrics = record.get("metrics") or {}
        lines.append(
            f"| {record['instance_id']} | {record['condition']} | "
            f"{metrics.get('tokens_total', 0)} | "
            f"{metrics.get('tokens_input', 0)} | "
            f"{metrics.get('tokens_output', 0)} | "
            f"{metrics.get('wall_time_sec', 0)} | "
            f"{metrics.get('tool_calls_total', 0)} | "
            f"{metrics.get('ast_mcp_calls', 0)} | "
            f"{metrics.get('patch_applies')} | "
            f"{record['success_proxy']} |"
        )

    lines += ["", "## Aggregate Summary (Mean)", "",
              "| metric | control_mean | treatment_mean | delta |",
              "|---|---|---|---|"]
    for metric in REPORT_METRICS:
        control_vals = [m.get(metric, 0) for m in metrics_of("control")]
        treatment_vals = [m.get(metric, 0) for m in metrics_of("treatment")]
        c_mean = _mean(control_vals)
        t_mean = _mean(treatment_vals)
        lines.append(f"| {metric} | {c_mean:.1f} | {t_mean:.1f} | "
                     f"{t_mean - c_mean:+.1f} |")

    lines += ["", "## Aggregate Summary (Median)", "",
              "| metric | control_median | treatment_median | delta |",
              "|---|---|---|---|"]
    for metric in REPORT_METRICS:
        control_vals = [m.get(metric, 0) for m in metrics_of("control")]
        treatment_vals = [m.get(metric, 0) for m in metrics_of("treatment")]
        c_med = median(control_vals)
        t_med = median(treatment_vals)
        lines.append(f"| {metric} | {c_med:.1f} | {t_med:.1f} | "
                     f"{t_med - c_med:+.1f} |")

    lines += ["", "## Success Rate", ""]
    for condition in CONDITIONS:
        records = by_condition.get(condition, [])
        if not records:
            lines.append(f"- **{condition}**: no results")
            continue
        counts = {
            verdict: sum(1 for r in records if r["success_proxy"] == verdict)
            for verdict in ("confirmed", "proxy_pass", "proxy_fail", "unverified")
        }
        rate = _success_rate(records)
        lines.append(
            f"- **{condition}**: {rate:.0%} pass proxy "
            f"({counts['confirmed']} confirmed, {counts['proxy_pass']} proxy_pass, "
            f"{counts['proxy_fail']} proxy_fail, {counts['unverified']} unverified) "
            f"of {len(records)}"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `confirmed` = patch applies cleanly and touches a file the gold patch touches.",
        "- `proxy_pass` = patch applies cleanly but touches no gold-patch file.",
        "- `proxy_fail` = no patch, or the patch does not apply to the base commit.",
        "- `unverified` = the opencode run produced no usable metrics.",
        "- Docker is not used: the official SWE-bench harness is required for test-verified "
        "success rates (infra follow-up).",
        f"- Raw per-run metrics: {output_dir / 'raw_sessions'}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def check_ast_mcp_health() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:7821/health", timeout=5) as resp:
            return resp.getcode() == 200
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SWE-bench Lite benchmark runner")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated instance_ids (runs in the order given)")
    parser.add_argument("--task-count", type=int, default=5,
                        help="Number of tasks to take from the dataset (default: 5)")
    parser.add_argument("--model", default="litellm/free",
                        help="opencode model id (default: litellm/free)")
    parser.add_argument("--output-dir", default="benchmarks/runs/latest",
                        help="Output directory (default: benchmarks/runs/latest)")
    parser.add_argument("--dataset", default=None,
                        help="Path to the SWE-bench Lite JSONL dataset")
    parser.add_argument("--fetch-dataset", action="store_true",
                        help="Download the dataset to the default path and exit")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-run opencode timeout in seconds (default: 900)")
    parser.add_argument("--context-limit", type=int, default=131072,
                        help="Context window advertised to opencode (default: 131072)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Set up condition directories and print the configs, "
                             "without invoking opencode")
    args = parser.parse_args(argv)

    if args.fetch_dataset:
        fetch_dataset(DEFAULT_DATASET)
        return 0

    dataset_path = resolve_dataset(args.dataset)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}")
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}")
        print("Run `python3 bench_runner.py --fetch-dataset` first.")
        return 1
    records = load_dataset(dataset_path)
    print(f"Dataset has {len(records)} instances")

    task_ids = None
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    tasks = select_tasks(records, args.task_count, task_ids)
    if not tasks:
        print("ERROR: no tasks selected")
        return 1
    print(f"Selected {len(tasks)} task(s)")

    healthy = check_ast_mcp_health()
    print(f"ast-mcp health check: {'200' if healthy else 'UNREACHABLE'}")
    if not healthy:
        print("WARNING: ast-mcp is not reachable at localhost:7821; the treatment "
              "condition will have no ast-context-cache tools.")

    results: list[dict] = []
    raw_sessions_dir = output_dir / "raw_sessions"
    raw_sessions_dir.mkdir(exist_ok=True)

    for index, instance in enumerate(tasks, 1):
        iid = instance["instance_id"]
        print(f"\n{'=' * 70}\nTask {index}/{len(tasks)}: {iid}")
        task_base = output_dir / "tasks" / iid
        task_base.mkdir(parents=True, exist_ok=True)

        for condition in CONDITIONS:
            print(f"\n  [{condition}] setting up ...")
            setup = setup_condition(task_base, instance, condition,
                                    args.model, args.context_limit)
            (task_base / condition / "config_snapshot.json").write_text(
                json.dumps(setup["config"], indent=2), encoding="utf-8"
            )

            if args.dry_run:
                print(json.dumps(setup["config"]["mcp"], indent=2))
                results.append({
                    "instance_id": iid,
                    "condition": condition,
                    "metrics": None,
                    "success_proxy": "unverified",
                })
                continue

            print(f"  [{condition}] running with model {args.model} ...")
            metrics = run_opencode(
                setup["repo_dir"], setup["config_home"], instance,
                args.model, args.timeout,
            )

            if metrics is None:
                print(f"  [{condition}] FAILED — no metrics collected")
                success_proxy = "unverified"
            else:
                patch = collect_patch(setup["repo_dir"])
                metrics["output_patch"] = patch
                metrics["patch_applies"] = patch_applies(setup["repo_dir"], patch)
                metrics["patch_files"] = sorted(extract_patch_files(patch))
                success_proxy = evaluate_success_proxy(instance, metrics)
                print(
                    f"  [{condition}] tokens={metrics['tokens_total']} "
                    f"time={metrics['wall_time_sec']}s "
                    f"tools={metrics['tool_calls_total']} "
                    f"ast={metrics['ast_mcp_calls']} "
                    f"patch={'yes' if patch.strip() else 'no'} "
                    f"applies={metrics['patch_applies']} "
                    f"success={success_proxy}"
                )

            results.append({
                "instance_id": iid,
                "condition": condition,
                "metrics": metrics,
                "success_proxy": success_proxy,
            })

            if metrics:
                raw_path = raw_sessions_dir / f"raw_{iid}_{condition}.json"
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "metrics": metrics,
                        "instance": {
                            "instance_id": iid,
                            "repo": instance["repo"],
                            "base_commit": instance["base_commit"],
                            "problem_statement": instance.get("problem_statement", ""),
                        },
                    }, f, indent=2)

    print(f"\n{'=' * 70}\nGenerating outputs ...")
    jsonl_path = output_dir / "benchmark_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    print(f"JSONL results: {jsonl_path}")

    report = generate_report(results, output_dir)
    print(f"Markdown report: {report}")
    print(f"\nDone. {len(results)} result(s) across {len(tasks)} task(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
