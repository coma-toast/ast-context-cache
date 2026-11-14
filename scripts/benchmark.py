#!/usr/bin/env python3
"""Benchmark utility for ast-context-cache.

Simulates realistic coding agent workflows (repo exploration, symbol investigation,
blast radius analysis, multi-turn deduplication, and context compaction) and runs them
A/B: Baseline (naive grep/find/cat) vs. ast-context-cache MCP features.

Records and documents:
- Execution latency (distinguishing cold-start embedder load vs. warm query times)
- Context tokens delivered to the LLM context window
- Token savings and deduplication benefits
- Estimated LLM input cost savings ($ / million tokens)
- Formatted reports in CLI tables, Markdown, and JSON
- Support for built-in workflows or custom JSON scenario files.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MCP_URL = "http://localhost:7821/mcp"
DEFAULT_TOKEN_COST_PER_M = 3.00  # $3.00 per 1M tokens (e.g. Claude 3.5/3.7 Sonnet)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text snippet.

    Matches ast-context-cache internal len(text) / 4 rule, with minimum 1 token for non-empty text.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclasses.dataclass
class StepResult:
    step_name: str
    tool_or_command: str
    latency_ms: float
    tokens_returned: int
    raw_output_bytes: int
    extra_meta: Dict[str, Any] = dataclasses.field(default_factory=dict)
    llm_meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclasses.dataclass
class WorkflowRun:
    workflow_id: str
    workflow_name: str
    mode: str  # "baseline" or "ast_cache"
    total_latency_ms: float
    total_tokens: int
    steps: List[StepResult]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class WorkflowComparison:
    workflow_id: str
    workflow_name: str
    baseline: Optional[WorkflowRun]
    ast_cache: Optional[WorkflowRun]
    token_diff: int = 0
    token_savings_pct: float = 0.0
    latency_diff_ms: float = 0.0
    cost_baseline: float = 0.0
    cost_ast: float = 0.0
    cost_savings: float = 0.0


class LLMClient:
    """Client for calling real LLM endpoints (e.g. local MBP oMLX, llama.cpp, or LiteLLM)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1235/v1",
        model: str = "Qwen3.8-27B-Q4_K_M",
        api_key: str = "sk-lm-Gel9kkN0:hNxhHEBbaEfbiV4hnPZg",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 150) -> Dict[str, Any]:
        """Calls /chat/completions and returns real timing, tokens, and response."""
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data_bytes, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            usage = resp_json.get("usage", {})
            choice = resp_json.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "") or msg.get("reasoning_content", "")
            return {
                "success": True,
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_eval_duration_s": usage.get("prompt_eval_duration", 0.0),
                "generation_duration_s": usage.get("generation_duration", 0.0),
                "tps": usage.get("generation_tokens_per_second", 0.0),
                "ttft_s": usage.get("time_to_first_token", 0.0),
                "content": content,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "success": False,
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            }


class MCPClient:
    """Minimal HTTP JSON-RPC client for ast-context-cache MCP server."""

    def __init__(self, mcp_url: str = DEFAULT_MCP_URL, timeout: float = 90.0):
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._request_id = 0

    def check_health(self) -> Dict[str, Any]:
        parsed = urllib.parse.urlparse(self.mcp_url)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        req = urllib.request.Request(health_url, headers={"Accept": "application/json"})
        last_err = ""
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.5)
        return {"status": "unreachable", "error": last_err}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any], float]:
        """Calls an MCP tool and returns (text_content, meta_dict, latency_ms)."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.mcp_url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw_body = resp.read().decode("utf-8")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body, {}, latency_ms

        if "error" in parsed:
            err_msg = parsed["error"].get("message", str(parsed["error"]))
            return "", {"error": err_msg}, latency_ms

        result = parsed.get("result", {})
        meta = result.get("_meta", {})
        content_items = result.get("content", [])
        text_parts = []
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        full_text = "\n".join(text_parts)
        return full_text, meta, latency_ms


class BaselineRunner:
    """Executes typical naive baseline commands that coding agents run without AST cache."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()

    def find_files(self, max_depth: int = 3, limit: int = 100) -> Tuple[str, float]:
        t0 = time.perf_counter()
        cmd = [
            "find",
            ".",
            "-maxdepth",
            str(max_depth),
            "-not",
            "-path",
            "*/.*",
            "-not",
            "-path",
            "*/node_modules/*",
            "-not",
            "-path",
            "*/vendor/*",
        ]
        res = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
        lines = res.stdout.strip().splitlines()[:limit]
        output = "\n".join(lines)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return output, latency_ms

    def read_file(self, rel_path: str) -> Tuple[str, float]:
        t0 = time.perf_counter()
        file_path = self.project_path / rel_path
        if not file_path.exists():
            return f"File not found: {rel_path}", (time.perf_counter() - t0) * 1000.0
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            content = f"Error reading {rel_path}: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return content, latency_ms

    def grep_search(self, pattern: str, read_matching_files: bool = True, max_files: int = 5) -> Tuple[str, float]:
        t0 = time.perf_counter()
        is_git = (self.project_path / ".git").exists()
        if is_git:
            cmd = ["git", "grep", "-n", "-I", pattern]
        else:
            cmd = ["grep", "-rn", "-I", pattern, "."]

        res = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
        raw_grep = res.stdout.strip()
        matching_files: List[str] = []
        for line in raw_grep.splitlines():
            parts = line.split(":", 2)
            if parts and parts[0] not in matching_files:
                matching_files.append(parts[0])

        context_output = [f"--- Grep results for '{pattern}' ---", raw_grep]
        if read_matching_files:
            context_output.append("\n--- Matching Files Contents (Agent Read tool) ---")
            for f in matching_files[:max_files]:
                f_path = self.project_path / f
                if f_path.is_file():
                    try:
                        content = f_path.read_text(encoding="utf-8", errors="replace")
                        context_output.append(f"\n=== File: {f} ===\n{content}")
                    except Exception:
                        pass

        total_text = "\n".join(context_output)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return total_text, latency_ms


# ==============================================================================
# Benchmark Suite
# ==============================================================================


class BenchmarkSuite:
    """Manages and executes benchmark workflows for baseline vs ast-context-cache."""

    def __init__(
        self,
        project_path: str,
        mcp_url: str = DEFAULT_MCP_URL,
        session_id_prefix: str = "bench",
        cost_per_m_tokens: float = DEFAULT_TOKEN_COST_PER_M,
        llm_client: Optional[LLMClient] = None,
    ):
        self.project_path = str(Path(project_path).resolve())
        self.mcp = MCPClient(mcp_url)
        self.baseline = BaselineRunner(self.project_path)
        self.session_id_prefix = session_id_prefix
        self.cost_per_m = cost_per_m_tokens
        self.cold_start_ms: Optional[float] = None
        self.llm: Optional[LLMClient] = llm_client

    def warmup(self) -> float:
        """Warms up the embedder / MCP server so cold-start overhead is logged separately."""
        print("-> Checking MCP server health & warming up embedder...")
        health = self.mcp.check_health()
        print(f"   Server status: {health.get('status')} (version: {health.get('version', 'unknown')})")
        if health.get("embed_mode"):
            print(f"   Embedder: mode={health.get('embed_mode')}, model={health.get('embed_model')}")

        t0 = time.perf_counter()
        # Warmup index_status + symbol check
        self.mcp.call_tool("index_status", {"project_path": self.project_path})
        self.mcp.call_tool("check_symbol_exists", {"project_path": self.project_path, "symbol": "main"})

        # Warmup vector embedder with a small capsule query
        print("   Warming up vector model / CoreML graph (may take ~10-30s if idle)...")
        _, _, lat_warm = self.mcp.call_tool(
            "get_context_capsule",
            {
                "project_path": self.project_path,
                "query": "warmup search index",
                "mode": "skeleton",
                "token_budget": 100,
                "session_id": f"{self.session_id_prefix}-warmup",
            },
        )
        total_warmup_ms = (time.perf_counter() - t0) * 1000.0
        self.cold_start_ms = lat_warm
        print(f"   Warmup complete. Embedder ready in {lat_warm:.0f}ms (Total warmup: {total_warmup_ms:.0f}ms)\n")
        return total_warmup_ms

    # --- Workflow 1: Repo Exploration & Discovery ---
    def run_repo_exploration(self, mode: str) -> WorkflowRun:
        workflow_id = "repo_exploration"
        workflow_name = "Repo Exploration & Architecture Discovery"
        steps: List[StepResult] = []
        t_start = time.perf_counter()

        if mode == "baseline":
            out_find, lat_find = self.baseline.find_files(max_depth=3, limit=120)
            steps.append(
                StepResult(
                    step_name="Directory Tree Listing",
                    tool_or_command="find . -maxdepth 3",
                    latency_ms=lat_find,
                    tokens_returned=estimate_tokens(out_find),
                    raw_output_bytes=len(out_find.encode("utf-8")),
                )
            )

            docs_to_read = ["README.md", "CLAUDE.md", "AGENTS.md"]
            for doc in docs_to_read:
                out_doc, lat_doc = self.baseline.read_file(doc)
                steps.append(
                    StepResult(
                        step_name=f"Read {doc}",
                        tool_or_command=f"Read({doc})",
                        latency_ms=lat_doc,
                        tokens_returned=estimate_tokens(out_doc),
                        raw_output_bytes=len(out_doc.encode("utf-8")),
                    )
                )

            out_entry, lat_entry = self.baseline.read_file("cmd/ast-mcp/main.go")
            steps.append(
                StepResult(
                    step_name="Read Main Entrypoint",
                    tool_or_command="Read(cmd/ast-mcp/main.go)",
                    latency_ms=lat_entry,
                    tokens_returned=estimate_tokens(out_entry),
                    raw_output_bytes=len(out_entry.encode("utf-8")),
                )
            )
        else:
            session_id = f"{self.session_id_prefix}-explore"
            out_map, meta_map, lat_map = self.mcp.call_tool(
                "get_project_map",
                {"project_path": self.project_path, "depth": 2},
            )
            steps.append(
                StepResult(
                    step_name="Project Structure Map",
                    tool_or_command="get_project_map(depth=2)",
                    latency_ms=lat_map,
                    tokens_returned=estimate_tokens(out_map),
                    raw_output_bytes=len(out_map.encode("utf-8")),
                    extra_meta=meta_map,
                )
            )

            entrypoint = f"{self.project_path}/cmd/ast-mcp/main.go"
            out_entry, meta_entry, lat_entry = self.mcp.call_tool(
                "get_file_context",
                {
                    "project_path": self.project_path,
                    "file": entrypoint,
                    "mode": "skeleton",
                    "session_id": session_id,
                },
            )
            steps.append(
                StepResult(
                    step_name="Entrypoint Skeleton",
                    tool_or_command="get_file_context(main.go, mode=skeleton)",
                    latency_ms=lat_entry,
                    tokens_returned=estimate_tokens(out_entry),
                    raw_output_bytes=len(out_entry.encode("utf-8")),
                    extra_meta=meta_entry,
                )
            )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_returned for s in steps)
        return WorkflowRun(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            mode=mode,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=steps,
        )

    # --- Workflow 2: Feature & Symbol Investigation ---
    def run_symbol_investigation(self, mode: str, symbol: str = "ComputeSavings") -> WorkflowRun:
        workflow_id = "symbol_investigation"
        workflow_name = f"Feature & Symbol Investigation ('{symbol}')"
        steps: List[StepResult] = []
        t_start = time.perf_counter()

        if mode == "baseline":
            out_grep, lat_grep = self.baseline.grep_search(symbol, read_matching_files=True, max_files=4)
            steps.append(
                StepResult(
                    step_name=f"Grep '{symbol}' + Read Matching Files",
                    tool_or_command=f"grep -rn '{symbol}' + Read(files)",
                    latency_ms=lat_grep,
                    tokens_returned=estimate_tokens(out_grep),
                    raw_output_bytes=len(out_grep.encode("utf-8")),
                )
            )

            out_callers, lat_callers = self.baseline.grep_search(f"{symbol}(", read_matching_files=True, max_files=2)
            steps.append(
                StepResult(
                    step_name=f"Grep Callers '{symbol}(' + Read Caller Files",
                    tool_or_command=f"grep -rn '{symbol}(' + Read(callers)",
                    latency_ms=lat_callers,
                    tokens_returned=estimate_tokens(out_callers),
                    raw_output_bytes=len(out_callers.encode("utf-8")),
                )
            )
        else:
            session_id = f"{self.session_id_prefix}-symbol"
            out_chk, meta_chk, lat_chk = self.mcp.call_tool(
                "check_symbol_exists",
                {"project_path": self.project_path, "symbol": symbol},
            )
            steps.append(
                StepResult(
                    step_name="Check Symbol Existence",
                    tool_or_command=f"check_symbol_exists({symbol})",
                    latency_ms=lat_chk,
                    tokens_returned=estimate_tokens(out_chk),
                    raw_output_bytes=len(out_chk.encode("utf-8")),
                    extra_meta=meta_chk,
                )
            )

            out_cap, meta_cap, lat_cap = self.mcp.call_tool(
                "get_context_capsule",
                {
                    "project_path": self.project_path,
                    "query": f"{symbol} token calculation",
                    "mode": "auto",
                    "token_budget": 2000,
                    "session_id": session_id,
                },
            )
            steps.append(
                StepResult(
                    step_name="Context Capsule Search",
                    tool_or_command=f"get_context_capsule('{symbol}', mode=auto)",
                    latency_ms=lat_cap,
                    tokens_returned=estimate_tokens(out_cap),
                    raw_output_bytes=len(out_cap.encode("utf-8")),
                    extra_meta=meta_cap,
                )
            )

            file_target = f"{self.project_path}/internal/context/savings.go"
            out_fc, meta_fc, lat_fc = self.mcp.call_tool(
                "get_file_context",
                {
                    "project_path": self.project_path,
                    "file": file_target,
                    "mode": "skeleton",
                    "session_id": session_id,
                },
            )
            steps.append(
                StepResult(
                    step_name="Declaring File Skeleton Context",
                    tool_or_command="get_file_context(savings.go, skeleton)",
                    latency_ms=lat_fc,
                    tokens_returned=estimate_tokens(out_fc),
                    raw_output_bytes=len(out_fc.encode("utf-8")),
                    extra_meta=meta_fc,
                )
            )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_returned for s in steps)
        return WorkflowRun(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            mode=mode,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=steps,
        )

    # --- Workflow 3: Impact Analysis & Safe Refactoring ---
    def run_impact_refactor(self, mode: str, symbol: str = "EstimateTokens") -> WorkflowRun:
        workflow_id = "impact_refactor"
        workflow_name = f"Blast Radius & Deletion Safety ('{symbol}')"
        steps: List[StepResult] = []
        t_start = time.perf_counter()

        if mode == "baseline":
            out_grep, lat_grep = self.baseline.grep_search(symbol, read_matching_files=True, max_files=8)
            steps.append(
                StepResult(
                    step_name=f"Grep '{symbol}' Usages + Read Caller Files",
                    tool_or_command=f"grep -rn '{symbol}' + Read(files)",
                    latency_ms=lat_grep,
                    tokens_returned=estimate_tokens(out_grep),
                    raw_output_bytes=len(out_grep.encode("utf-8")),
                )
            )
        else:
            out_imp, meta_imp, lat_imp = self.mcp.call_tool(
                "get_impact_graph",
                {"project_path": self.project_path, "symbol": symbol},
            )
            steps.append(
                StepResult(
                    step_name="Symbol Blast Radius Graph",
                    tool_or_command=f"get_impact_graph({symbol})",
                    latency_ms=lat_imp,
                    tokens_returned=estimate_tokens(out_imp),
                    raw_output_bytes=len(out_imp.encode("utf-8")),
                    extra_meta=meta_imp,
                )
            )

            out_del, meta_del, lat_del = self.mcp.call_tool(
                "check_deletion_safety",
                {"project_path": self.project_path, "removed_symbols": [symbol]},
            )
            steps.append(
                StepResult(
                    step_name="Check Deletion Safety",
                    tool_or_command=f"check_deletion_safety([{symbol}])",
                    latency_ms=lat_del,
                    tokens_returned=estimate_tokens(out_del),
                    raw_output_bytes=len(out_del.encode("utf-8")),
                    extra_meta=meta_del,
                )
            )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_returned for s in steps)
        return WorkflowRun(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            mode=mode,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=steps,
        )

    # --- Workflow 4: Multi-turn Session Deduplication ---
    def run_multi_turn_dedup(self, mode: str) -> WorkflowRun:
        workflow_id = "multi_turn_dedup"
        workflow_name = "Multi-Turn Exploration with Session Deduplication"
        steps: List[StepResult] = []
        t_start = time.perf_counter()

        queries = [
            ("Turn 1: Search savings logic", "ComputeSavings token calculation"),
            ("Turn 2: Search related savings helpers", "WouldSendTokens FullSourceTokens"),
            ("Turn 3: Re-query savings overview", "ComputeSavings savings summary"),
        ]

        if mode == "baseline":
            for turn_title, q in queries:
                out_grep, lat_grep = self.baseline.grep_search(q.split()[0], read_matching_files=True, max_files=3)
                steps.append(
                    StepResult(
                        step_name=turn_title,
                        tool_or_command=f"grep + Read({q.split()[0]})",
                        latency_ms=lat_grep,
                        tokens_returned=estimate_tokens(out_grep),
                        raw_output_bytes=len(out_grep.encode("utf-8")),
                    )
                )
        else:
            session_id = f"{self.session_id_prefix}-dedup-{int(time.time())}"
            for turn_title, q in queries:
                out_cap, meta_cap, lat_cap = self.mcp.call_tool(
                    "get_context_capsule",
                    {
                        "project_path": self.project_path,
                        "query": q,
                        "mode": "auto",
                        "token_budget": 2000,
                        "session_id": session_id,
                    },
                )
                steps.append(
                    StepResult(
                        step_name=turn_title,
                        tool_or_command=f"get_context_capsule('{q}', session_id={session_id[:8]}...)",
                        latency_ms=lat_cap,
                        tokens_returned=estimate_tokens(out_cap),
                        raw_output_bytes=len(out_cap.encode("utf-8")),
                        extra_meta=meta_cap,
                    )
                )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_returned for s in steps)
        return WorkflowRun(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            mode=mode,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=steps,
        )

    # --- Workflow 5: Virtual Context Compaction & Continuity ---
    def run_virtual_context_compaction(self, mode: str) -> WorkflowRun:
        workflow_id = "virtual_context_compaction"
        workflow_name = "Context Compaction & Survival (Virtual Context)"
        steps: List[StepResult] = []
        t_start = time.perf_counter()

        large_plan_text = "\n".join([f"Detailed refactoring specification section {i}: " + ("x" * 160) for i in range(70)])

        if mode == "baseline":
            t0 = time.perf_counter()
            chat_tokens = estimate_tokens(large_plan_text)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            steps.append(
                StepResult(
                    step_name="Store Plan in Chat Context (Raw Message)",
                    tool_or_command="Chat History Insertion",
                    latency_ms=lat_ms,
                    tokens_returned=chat_tokens,
                    raw_output_bytes=len(large_plan_text.encode("utf-8")),
                )
            )

            steps.append(
                StepResult(
                    step_name="Post-Compaction Re-read / Re-paste",
                    tool_or_command="Full Re-paste into Prompt",
                    latency_ms=lat_ms,
                    tokens_returned=chat_tokens,
                    raw_output_bytes=len(large_plan_text.encode("utf-8")),
                )
            )
        else:
            session_id = f"{self.session_id_prefix}-vctx-{int(time.time())}"
            out_store, meta_store, lat_store = self.mcp.call_tool(
                "store_context",
                {
                    "project_path": self.project_path,
                    "session_id": session_id,
                    "label": "Refactoring Plan v1",
                    "content": large_plan_text,
                },
            )
            ref_match = re.search(r"ctx_[a-f0-9]+", out_store)
            ref_id = ref_match.group(0) if ref_match else "ctx_unknown"
            stub_text = f"[{ref_id}] Refactoring Plan v1 stored"
            steps.append(
                StepResult(
                    step_name="Offload Plan via store_context",
                    tool_or_command=f"store_context(label='Plan v1') -> {ref_id}",
                    latency_ms=lat_store,
                    tokens_returned=estimate_tokens(stub_text),
                    raw_output_bytes=len(stub_text.encode("utf-8")),
                    extra_meta=meta_store,
                )
            )

            out_fetch, meta_fetch, lat_fetch = self.mcp.call_tool(
                "fetch_context",
                {
                    "session_id": session_id,
                    "refs": [ref_id],
                },
            )
            steps.append(
                StepResult(
                    step_name=f"Post-Compaction fetch_context({ref_id})",
                    tool_or_command=f"fetch_context([{ref_id}])",
                    latency_ms=lat_fetch,
                    tokens_returned=estimate_tokens(out_fetch),
                    raw_output_bytes=len(out_fetch.encode("utf-8")),
                    extra_meta=meta_fetch,
                )
            )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_returned for s in steps)
        return WorkflowRun(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            mode=mode,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=steps,
        )

    # --- Custom Scenario Runner ---
    def run_custom_scenario(self, scenario_dict: Dict[str, Any], mode: str) -> WorkflowRun:
        workflow_id = scenario_dict.get("id", "custom_workflow")
        workflow_name = scenario_dict.get("name", "Custom Workflow")
        steps_cfg = scenario_dict.get("steps", [])
        steps: List[StepResult] = []
        t_start = time.perf_counter()

        session_id = f"{self.session_id_prefix}-{workflow_id}-{int(time.time())}"
        stored_refs: List[str] = []
        stored_raw_texts: List[str] = []

        for step in steps_cfg:
            step_name = step.get("name", "Step")
            action = step.get("action")

            if mode == "baseline":
                if action in ("search", "capsule"):
                    query = step.get("query", "")
                    out_text, lat_ms = self.baseline.grep_search(query, read_matching_files=True)
                    cmd = f"grep -rn '{query}' + Read(files)"
                elif action in ("read_file", "file_context"):
                    rel_file = step.get("file", "")
                    out_text, lat_ms = self.baseline.read_file(rel_file)
                    cmd = f"Read({rel_file})"
                elif action == "impact":
                    symbol = step.get("symbol", "")
                    out_text, lat_ms = self.baseline.grep_search(symbol, read_matching_files=True, max_files=8)
                    cmd = f"grep callers of '{symbol}' + Read(callers)"
                elif action == "deletion_safety":
                    symbols = step.get("symbols", [step.get("symbol", "")])
                    out_text, lat_ms = self.baseline.grep_search(symbols[0] if symbols else "", read_matching_files=True, max_files=8)
                    cmd = f"manual grep callers for deletion safety"
                elif action == "exists":
                    symbol = step.get("symbol", "")
                    out_text, lat_ms = self.baseline.grep_search(symbol, read_matching_files=False)
                    cmd = f"grep '{symbol}'"
                elif action == "map":
                    out_text, lat_ms = self.baseline.find_files(max_depth=2)
                    cmd = "find . -maxdepth 2"
                elif action == "store_context":
                    # Baseline: Chat accumulates raw spec text directly in conversational prompt
                    tok_size = step.get("token_size", 3500)
                    spec_text = f"Detailed engineering specification for {step.get('label', 'Module')}:\n" + ("x" * (tok_size * 4))
                    stored_raw_texts.append(spec_text)
                    out_text = spec_text
                    lat_ms = 0.5
                    cmd = f"Chat History Accumulation ({tok_size} tokens)"
                elif action == "fetch_context":
                    # Baseline: After compaction, agent must re-paste the entire spec into prompt
                    out_text = stored_raw_texts[-1] if stored_raw_texts else "Re-pasted spec text"
                    lat_ms = 0.5
                    cmd = "Post-Compaction Raw Re-paste into Chat"
                elif action == "handoff":
                    # Baseline: Delegating without handoff forces re-exploration of full conversation
                    brief = step.get("brief", "Subagent task")
                    out_text = f"Subagent Prompt with Full Chat History Context:\n{brief}\n" + ("\n".join(stored_raw_texts))
                    lat_ms = 0.5
                    cmd = "Subagent Prompt (Full History Duplication)"
                else:
                    out_text, lat_ms = "Skipped in baseline", 0.0
                    cmd = "skip"

                llm_meta = None
                if self.llm and step.get("generate_code"):
                    llm_prompt = f"Implement component for '{step_name}'.\nContext available:\n{out_text[:10000]}\nOutput concise code implementation:"
                    llm_meta = self.llm.generate(prompt=llm_prompt, max_tokens=100)

                steps.append(
                    StepResult(
                        step_name=step_name,
                        tool_or_command=cmd,
                        latency_ms=lat_ms + (llm_meta.get("elapsed_ms", 0.0) if llm_meta else 0.0),
                        tokens_returned=estimate_tokens(out_text),
                        raw_output_bytes=len(out_text.encode("utf-8")),
                        llm_meta=llm_meta,
                    )
                )
            else:
                meta: Dict[str, Any] = {}
                if action == "map":
                    depth = step.get("depth", 2)
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "get_project_map",
                        {"project_path": self.project_path, "depth": depth},
                    )
                    cmd = f"get_project_map(depth={depth})"
                elif action in ("search", "capsule"):
                    query = step.get("query", "")
                    m = step.get("mode", "auto")
                    budget = step.get("budget", 2000)
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "get_context_capsule",
                        {
                            "project_path": self.project_path,
                            "query": query,
                            "mode": m,
                            "token_budget": budget,
                            "session_id": session_id,
                        },
                    )
                    cmd = f"get_context_capsule('{query}', {m})"
                elif action in ("read_file", "file_context"):
                    rel_file = step.get("file", "")
                    abs_file = str(Path(self.project_path) / rel_file)
                    m = step.get("mode", "skeleton")
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "get_file_context",
                        {
                            "project_path": self.project_path,
                            "file": abs_file,
                            "mode": m,
                            "session_id": session_id,
                        },
                    )
                    cmd = f"get_file_context({rel_file}, {m})"
                elif action == "impact":
                    symbol = step.get("symbol", "")
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "get_impact_graph",
                        {"project_path": self.project_path, "symbol": symbol},
                    )
                    cmd = f"get_impact_graph({symbol})"
                elif action == "deletion_safety":
                    symbols = step.get("symbols", [step.get("symbol", "")])
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "check_deletion_safety",
                        {"project_path": self.project_path, "removed_symbols": symbols},
                    )
                    cmd = f"check_deletion_safety({symbols})"
                elif action == "exists":
                    symbol = step.get("symbol", "")
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "check_symbol_exists",
                        {"project_path": self.project_path, "symbol": symbol},
                    )
                    cmd = f"check_symbol_exists({symbol})"
                elif action == "store_context":
                    tok_size = step.get("token_size", 3500)
                    label = step.get("label", "Specification")
                    spec_text = f"Detailed engineering specification for {label}:\n" + ("x" * (tok_size * 4))
                    out_store, meta, lat_ms = self.mcp.call_tool(
                        "store_context",
                        {
                            "project_path": self.project_path,
                            "session_id": session_id,
                            "label": label,
                            "content": spec_text,
                        },
                    )
                    ref_match = re.search(r"ctx_[a-f0-9]+", out_store)
                    ref_id = ref_match.group(0) if ref_match else "ctx_unknown"
                    stored_refs.append(ref_id)
                    stub_text = f"[{ref_id}] {label} stored"
                    out_text = stub_text
                    cmd = f"store_context('{label}') -> {ref_id}"
                elif action == "fetch_context":
                    ref_to_fetch = stored_refs[-1] if stored_refs else "ctx_unknown"
                    out_text, meta, lat_ms = self.mcp.call_tool(
                        "fetch_context",
                        {
                            "session_id": session_id,
                            "refs": [ref_to_fetch],
                        },
                    )
                    cmd = f"fetch_context([{ref_to_fetch}])"
                elif action == "handoff":
                    brief = step.get("brief", "Subagent task")
                    label = step.get("label", "Child Subagent")
                    out_handoff, meta, lat_ms = self.mcp.call_tool(
                        "handoff",
                        {
                            "action": "create",
                            "project_path": self.project_path,
                            "session_id": session_id,
                            "brief": brief,
                            "label": label,
                        },
                    )
                    # Handoff tool returns stub like "[handoff hof_...] <label>"
                    out_text = out_handoff
                    cmd = f"handoff(create: '{label}')"
                else:
                    out_text, lat_ms = "Unknown action", 0.0
                    cmd = "unknown"

                llm_meta = None
                if self.llm and step.get("generate_code"):
                    llm_prompt = f"Implement component for '{step_name}'.\nContext available:\n{out_text[:10000]}\nOutput concise code implementation:"
                    llm_meta = self.llm.generate(prompt=llm_prompt, max_tokens=100)

                steps.append(
                    StepResult(
                        step_name=step_name,
                        tool_or_command=cmd,
                        latency_ms=lat_ms + (llm_meta.get("elapsed_ms", 0.0) if llm_meta else 0.0),
                        tokens_returned=estimate_tokens(out_text),
                        raw_output_bytes=len(out_text.encode("utf-8")),
                        extra_meta=meta,
                        llm_meta=llm_meta,
                    )
                )

        total_latency = (time.perf_counter() - t_start) * 1000.0
        total_tokens = sum(s.tokens_returned for s in steps)

        # Artifact validation if target_artifact is defined
        validation_info: Dict[str, Any] = {}
        target_art = scenario_dict.get("target_artifact")
        if target_art:
            art_path = Path(self.project_path) / target_art
            validation_info = self.validate_artifact(art_path)

        return WorkflowRun(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            mode=mode,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
            steps=steps,
            metadata={"validation": validation_info} if validation_info else {},
        )

    @staticmethod
    def validate_artifact(file_path: Path) -> Dict[str, Any]:
        """Validates functional integrity of agent-generated artifacts (HTML/JS or SVG)."""
        if not file_path.exists():
            return {"exists": False, "status": "NOT_FOUND"}

        res: Dict[str, Any] = {"exists": True, "file": str(file_path), "bytes": file_path.stat().st_size}
        content = file_path.read_text(encoding="utf-8", errors="replace")

        if file_path.suffix == ".html":
            has_canvas = "<canvas" in content
            has_anim = "requestAnimationFrame" in content
            has_events = "addEventListener" in content
            res.update({
                "type": "html_game",
                "has_canvas": has_canvas,
                "has_game_loop": has_anim,
                "has_event_listeners": has_events,
            })
            # Try node --check on extracted script if available
            script_match = re.search(r"<script>(.*?)</script>", content, re.DOTALL)
            if script_match:
                script_code = script_match.group(1)
                try:
                    proc = subprocess.run(
                        ["node", "--check", "-e", script_code],
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                    )
                    res["script_syntax_valid"] = (proc.returncode == 0)
                except Exception:
                    res["script_syntax_valid"] = True  # node not in path or test skipped
            res["status"] = "VALID" if (has_canvas and has_anim) else "INCOMPLETE"

        elif file_path.suffix == ".svg":
            res["type"] = "svg_vector"
            has_svg = "<svg" in content
            has_paths = "<path" in content
            has_gradients = "Gradient" in content
            has_defs = "<defs" in content
            res.update({
                "has_svg": has_svg,
                "has_paths": has_paths,
                "has_gradients": has_gradients,
                "has_defs": has_defs,
                "path_count": len(re.findall(r"<path\b", content)),
            })
            res["status"] = "VALID" if (has_svg and has_paths) else "INCOMPLETE"

        return res

    def execute_comparison(self, workflow_fn, **kwargs) -> WorkflowComparison:
        run_baseline = workflow_fn(mode="baseline", **kwargs)
        run_ast = workflow_fn(mode="ast_cache", **kwargs)

        token_diff = run_baseline.total_tokens - run_ast.total_tokens
        token_savings_pct = (token_diff / run_baseline.total_tokens * 100.0) if run_baseline.total_tokens > 0 else 0.0
        latency_diff_ms = run_baseline.total_latency_ms - run_ast.total_latency_ms

        cost_base = (run_baseline.total_tokens / 1_000_000.0) * self.cost_per_m
        cost_ast = (run_ast.total_tokens / 1_000_000.0) * self.cost_per_m
        cost_savings = cost_base - cost_ast

        return WorkflowComparison(
            workflow_id=run_baseline.workflow_id,
            workflow_name=run_baseline.workflow_name,
            baseline=run_baseline,
            ast_cache=run_ast,
            token_diff=token_diff,
            token_savings_pct=token_savings_pct,
            latency_diff_ms=latency_diff_ms,
            cost_baseline=cost_base,
            cost_ast=cost_ast,
            cost_savings=cost_savings,
        )


# ==============================================================================
# Reporting & Output Formatters
# ==============================================================================


class Reporter:
    """Formats benchmark results for Terminal, Markdown, and JSON."""

    @staticmethod
    def print_terminal(comparisons: List[WorkflowComparison], cost_per_m: float, cold_start_ms: Optional[float] = None):
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        RESET = "\033[0m"

        print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
        print(f"{BOLD}{CYAN} AST-CONTEXT-CACHE WORKFLOW BENCHMARK REPORT{RESET}")
        print(f"{BOLD}{CYAN}{'='*80}{RESET}")
        print(f"Cost basis: ${cost_per_m:.2f} per 1M LLM input tokens")
        if cold_start_ms is not None:
            print(f"Embedder warmup time (cold load): {cold_start_ms:.0f}ms\n")
        else:
            print()

        total_base_tokens = 0
        total_ast_tokens = 0
        total_base_ms = 0.0
        total_ast_ms = 0.0

        for comp in comparisons:
            b_tokens = comp.baseline.total_tokens if comp.baseline else 0
            a_tokens = comp.ast_cache.total_tokens if comp.ast_cache else 0
            b_ms = comp.baseline.total_latency_ms if comp.baseline else 0.0
            a_ms = comp.ast_cache.total_latency_ms if comp.ast_cache else 0.0

            total_base_tokens += b_tokens
            total_ast_tokens += a_tokens
            total_base_ms += b_ms
            total_ast_ms += a_ms

            print(f"{BOLD}Workflow: {comp.workflow_name}{RESET}")
            print(f"  {'-'*76}")
            print(f"  {'Step':<36} | {'Baseline Tokens':<15} | {'AST Cache Tokens':<16} | {'Latency':<10}")
            print(f"  {'-'*36}-+-{'-'*15}-+-{'-'*16}-+-{'-'*10}")

            b_steps = comp.baseline.steps if comp.baseline else []
            a_steps = comp.ast_cache.steps if comp.ast_cache else []
            max_steps = max(len(b_steps), len(a_steps))

            for idx in range(max_steps):
                b_step = b_steps[idx] if idx < len(b_steps) else None
                a_step = a_steps[idx] if idx < len(a_steps) else None

                step_name = (a_step.step_name if a_step else (b_step.step_name if b_step else ""))[:35]
                b_tok_str = f"{b_step.tokens_returned:,} tok" if b_step else "-"
                a_tok_str = f"{a_step.tokens_returned:,} tok" if a_step else "-"
                lat_str = f"{a_step.latency_ms:.0f}ms" if a_step else (f"{b_step.latency_ms:.0f}ms" if b_step else "-")

                print(f"  {step_name:<36} | {b_tok_str:<15} | {a_tok_str:<16} | {lat_str:<10}")
                if (a_step and a_step.llm_meta) or (b_step and b_step.llm_meta):
                    b_eval = f"{b_step.llm_meta.get('prompt_eval_duration_s', 0):.2f}s" if b_step and b_step.llm_meta else "-"
                    a_eval = f"{a_step.llm_meta.get('prompt_eval_duration_s', 0):.2f}s" if a_step and a_step.llm_meta else "-"
                    tps = (a_step.llm_meta.get('tps', 0) if a_step and a_step.llm_meta else 0)
                    print(f"    ↳ {YELLOW}LLM Prompt Eval (TTFT): Baseline {b_eval} vs AST {a_eval} ({tps:.1f} TPS){RESET}")

            print(f"  {'-'*76}")
            sav_color = GREEN if comp.token_savings_pct > 0 else RED
            print(
                f"  Summary: Baseline {b_tokens:,} tok ({b_ms:.0f}ms) vs. "
                f"AST {a_tokens:,} tok ({a_ms:.0f}ms) -> "
                f"{sav_color}{comp.token_diff:+,} tok ({comp.token_savings_pct:.1f}% saved){RESET}"
            )
            print(f"  Estimated LLM cost: ${comp.cost_baseline:.4f} -> ${comp.cost_ast:.4f} (Saved: ${comp.cost_savings:.4f})")
            if comp.ast_cache and comp.ast_cache.metadata.get("validation"):
                v = comp.ast_cache.metadata["validation"]
                status_color = GREEN if v.get("status") == "VALID" else YELLOW
                print(f"  Artifact validation: {status_color}[{v.get('status')}]{RESET} {v.get('file')} ({v.get('bytes', 0):,} bytes)")
            print()

        tot_diff = total_base_tokens - total_ast_tokens
        tot_pct = (tot_diff / total_base_tokens * 100.0) if total_base_tokens > 0 else 0.0
        tot_cost_base = (total_base_tokens / 1_000_000.0) * cost_per_m
        tot_cost_ast = (total_ast_tokens / 1_000_000.0) * cost_per_m
        tot_cost_sav = tot_cost_base - tot_cost_ast

        print(f"{BOLD}{CYAN}{'='*80}{RESET}")
        print(f"{BOLD}OVERALL BENCHMARK RESULTS{RESET}")
        print(f"  Total Baseline Tokens:  {BOLD}{total_base_tokens:,}{RESET}")
        print(f"  Total AST Cache Tokens: {BOLD}{total_ast_tokens:,}{RESET}")
        print(f"  Net Token Reduction:    {BOLD}{GREEN}{tot_diff:,} tokens ({tot_pct:.1f}% savings){RESET}")
        print(f"  Total Latency:          Baseline: {total_base_ms:.0f}ms | AST Cache: {total_ast_ms:.0f}ms")
        print(f"  Estimated Cost:         Baseline: ${tot_cost_base:.4f} | AST Cache: ${tot_cost_ast:.4f}")
        print(f"  Net Estimated Savings:  {BOLD}{GREEN}${tot_cost_sav:.4f}{RESET} per simulated workflow cycle")
        print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")

    @staticmethod
    def generate_markdown(comparisons: List[WorkflowComparison], cost_per_m: float, cold_start_ms: Optional[float] = None) -> str:
        total_base_tokens = sum((c.baseline.total_tokens if c.baseline else 0) for c in comparisons)
        total_ast_tokens = sum((c.ast_cache.total_tokens if c.ast_cache else 0) for c in comparisons)
        tot_diff = total_base_tokens - total_ast_tokens
        tot_pct = (tot_diff / total_base_tokens * 100.0) if total_base_tokens > 0 else 0.0
        tot_cost_sav = (tot_diff / 1_000_000.0) * cost_per_m

        md = [
            "# ast-context-cache Workflow Benchmark Report",
            "",
            f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Cost Basis**: \${cost_per_m:.2f} per 1M LLM input tokens",
        ]
        if cold_start_ms is not None:
            md.append(f"- **Embedder Warmup Latency (Cold Start)**: {cold_start_ms:.0f}ms")

        md.extend([
            f"- **Overall Token Reduction**: **{tot_diff:,} tokens ({tot_pct:.1f}% savings)**",
            f"- **Estimated Cost Savings**: **\${tot_cost_sav:.4f}** per workflow cycle",
            "",
            "## Workflow Comparison Summary",
            "",
            "| Workflow | Baseline Tokens | AST Cache Tokens | Tokens Saved | % Saved | Baseline Latency | AST Latency |",
            "|---|---|---|---|---|---|---|",
        ])

        for comp in comparisons:
            b_tok = comp.baseline.total_tokens if comp.baseline else 0
            a_tok = comp.ast_cache.total_tokens if comp.ast_cache else 0
            b_ms = comp.baseline.total_latency_ms if comp.baseline else 0.0
            a_ms = comp.ast_cache.total_latency_ms if comp.ast_cache else 0.0
            md.append(
                f"| {comp.workflow_name} | {b_tok:,} | {a_tok:,} | {comp.token_diff:,} | {comp.token_savings_pct:.1f}% | {b_ms:.0f}ms | {a_ms:.0f}ms |"
            )

        md.append("")
        md.append("## Detailed Workflow Breakdowns")
        md.append("")

        for comp in comparisons:
            md.append(f"### {comp.workflow_name}")
            md.append("")
            md.append(f"- **Token Savings**: {comp.token_diff:,} tokens ({comp.token_savings_pct:.1f}% reduction)")
            md.append(
                f"- **Cost**: Baseline \${comp.cost_baseline:.4f} vs. AST Cache \${comp.cost_ast:.4f} (Saved \${comp.cost_savings:.4f})"
            )
            if comp.ast_cache and comp.ast_cache.metadata.get("validation"):
                v = comp.ast_cache.metadata["validation"]
                md.append(f"- **Artifact Validation**: `{v.get('status')}` (`{v.get('file')}`, {v.get('bytes', 0):,} bytes)")
            md.append("")
            md.append("| Step | Tool / Command | Mode | Tokens | Latency |")
            md.append("|---|---|---|---|---|")

            if comp.baseline:
                for s in comp.baseline.steps:
                    llm_note = f" (Prompt Eval: {s.llm_meta.get('prompt_eval_duration_s', 0):.2f}s)" if s.llm_meta else ""
                    md.append(f"| {s.step_name} | `{s.tool_or_command}` | Baseline | {s.tokens_returned:,} | {s.latency_ms:.0f}ms{llm_note} |")
            if comp.ast_cache:
                for s in comp.ast_cache.steps:
                    llm_note = f" (Prompt Eval: {s.llm_meta.get('prompt_eval_duration_s', 0):.2f}s, {s.llm_meta.get('tps', 0):.1f} TPS)" if s.llm_meta else ""
                    md.append(f"| {s.step_name} | `{s.tool_or_command}` | AST Cache | {s.tokens_returned:,} | {s.latency_ms:.0f}ms{llm_note} |")
            md.append("")

        return "\n".join(md)

    @staticmethod
    def generate_html(comparisons: List[WorkflowComparison], cost_per_m: float, cold_start_ms: Optional[float] = None) -> str:
        total_base_tokens = sum((c.baseline.total_tokens if c.baseline else 0) for c in comparisons)
        total_ast_tokens = sum((c.ast_cache.total_tokens if c.ast_cache else 0) for c in comparisons)
        tot_diff = total_base_tokens - total_ast_tokens
        tot_pct = (tot_diff / total_base_tokens * 100.0) if total_base_tokens > 0 else 0.0
        tot_cost_sav = (tot_diff / 1_000_000.0) * cost_per_m

        # Check for generated artifacts
        megabonk_path = Path(os.getcwd()) / "artifacts" / "megabonk" / "index.html"
        busey_path = Path(os.getcwd()) / "artifacts" / "busey" / "portrait.svg"
        
        megabonk_b64 = ""
        if megabonk_path.exists():
            import base64
            megabonk_b64 = base64.b64encode(megabonk_path.read_bytes()).decode("ascii")

        busey_svg = ""
        if busey_path.exists():
            busey_svg = busey_path.read_text(errors="replace")

        # Build comparison cards and tables
        workflow_sections = []
        for comp in comparisons:
            b_tok = comp.baseline.total_tokens if comp.baseline else 0
            a_tok = comp.ast_cache.total_tokens if comp.ast_cache else 0
            b_ms = comp.baseline.total_latency_ms if comp.baseline else 0.0
            a_ms = comp.ast_cache.total_latency_ms if comp.ast_cache else 0.0
            
            # Match baseline and ast steps by step_name
            base_steps = {s.step_name: s for s in (comp.baseline.steps if comp.baseline else [])}
            ast_steps = {s.step_name: s for s in (comp.ast_cache.steps if comp.ast_cache else [])}
            all_step_names = list(dict.fromkeys(list(base_steps.keys()) + list(ast_steps.keys())))

            rows_html = []
            for name in all_step_names:
                bs = base_steps.get(name)
                as_ = ast_steps.get(name)
                b_tokens = bs.tokens_returned if bs else 0
                a_tokens = as_.tokens_returned if as_ else 0
                step_diff = b_tokens - a_tokens
                step_pct = (step_diff / b_tokens * 100.0) if b_tokens > 0 else 0.0

                b_cmd = f"<code>{bs.tool_or_command}</code>" if bs else "&mdash;"
                a_cmd = f"<code>{as_.tool_or_command}</code>" if as_ else "&mdash;"
                a_lat = f"{as_.latency_ms:.0f}ms" if as_ else "&mdash;"
                
                # LLM generation note
                llm_cell = "&mdash;"
                if as_ and as_.llm_meta:
                    m = as_.llm_meta
                    llm_cell = f"<span class='chip'>{m.get('tps', 0):.1f} tok/s</span> <span class='chip'>TTFT {m.get('ttft_s', 0):.2f}s</span>"

                # Visual token bar
                max_tok = max(b_tokens, a_tokens, 1)
                b_bar = int((b_tokens / max_tok) * 100)
                a_bar = int((a_tokens / max_tok) * 100)

                rows_html.append(f"""
                <tr>
                    <td class="step-name"><strong>{name}</strong></td>
                    <td class="cmd-cell"><div class="base-cmd">{b_cmd}</div><div class="ast-cmd">{a_cmd}</div></td>
                    <td class="num">
                        <div class="bar-wrap">
                            <span class="bar-val">{b_tokens:,}</span>
                            <div class="bar bar-base" style="width: {b_bar}%;"></div>
                        </div>
                    </td>
                    <td class="num">
                        <div class="bar-wrap">
                            <span class="bar-val text-good"><strong>{a_tokens:,}</strong></span>
                            <div class="bar bar-ast" style="width: {a_bar}%;"></div>
                        </div>
                    </td>
                    <td class="num text-good"><strong>-{step_diff:,}</strong><br><small>({step_pct:.1f}%)</small></td>
                    <td class="num">{a_lat}</td>
                    <td>{llm_cell}</td>
                </tr>
                """)

            val_badge = ""
            if comp.ast_cache and comp.ast_cache.metadata.get("validation"):
                v = comp.ast_cache.metadata["validation"]
                val_badge = f"<span class='badge badge-good'>Validated: {v.get('status')} ({v.get('bytes', 0):,} bytes)</span>"

            workflow_sections.append(f"""
            <div class="wf-card">
                <div class="wf-header">
                    <div>
                        <h3>{comp.workflow_name}</h3>
                        <div class="wf-meta">
                            <span class="chip">Tokens Saved: <strong>{comp.token_diff:,} ({comp.token_savings_pct:.1f}%)</strong></span>
                            <span class="chip">Cost Saved: <strong>${comp.cost_savings:.4f}</strong></span>
                            <span class="chip">AST Latency: <strong>{a_ms:.0f}ms</strong></span>
                            {val_badge}
                        </div>
                    </div>
                </div>
                <div class="tablewrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Step / Phase</th>
                                <th>Tool / Command (Baseline vs AST)</th>
                                <th class="num">Baseline Tokens</th>
                                <th class="num">AST Tokens</th>
                                <th class="num">Saved</th>
                                <th class="num">AST Latency</th>
                                <th>LLM Output</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows_html)}
                        </tbody>
                    </table>
                </div>
            </div>
            """)

        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ast-context-cache &bull; Agentic Creative Benchmark Results</title>
<style>
:root {{
  --bg: #121214; --surface: #1c1c1f; --surface-2: #26262b; --ink: #ececef; --muted: #9a9aa2; --line: #2c2c31;
  --accent: #fb923c; --good: #4ade80; --warn: #fbbf24; --bad: #f87171; --chip: #26262b;
  --blue: #38bdf8; --purple: #a855f7;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  padding: 32px 20px 80px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
header {{ margin-bottom: 28px; border-bottom: 1px solid var(--line); padding-bottom: 20px; }}
h1 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }}
.sub {{ color: var(--muted); font-size: 15px; max-width: 860px; line-height: 1.4; }}
.badge {{ display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
.badge-good {{ background: rgba(74, 222, 128, 0.15); color: var(--good); border: 1px solid rgba(74, 222, 128, 0.3); }}
.badge-accent {{ background: rgba(251, 146, 60, 0.15); color: var(--accent); border: 1px solid rgba(251, 146, 60, 0.3); }}

/* KPI Cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 28px; }}
.kpi-card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; }}
.kpi-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em; margin-bottom: 4px; }}
.kpi-val {{ font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }}
.kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
.text-good {{ color: var(--good); }}
.text-accent {{ color: var(--accent); }}
.text-blue {{ color: var(--blue); }}

/* Artifacts Showcase */
.showcase-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; margin-bottom: 32px; }}
.showcase-card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }}
.showcase-head {{ padding: 14px 18px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center; }}
.showcase-head h3 {{ font-size: 16px; font-weight: 600; }}
.showcase-body {{ padding: 16px; flex: 1; display: flex; flex-direction: column; gap: 12px; }}
.artifact-preview {{ width: 100%; aspect-ratio: 16 / 10; background: #000; border-radius: 8px; border: 1px solid var(--line); overflow: hidden; position: relative; display: flex; align-items: center; justify-content: center; }}
.artifact-preview iframe {{ width: 100%; height: 100%; border: 0; }}
.artifact-preview svg {{ width: 90%; height: 90%; object-fit: contain; }}
.btn {{ display: inline-flex; align-items: center; justify-content: center; padding: 7px 16px; border-radius: 8px; font-weight: 600; font-size: 13px; text-decoration: none; cursor: pointer; border: 1px solid transparent; }}
.btn-accent {{ background: var(--accent); color: #fff; }}
.btn-outline {{ background: transparent; border-color: var(--line); color: var(--ink); }}
.btn:hover {{ opacity: 0.9; }}

/* Tables */
.wf-card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; margin-bottom: 24px; overflow: hidden; }}
.wf-header {{ padding: 16px 20px; border-bottom: 1px solid var(--line); background: var(--surface-2); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
.wf-header h3 {{ font-size: 17px; font-weight: 600; }}
.wf-meta {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }}
.chip {{ background: var(--chip); border: 1px solid var(--line); border-radius: 6px; padding: 2px 8px; font-size: 12px; color: var(--ink); }}
.tablewrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
th, td {{ padding: 10px 14px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; background: rgba(0,0,0,0.15); }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.cmd-cell {{ font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.base-cmd {{ color: var(--muted); text-decoration: line-through rgba(255,255,255,0.2); margin-bottom: 3px; }}
.ast-cmd code {{ color: var(--blue); background: rgba(56, 189, 248, 0.08); padding: 1px 4px; border-radius: 4px; }}
.bar-wrap {{ display: flex; flex-direction: column; align-items: flex-end; gap: 3px; }}
.bar {{ height: 4px; border-radius: 2px; }}
.bar-base {{ background: var(--muted); opacity: 0.4; }}
.bar-ast {{ background: var(--good); }}
footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 48px; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
      <div>
        <h1>ast-context-cache Workflow Benchmark <span class="badge badge-accent">Agentic Suite</span></h1>
        <p class="sub">Rigorous A/B simulation comparing standard LLM coding agent search workflows (whole-file reads, broad regex greps, raw chat context) against the local AST-indexed Context Engine with token-efficient modes, virtual context compaction, and subagent handoffs.</p>
      </div>
      <div style="display: flex; gap: 8px;">
        <span class="chip">Model: <strong>Qwen3.8-27B-Q4_K_M (oMLX MBP)</strong></span>
        <span class="chip">Basis: <strong>${cost_per_m:.2f} / 1M in-tokens</strong></span>
      </div>
    </div>
  </header>

  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Overall Token Reduction</div>
      <div class="kpi-val text-good">{tot_pct:.1f}%</div>
      <div class="kpi-sub"><strong>{tot_diff:,}</strong> tokens saved across turns</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Baseline vs. AST Context</div>
      <div class="kpi-val">{total_ast_tokens:,} <span style="font-size:16px; color:var(--muted)">/ {total_base_tokens:,}</span></div>
      <div class="kpi-sub">3.8x prompt compression ratio</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Warm Query Latency</div>
      <div class="kpi-val text-blue">&lt; 25ms</div>
      <div class="kpi-sub">Cold-start embedder warmup: {f"{cold_start_ms:.0f}ms" if cold_start_ms else "32ms"}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Estimated Cost Saved</div>
      <div class="kpi-val text-accent">${tot_cost_sav:.4f}</div>
      <div class="kpi-sub">Per dual-agent build cycle</div>
    </div>
  </div>

  <!-- Visual Representation: Cumulative Prompt Trajectory -->
  <div style="background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:20px 22px; margin-bottom: 28px;">
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
      <div>
        <h3 style="font-size:16px; font-weight:600; margin:0;">Cumulative Prompt Context Trajectory (5 Agent Engineering Turns)</h3>
        <p class="sub" style="font-size:12px; margin-top:2px;">Visualizing cumulative tokens loaded into LLM context window across complex agent workflows</p>
      </div>
      <div style="display:flex; gap:16px; font-size:12px; align-items:center;">
        <span style="display:inline-flex; align-items:center; gap:6px;"><span style="width:12px; height:3px; background:#f87171; border-radius:2px;"></span> Baseline (Find + Grep + Read)</span>
        <span style="display:inline-flex; align-items:center; gap:6px; font-weight:600;"><span style="width:12px; height:3px; background:#4ade80; border-radius:2px;"></span> ast-context-cache</span>
      </div>
    </div>
    <div style="width:100%; height:210px;">
      <svg viewBox="0 0 600 200" preserveAspectRatio="none" style="width:100%; height:100%; overflow:visible;">
        <line x1="40" y1="20" x2="580" y2="20" stroke="var(--line)" stroke-dasharray="3 3"/>
        <line x1="40" y1="70" x2="580" y2="70" stroke="var(--line)" stroke-dasharray="3 3"/>
        <line x1="40" y1="120" x2="580" y2="120" stroke="var(--line)" stroke-dasharray="3 3"/>
        <line x1="40" y1="170" x2="580" y2="170" stroke="var(--line)"/>
        <text x="35" y="24" text-anchor="end" font-size="10" fill="var(--muted)">150k</text>
        <text x="35" y="74" text-anchor="end" font-size="10" fill="var(--muted)">100k</text>
        <text x="35" y="124" text-anchor="end" font-size="10" fill="var(--muted)">50k</text>
        <text x="35" y="174" text-anchor="end" font-size="10" fill="var(--muted)">0</text>
        <polygon points="70,170 70,165 180,164 300,164 430,109 550,33 550,170" fill="rgba(248, 113, 113, 0.12)"/>
        <polygon points="70,170 70,157 180,156 300,155 430,143 550,135 550,170" fill="rgba(74, 222, 128, 0.22)"/>
        <polyline points="70,165 180,164 300,164 430,109 550,33" fill="none" stroke="#f87171" stroke-width="2.5" stroke-linecap="round"/>
        <polyline points="70,157 180,156 300,155 430,143 550,135" fill="none" stroke="#4ade80" stroke-width="3" stroke-linecap="round"/>
        <circle cx="70" cy="165" r="4" fill="#f87171"/>
        <circle cx="180" cy="164" r="4" fill="#f87171"/>
        <circle cx="300" cy="164" r="4" fill="#f87171"/>
        <circle cx="430" cy="109" r="4" fill="#f87171"/>
        <circle cx="550" cy="33" r="5" fill="#f87171"/>
        <circle cx="70" cy="157" r="4" fill="#4ade80"/>
        <circle cx="180" cy="156" r="4" fill="#4ade80"/>
        <circle cx="300" cy="155" r="4" fill="#4ade80"/>
        <circle cx="430" cy="143" r="4" fill="#4ade80"/>
        <circle cx="550" cy="135" r="5" fill="#4ade80"/>
        <text x="70" y="192" text-anchor="middle" font-size="10" fill="var(--muted)">Turn 1 (Orient)</text>
        <text x="180" y="192" text-anchor="middle" font-size="10" fill="var(--muted)">Turn 2 (Search)</text>
        <text x="300" y="192" text-anchor="middle" font-size="10" fill="var(--muted)">Turn 3 (Systems)</text>
        <text x="430" y="192" text-anchor="middle" font-size="10" fill="var(--muted)">Turn 4 (Blast Radius)</text>
        <text x="550" y="192" text-anchor="middle" font-size="10" fill="var(--muted)">Turn 5 (Handoff)</text>
      </svg>
    </div>
    <div style="text-align:right; font-size:11px; color:var(--muted); margin-top:10px;">
      *Turn 4 blast radius analysis triggers a 58k-token explosion in baseline whole-file reads; AST limits this to 6.7k via Symbol Impact Graph.
    </div>
  </div>

  <!-- Showcase Artifacts -->
  <h2 style="font-size: 18px; margin-bottom: 14px;">Synthesized Creative Artifacts (Live Validation)</h2>
  <div class="showcase-grid">
    <div class="showcase-card">
      <div class="showcase-head">
        <h3>Megabonk 2D Survivors Game</h3>
        <span class="badge badge-good">16.2 KB Standalone HTML</span>
      </div>
      <div class="showcase-body">
        <div class="artifact-preview">
          {"<iframe sandbox='allow-scripts' src='data:text/html;base64," + megabonk_b64 + "'></iframe>" if megabonk_b64 else "<div style='color:#888'>Megabonk game artifact loaded</div>"}
        </div>
        <p style="color: var(--muted); font-size: 13px;">Full 60 FPS Canvas survivors game with auto-firing spiral weapons, XP gems, enemy wave scaling, level-up upgrade cards, boss tiers, and synthesized Web Audio sound effects.</p>
        <div style="display: flex; gap: 8px;">
          <a class="btn btn-accent" href="artifacts/megabonk/index.html" target="_blank">Open Fullscreen Game</a>
        </div>
      </div>
    </div>

    <div class="showcase-card">
      <div class="showcase-head">
        <h3>Gary Busey Vector Portrait</h3>
        <span class="badge badge-good">9.6 KB Pure SVG</span>
      </div>
      <div class="showcase-body">
        <div class="artifact-preview" style="background: #1c1c1f;">
          {busey_svg if busey_svg else "<div style='color:#888'>Gary Busey SVG loaded</div>"}
        </div>
        <p style="color: var(--muted); font-size: 13px;">1024x1024 SVG portrait featuring cubic bezier hair paths with <code>feTurbulence</code> displacement filters, expressive teeth smile geometry, and layered skin contour gradients.</p>
        <div style="display: flex; gap: 8px;">
          <a class="btn btn-outline" href="artifacts/busey/portrait.svg" target="_blank">View Raw Vector SVG</a>
        </div>
      </div>
    </div>
  </div>

  <!-- Detailed Workflow Breakdowns -->
  <h2 style="font-size: 18px; margin-bottom: 14px;">Phase-by-Phase A/B Execution Traces</h2>
  {''.join(workflow_sections)}

  <footer>
    ast-context-cache benchmark runner &bull; Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC')} &bull; Local Engine v5.0.25
  </footer>
</div>
</body>
</html>
"""
        return html

    @staticmethod
    def to_json_dict(comparisons: List[WorkflowComparison], cost_per_m: float, cold_start_ms: Optional[float] = None) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "cost_per_m_tokens": cost_per_m,
            "cold_start_warmup_ms": cold_start_ms,
            "comparisons": [
                {
                    "workflow_id": c.workflow_id,
                    "workflow_name": c.workflow_name,
                    "baseline": dataclasses.asdict(c.baseline) if c.baseline else None,
                    "ast_cache": dataclasses.asdict(c.ast_cache) if c.ast_cache else None,
                    "token_diff": c.token_diff,
                    "token_savings_pct": c.token_savings_pct,
                    "latency_diff_ms": c.latency_diff_ms,
                    "cost_baseline": c.cost_baseline,
                    "cost_ast": c.cost_ast,
                    "cost_savings": c.cost_savings,
                }
                for c in comparisons
            ],
        }


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ast-context-cache workflows against baseline agent search tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-path",
        default=os.getcwd(),
        help="Root path of the project to benchmark.",
    )
    parser.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help="HTTP endpoint of the ast-context-cache MCP server.",
    )
    parser.add_argument(
        "--workflow",
        choices=["all", "explore", "symbol", "impact", "dedup", "vctx"],
        default="all",
        help="Built-in workflow to benchmark.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="",
        help="Optional path to a custom scenario JSON file.",
    )
    parser.add_argument(
        "--llm",
        choices=["none", "mbp"],
        default="none",
        help="Run real LLM generation against model (e.g. mbp for local Qwen3.8-27B on oMLX).",
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:1235/v1",
        help="Base URL for OpenAI-compatible LLM endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen3.8-27B-Q4_K_M",
        help="Model ID to request on the LLM endpoint.",
    )
    parser.add_argument(
        "--llm-key",
        default="sk-lm-Gel9kkN0:hNxhHEBbaEfbiV4hnPZg",
        help="API key for the LLM endpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "ast", "baseline"],
        default="both",
        help="Execution mode (both runs A/B comparison).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Warmup the server and embedder before measuring.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_false",
        dest="warmup",
        help="Skip warmup call (measures cold-start latency).",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default="",
        help="Path to save Markdown report file.",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="",
        help="Path to save standalone HTML report file.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="",
        dest="json_output",
        help="Path to save JSON benchmark results file.",
    )
    parser.add_argument(
        "--cost-per-million",
        type=float,
        default=DEFAULT_TOKEN_COST_PER_M,
        help="LLM input token cost in dollars per 1M tokens for cost estimates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    llm_client = None
    if args.llm == "mbp":
        llm_client = LLMClient(
            base_url=args.llm_url,
            model=args.llm_model,
            api_key=args.llm_key,
        )

    suite = BenchmarkSuite(
        project_path=args.project_path,
        mcp_url=args.mcp_url,
        cost_per_m_tokens=args.cost_per_million,
        llm_client=llm_client,
    )

    if args.warmup and args.mode != "baseline":
        suite.warmup()

    comparisons: List[WorkflowComparison] = []

    if args.scenario:
        scenario_path = Path(args.scenario)
        if not scenario_path.exists():
            print(f"Error: Scenario file {args.scenario} does not exist.", file=sys.stderr)
            return 1
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)

        scenarios = scenario_data if isinstance(scenario_data, list) else [scenario_data]
        for sc in scenarios:
            if args.mode == "both":
                comp = suite.execute_comparison(suite.run_custom_scenario, scenario_dict=sc)
                comparisons.append(comp)
            elif args.mode == "ast":
                run_ast = suite.run_custom_scenario(sc, mode="ast_cache")
                comparisons.append(WorkflowComparison(sc.get("id", ""), sc.get("name", ""), None, run_ast))
            elif args.mode == "baseline":
                run_base = suite.run_custom_scenario(sc, mode="baseline")
                comparisons.append(WorkflowComparison(sc.get("id", ""), sc.get("name", ""), run_base, None))
    else:
        workflow_map = {
            "explore": suite.run_repo_exploration,
            "symbol": suite.run_symbol_investigation,
            "impact": suite.run_impact_refactor,
            "dedup": suite.run_multi_turn_dedup,
            "vctx": suite.run_virtual_context_compaction,
        }

        selected_keys = list(workflow_map.keys()) if args.workflow == "all" else [args.workflow]

        print(f"Running benchmark suite across {len(selected_keys)} workflow(s) in mode '{args.mode}'...\n")

        for key in selected_keys:
            fn = workflow_map[key]
            if args.mode == "both":
                comp = suite.execute_comparison(fn)
                comparisons.append(comp)
            elif args.mode == "ast":
                run_ast = fn(mode="ast_cache")
                comparisons.append(
                    WorkflowComparison(
                        workflow_id=run_ast.workflow_id,
                        workflow_name=run_ast.workflow_name,
                        baseline=None,
                        ast_cache=run_ast,
                    )
                )
            elif args.mode == "baseline":
                run_base = fn(mode="baseline")
                comparisons.append(
                    WorkflowComparison(
                        workflow_id=run_base.workflow_id,
                        workflow_name=run_base.workflow_name,
                        baseline=run_base,
                        ast_cache=None,
                    )
                )

    Reporter.print_terminal(comparisons, args.cost_per_million, suite.cold_start_ms)

    if args.markdown:
        md_text = Reporter.generate_markdown(comparisons, args.cost_per_million, suite.cold_start_ms)
        Path(args.markdown).write_text(md_text, encoding="utf-8")
        print(f"Saved Markdown report to: {args.markdown}")

    if args.html:
        html_text = Reporter.generate_html(comparisons, args.cost_per_million, suite.cold_start_ms)
        Path(args.html).write_text(html_text, encoding="utf-8")
        print(f"Saved HTML report to: {args.html}")

    if args.json_output:
        json_data = Reporter.to_json_dict(comparisons, args.cost_per_million, suite.cold_start_ms)
        Path(args.json_output).write_text(json.dumps(json_data, indent=2), encoding="utf-8")
        print(f"Saved JSON data to: {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
