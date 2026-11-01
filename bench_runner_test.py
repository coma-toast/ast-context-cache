#!/usr/bin/env python3
"""Unit tests for bench_runner.py.

Run with:  python3 -m unittest bench_runner_test -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import bench_runner as br


# A real (trimmed) opencode --format json event stream, captured with
# opencode 1.18.34.
REAL_EVENTS = [
    {"type": "step_start", "timestamp": 1793507663182,
     "sessionID": "ses_test", "part": {"type": "step-start", "sessionID": "ses_test"}},
    {"type": "tool_use", "timestamp": 1793507663886, "sessionID": "ses_test",
     "part": {"type": "tool", "tool": "write", "callID": "call-1",
              "state": {"status": "completed", "input": {}, "output": "ok"}}},
    {"type": "step_finish", "timestamp": 1793507663886, "sessionID": "ses_test",
     "part": {"type": "step-finish", "reason": "tool-calls",
              "tokens": {"total": 32513, "input": 32423, "output": 45,
                         "reasoning": 45, "cache": {"write": 0, "read": 100}},
              "cost": 0}},
    {"type": "tool_use", "timestamp": 1793507667176, "sessionID": "ses_test",
     "part": {"type": "tool", "tool": "ast-context-cache_get_project_map",
              "callID": "call-2", "state": {"status": "completed"}}},
    {"type": "step_finish", "timestamp": 1793507667177, "sessionID": "ses_test",
     "part": {"type": "step-finish", "reason": "tool-calls",
              "tokens": {"total": 32582, "input": 32534, "output": 34,
                         "reasoning": 14, "cache": {"write": 5, "read": 0}},
              "cost": 0}},
    {"type": "text", "timestamp": 1793507669069, "sessionID": "ses_test",
     "part": {"type": "text", "text": "done"}},
    {"type": "step_finish", "timestamp": 1793507669069, "sessionID": "ses_test",
     "part": {"type": "step-finish", "reason": "stop",
              "tokens": {"total": 22822, "input": 22793, "output": 13,
                         "reasoning": 16, "cache": {"write": 0, "read": 7}},
              "cost": 0}},
]


GOLD_PATCH = """diff --git a/src/core.py b/src/core.py
--- a/src/core.py
+++ b/src/core.py
@@ -1 +1 @@
-old
+new
"""


class TestMetricsExtraction(unittest.TestCase):
    def test_extracts_real_opencode_schema(self):
        metrics = br._extract_metrics(REAL_EVENTS, 12.5, returncode=0)

        self.assertEqual(metrics["tokens_input"], 32423 + 32534 + 22793)
        self.assertEqual(metrics["tokens_output"], 45 + 34 + 13)
        self.assertEqual(metrics["tokens_reasoning"], 45 + 14 + 16)
        self.assertEqual(metrics["tokens_total"], 32513 + 32582 + 22822)
        self.assertEqual(metrics["tokens_cache_read"], 107)
        self.assertEqual(metrics["tokens_cache_write"], 5)
        self.assertEqual(metrics["session_id"], "ses_test")
        self.assertEqual(metrics["tool_calls_total"], 2)
        self.assertEqual(metrics["ast_mcp_calls"], 1)
        self.assertEqual(metrics["tool_calls_by_name"]["write"], 1)
        self.assertEqual(metrics["wall_time_sec"], 12.5)
        self.assertEqual(metrics["event_span_sec"], 5.89)
        self.assertFalse(metrics["timed_out"])
        self.assertEqual(metrics["events_count"], len(REAL_EVENTS))

    def test_ast_tool_detection(self):
        for name in ("ast-context-cache_read", "ast_context_cache_read",
                     "ast-cache-read", "ast_cache_read"):
            self.assertTrue(br.is_ast_tool(name), name)
        for name in ("read", "bash", "grep"):
            self.assertFalse(br.is_ast_tool(name), name)

    def test_timeout_is_recorded(self):
        metrics = br._extract_metrics(REAL_EVENTS[:3], 900.0, returncode=None,
                                      timed_out=True)
        self.assertTrue(metrics["timed_out"])
        self.assertIsNone(metrics["returncode"])

    def test_error_events_are_captured(self):
        events = [{"type": "error", "sessionID": "s",
                   "error": {"name": "APIError",
                             "data": {"message": "Invalid model name"}}}]
        metrics = br._extract_metrics(events, 1.0, returncode=1)
        self.assertEqual(len(metrics["errors"]), 1)
        self.assertIn("Invalid model name", metrics["errors"][0])


class TestDataset(unittest.TestCase):
    def test_load_and_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.jsonl"
            path.write_text("\n".join(
                json.dumps({"instance_id": f"repo__x-{i}", "repo": "r",
                            "base_commit": "abc", "patch": ""})
                for i in range(4)
            ), encoding="utf-8")
            records = br.load_dataset(path)
            self.assertEqual(len(records), 4)
            self.assertEqual(len(br.select_tasks(records, 2, None)), 2)

    def test_task_ids_preserve_requested_order(self):
        records = [{"instance_id": x} for x in ("a", "b", "c")]
        selected = br.select_tasks(records, 5, ["c", "a"])
        self.assertEqual([r["instance_id"] for r in selected], ["c", "a"])

    def test_missing_task_ids_are_skipped(self):
        records = [{"instance_id": "a"}]
        self.assertEqual(br.select_tasks(records, 5, ["a", "zzz"]), [{"instance_id": "a"}])


class TestConfigIsolation(unittest.TestCase):
    def test_control_disables_ast_mcp(self):
        config = br.build_opencode_config("control", "litellm/free")
        self.assertFalse(config["mcp"]["ast-context-cache"]["enabled"])

    def test_treatment_enables_ast_mcp(self):
        config = br.build_opencode_config("treatment", "litellm/free")
        self.assertTrue(config["mcp"]["ast-context-cache"]["enabled"])
        self.assertEqual(config["mcp"]["ast-context-cache"]["url"], br.AST_MCP_URL)

    def test_only_ast_mcp_is_declared(self):
        for condition in br.CONDITIONS:
            config = br.build_opencode_config(condition, "litellm/free")
            self.assertEqual(list(config["mcp"]), ["ast-context-cache"])

    def test_no_secret_is_written(self):
        config = br.build_opencode_config("treatment", "litellm/free")
        text = json.dumps(config)
        self.assertIn("{env:LITELLM_API_KEY}", text)
        self.assertNotIn("sk-", text)

    def test_control_agents_md_has_no_ast_block(self):
        self.assertNotIn("ast-context-cache", br.agents_md_for("control"))
        self.assertIn("ast-context-cache", br.agents_md_for("treatment"))

    def test_leaky_env_is_removed(self):
        env = {"PATH": "/bin", "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
               "OPENCODE": "1", "OPENCODE_PID": "123"}
        cleaned = {"PATH": "/bin"}
        import os
        saved = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update(env)
            result = br._clean_env(Path("/tmp/cfg"))
        finally:
            os.environ.clear()
            os.environ.update(saved)
        for key in ("OPENCODE_DISABLE_PROJECT_CONFIG", "OPENCODE", "OPENCODE_PID"):
            self.assertNotIn(key, result)
        self.assertIn("PATH", result)
        self.assertEqual(result["XDG_CONFIG_HOME"], "/tmp/cfg")


class TestPatchHandling(unittest.TestCase):
    def test_extract_patch_files(self):
        files = br.extract_patch_files(GOLD_PATCH)
        self.assertEqual(files, {"src/core.py"})

    def test_detects_new_files(self):
        patch = ("diff --git a/new.py b/new.py\n"
                 "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n")
        self.assertEqual(br.extract_patch_files(patch), {"new.py"})

    def test_proxy_requires_patch(self):
        self.assertEqual(br.evaluate_success_proxy({}, {}), "proxy_fail")
        self.assertEqual(
            br.evaluate_success_proxy({}, {"output_patch": "", "patch_applies": True}),
            "proxy_fail")

    def test_proxy_fails_when_patch_does_not_apply(self):
        metrics = {"output_patch": GOLD_PATCH, "patch_applies": False}
        self.assertEqual(br.evaluate_success_proxy({"patch": GOLD_PATCH}, metrics),
                         "proxy_fail")

    def test_confirmed_on_gold_file_overlap(self):
        metrics = {"output_patch": GOLD_PATCH, "patch_applies": True}
        self.assertEqual(br.evaluate_success_proxy({"patch": GOLD_PATCH}, metrics),
                         "confirmed")

    def test_proxy_pass_without_overlap(self):
        other = GOLD_PATCH.replace("src/core.py", "src/other.py")
        metrics = {"output_patch": other, "patch_applies": True}
        self.assertEqual(br.evaluate_success_proxy({"patch": GOLD_PATCH}, metrics),
                         "proxy_pass")


class TestReport(unittest.TestCase):
    def test_report_contains_tables_and_success_rates(self):
        results = [
            {"instance_id": "repo__a-1", "condition": "control",
             "metrics": {"tokens_total": 100, "tokens_input": 90, "tokens_output": 10,
                         "wall_time_sec": 5, "tool_calls_total": 2,
                         "ast_mcp_calls": 0, "patch_applies": False},
             "success_proxy": "proxy_fail"},
            {"instance_id": "repo__a-1", "condition": "treatment",
             "metrics": {"tokens_total": 60, "tokens_input": 50, "tokens_output": 10,
                         "wall_time_sec": 4, "tool_calls_total": 3,
                         "ast_mcp_calls": 1, "patch_applies": True},
             "success_proxy": "confirmed"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = br.generate_report(results, Path(tmp))
            text = report.read_text(encoding="utf-8")
        self.assertIn("| tokens_total | 100.0 | 60.0 | -40.0 |", text)
        self.assertIn("**control**: 0% pass proxy", text)
        self.assertIn("**treatment**: 100% pass proxy", text)
        self.assertIn("proxy_pass", text)


class TestMedian(unittest.TestCase):
    def test_median(self):
        self.assertEqual(br.median([]), 0)
        self.assertEqual(br.median([3, 1, 2]), 2)
        self.assertEqual(br.median([4, 1, 3, 2]), 2.5)


if __name__ == "__main__":
    unittest.main()
