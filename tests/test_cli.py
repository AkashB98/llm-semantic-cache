"""Tests for the CLI: exit codes and command behavior (hermetic via temp dir)."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import cli


def run_cli(argv, tmpdir):
    with mock.patch.dict(os.environ, {"LLM_CACHE_DIR": tmpdir}):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(argv)
    return code, buf.getvalue()


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_demo_exit_zero(self):
        code, out = run_cli(["demo"], self.tmp.name)
        self.assertEqual(code, 0)
        self.assertIn("EXACT-HIT", out)

    def test_bench_exit_zero_and_table(self):
        code, out = run_cli(["bench"], self.tmp.name)
        self.assertEqual(code, 0)
        self.assertIn("hit_rate", out)
        self.assertIn("operating point", out)

    def test_stats_exit_zero_json(self):
        code, out = run_cli(["stats"], self.tmp.name)
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIn("hit_rate", data)
        self.assertIn("est_cost_saved_usd", data)

    def test_clear_exit_zero(self):
        run_cli(["demo"], self.tmp.name)  # create cache + trace files
        code, out = run_cli(["clear"], self.tmp.name)
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "cache.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "trace.jsonl")))

    def test_stats_after_demo_shows_hits(self):
        run_cli(["demo"], self.tmp.name)
        code, out = run_cli(["stats"], self.tmp.name)
        data = json.loads(out)
        self.assertGreater(data["hits"], 0)
        self.assertGreater(data["trace_records"], 0)

    def test_unknown_command_exit_nonzero(self):
        import contextlib
        with mock.patch.dict(os.environ, {"LLM_CACHE_DIR": self.tmp.name}):
            with self.assertRaises(SystemExit) as cm:
                with redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    cli.main(["nope"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_demo_is_deterministic(self):
        _, out1 = run_cli(["demo"], self.tmp.name)
        _, out2 = run_cli(["demo"], self.tmp.name)
        # Scores and hit/miss pattern must be identical across runs.
        kinds1 = [l.split("]")[0] for l in out1.splitlines() if l.startswith("[")]
        kinds2 = [l.split("]")[0] for l in out2.splitlines() if l.startswith("[")]
        self.assertEqual(kinds1, kinds2)


if __name__ == "__main__":
    unittest.main()
