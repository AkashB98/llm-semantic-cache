"""Tests for the JSONL trace writer."""

import json
import os
import tempfile
import threading
import unittest

from semantic_cache import SemanticCache, CacheConfig
from semantic_cache.simulator import ScriptedLLM
from semantic_cache.trace import TraceWriter

REQUIRED_FIELDS = {
    "timestamp", "model", "prompt_hash", "hit", "kind", "matched_score",
    "latency_ms", "prompt_tokens", "completion_tokens", "est_cost_saved",
    "cost_incurred", "fallback_pricing",
}


class TestTraceWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "trace.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_read_roundtrip(self):
        tw = TraceWriter(self.path)
        tw.write({"a": 1, "hit": True})
        self.assertEqual(tw.read_all(), [{"a": 1, "hit": True}])

    def test_one_record_per_line(self):
        tw = TraceWriter(self.path)
        tw.write({"n": 1})
        tw.write({"n": 2})
        with open(self.path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(l)["n"] for l in lines], [1, 2])

    def test_missing_file_reads_empty(self):
        tw = TraceWriter(os.path.join(self.tmp.name, "nope.jsonl"))
        self.assertEqual(tw.read_all(), [])

    def test_concurrent_writes_no_interleave(self):
        tw = TraceWriter(self.path)
        def writer(n):
            for i in range(50):
                tw.write({"thread": n, "i": i})
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records = tw.read_all()
        self.assertEqual(len(records), 200)
        self.assertEqual(len({(r["thread"], r["i"]) for r in records}), 200)


class TestTraceSchema(unittest.TestCase):
    def test_record_schema_on_hit_and_miss(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "t.jsonl")
        llm = ScriptedLLM()
        cache = SemanticCache(llm, config=CacheConfig(), trace=TraceWriter(path))
        cache.chat("gpt-4o-mini", [{"role": "user", "content": "what is the capital of france"}])
        cache.chat("gpt-4o-mini", [{"role": "user", "content": "what is the capital of france"}])
        records = TraceWriter(path).read_all()
        self.assertEqual(len(records), 2)
        kinds = {r["kind"] for r in records}
        self.assertEqual(kinds, {"miss", "exact"})
        for r in records:
            self.assertTrue(REQUIRED_FIELDS.issubset(r.keys()), f"missing: {REQUIRED_FIELDS - set(r.keys())}")
            self.assertIsInstance(r["hit"], bool)
            self.assertEqual(len(r["prompt_hash"]), 16)
        hit_rec = next(r for r in records if r["hit"])
        miss_rec = next(r for r in records if not r["hit"])
        self.assertGreater(hit_rec["est_cost_saved"], 0)
        self.assertEqual(hit_rec["cost_incurred"], 0)
        self.assertGreater(miss_rec["cost_incurred"], 0)
        self.assertEqual(miss_rec["est_cost_saved"], 0)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
