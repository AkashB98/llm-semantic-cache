"""Tests for SemanticCache: paths, params, eviction, thread-safety, persistence."""

import math
import os
import tempfile
import threading
import unittest

from semantic_cache import SemanticCache, CacheConfig
from semantic_cache.embeddings import EmbeddingModel
from semantic_cache.simulator import ScriptedLLM


class FakeClock:
    """Injectable clock: tests control time exactly, no wall-clock dependence."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class StubEmbedding(EmbeddingModel):
    """Returns fixed vectors per text so cosine scores are exactly controlled."""

    def __init__(self, mapping, dim=4):
        self.mapping = mapping
        self.dim = dim

    def embed(self, text):
        vec = [0.0] * self.dim
        for i, v in enumerate(self.mapping.get(text, [1.0] + [0.0] * (self.dim - 1))):
            vec[i] = v
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


def make_cache(llm=None, clock=None, **cfg_kwargs):
    llm = llm or ScriptedLLM()
    clock = clock or FakeClock()
    cfg = CacheConfig(**cfg_kwargs)
    return SemanticCache(llm, config=cfg, clock=clock, mono=clock), llm, clock


def msg(text):
    return [{"role": "user", "content": text}]


class TestExactPath(unittest.TestCase):
    def test_first_call_misses_second_hits_exact(self):
        cache, llm, _ = make_cache()
        r1 = cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        r2 = cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        self.assertFalse(r1.hit)
        self.assertEqual(r1.kind, "miss")
        self.assertTrue(r2.hit)
        self.assertEqual(r2.kind, "exact")
        self.assertEqual(r2.score, 1.0)
        self.assertEqual(r1.answer, r2.answer)
        self.assertEqual(llm.calls, 1)

    def test_case_and_whitespace_normalized(self):
        cache, _, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("What is the CAPITAL of France?  "))
        r = cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        self.assertEqual(r.kind, "exact")

    def test_system_messages_ignored_in_key(self):
        cache, llm, _ = make_cache()
        cache.chat("m", [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}])
        r = cache.chat("m", [{"role": "user", "content": "hi"}])
        self.assertEqual(r.kind, "exact")
        self.assertEqual(llm.calls, 1)


class TestSemanticPath(unittest.TestCase):
    def test_paraphrase_hits_semantic(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        r = cache.chat("gpt-4o-mini", msg("which city is the capital of france"))
        self.assertTrue(r.hit)
        self.assertEqual(r.kind, "semantic")
        self.assertGreaterEqual(r.score, cache.config.threshold)
        self.assertEqual(llm.calls, 1)

    def test_dissimilar_prompt_misses(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        r = cache.chat("gpt-4o-mini", msg("best way to train for a marathon"))
        self.assertFalse(r.hit)
        self.assertEqual(r.kind, "miss")
        self.assertEqual(llm.calls, 2)

    def test_threshold_boundary_exact(self):
        # cos("base","probe") is engineered to exactly 0.8.
        c, s = 0.8, 0.6
        emb = StubEmbedding({"base": [1, 0], "probe": [c, s]}, dim=2)
        llm = ScriptedLLM()
        clock = FakeClock()
        cache = SemanticCache(llm, embedding_model=emb,
                              config=CacheConfig(threshold=0.8), clock=clock, mono=clock)
        cache.chat("m", msg("base"))
        r = cache.chat("m", msg("probe"))
        self.assertTrue(r.hit, "score == threshold must hit")
        self.assertAlmostEqual(r.score, 0.8, places=6)

    def test_threshold_boundary_below(self):
        c, s = 0.8, 0.6
        emb = StubEmbedding({"base": [1, 0], "probe": [c, s]}, dim=2)
        llm = ScriptedLLM()
        clock = FakeClock()
        cache = SemanticCache(llm, embedding_model=emb,
                              config=CacheConfig(threshold=0.800001), clock=clock, mono=clock)
        cache.chat("m", msg("base"))
        r = cache.chat("m", msg("probe"))
        self.assertFalse(r.hit, "score just below threshold must miss")

    def test_semantic_hit_serves_original_answer(self):
        cache, _, _ = make_cache()
        r1 = cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        r2 = cache.chat("gpt-4o-mini", msg("which city is the capital of france"))
        self.assertEqual(r2.answer, r1.answer)
        self.assertIn("Paris", r2.answer)


class TestParamIsolation(unittest.TestCase):
    def test_different_temperature_misses(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"), temperature=0.0)
        r = cache.chat("gpt-4o-mini", msg("what is the capital of france"), temperature=0.9)
        self.assertFalse(r.hit)
        self.assertEqual(llm.calls, 2)

    def test_same_temperature_hits(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"), temperature=0.2)
        r = cache.chat("gpt-4o-mini", msg("what is the capital of france"), temperature=0.2)
        self.assertTrue(r.hit)
        self.assertEqual(llm.calls, 1)

    def test_different_top_p_misses(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("hello there"), top_p=0.5)
        r = cache.chat("gpt-4o-mini", msg("hello there"), top_p=1.0)
        self.assertFalse(r.hit)

    def test_semantic_hit_respects_params(self):
        # A paraphrase at a different temperature must not semantic-hit.
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"), temperature=0.0)
        r = cache.chat("gpt-4o-mini", msg("which city is the capital of france"), temperature=0.7)
        self.assertFalse(r.hit)

    def test_different_model_misses(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        r = cache.chat("gpt-4o", msg("what is the capital of france"))
        self.assertFalse(r.hit)

    def test_different_model_no_semantic_leak(self):
        cache, llm, _ = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        r = cache.chat("gpt-4o", msg("which city is the capital of france"))
        self.assertFalse(r.hit, "cross-model semantic hit would serve the wrong model")


class TestEviction(unittest.TestCase):
    def test_ttl_expiry(self):
        cache, llm, clock = make_cache(ttl_seconds=60.0)
        cache.chat("m", msg("ttl probe"))
        clock.advance(61.0)
        r = cache.chat("m", msg("ttl probe"))
        self.assertFalse(r.hit)
        self.assertEqual(r.kind, "miss")
        self.assertEqual(llm.calls, 2)

    def test_ttl_not_expired_hits(self):
        cache, llm, clock = make_cache(ttl_seconds=60.0)
        cache.chat("m", msg("ttl probe"))
        clock.advance(59.0)
        r = cache.chat("m", msg("ttl probe"))
        self.assertTrue(r.hit)

    def test_lru_eviction(self):
        cache, llm, _ = make_cache(max_entries=3, ttl_seconds=0)
        prompts = ["astronomy lecture notes", "sourdough baking tips",
                   "chess opening theory", "marathon training plan"]
        for p in prompts:
            cache.chat("m", msg(p))
        self.assertEqual(cache.stats()["entries"], 3)
        # Oldest entry was evicted -> miss on re-query.
        r = cache.chat("m", msg("astronomy lecture notes"))
        self.assertFalse(r.hit)

    def test_lru_refresh_on_hit(self):
        cache, llm, _ = make_cache(max_entries=2, ttl_seconds=0)
        cache.chat("m", msg("astronomy lecture notes"))
        cache.chat("m", msg("sourdough baking tips"))
        cache.chat("m", msg("astronomy lecture notes"))  # refresh
        cache.chat("m", msg("chess opening theory"))  # evicts sourdough
        r = cache.chat("m", msg("astronomy lecture notes"))
        self.assertTrue(r.hit, "recently used entry must survive eviction")
        r2 = cache.chat("m", msg("sourdough baking tips"))
        self.assertFalse(r2.hit, "least-recently-used entry must be evicted")

    def test_ttl_disabled_when_zero(self):
        cache, llm, clock = make_cache(ttl_seconds=0)
        cache.chat("m", msg("forever prompt"))
        clock.advance(10_000_000.0)
        r = cache.chat("m", msg("forever prompt"))
        self.assertTrue(r.hit)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_writes_no_corruption(self):
        cache, llm, _ = make_cache(max_entries=500, ttl_seconds=0)
        errors = []

        def worker(n):
            try:
                for i in range(25):
                    cache.chat("m", msg(f"thread {n} prompt {i} concurrent work"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        s = cache.stats()
        # Similar prompts may semantic-hit each other; what matters is the
        # accounting is consistent and nothing was lost or double-counted.
        self.assertEqual(s["hits"] + s["misses"], 200)
        self.assertEqual(s["entries"], s["misses"])
        self.assertEqual(llm.calls, s["misses"])

    def test_concurrent_same_prompt_consistent(self):
        llm = ScriptedLLM()
        clock = FakeClock()
        cache = SemanticCache(llm, config=CacheConfig(), clock=clock, mono=clock)
        cache.chat("m", msg("shared prompt"))
        results = []
        errors = []
        def reader():
            try:
                results.append(cache.chat("m", msg("shared prompt")))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=reader) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 16)
        self.assertTrue(all(r.hit for r in results))
        self.assertEqual(len({r.answer for r in results}), 1)
        self.assertEqual(llm.calls, 1)


class TestStatsAndMaintenance(unittest.TestCase):
    def test_stats_hit_rate(self):
        cache, _, _ = make_cache()
        cache.chat("m", msg("aaa bbb"))
        cache.chat("m", msg("aaa bbb"))
        cache.chat("m", msg("ccc ddd"))
        s = cache.stats()
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 2)
        self.assertAlmostEqual(s["hit_rate"], 1 / 3)

    def test_clear_resets(self):
        cache, _, _ = make_cache()
        cache.chat("m", msg("aaa bbb"))
        cache.clear()
        s = cache.stats()
        self.assertEqual(s["entries"], 0)
        self.assertEqual(s["hits"], 0)
        self.assertEqual(s["misses"], 0)

    def test_save_load_roundtrip(self):
        cache, _, clock = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cache.json")
            cache.save(path)
            llm2 = ScriptedLLM()
            cache2 = SemanticCache(llm2, config=CacheConfig(), clock=clock, mono=clock)
            cache2.load(path)
            r = cache2.chat("gpt-4o-mini", msg("what is the capital of france"))
            self.assertEqual(r.kind, "exact")
            self.assertEqual(llm2.calls, 0)

    def test_loaded_entry_no_cross_model_semantic(self):
        cache, _, clock = make_cache()
        cache.chat("gpt-4o-mini", msg("what is the capital of france"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cache.json")
            cache.save(path)
            cache2 = SemanticCache(ScriptedLLM(), config=CacheConfig(), clock=clock, mono=clock)
            cache2.load(path)
            r = cache2.chat("gpt-4o", msg("which city is the capital of france"))
            self.assertFalse(r.hit)


class TestLatencyAndTokens(unittest.TestCase):
    def test_latency_measured_with_injected_clock(self):
        class StepMono:
            def __init__(self):
                self.t = 0.0
            def __call__(self):
                self.t += 0.025
                return self.t
        llm = ScriptedLLM()
        mono = StepMono()
        cache = SemanticCache(llm, config=CacheConfig(), clock=FakeClock(), mono=mono)
        r = cache.chat("m", msg("latency probe"))
        self.assertGreater(r.latency_ms, 0)

    def test_tokens_recorded(self):
        cache, _, _ = make_cache()
        r = cache.chat("m", msg("what is the capital of france"))
        self.assertGreater(r.prompt_tokens, 0)
        self.assertGreater(r.completion_tokens, 0)


if __name__ == "__main__":
    unittest.main()
