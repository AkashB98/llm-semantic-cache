"""Tests for pricing math, including the unknown-model fallback flag."""

import unittest

from semantic_cache.pricing import (
    FALLBACK_RATE_PER_1M,
    estimate_cost,
    load_pricing,
)


class TestPricing(unittest.TestCase):
    def test_known_model_math(self):
        c = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(c["input"], 0.15)
        self.assertAlmostEqual(c["output"], 0.60)
        self.assertAlmostEqual(c["total"], 0.75)
        self.assertFalse(c["fallback"])

    def test_zero_tokens_zero_cost(self):
        c = estimate_cost("gpt-4o", 0, 0)
        self.assertEqual(c["total"], 0.0)
        self.assertFalse(c["fallback"])

    def test_unknown_model_fallback_flagged(self):
        c = estimate_cost("mystery-model-9000", 1_000_000, 1_000_000)
        self.assertTrue(c["fallback"])
        self.assertAlmostEqual(c["total"], 2 * FALLBACK_RATE_PER_1M)
        self.assertEqual(c["rate_input_per_1m"], FALLBACK_RATE_PER_1M)

    def test_all_sample_models_priced(self):
        pricing = load_pricing()
        models = pricing["models"]
        self.assertGreaterEqual(len(models), 6)
        for name, rates in models.items():
            self.assertGreater(rates["input_per_1m"], 0, name)
            self.assertGreater(rates["output_per_1m"], 0, name)

    def test_sample_data_disclaimer(self):
        pricing = load_pricing()
        self.assertIn("SAMPLE", pricing.get("_note", ""))

    def test_cost_scales_linearly(self):
        a = estimate_cost("deepseek-chat", 1000, 0)
        b = estimate_cost("deepseek-chat", 2000, 0)
        self.assertAlmostEqual(b["total"], 2 * a["total"])

    def test_cache_hit_saves_full_call_cost(self):
        # A hit avoids exactly the cost the miss incurred.
        from semantic_cache import SemanticCache, CacheConfig
        from semantic_cache.simulator import ScriptedLLM
        llm = ScriptedLLM()
        cache = SemanticCache(llm, config=CacheConfig())
        miss = cache.chat("gpt-4o-mini", [{"role": "user", "content": "what is the capital of france"}])
        hit = cache.chat("gpt-4o-mini", [{"role": "user", "content": "what is the capital of france"}])
        self.assertAlmostEqual(hit.cost_saved, miss.cost_incurred, places=9)
        self.assertGreater(hit.cost_saved, 0)


if __name__ == "__main__":
    unittest.main()
