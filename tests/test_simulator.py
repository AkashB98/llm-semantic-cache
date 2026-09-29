"""Tests for the deterministic scripted LLM."""

import unittest

from semantic_cache.simulator import (
    ADVERSARIAL_PAIRS,
    CORPUS,
    DISSIMILAR_PROMPTS,
    PARAPHRASE_PAIRS,
    ScriptedLLM,
    estimate_tokens,
)


class TestScriptedLLM(unittest.TestCase):
    def test_corpus_answer(self):
        llm = ScriptedLLM()
        answer, _, _ = llm("m", [{"role": "user", "content": "what is the capital of france"}])
        self.assertIn("Paris", answer)

    def test_deterministic_across_instances(self):
        a = ScriptedLLM()("m", [{"role": "user", "content": "brand new question here"}])
        b = ScriptedLLM()("m", [{"role": "user", "content": "brand new question here"}])
        self.assertEqual(a, b)

    def test_deterministic_repeated_calls(self):
        llm = ScriptedLLM()
        first = llm("m", [{"role": "user", "content": "zzz unknown prompt"}])
        for _ in range(5):
            self.assertEqual(llm("m", [{"role": "user", "content": "zzz unknown prompt"}]), first)

    def test_calls_counted(self):
        llm = ScriptedLLM()
        llm("m", [{"role": "user", "content": "a"}])
        llm("m", [{"role": "user", "content": "b"}])
        self.assertEqual(llm.calls, 2)

    def test_tokens_positive(self):
        _, pt, ct = ScriptedLLM()("m", [{"role": "user", "content": "hello world"}])
        self.assertGreater(pt, 0)
        self.assertGreater(ct, 0)

    def test_params_do_not_change_answer(self):
        llm = ScriptedLLM()
        a = llm("m", [{"role": "user", "content": "hi"}], temperature=0.0)
        b = llm("m", [{"role": "user", "content": "hi"}], temperature=0.9)
        self.assertEqual(a[0], b[0])

    def test_corpus_nonempty(self):
        self.assertGreaterEqual(len(CORPUS), 10)

    def test_paraphrase_pairs_reference_corpus(self):
        questions = {q for q, _ in CORPUS}
        for q, _ in PARAPHRASE_PAIRS:
            self.assertIn(q, questions)

    def test_guard_corpus_disjoint_from_paraphrases(self):
        paras = {p for _, p in PARAPHRASE_PAIRS}
        for d in DISSIMILAR_PROMPTS:
            self.assertNotIn(d, paras)

    def test_adversarial_pairs_wellformed(self):
        self.assertTrue(len(ADVERSARIAL_PAIRS) >= 3)
        for a, b in ADVERSARIAL_PAIRS:
            self.assertNotEqual(a, b)


class TestEstimateTokens(unittest.TestCase):
    def test_scales_with_words(self):
        self.assertGreater(estimate_tokens("a b c d e f"), estimate_tokens("a b"))

    def test_minimum_one(self):
        self.assertEqual(estimate_tokens(""), 1)


if __name__ == "__main__":
    unittest.main()
