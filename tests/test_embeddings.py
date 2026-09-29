"""Tests for the embedding models: determinism, norms, separability."""

import math
import unittest

from semantic_cache.embeddings import (
    HashedEmbedding,
    HttpEmbeddingModel,
    embedding_from_env,
    normalize_text,
)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class TestNormalizeText(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize_text("  Hello, WORLD!  "), "hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("a\tb\nc"), "a b c")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")


class TestHashedEmbedding(unittest.TestCase):
    def setUp(self):
        self.emb = HashedEmbedding()

    def test_dimension(self):
        self.assertEqual(len(self.emb.embed("hello world")), 1024)

    def test_custom_dimension(self):
        self.assertEqual(len(HashedEmbedding(dim=64).embed("hello")), 64)

    def test_l2_norm_is_one(self):
        for text in ["hello world", "a", "", "The quick brown fox! 123"]:
            vec = self.emb.embed(text)
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in vec)), 1.0, places=9)

    def test_deterministic_same_process(self):
        a = self.emb.embed("deterministic test string")
        b = self.emb.embed("deterministic test string")
        self.assertEqual(a, b)

    def test_deterministic_across_instances(self):
        other = HashedEmbedding()
        self.assertEqual(self.emb.embed("same input"), other.embed("same input"))

    def test_identical_texts_cosine_one(self):
        self.assertAlmostEqual(
            cosine(self.emb.embed("abc"), self.emb.embed("abc")), 1.0, places=9
        )

    def test_different_texts_cosine_below_one(self):
        self.assertLess(cosine(self.emb.embed("apple"), self.emb.embed("quantum physics")), 0.99)

    def test_seed_changes_embedding(self):
        other = HashedEmbedding(seed=12345)
        self.assertNotEqual(self.emb.embed("hello"), other.embed("hello"))

    def test_empty_text_returns_unit_vector(self):
        vec = self.emb.embed("")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vec)), 1.0, places=9)
        self.assertEqual(vec[0], 1.0)

    def test_invalid_dim_rejected(self):
        with self.assertRaises(ValueError):
            HashedEmbedding(dim=0)

    def test_invalid_ngram_rejected(self):
        with self.assertRaises(ValueError):
            HashedEmbedding(ngram=1)

    def test_word_features_weighted(self):
        feats = list(self.emb.features("reset password"))
        word_feats = [f for f, w in feats if f.startswith("w:")]
        self.assertTrue(word_feats)
        for f, w in feats:
            if f.startswith("w:") or f.startswith("b:"):
                self.assertEqual(w, self.emb.word_weight)

    def test_stopwords_dropped_from_word_features(self):
        feats = [f for f, _ in self.emb.features("what is the speed of light")]
        self.assertNotIn("w:what", feats)
        self.assertNotIn("w:the", feats)
        self.assertIn("w:light", feats)

    def test_paraphrase_scores_above_guard_corpus(self):
        from semantic_cache.simulator import PARAPHRASE_PAIRS, DISSIMILAR_PROMPTS, CORPUS
        qs = [q for q, _ in CORPUS]
        para_min = min(cosine(self.emb.embed(a), self.emb.embed(b)) for a, b in PARAPHRASE_PAIRS)
        guard_max = max(cosine(self.emb.embed(a), self.emb.embed(b)) for a in qs for b in DISSIMILAR_PROMPTS)
        self.assertGreater(para_min, guard_max,
                           f"paraphrase min {para_min:.3f} must exceed guard max {guard_max:.3f}")


class TestEmbeddingFromEnv(unittest.TestCase):
    def test_default_is_hashed(self):
        import os
        os.environ.pop("LLM_CACHE_EMBEDDINGS_URL", None)
        self.assertIsInstance(embedding_from_env(), HashedEmbedding)

    def test_url_selects_http_model(self):
        import os
        os.environ["LLM_CACHE_EMBEDDINGS_URL"] = "http://localhost:9/v1/embeddings"
        try:
            m = embedding_from_env()
            self.assertIsInstance(m, HttpEmbeddingModel)
            self.assertEqual(m.url, "http://localhost:9/v1/embeddings")
        finally:
            del os.environ["LLM_CACHE_EMBEDDINGS_URL"]


if __name__ == "__main__":
    unittest.main()
