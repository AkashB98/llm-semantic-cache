"""Pluggable embedding models.

Default: HashedEmbedding -- a deterministic, offline embedding built from
character n-gram feature hashing. It runs with zero keys and zero network,
which makes every test and demo fully hermetic and reproducible.

Design notes:
- Python's built-in hash() is salted per process (PYTHONHASHSEED), so we never
  use it. All hashing goes through hashlib.blake2b with an explicit seed, so
  embeddings are byte-identical across processes, machines and runs.
- Text is normalized (lowercase, whitespace collapsed, punctuation stripped)
  before n-gram extraction so trivial surface differences don't dominate.
- Signed feature hashing (sign +/-1 per bucket) reduces collision bias vs
  plain counting, then the vector is L2-normalized so cosine == dot product.

Optional: set LLM_CACHE_EMBEDDINGS_URL to point at an OpenAI-compatible
embeddings endpoint (POST {url} with {"input": text, "model": name}) and
HttpEmbeddingModel will call it with stdlib urllib. It is documented but
never required -- the default path never touches the network.
"""

import hashlib
import json
import math
import os
import re
import urllib.request


class EmbeddingModel:
    """Interface: embed(text) -> list[float], L2-normalized."""

    dim = 0

    def embed(self, text: str):
        raise NotImplementedError


def normalize_text(text: str) -> str:
    """Canonical normalization shared by exact keys and embeddings."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STOPWORDS = frozenset({
    "what", "is", "the", "a", "an", "of", "to", "in", "on", "for", "how",
    "do", "i", "my", "it", "and", "or", "me", "you", "your", "tell", "give",
    "can", "does", "are", "be", "by", "as", "at", "with", "from", "that",
    "this", "s", "re", "ll", "d",
})


class HashedEmbedding(EmbeddingModel):
    """Deterministic offline embedding via character n-gram feature hashing.

    **hashing trick** -- a memory-savvy way to turn sparse features into a
    fixed-size vector: each feature is hashed to a bucket index, and counts
    accumulate in the buckets. Collisions are tolerated because they're rare
    and statistically cancel out (signed hashing makes positive/negative
    collisions balance). No vocabulary to train, no model to download.

    Feature mix (weights sum before L2 normalization):
      - character 3-grams (weight 1): typo/punctuation robustness
      - content-word unigrams (weight 4): what the question is *about*
      - content-word bigrams (weight 4): short phrases ("reset password")
    Stopwords are dropped from word features so template overlap
    ("what is the X of Y") doesn't inflate similarity between genuinely
    different questions.
    """

    def __init__(self, dim: int = 1024, ngram: int = 3, seed: int = 0xCACE,
                 word_weight: float = 4.0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if ngram < 2:
            raise ValueError("ngram must be >= 2")
        self.dim = dim
        self.ngram = ngram
        self.seed = seed
        self.word_weight = word_weight

    def _bucket(self, gram: str):
        digest = hashlib.blake2b(
            gram.encode("utf-8"),
            digest_size=8,
            person=str(self.seed).encode("utf-8"),
        ).digest()
        as_int = int.from_bytes(digest, "big")
        return as_int % self.dim, 1 if (as_int >> 63) & 1 else -1

    def features(self, text: str):
        """Yield (feature_string, weight) pairs -- exposed for tests."""
        norm = normalize_text(text)
        words = norm.split()
        content = [w for w in words if w not in STOPWORDS]
        padded = " " + norm + " "
        for i in range(len(padded) - self.ngram + 1):
            yield padded[i : i + self.ngram], 1.0
        for w in content:
            yield "w:" + w, self.word_weight
        for i in range(len(content) - 1):
            yield "b:" + " ".join(content[i : i + 2]), self.word_weight

    def embed(self, text: str):
        vec = [0.0] * self.dim
        for gram, weight in self.features(text):
            idx, sign = self._bucket(gram)
            vec[idx] += sign * weight
        norm_len = math.sqrt(sum(x * x for x in vec))
        if norm_len == 0.0:
            # Empty/degenerate input -> fixed unit vector so cosine stays valid.
            vec[0] = 1.0
            return vec
        return [x / norm_len for x in vec]


class HttpEmbeddingModel(EmbeddingModel):
    """OpenAI-compatible HTTP embeddings (optional, never required).

    Expects POST {url} with JSON {"input": text, "model": model} returning
    {"data": [{"embedding": [...]}]}. Uses stdlib only.
    """

    def __init__(self, url: str, model: str = "text-embedding-3-small", dim: int = 1536, timeout: int = 10):
        self.url = url
        self.model = model
        self.dim = dim
        self.timeout = timeout

    def embed(self, text: str):
        payload = json.dumps({"input": text, "model": self.model}).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        emb = body["data"][0]["embedding"]
        norm_len = math.sqrt(sum(x * x for x in emb))
        if norm_len == 0.0:
            raise ValueError("embedding endpoint returned zero vector")
        return [x / norm_len for x in emb]


def embedding_from_env(default: EmbeddingModel = None) -> EmbeddingModel:
    """Pick the embedding model: HTTP hook when LLM_CACHE_EMBEDDINGS_URL is set.

    Env knobs (all optional):
      LLM_CACHE_EMBEDDINGS_URL  -- OpenAI-compatible embeddings endpoint
      LLM_CACHE_EMBEDDINGS_MODEL -- model name sent to the endpoint
      LLM_CACHE_EMBEDDINGS_DIM   -- expected dimension (default 1536)
    """
    url = os.environ.get("LLM_CACHE_EMBEDDINGS_URL")
    if url:
        return HttpEmbeddingModel(
            url,
            model=os.environ.get("LLM_CACHE_EMBEDDINGS_MODEL", "text-embedding-3-small"),
            dim=int(os.environ.get("LLM_CACHE_EMBEDDINGS_DIM", "1536")),
        )
    return default or HashedEmbedding()
