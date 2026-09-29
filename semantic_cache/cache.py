"""SemanticCache: the cache layer in front of any chat(model=, messages=) call.

Lookup has two paths:
  1. Exact path -- SHA256 of (model, normalized prompt, generation params).
  2. Semantic path -- cosine similarity over L2-normalized embeddings,
     but ONLY among entries with the same model AND the same generation
     params fingerprint, so a different temperature can never serve a
     stale/wrong-param response.

Eviction: TTL expiry (lazy on access) + max-entries LRU. All state mutations
are guarded by a re-entrant lock, so concurrent chat calls are safe.

Tracing: every request appends one JSONL record via trace.TraceWriter.
Clocks are injectable (clock=, mono=) so tests never depend on wall time.
"""

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .embeddings import HashedEmbedding, normalize_text
from .pricing import estimate_cost
from .trace import TraceWriter


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _params_fingerprint(params: dict) -> str:
    """Stable fingerprint of the generation params that affect the output."""
    relevant = {k: params.get(k) for k in ("temperature", "top_p") if k in params}
    return json.dumps(relevant, sort_keys=True, default=str)


@dataclass
class CacheConfig:
    """Configuration knobs for SemanticCache.

    **cosine similarity** -- plain-English: a 0-to-1 score of how alike two
    embedding vectors point; 1.0 = identical direction, 0 = unrelated.
    """

    threshold: float = 0.55  # semantic hit boundary on cosine similarity.
    # Tuned by the threshold sweep in evals/ for the bundled offline hashed
    # embedding (paraphrase min ~0.58, guard-corpus max ~0.20). With a real
    # embedding model via LLM_CACHE_EMBEDDINGS_URL, raise toward 0.90-0.95:
    # production embeddings separate paraphrases from near-misses far better.
    ttl_seconds: float = 3600.0  # 0 or negative disables TTL
    max_entries: int = 1000  # 0 or negative disables the LRU cap
    embedding_dim: int = 1024


@dataclass
class CacheResult:
    hit: bool
    kind: str  # "exact", "semantic", or "miss"
    answer: str = ""
    score: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_incurred: float = 0.0
    cost_saved: float = 0.0
    fallback_pricing: bool = False


@dataclass
class _Entry:
    embedding: list
    answer: str
    prompt_tokens: int
    completion_tokens: int
    created_at: float
    last_used_at: float
    sem_key: str = ""  # key of this entry's mirror in the semantic store
    uses: int = 0


class SemanticCache:
    """Thread-safe semantic cache wrapping an LLM chat callable."""

    def __init__(
        self,
        llm_call,
        embedding_model=None,
        config: CacheConfig = None,
        pricing_path: str = None,
        trace: TraceWriter = None,
        clock=None,
        mono=None,
    ):
        self.llm_call = llm_call
        self.embedding = embedding_model or HashedEmbedding()
        self.config = config or CacheConfig()
        self.pricing_path = pricing_path
        self.trace = trace
        self._clock = clock or time.time
        self._mono = mono or time.perf_counter
        self._lock = threading.RLock()
        self._exact = OrderedDict()  # exact_key -> _Entry
        self._semantic = OrderedDict()  # semantic_key -> _Entry
        self.hits = 0
        self.misses = 0

    # -- key construction -------------------------------------------------

    def _keys(self, model: str, prompt: str, params: dict):
        norm = normalize_text(prompt)
        pf = _params_fingerprint(params)
        exact_key = _sha256_hex("\x1f".join([model, norm, pf]))
        semantic_key = _sha256_hex("\x1f".join([model, norm, pf, "sem"]))
        return norm, pf, exact_key, semantic_key

    def _expired(self, entry: _Entry, now: float) -> bool:
        ttl = self.config.ttl_seconds
        return ttl > 0 and (now - entry.created_at) >= ttl

    def _prune(self, now: float):
        """Drop TTL-expired entries and enforce the LRU cap."""
        for store in (self._exact, self._semantic):
            for key in [k for k, e in store.items() if self._expired(e, now)]:
                del store[key]
        cap = self.config.max_entries
        if cap > 0:
            for store in (self._exact, self._semantic):
                while len(store) > cap:
                    store.popitem(last=False)

    # -- lookup -----------------------------------------------------------

    @staticmethod
    def _cosine(a, b) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _semantic_best(self, model: str, pf: str, vec, now: float):
        """Best candidate among entries with matching model+params.

        Each semantic key ends with "|" + sha256(model, pf), so filtering is
        an exact suffix match: a different model or different generation
        params can never serve a stale/wrong-param response.
        """
        suffix = "|" + _sha256_hex("\x1f".join([model, pf]))
        best_key, best, best_score = None, None, -1.0
        for key, entry in self._semantic.items():
            if key.endswith(suffix) and not self._expired(entry, now):
                s = self._cosine(vec, entry.embedding)
                if s > best_score:
                    best_key, best, best_score = key, entry, s
        return best_key, best, best_score

    def chat(self, model: str, messages: list, **params) -> CacheResult:
        """Cached chat completion. Signature mirrors OpenAI chat calls."""
        t0 = self._mono()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
        norm, pf, exact_key, _ = self._keys(model, prompt, params)
        prompt_hash = _sha256_hex(norm)[:16]
        vec = self.embedding.embed(norm)

        with self._lock:
            now = self._clock()
            self._prune(now)

            entry = self._exact.get(exact_key)
            if entry is not None and not self._expired(entry, now):
                return self._record_hit("exact", entry, exact_key, entry.sem_key,
                                        model, prompt_hash, 1.0, t0, params)

            sem_key, best, score = self._semantic_best(model, pf, vec, now)
            if best is not None and score >= self.config.threshold:
                return self._record_hit("semantic", best, None, sem_key,
                                        model, prompt_hash, score, t0, params)

            # Miss: call through, then store.

        # The actual LLM call happens outside the lock so concurrent
        # misses don't serialize on network latency.
        answer, prompt_tokens, completion_tokens = self.llm_call(model, messages, **params)

        with self._lock:
            now = self._clock()
            cost = estimate_cost(model, prompt_tokens, completion_tokens, self.pricing_path)
            # Semantic key: digest of (model, norm, pf) plus a "|" + digest
            # of (model, pf) suffix, so the lookup can filter candidates by
            # model+params with an exact suffix match -- no side index needed.
            semantic_key = (
                _sha256_hex("\x1f".join([model, norm, pf]))
                + "|"
                + _sha256_hex("\x1f".join([model, pf]))
            )
            entry = _Entry(
                embedding=vec,
                answer=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                created_at=now,
                last_used_at=now,
                sem_key=semantic_key,
            )
            self._exact[exact_key] = entry
            self._semantic[semantic_key] = entry
            self.misses += 1
            latency_ms = (self._mono() - t0) * 1000.0
            result = CacheResult(
                hit=False, kind="miss", answer=answer, latency_ms=latency_ms,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cost_incurred=cost["total"], fallback_pricing=cost["fallback"],
            )
            self._write_trace(model, prompt_hash, result, t0, params)
            self._prune(now)
            return result

    def _record_hit(self, kind, entry, exact_key, sem_key, model, prompt_hash, score, t0, params):
        now = self._clock()
        entry.last_used_at = now
        entry.uses += 1
        self.hits += 1
        # Move-to-end on BOTH stores so their LRU orders stay consistent.
        # (Skipping the semantic store here once caused it to evict a
        # recently-used entry while the exact store evicted the right one.)
        if exact_key is not None and exact_key in self._exact:
            self._exact.move_to_end(exact_key)
        if sem_key and sem_key in self._semantic:
            self._semantic.move_to_end(sem_key)
        latency_ms = (self._mono() - t0) * 1000.0
        cost = estimate_cost(model, entry.prompt_tokens, entry.completion_tokens, self.pricing_path)
        result = CacheResult(
            hit=True, kind=kind, answer=entry.answer, score=score,
            latency_ms=latency_ms,
            prompt_tokens=entry.prompt_tokens, completion_tokens=entry.completion_tokens,
            cost_saved=cost["total"], fallback_pricing=cost["fallback"],
        )
        self._write_trace(model, prompt_hash, result, t0, params)
        return result

    def _write_trace(self, model, prompt_hash, result: CacheResult, t0, params):
        if self.trace is None:
            return
        self.trace.write({
            "timestamp": self._clock_iso(),
            "model": model,
            "prompt_hash": prompt_hash,
            "hit": result.hit,
            "kind": result.kind,
            "matched_score": round(result.score, 6),
            "latency_ms": round(result.latency_ms, 3),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "est_cost_saved": round(result.cost_saved, 8),
            "cost_incurred": round(result.cost_incurred, 8),
            "fallback_pricing": result.fallback_pricing,
        })

    def _clock_iso(self) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(self._clock(), tz=datetime.timezone.utc).isoformat()

    # -- persistence / maintenance ----------------------------------------

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._exact),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "threshold": self.config.threshold,
                "ttl_seconds": self.config.ttl_seconds,
                "max_entries": self.config.max_entries,
            }

    def save(self, path: str):
        """Persist entries (JSON). Embeddings are plain lists, no pickle."""
        with self._lock:
            data = {
                "exact_keys": list(self._exact.keys()),
                "hits": self.hits,
                "misses": self.misses,
                "entries": [
                    {
                        "exact_key": k,
                        "embedding": e.embedding,
                        "answer": e.answer,
                        "prompt_tokens": e.prompt_tokens,
                        "completion_tokens": e.completion_tokens,
                        "created_at": e.created_at,
                        "last_used_at": e.last_used_at,
                        "uses": e.uses,
                    }
                    for k, e in self._exact.items()
                ],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._exact.clear()
            self._semantic.clear()
            self.hits = data.get("hits", 0)
            self.misses = data.get("misses", 0)
            for item in data.get("entries", []):
                entry = _Entry(
                    embedding=item["embedding"], answer=item["answer"],
                    prompt_tokens=item["prompt_tokens"],
                    completion_tokens=item["completion_tokens"],
                    created_at=item["created_at"], last_used_at=item["last_used_at"],
                    uses=item.get("uses", 0),
                )
                self._exact[item["exact_key"]] = entry
                # Semantic keys are not recoverable exactly (they embed the
                # params suffix), so loaded entries are exact-hit only: they
                # get a sentinel suffix that never matches a live query's
                # model+params fingerprint. That keeps cross-model and
                # cross-param isolation airtight after a restart.
                sem_key = item["exact_key"] + "|\x00loaded"
                entry.sem_key = sem_key
                self._semantic[sem_key] = entry

    def clear(self):
        with self._lock:
            self._exact.clear()
            self._semantic.clear()
            self.hits = 0
            self.misses = 0
