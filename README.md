# llm-semantic-cache

**Production cost engineering for AI apps: a semantic cache that sits in front of LLM chat calls and serves repeated or reworded prompts without paying for another inference.**

Every team shipping LLM features to customers eventually builds one of these: support copilots, RAG assistants, and code helpers all see the same questions asked slightly differently, hundreds of times a day. A semantic cache turns those repeats into sub-millisecond cache hits — cutting both latency and the inference bill. This project is a complete, dependency-free implementation of that layer: exact-match fast path, embedding-similarity matching, param-aware cache keys, TTL + LRU eviction, per-request cost tracing, and a golden eval suite that proves the hit/miss behavior.

Pairs naturally with [llm-cost-sidecar](../llm-cost-sidecar): the sidecar *measures* LLM spend, this cache *reduces* it.

> **Sample data only.** All Q&A content is fictional. Pricing in `pricing.json` is illustrative sample data, not real provider rates. No API keys, no network, no personal data — everything runs offline.

## Quickstart

```bash
python3 demo.py            # one-command end-to-end: 37 prompts, hits/misses, cost saved
python3 cli.py demo        # scripted demo through the CLI
python3 cli.py bench       # threshold sweep: hit-rate vs false-hit-rate table
python3 cli.py stats       # cache + trace summary
python3 cli.py clear       # wipe cache and trace
python3 cli.py serve       # tiny JSON API on 127.0.0.1:8471 (POST /chat)
python3 -m unittest discover -s tests   # 77 hermetic tests
python3 evals/run_evals.py              # 8 golden evals -> evals/eval_report.json
```

Wrap any OpenAI-compatible chat callable:

```python
from semantic_cache import SemanticCache, CacheConfig

cache = SemanticCache(llm_call, config=CacheConfig(threshold=0.55))
result = cache.chat("gpt-4o-mini", [{"role": "user", "content": "reset my password?"}])
# result.hit / result.kind ("exact" | "semantic" | "miss") / result.answer
# result.cost_saved -- dollars of inference avoided on a hit (sample pricing)
```

## Architecture

```
                    +------------------+
  chat(model,       |  SemanticCache   |
  messages,        |                  |
  temperature, +--->+  1. exact key:  +-- HIT (exact, score=1.0)
  top_p)           |  sha256(model |  |
                    |  norm(prompt) |  +-- HIT (semantic, score>=thr)
                    |  | temp/top_p) |        candidates filtered to
                    |                  |        SAME model + SAME params
                    |  2. cosine over  |
                    |  L2-normalized  +-- MISS -> llm_call() -> store
                    |  embeddings     |          (exact + semantic keys)
                    +--------+---------+
                             |  JSONL trace per request:
                             |  timestamp, model, prompt_hash, hit/miss,
                             |  matched score, latency_ms, tokens,
                             |  est. cost saved, fallback-pricing flag
                             v
                    +------------------+
                    | HashedEmbedding  |  default: offline, deterministic
                    | (char 3-grams + |  (blake2b-seeded feature hashing,
                    |  word 1/2-grams |   no PYTHONHASHSEED dependence)
                    |  -> 1024 dims)  |
                    +------------------+
  Optional: LLM_CACHE_EMBEDDINGS_URL -> OpenAI-compatible embeddings endpoint
```

**Key design decisions**

- **Cache key = model + normalized prompt + generation params.** Temperature and `top_p` are part of the key, and the semantic lookup only considers entries with the *same* model and *same* params fingerprint — a different temperature can never serve a stale or wrong-param response.
- **Pluggable embeddings.** Default is a deterministic offline hashed embedding (character 3-grams + stopword-filtered word unigrams/bigrams, signed hashing into 1024 dims, L2-normalized), so the whole project runs with zero keys and zero network. Set `LLM_CACHE_EMBEDDINGS_URL` to swap in any OpenAI-compatible embeddings endpoint.
- **Eviction:** TTL expiry (lazy, checked on access) + max-entries LRU, all under a re-entrant lock. The LLM call itself happens *outside* the lock so concurrent misses don't serialize.
- **Tracing:** one JSONL record per request with latency, tokens, and cost math — the raw material for "how much did the cache save us?" dashboards.

## Eval results

`python3 evals/run_evals.py` — 8 golden evals, all deterministic (two full runs must be byte-identical):

| Eval | Result |
|---|---|
| determinism — two runs byte-identical | ✅ PASS |
| exact-match recall (12/12 corpus questions re-queried) | ✅ PASS |
| semantic recall (12/12 hand-written paraphrases hit, min score 0.58) | ✅ PASS |
| false-hit guard (12/12 dissimilar prompts miss, max score 0.20) | ✅ PASS |
| threshold sweep — operating point picked (below) | ✅ PASS |
| param isolation (different temperature/top_p → miss, incl. semantic path) | ✅ PASS |
| TTL expiry + LRU eviction behavior | ✅ PASS |
| cost-savings reconciliation (trace sums == pricing math) | ✅ PASS |

Threshold sweep (hit-rate on paraphrases vs false-hit-rate on dissimilar prompts):

| threshold | hit_rate | false_hit_rate | adversarial* false_hit |
|---|---|---|---|
| 0.30–0.45 | 100% | 0% | 60–80% |
| **0.55** ✅ | **100%** | **0%** | **0%** |
| 0.60 | 92% | 0% | 0% |
| 0.70 | 58% | 0% | 0% |
| 0.90 | 0% | 0% | 0% |

\*Adversarial = lexically near-identical but semantically different ("speed of light" vs "speed of sound"). At the operating point the hashed embedding separates even these (max score 0.49 < 0.55) — see *Known limitations* below.

## Config / threshold guidance

- `threshold` (default **0.55**): the operating point from the sweep above — 100% paraphrase recall at 0% false hits on this corpus. Raise it if you see false hits in production; lower it if paraphrases miss.
- Why not 0.93? Textbook semantic-cache defaults (~0.9+) assume production embedding models, which separate paraphrases from near-misses far better than any offline hash. The bundled hashed embedding is tuned for zero-dependency demos; its *measured* operating point is 0.55. If you plug in a real embedding model via `LLM_CACHE_EMBEDDINGS_URL`, re-run `cli.py bench` and expect the sweet spot to move toward 0.90–0.95.
- `ttl_seconds` (default 3600): how long an entry stays valid. Shorten for fast-changing knowledge.
- `max_entries` (default 1000): LRU cap. Size it to your working set; each entry holds one 1024-dim vector (~8 KB).

## Dev loop: bugs the tests actually caught

1. **LRU desync between the exact and semantic stores.** The cache keeps two indexes over the same entries. On a hit I called `move_to_end` on the exact store only — so under an LRU cap, the semantic store evicted a *recently used* entry while the exact store evicted the right one. `test_lru_refresh_on_hit` caught it: a re-query returned a "semantic, score=1.0" hit for a prompt that had just been evicted. Fix: each entry now tracks its semantic key, and hits move-to-end in *both* stores.
2. **Hit/miss counters weren't persisted.** `save()`/`load()` persisted entries but not counters, so `cli.py stats` reported `hits=0` after a restart even with a warm cache. `test_stats_after_demo_shows_hits` caught it. Fix: counters are serialized with the snapshot.
3. **Param isolation hole in the first semantic-filter design.** The first draft filtered semantic candidates by params-fingerprint suffix only — a different *model* with the same params could have served a cross-model hit. Caught while writing `test_different_model_no_semantic_leak`. Fix: the filter suffix is now `sha256(model, params)`, an exact match on both.
4. **Eval-harness ordering bug in param isolation.** The eval primed the original prompt at temperature=0.9 in an earlier step, then asserted a paraphrase at 0.9 must miss — but that hit was *legitimate* (the 0.9 entry existed). The failure was in the test setup, not the cache. Fix: the cross-param semantic check now uses a fresh cache primed at one temperature only.

## Known limitations

- The hashed embedding is lexical, not semantic: it measures word/character overlap, so it can't distinguish meanings the way a transformer embedding can. It is honest about this — the adversarial pairs and the sweep exist to show exactly where the boundary lies.
- Token counts are word-based estimates (`1 word ≈ 1.3 tokens`), and pricing is sample data — cost-saved figures are directional, not billing-grade.
- `serve` is a minimal stdlib HTTP server for demos, not production infrastructure.

## Layout

```
semantic_cache/      the library: cache.py, embeddings.py, pricing.py, trace.py, simulator.py
cli.py               demo | bench | stats | clear | serve
demo.py              one-command end-to-end (no keys, no network)
evals/run_evals.py   8 golden evals -> evals/eval_report.json (committed)
tests/               77 hermetic unittest tests (injected clocks, fixed seeds)
```

## License

MIT — see [LICENSE](LICENSE).
