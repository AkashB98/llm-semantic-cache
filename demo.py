#!/usr/bin/env python3
"""One-command end-to-end demo: zero keys, zero network.

Runs a scripted workload through SemanticCache backed by the deterministic
ScriptedLLM, prints per-request hits/misses with scores and the cost saved,
then verifies the exact-match recall of the second pass.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_cache import SemanticCache, CacheConfig
from semantic_cache.simulator import ScriptedLLM, CORPUS, PARAPHRASE_PAIRS, DISSIMILAR_PROMPTS
from semantic_cache.trace import TraceWriter

MODEL = "gpt-4o-mini"


def main() -> int:
    trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_trace.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)
    llm = ScriptedLLM()
    cache = SemanticCache(
        llm, config=CacheConfig(threshold=0.55), trace=TraceWriter(trace_path)
    )

    workload = [q for q, _ in CORPUS[:8]]
    workload += [q for q, _ in CORPUS[:8]]          # exact repeats
    workload += [p for _, p in PARAPHRASE_PAIRS[:8]]  # paraphrases
    workload += DISSIMILAR_PROMPTS[:4]              # brand-new topics

    print(f"demo: {len(workload)} prompts through the semantic cache (model={MODEL})")
    print("=" * 70)
    saved_total = 0.0
    for i, q in enumerate(workload, 1):
        r = cache.chat(MODEL, [{"role": "user", "content": q}])
        tag = {"exact": "EXACT HIT", "semantic": "SEMANTIC HIT", "miss": "MISS"}[r.kind]
        saved_total += r.cost_saved
        print(f"{i:2d}. [{tag:12s}] score={r.score:0.3f} saved=${r.cost_saved:0.7f}  {q[:52]}")

    # Param isolation: same prompt, different temperature must miss.
    r = cache.chat(MODEL, [{"role": "user", "content": workload[0]}], temperature=0.9)
    assert not r.hit, "param isolation failed: different temperature must miss"
    print("param isolation check: same prompt @ temperature=0.9 -> MISS (correct)")

    # Second pass over the originals: everything must be an exact hit now.
    second = [cache.chat(MODEL, [{"role": "user", "content": q}]) for q, _ in CORPUS[:8]]
    assert all(x.hit and x.kind == "exact" for x in second), "exact recall failed"
    print("exact-match recall check: 8/8 second-pass prompts -> EXACT HIT")

    s = cache.stats()
    print("=" * 70)
    print(f"LLM calls: {llm.calls} for {len(workload) + 1 + 8} requests "
          f"-> {len(workload) + 9 - llm.calls} served from cache")
    print(f"hits={s['hits']} misses={s['misses']} hit_rate={s['hit_rate']:.1%}")
    print(f"estimated cost saved: ${saved_total:.6f} (sample pricing, not real rates)")
    print(f"trace: {trace_path} ({len(TraceWriter(trace_path).read_all())} records)")
    print("demo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
