#!/usr/bin/env python3
"""Golden evals for llm-semantic-cache. Fully deterministic: fixed corpus,
injected fake clock, no network, no randomness. Two full pipeline runs must
produce byte-identical reports (eval 1).

Writes evals/eval_report.json on success.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from semantic_cache import SemanticCache, CacheConfig
from semantic_cache.embeddings import HashedEmbedding
from semantic_cache.pricing import estimate_cost
from semantic_cache.simulator import (
    ADVERSARIAL_PAIRS,
    CORPUS,
    DISSIMILAR_PROMPTS,
    PARAPHRASE_PAIRS,
    ScriptedLLM,
)
from semantic_cache.trace import TraceWriter

MODEL = "gpt-4o-mini"
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.json")


class FakeClock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        tick = self.t
        self.t += 0.001  # 1ms per read: deterministic latency numbers
        return tick

    def advance(self, s):
        self.t += s


def msg(text):
    return [{"role": "user", "content": text}]


def fresh_cache(clock, trace_path=None, **cfg):
    trace = TraceWriter(trace_path) if trace_path else None
    return SemanticCache(
        ScriptedLLM(), embedding_model=HashedEmbedding(),
        config=CacheConfig(**cfg), trace=trace, clock=clock, mono=clock,
    )


def eval_exact_match_recall():
    clock = FakeClock()
    cache = fresh_cache(clock)
    for q, _ in CORPUS:
        cache.chat(MODEL, msg(q))
    results = [cache.chat(MODEL, msg(q)) for q, _ in CORPUS]
    recalled = sum(1 for r in results if r.hit and r.kind == "exact")
    return {
        "pass": recalled == len(CORPUS),
        "recalled": recalled, "total": len(CORPUS),
    }


def eval_semantic_recall():
    clock = FakeClock()
    cache = fresh_cache(clock)
    for q, _ in CORPUS:
        cache.chat(MODEL, msg(q))
    results = [(p, cache.chat(MODEL, msg(p))) for _, p in PARAPHRASE_PAIRS]
    hits = [(p, r) for p, r in results if r.hit]
    scores = {p: round(r.score, 4) for p, r in hits}
    return {
        "pass": len(hits) == len(PARAPHRASE_PAIRS),
        "hits": len(hits), "total": len(PARAPHRASE_PAIRS),
        "min_score": round(min(scores.values()), 4) if scores else 0.0,
        "scores": scores,
    }


def eval_false_hit_guard():
    clock = FakeClock()
    cache = fresh_cache(clock)
    for q, _ in CORPUS:
        cache.chat(MODEL, msg(q))
    results = [(d, cache.chat(MODEL, msg(d))) for d in DISSIMILAR_PROMPTS]
    false_hits = [d for d, r in results if r.hit]
    max_score = max((r.score for _, r in results), default=0.0)
    return {
        "pass": not false_hits,
        "false_hits": false_hits,
        "total": len(DISSIMILAR_PROMPTS),
        "max_dissimilar_score": round(max_score, 4),
    }


def eval_threshold_sweep():
    rows = []
    thresholds = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30..0.95
    for thr in thresholds:
        clock = FakeClock()
        cache = fresh_cache(clock, threshold=thr)
        for q, _ in CORPUS:
            cache.chat(MODEL, msg(q))
        hits = sum(1 for _, p in PARAPHRASE_PAIRS if cache.chat(MODEL, msg(p)).hit)
        false_hits = sum(1 for d in DISSIMILAR_PROMPTS if cache.chat(MODEL, msg(d)).hit)
        adv_hits = sum(1 for _, d in ADVERSARIAL_PAIRS if cache.chat(MODEL, msg(d)).hit)
        hr = hits / len(PARAPHRASE_PAIRS)
        fhr = false_hits / len(DISSIMILAR_PROMPTS)
        rows.append({
            "threshold": thr,
            "hit_rate": round(hr, 4),
            "false_hit_rate": round(fhr, 4),
            "adversarial_false_hit_rate": round(adv_hits / len(ADVERSARIAL_PAIRS), 4),
            "youden_j": round(hr - fhr, 4),
        })
    # Operating point: max Youden's J, tie-break toward the higher threshold
    # (fewer false hits is the production-safe direction).
    best = max(rows, key=lambda r: (r["youden_j"], r["threshold"]))
    adv_scores = {}
    clock = FakeClock()
    emb = HashedEmbedding()
    for a, b in ADVERSARIAL_PAIRS:
        va, vb = emb.embed(a), emb.embed(b)
        adv_scores[f"{a} <-> {b}"] = round(sum(x * y for x, y in zip(va, vb)), 4)
    default_ok = (
        eval_semantic_recall()["pass"]
        and eval_false_hit_guard()["pass"]
        and abs(best["threshold"] - CacheConfig.threshold) < 1e-9
    )
    return {
        "pass": default_ok,
        "rows": rows,
        "operating_point": best["threshold"],
        "default_threshold": CacheConfig.threshold,
        "adversarial_scores": adv_scores,
    }


def eval_param_isolation():
    clock = FakeClock()
    cache = fresh_cache(clock)
    q = CORPUS[0][0]
    cache.chat(MODEL, msg(q), temperature=0.0)
    r_temp = cache.chat(MODEL, msg(q), temperature=0.9)
    r_topp = cache.chat(MODEL, msg(q), temperature=0.0, top_p=0.5)
    r_same = cache.chat(MODEL, msg(q), temperature=0.0)
    # Semantic path must also respect params: fresh cache primed at
    # temperature=0.0 only, then a paraphrase at 0.9 must miss even though
    # it would semantic-hit at 0.0.
    clock2 = FakeClock()
    cache2 = fresh_cache(clock2)
    cache2.chat(MODEL, msg(q), temperature=0.0)
    para = PARAPHRASE_PAIRS[0][1]
    r_sem = cache2.chat(MODEL, msg(para), temperature=0.9)
    r_sem_same = cache2.chat(MODEL, msg(para), temperature=0.0)
    return {
        "pass": (not r_temp.hit) and (not r_topp.hit) and (not r_sem.hit)
                and r_same.hit and r_sem_same.hit,
        "different_temperature_miss": not r_temp.hit,
        "different_top_p_miss": not r_topp.hit,
        "cross_param_semantic_miss": not r_sem.hit,
        "same_param_semantic_hit": r_sem_same.hit,
        "same_params_hit": r_same.hit,
    }


def eval_ttl_lru():
    clock = FakeClock()
    cache = fresh_cache(clock, ttl_seconds=60.0, max_entries=3)
    prompts = ["astronomy lecture notes", "sourdough baking tips", "chess opening theory"]
    for p in prompts:
        cache.chat(MODEL, msg(p))
    clock.advance(61.0)
    r_expired = cache.chat(MODEL, msg(prompts[0]))
    ttl_ok = not r_expired.hit

    clock2 = FakeClock()
    cache2 = fresh_cache(clock2, ttl_seconds=0, max_entries=2)
    cache2.chat(MODEL, msg("astronomy lecture notes"))
    cache2.chat(MODEL, msg("sourdough baking tips"))
    cache2.chat(MODEL, msg("astronomy lecture notes"))  # refresh
    cache2.chat(MODEL, msg("chess opening theory"))  # evicts sourdough
    r_kept = cache2.chat(MODEL, msg("astronomy lecture notes"))
    r_evicted = cache2.chat(MODEL, msg("sourdough baking tips"))
    lru_ok = r_kept.hit and not r_evicted.hit
    return {
        "pass": ttl_ok and lru_ok,
        "ttl_expiry_miss": ttl_ok,
        "lru_keeps_recent": r_kept.hit,
        "lru_evicts_stale": not r_evicted.hit,
    }


def eval_cost_reconciliation(tmp_trace):
    clock = FakeClock()
    cache = fresh_cache(clock, trace_path=tmp_trace)
    workload = [q for q, _ in CORPUS[:8]] + [q for q, _ in CORPUS[:8]]
    workload += [p for _, p in PARAPHRASE_PAIRS[:8]]
    workload += DISSIMILAR_PROMPTS[:4]
    for q in workload:
        cache.chat(MODEL, msg(q))
    records = TraceWriter(tmp_trace).read_all()
    ok = len(records) == len(workload)
    saved_sum = incurred_sum = 0.0
    for r in records:
        expect = estimate_cost(r["model"], r["prompt_tokens"], r["completion_tokens"])
        if r["hit"]:
            ok &= abs(r["est_cost_saved"] - round(expect["total"], 8)) < 1e-9
            ok &= r["cost_incurred"] == 0
            saved_sum += r["est_cost_saved"]
        else:
            ok &= abs(r["cost_incurred"] - round(expect["total"], 8)) < 1e-9
            ok &= r["est_cost_saved"] == 0
            incurred_sum += r["cost_incurred"]
        ok &= r["fallback_pricing"] == expect["fallback"]
    # No-cache counterfactual: every request pays full price.
    no_cache_total = saved_sum + incurred_sum
    return {
        "pass": bool(ok),
        "records": len(records),
        "est_cost_saved_usd": round(saved_sum, 8),
        "cost_incurred_usd": round(incurred_sum, 8),
        "no_cache_counterfactual_usd": round(no_cache_total, 8),
        "savings_rate": round(saved_sum / no_cache_total, 4) if no_cache_total else 0.0,
    }


def run_pipeline(tmp_trace):
    report = {
        "model": MODEL,
        "default_threshold": CacheConfig.threshold,
        "embedding": {"type": "hashed", "dim": 1024, "seed": "0xCACE"},
        "evals": {
            "exact_match_recall": eval_exact_match_recall(),
            "semantic_recall": eval_semantic_recall(),
            "false_hit_guard": eval_false_hit_guard(),
            "threshold_sweep": eval_threshold_sweep(),
            "param_isolation": eval_param_isolation(),
            "ttl_lru_eviction": eval_ttl_lru(),
            "cost_reconciliation": eval_cost_reconciliation(tmp_trace),
        },
    }
    return report


def main() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        trace1 = os.path.join(tmp, "t1.jsonl")
        trace2 = os.path.join(tmp, "t2.jsonl")
        run1 = run_pipeline(trace1)
        run2 = run_pipeline(trace2)
        blob1 = json.dumps(run1, indent=2, sort_keys=True)
        blob2 = json.dumps(run2, indent=2, sort_keys=True)
        deterministic = blob1 == blob2
        run1["evals"]["determinism"] = {
            "pass": deterministic,
            "runs_byte_identical": deterministic,
        }
        all_pass = all(e["pass"] for e in run1["evals"].values())
        run1["summary"] = {
            "evals_total": len(run1["evals"]),
            "evals_passed": sum(1 for e in run1["evals"].values() if e["pass"]),
            "all_pass": all_pass,
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(run1, indent=2, sort_keys=True) + "\n")
    for name, e in run1["evals"].items():
        print(f"[{'PASS' if e['pass'] else 'FAIL'}] {name}")
    print(f"report: {REPORT_PATH}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
