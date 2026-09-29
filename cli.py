#!/usr/bin/env python3
"""CLI for llm-semantic-cache.

Commands:
  demo   scripted end-to-end run showing hits/misses (zero keys/network)
  bench  threshold sweep: hit-rate vs false-hit-rate table
  stats  cache + trace summary
  clear  wipe cache file and trace
  serve  tiny JSON API on localhost (POST /chat)

Exit codes: 0 ok, 1 usage error, 2 runtime error.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_cache import SemanticCache, CacheConfig, embedding_from_env
from semantic_cache.simulator import (
    ScriptedLLM, CORPUS, PARAPHRASE_PAIRS, DISSIMILAR_PROMPTS, ADVERSARIAL_PAIRS,
)
from semantic_cache.trace import TraceWriter

BASE_DIR = os.environ.get("LLM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".llm-semantic-cache"))
CACHE_FILE = os.path.join(BASE_DIR, "cache.json")
TRACE_FILE = os.path.join(BASE_DIR, "trace.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"


def _ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def _make_cache(threshold=None, ttl=None, max_entries=None):
    _ensure_dir()
    trace = TraceWriter(TRACE_FILE)
    llm = ScriptedLLM()
    cfg = CacheConfig(
        threshold=CacheConfig.threshold if threshold is None else threshold,
        ttl_seconds=CacheConfig.ttl_seconds if ttl is None else ttl,
        max_entries=CacheConfig.max_entries if max_entries is None else max_entries,
    )
    cache = SemanticCache(llm, embedding_model=embedding_from_env(), config=cfg, trace=trace)
    if os.path.exists(CACHE_FILE):
        try:
            cache.load(CACHE_FILE)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return cache, llm


def _save(cache):
    _ensure_dir()
    cache.save(CACHE_FILE)


def cmd_demo(args):
    """Scripted deterministic demo: repeat prompts, paraphrases, new topics."""
    cache, llm = _make_cache()
    cache.clear()
    model = DEFAULT_MODEL
    print(f"model={model}  threshold={cache.config.threshold}")
    print("-" * 64)
    script = []
    for q, _ in CORPUS[:6]:
        script.append(q)
    for q, _ in CORPUS[:6]:  # exact repeats -> exact hits
        script.append(q)
    for _, para in PARAPHRASE_PAIRS[:6]:  # paraphrases -> semantic hits
        script.append(para)
    script += DISSIMILAR_PROMPTS[:4]  # new topics -> misses
    for i, q in enumerate(script):
        r = cache.chat(model, [{"role": "user", "content": q}])
        mark = {"exact": "EXACT-HIT", "semantic": "SEM-HIT ", "miss": "MISS    "}[r.kind]
        print(f"[{i+1:2d}] {mark} score={r.score:.3f} lat={r.latency_ms:6.2f}ms  {q[:48]}")
    _save(cache)
    s = cache.stats()
    print("-" * 64)
    print(f"LLM calls made: {llm.calls} / {len(script)} prompts  "
          f"(saved {len(script) - llm.calls} calls)")
    print(f"hits={s['hits']} misses={s['misses']} hit_rate={s['hit_rate']:.2%}")
    return 0


def cmd_bench(args):
    """Threshold sweep: for each threshold, hit-rate on paraphrases vs
    false-hit-rate on dissimilar prompts (and adversarial near-duplicates)."""
    thresholds = [round(0.30 + 0.05 * i, 2) for i in range(15)]  # 0.30..1.00
    print(f"{'thr':>5} {'hit_rate':>8} {'false_hit':>9} {'adv_false':>9}  J")
    print("-" * 48)
    rows = []
    for thr in thresholds:
        cache, _ = _make_cache(threshold=thr)
        cache.clear()
        model = DEFAULT_MODEL
        for q, _ in CORPUS:
            cache.chat(model, [{"role": "user", "content": q}])
        hits = sum(
            1 for _, p in PARAPHRASE_PAIRS
            if cache.chat(model, [{"role": "user", "content": p}]).hit
        )
        false_hits = sum(
            1 for d in DISSIMILAR_PROMPTS
            if cache.chat(model, [{"role": "user", "content": d}]).hit
        )
        adv_hits = sum(
            1 for _, d in ADVERSARIAL_PAIRS
            if cache.chat(model, [{"role": "user", "content": d}]).hit
        )
        hr = hits / len(PARAPHRASE_PAIRS)
        fhr = false_hits / len(DISSIMILAR_PROMPTS)
        afhr = adv_hits / len(ADVERSARIAL_PAIRS)
        j = hr - fhr  # Youden's J: operating-point score
        rows.append((thr, hr, fhr, afhr, j))
        print(f"{thr:5.2f} {hr:8.2%} {fhr:9.2%} {afhr:9.2%}  {j:.3f}")
    best = max(rows, key=lambda r: (r[4], -r[0]))
    print("-" * 48)
    print(f"operating point: threshold={best[0]:.2f} "
          f"(hit_rate={best[1]:.0%}, false_hit_rate={best[2]:.0%}, J={best[4]:.3f})")
    return 0


def cmd_stats(args):
    cache, _ = _make_cache()
    s = cache.stats()
    trace = TraceWriter(TRACE_FILE)
    records = trace.read_all()
    saved = sum(r.get("est_cost_saved", 0.0) for r in records)
    spent = sum(r.get("cost_incurred", 0.0) for r in records)
    fallback = sum(1 for r in records if r.get("fallback_pricing"))
    print(json.dumps({
        "entries": s["entries"], "hits": s["hits"], "misses": s["misses"],
        "hit_rate": round(s["hit_rate"], 4),
        "trace_records": len(records),
        "est_cost_saved_usd": round(saved, 6),
        "cost_incurred_usd": round(spent, 6),
        "fallback_pricing_records": fallback,
        "trace_file": TRACE_FILE, "cache_file": CACHE_FILE,
    }, indent=2))
    return 0


def cmd_clear(args):
    cache, _ = _make_cache()
    cache.clear()
    for path in (CACHE_FILE, TRACE_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    print("cache and trace cleared")
    return 0


def cmd_serve(args):
    """Tiny JSON API: POST /chat {"model","messages","temperature","top_p"}."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    cache, _ = _make_cache()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/chat":
                return self._send(404, {"error": "use POST /chat"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                model = payload.get("model", DEFAULT_MODEL)
                messages = payload.get("messages", [])
                params = {k: payload[k] for k in ("temperature", "top_p") if k in payload}
                r = cache.chat(model, messages, **params)
                _save(cache)
                self._send(200, {
                    "hit": r.hit, "kind": r.kind, "answer": r.answer,
                    "score": r.score, "latency_ms": r.latency_ms,
                    "cost_saved": r.cost_saved, "fallback_pricing": r.fallback_pricing,
                })
            except Exception as exc:  # noqa: BLE001 -- tiny demo server
                self._send(400, {"error": str(exc)})

        def log_message(self, *a):
            pass

    port = args.port
    print(f"serving on http://127.0.0.1:{port}  (POST /chat)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cli.py", description="llm-semantic-cache CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="scripted end-to-end demo (no keys/network)")
    sub.add_parser("bench", help="threshold sweep: hit-rate vs false-hit-rate")
    sub.add_parser("stats", help="cache + trace summary")
    sub.add_parser("clear", help="wipe cache file and trace")
    serve_p = sub.add_parser("serve", help="tiny JSON API on localhost")
    serve_p.add_argument("--port", type=int, default=8471)
    args = parser.parse_args(argv)
    try:
        return {
            "demo": cmd_demo, "bench": cmd_bench, "stats": cmd_stats,
            "clear": cmd_clear, "serve": cmd_serve,
        }[args.command](args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 -- report, don't traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
