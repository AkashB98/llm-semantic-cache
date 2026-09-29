"""Token pricing math over the bundled SAMPLE pricing.json.

All rates are illustrative sample data, NOT real provider prices. Unknown
models fall back to a flat fallback rate and are flagged with
fallback=True so "cost saved" numbers never silently mislead.
"""

import json
import os

DEFAULT_PRICING_PATH = os.path.join(os.path.dirname(__file__), "pricing.json")

# Flat fallback rate per 1M tokens (input and output) for unlisted models.
FALLBACK_RATE_PER_1M = 5.00

_cache = {}


def load_pricing(path: str = None) -> dict:
    path = path or DEFAULT_PRICING_PATH
    if path not in _cache:
        with open(path, "r", encoding="utf-8") as f:
            _cache[path] = json.load(f)
    return _cache[path]


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, path: str = None) -> dict:
    """Return {"input": $, "output": $, "total": $, "fallback": bool}."""
    pricing = load_pricing(path)
    entry = pricing.get("models", {}).get(model)
    if entry is None:
        rate_in = rate_out = FALLBACK_RATE_PER_1M
        fallback = True
    else:
        rate_in = entry["input_per_1m"]
        rate_out = entry["output_per_1m"]
        fallback = False
    cost_in = prompt_tokens / 1_000_000 * rate_in
    cost_out = completion_tokens / 1_000_000 * rate_out
    return {
        "input": cost_in,
        "output": cost_out,
        "total": cost_in + cost_out,
        "fallback": fallback,
        "rate_input_per_1m": rate_in,
        "rate_output_per_1m": rate_out,
    }
