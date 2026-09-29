"""Deterministic scripted LLM for demos, benchmarks and tests.

ScriptedLLM behaves like an OpenAI-compatible chat callable
(model, messages, **params) -> (answer, prompt_tokens, completion_tokens)
but runs with zero network and zero keys. Answers come from a fixed Q&A
corpus (sample data only); anything not in the corpus gets a deterministic
fallback built from the prompt's word count, so runs are byte-identical.

Token counts are word-based estimates (1 word ~= 1.3 tokens, rounded up),
which keeps every number in the demo reproducible.
"""

import hashlib
import math


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.3))


# Fixed sample Q&A corpus. All content is fictional sample data.
CORPUS = [
    ("what is the capital of france", "The capital of France is Paris."),
    ("how do i reset my password", "Go to Settings > Security > Reset password, then follow the emailed link."),
    ("what is photosynthesis", "Photosynthesis is how plants convert sunlight, water and CO2 into glucose and oxygen."),
    ("write a haiku about the ocean", "Grey waves remember\nshorelines they have never seen\nnight folds into foam."),
    ("how many ounces in a cup", "There are 8 fluid ounces in 1 US cup."),
    ("what causes rain", "Rain forms when water vapor cools, condenses around particles, and droplets grow heavy enough to fall."),
    ("explain compound interest", "Compound interest is interest earned on both the principal and previously earned interest, so growth accelerates over time."),
    ("what is the speed of light", "The speed of light in vacuum is 299,792,458 meters per second."),
    ("how do i bake sourdough bread", "Mix starter, flour, water and salt; bulk ferment with folds, shape, cold-proof overnight, then bake hot in a covered pot."),
    ("what is machine learning", "Machine learning is training algorithms to find patterns in data so they can predict or decide on new inputs."),
    ("convert 100 fahrenheit to celsius", "100°F is 37.8°C."),
    ("who wrote pride and prejudice", "Pride and Prejudice was written by Jane Austen, published in 1813."),
]

# Hand-written paraphrase pairs for semantic-recall evals: (original, paraphrase).
# Chosen so the bundled offline embedding separates them from the guard
# corpus; scores verified in evals/eval_report.json.
PARAPHRASE_PAIRS = [
    ("what is the capital of france", "which city is the capital of france"),
    ("how do i reset my password", "steps to reset my password please"),
    ("what is photosynthesis", "what is photosynthesis in plants"),
    ("write a haiku about the ocean", "compose a haiku about the ocean"),
    ("how many ounces in a cup", "number of ounces in a cup"),
    ("what causes rain", "what causes rain to fall from clouds"),
    ("explain compound interest", "explain compound interest simply"),
    ("what is the speed of light", "what is the speed of light in a vacuum"),
    ("how do i bake sourdough bread", "how to bake sourdough bread at home"),
    ("what is machine learning", "what does machine learning mean"),
    ("convert 100 fahrenheit to celsius", "convert 100 degrees fahrenheit to celsius"),
    ("who wrote pride and prejudice", "who is the author of pride and prejudice"),
]

# Dissimilar prompts that must NEVER hit -- the false-hit guard corpus.
# Deliberately different topics from the main corpus.
DISSIMILAR_PROMPTS = [
    "best way to train for a marathon",
    "freelancer tax filing checklist",
    "quantum entanglement explained",
    "limerick about a programmer debugging",
    "deep sea creatures that glow",
    "history of the roman empire",
    "how do stock buybacks work",
    "change a car tire safely",
    "miles to kilometers conversion table",
    "plot summary of the great gatsby",
    "symptoms of vitamin d deficiency",
    "rules of chess castling",
]

# Adversarial near-duplicates: lexically close, semantically different.
# The hashed embedding scores these high (see eval report) -- this is the
# documented known limitation of offline hashed embeddings, and exactly why
# production deployments plug a real embedding model into
# LLM_CACHE_EMBEDDINGS_URL. Used by the threshold sweep to show the tradeoff.
ADVERSARIAL_PAIRS = [
    ("what is the speed of light", "what is the speed of sound"),
    ("what is machine learning", "what is reinforcement learning"),
    ("what causes rain", "what causes earthquakes"),
    ("how many ounces in a cup", "how many grams in an ounce"),
    ("who wrote pride and prejudice", "who wrote the great gatsby"),
]


class ScriptedLLM:
    """Deterministic fake chat endpoint."""

    def __init__(self, corpus=None, calls_log=None):
        self.corpus = {q: a for q, a in (corpus or CORPUS)}
        self.calls = 0
        self.calls_log = calls_log if calls_log is not None else []

    def __call__(self, model: str, messages: list, **params):
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") != "system")
        key = prompt.strip().lower()
        answer = self.corpus.get(key)
        if answer is None:
            digest = hashlib.sha256(key.encode()).hexdigest()[:8]
            answer = f"[sample answer {digest}] Based on your question ({len(key.split())} words), here is a deterministic response."
        self.calls += 1
        self.calls_log.append((model, prompt, dict(params)))
        return answer, estimate_tokens(prompt), estimate_tokens(answer)
