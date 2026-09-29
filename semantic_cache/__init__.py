"""llm-semantic-cache: an embedding-similarity semantic cache for LLM chat calls.

Sample data only. Everything runs locally with zero keys and zero network.
"""

from .cache import SemanticCache, CacheConfig, CacheResult
from .embeddings import HashedEmbedding, EmbeddingModel, embedding_from_env

__all__ = [
    "SemanticCache",
    "CacheConfig",
    "CacheResult",
    "HashedEmbedding",
    "EmbeddingModel",
    "embedding_from_env",
]

__version__ = "0.1.0"
