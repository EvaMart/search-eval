"""
What the retrievers share: the client of the Qdrant indexes (`tool_search` and
the experimental `tool_search_exp_names`), the query
encoder and the shape of a hit (docs/qdrant.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache, lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from search_eval.common import DEFAULT_MODEL, load_model

COLLECTION = "tool_search"
# Experimental, BM25-only, one point per tool (`rsetl search-index --names`).
NAMES_COLLECTION = "tool_search_exp_names"


@dataclass(frozen=True)
class Hit:
    tool_id: str
    name: str
    score: float
    kind: str
    text: str


@cache
def client() -> QdrantClient:
    load_dotenv()
    return QdrantClient(
        # Always pass url: with an API key and no url the client tries https, and
        # the SSH tunnel speaks plain http.
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,  # BM25 queries are sent as text
        timeout=60,
    )


@cache
def encoder(name: str = DEFAULT_MODEL):
    return load_model(name)


_PRIMED: dict[tuple[str, str], list[float]] = {}


def prime(queries: list[str], model_name: str = DEFAULT_MODEL) -> None:
    """Encode many queries in one batch, for `encode` to return (an evaluation run)."""
    todo = [q for q in dict.fromkeys(queries) if (q, model_name) not in _PRIMED]
    if todo:
        vectors = encoder(model_name).encode(todo, normalize_embeddings=True, batch_size=64)
        _PRIMED.update({(q, model_name): v.tolist() for q, v in zip(todo, vectors)})


@lru_cache(maxsize=4096)
def encode(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """
    As the index was built: no prefix, L2-normalised. Cached, so the two dense
    retrievers of a hybrid search encode a query once. Don't mutate the result.
    """
    if (query, model_name) in _PRIMED:
        return _PRIMED[(query, model_name)]
    return encoder(model_name).encode(query, normalize_embeddings=True).tolist()


def kind_filter(kind: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key="kind", match=models.MatchValue(value=kind))])


def to_hit(point) -> Hit:
    p = point.payload
    return Hit(p["tool_id"], p["name"], point.score, p["kind"], p["text"])
