"""
Hybrid retrieval: dense + BM25 sparse vectors merged with RRF.
Queries both ticket_resolutions (70% weight) and ticket_problems (30% weight).
"""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery,
    NamedSparseVector,
    NamedVector,
    Prefetch,
    Query,
    SparseVector,
)

logger = logging.getLogger(__name__)

RRF_K = 60
RESOLUTION_WEIGHT = 0.65
PROBLEM_WEIGHT = 0.25
DOCS_WEIGHT = 0.10


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _merge_rrf(
    resolution_results: list[dict],
    problem_results: list[dict],
    docs_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Merge results from three collections using Reciprocal Rank Fusion.
    Tickets keyed by ticket_id, docs keyed by chunk_id.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(resolution_results):
        key = hit["ticket_id"]
        scores[key] = scores.get(key, 0) + _rrf_score(rank) * RESOLUTION_WEIGHT
        payloads[key] = hit

    for rank, hit in enumerate(problem_results):
        key = hit["ticket_id"]
        boost = hit.get("retrieval_boost", 1.0)
        scores[key] = scores.get(key, 0) + _rrf_score(rank) * PROBLEM_WEIGHT * boost
        if key not in payloads:
            payloads[key] = hit

    for rank, hit in enumerate(docs_results):
        key = hit["chunk_id"]  # Docs deduplicated by chunk_id
        scores[key] = scores.get(key, 0) + _rrf_score(rank) * DOCS_WEIGHT
        payloads[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for key, score in ranked:
        hit = dict(payloads[key])
        hit["rrf_score"] = round(score, 4)
        results.append(hit)

    return results


def _qdrant_hits_to_dicts(hits) -> list[dict]:
    return [
        {**point.payload, "similarity": point.score}
        for point in hits
        if point.payload
    ]


def retrieve(
    client: QdrantClient,
    dense_vector: list[float],
    sparse_vector: dict,
    top_k: int = 20,
    filters: dict[str, str] = None,
) -> list[dict]:
    """
    Hybrid retrieval across both collections.
    filters: optional metadata pre-filter e.g. {"product": "voice_api"}
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    sparse_vec = SparseVector(
        indices=sparse_vector["indices"],
        values=sparse_vector["values"],
    )

    def _search(collection: str) -> list[dict]:
        try:
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k,
                        filter=qdrant_filter,
                    ),
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=top_k,
                        filter=qdrant_filter,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=top_k,
            )
            return _qdrant_hits_to_dicts(results.points)
        except Exception as e:
            logger.error(f"Search failed on '{collection}': {e}")
            return []

    resolution_hits = _search("ticket_resolutions")
    problem_hits = _search("ticket_problems")
    docs_hits = _search("support_docs")

    merged = _merge_rrf(resolution_hits, problem_hits, docs_hits, top_k)
    logger.info(f"Retrieved {len(merged)} candidates after RRF merge")
    return merged
