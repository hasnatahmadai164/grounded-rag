from collections.abc import Sequence

from grounded_rag.domain.retrieval import Hit
from grounded_rag.errors import InvalidRequestError


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], *, k: int = 60, limit: int | None = None
) -> list[Hit]:
    """Merge ranked lists by summing 1 / (k + rank) for every list a chunk appears in.

    Only ranks matter, so scores from different retrievers need no calibration. A chunk
    repeated inside one list is counted once, at its best rank. Ties break on the best
    rank any list gave the chunk, then on id, so the order is deterministic.
    """
    if k <= 0:
        raise InvalidRequestError("RRF constant k must be positive.")
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best_rank[hit.chunk_id] = min(best_rank.get(hit.chunk_id, rank), rank)
    ordered = sorted(fused, key=lambda cid: (-fused[cid], best_rank[cid], cid))
    hits = [Hit(chunk_id=cid, score=fused[cid]) for cid in ordered]
    return hits if limit is None else hits[:limit]
