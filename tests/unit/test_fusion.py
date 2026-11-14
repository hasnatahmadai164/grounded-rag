import pytest

from grounded_rag.domain.retrieval import Hit
from grounded_rag.errors import InvalidRequestError
from grounded_rag.retrieval.fusion import reciprocal_rank_fusion


def ranked(*ids: str) -> list[Hit]:
    return [Hit(chunk_id=i, score=1.0 / (n + 1)) for n, i in enumerate(ids)]


def ids(hits: list[Hit]) -> list[str]:
    return [h.chunk_id for h in hits]


def test_a_single_ranking_keeps_its_order() -> None:
    assert ids(reciprocal_rank_fusion([ranked("a", "b", "c")])) == ["a", "b", "c"]


def test_chunks_found_by_both_retrievers_outrank_chunks_found_by_one() -> None:
    fused = reciprocal_rank_fusion([ranked("a", "x", "b"), ranked("y", "b", "z")])
    assert ids(fused)[0] == "b"


def test_scores_follow_the_reciprocal_rank_formula() -> None:
    fused = reciprocal_rank_fusion([ranked("a", "b"), ranked("b", "a")], k=60)
    expected = 1 / 61 + 1 / 62
    assert fused[0].score == pytest.approx(expected)
    assert fused[1].score == pytest.approx(expected)


def test_ties_are_broken_by_best_rank_then_by_id() -> None:
    fused = reciprocal_rank_fusion([ranked("b", "a"), ranked("a", "b")])
    assert ids(fused) == ["a", "b"]


def test_ties_between_equal_ranks_are_ordered_by_id() -> None:
    fused = reciprocal_rank_fusion([ranked("z"), ranked("m")])
    assert ids(fused) == ["m", "z"]


def test_empty_inputs_produce_no_hits() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_a_chunk_repeated_in_one_list_counts_once_at_its_best_rank() -> None:
    fused = reciprocal_rank_fusion([ranked("a", "b", "a")], k=60)
    assert ids(fused) == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61)


def test_limit_truncates_the_fused_list() -> None:
    assert ids(reciprocal_rank_fusion([ranked("a", "b", "c")], limit=2)) == ["a", "b"]


def test_a_larger_k_flattens_the_difference_between_ranks() -> None:
    steep = reciprocal_rank_fusion([ranked("a", "b")], k=1)
    flat = reciprocal_rank_fusion([ranked("a", "b")], k=1000)
    assert (steep[0].score - steep[1].score) > (flat[0].score - flat[1].score)


@pytest.mark.parametrize("k", [0, -5])
def test_non_positive_k_is_rejected(k: int) -> None:
    with pytest.raises(InvalidRequestError):
        reciprocal_rank_fusion([ranked("a")], k=k)
