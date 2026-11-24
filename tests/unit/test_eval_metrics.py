import pytest

from grounded_rag.evals.metrics import (
    Span,
    abstention_scores,
    hit_at_k,
    mean,
    overlaps,
    percentile,
    proportion,
    recall_at_k,
    reciprocal_rank,
)

GOLD = [Span(100, 200), Span(500, 600)]


def test_spans_that_share_a_character_overlap() -> None:
    assert overlaps(Span(0, 10), Span(9, 20))
    assert overlaps(Span(5, 8), Span(0, 100))


def test_spans_that_only_touch_at_the_boundary_do_not_overlap() -> None:
    assert not overlaps(Span(0, 100), Span(100, 200))
    assert not overlaps(Span(300, 400), Span(100, 200))


def test_hit_at_k_looks_only_at_the_first_k_chunks() -> None:
    chunks = [Span(0, 50), Span(300, 400), Span(150, 180)]

    assert not hit_at_k(chunks, GOLD, 2)
    assert hit_at_k(chunks, GOLD, 3)


def test_hit_at_k_is_false_without_chunks_or_gold() -> None:
    assert not hit_at_k([], GOLD, 5)
    assert not hit_at_k([Span(0, 1000)], [], 5)


def test_k_larger_than_the_list_is_fine() -> None:
    assert hit_at_k([Span(150, 160)], GOLD, 50)


def test_recall_is_the_share_of_gold_spans_touched() -> None:
    chunks = [Span(120, 130), Span(0, 10)]

    assert recall_at_k(chunks, GOLD, 2) == 0.5
    assert recall_at_k([*chunks, Span(550, 560)], GOLD, 3) == 1.0
    assert recall_at_k(chunks, GOLD, 1) == 0.5


def test_one_chunk_covering_two_gold_spans_counts_both() -> None:
    assert recall_at_k([Span(0, 1000)], GOLD, 1) == 1.0


def test_recall_without_gold_is_zero_rather_than_undefined() -> None:
    assert recall_at_k([Span(0, 10)], [], 5) == 0.0


def test_duplicate_chunks_do_not_inflate_recall() -> None:
    chunk = Span(120, 130)
    assert recall_at_k([chunk, chunk, chunk], GOLD, 3) == 0.5


def test_reciprocal_rank_uses_the_first_relevant_position() -> None:
    assert reciprocal_rank([Span(0, 1), Span(150, 160)], GOLD) == 0.5
    assert reciprocal_rank([Span(150, 160)], GOLD) == 1.0
    assert reciprocal_rank([Span(0, 1)], GOLD) == 0.0
    assert reciprocal_rank([], GOLD) == 0.0


def test_reciprocal_rank_can_be_cut_off_at_k() -> None:
    assert reciprocal_rank([Span(0, 1), Span(2, 3), Span(150, 160)], GOLD, k=2) == 0.0


def test_mean_of_nothing_is_none() -> None:
    assert mean([]) is None
    assert mean([1.0, 2.0, 6.0]) == 3.0


@pytest.mark.parametrize(("p", "expected"), [(0, 1), (50, 50), (95, 95), (99, 99), (100, 100)])
def test_nearest_rank_percentiles(p: float, expected: float) -> None:
    assert percentile([float(n) for n in range(1, 101)], p) == expected


def test_percentile_does_not_depend_on_input_order() -> None:
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == 3.0


def test_percentile_of_one_value_and_of_none() -> None:
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None


def test_proportion_reports_a_wilson_interval() -> None:
    result = proportion(8, 10)

    assert result.value == 0.8
    assert result.low == pytest.approx(0.4902, abs=1e-3)
    assert result.high == pytest.approx(0.9433, abs=1e-3)


def test_proportion_intervals_stay_within_zero_and_one() -> None:
    assert proportion(0, 5).low == 0.0
    assert proportion(5, 5).high == 1.0


def test_proportion_of_nothing_is_undefined_not_zero() -> None:
    result = proportion(0, 0)
    assert (result.value, result.low, result.high) == (None, None, None)


def test_more_data_narrows_the_interval() -> None:
    small, large = proportion(8, 10), proportion(80, 100)
    assert (large.high - large.low) < (small.high - small.low)


def test_abstention_precision_recall_and_false_abstention() -> None:
    outcomes = [
        (False, True),
        (False, True),
        (False, False),
        (True, True),
        (True, False),
        (True, False),
        (True, False),
    ]

    scores = abstention_scores(outcomes)

    assert (scores.precision.successes, scores.precision.total) == (2, 3)
    assert (scores.recall.successes, scores.recall.total) == (2, 3)
    assert (scores.false_abstention.successes, scores.false_abstention.total) == (1, 4)


def test_abstention_scores_are_undefined_when_the_class_is_empty() -> None:
    never_abstains = abstention_scores([(True, False), (True, False)])

    assert never_abstains.precision.value is None
    assert never_abstains.recall.value is None
    assert never_abstains.false_abstention.value == 0.0
