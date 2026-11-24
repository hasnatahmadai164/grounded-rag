import math
from collections.abc import Sequence
from dataclasses import dataclass

_Z_95 = 1.959964


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open character range [start, end) in a document's stored text."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Proportion:
    """A rate with its Wilson 95% interval. `value` is None when nothing was counted."""

    successes: int
    total: int
    value: float | None
    low: float | None
    high: float | None


def overlaps(first: Span, second: Span) -> bool:
    return first.start < second.end and second.start < first.end


def hit_at_k(chunks: Sequence[Span], gold: Sequence[Span], k: int) -> bool:
    """Does any of the first k retrieved chunks overlap a gold span?"""
    return any(overlaps(chunk, span) for chunk in chunks[:k] for span in gold)


def recall_at_k(chunks: Sequence[Span], gold: Sequence[Span], k: int) -> float:
    """Share of gold spans touched by at least one of the first k chunks."""
    if not gold:
        return 0.0
    covered = sum(1 for span in gold if any(overlaps(chunk, span) for chunk in chunks[:k]))
    return covered / len(gold)


def reciprocal_rank(chunks: Sequence[Span], gold: Sequence[Span], k: int | None = None) -> float:
    """1 / rank of the first chunk overlapping a gold span, or 0 when there is none."""
    for rank, chunk in enumerate(chunks[:k], start=1):
        if any(overlaps(chunk, span) for span in gold):
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p% of the data at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def proportion(successes: int, total: int) -> Proportion:
    if total == 0:
        return Proportion(successes, total, None, None, None)
    rate = successes / total
    z2 = _Z_95**2
    denominator = 1 + z2 / total
    centre = (rate + z2 / (2 * total)) / denominator
    margin = _Z_95 * math.sqrt(rate * (1 - rate) / total + z2 / (4 * total**2)) / denominator
    return Proportion(successes, total, rate, max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True, slots=True)
class AbstentionScores:
    """Abstaining is the positive class. Precision asks: when the system abstains, was the
    question really unanswerable? Recall asks: of the unanswerable questions, how many did it
    abstain on? False abstention is the share of answerable questions it refused."""

    precision: Proportion
    recall: Proportion
    false_abstention: Proportion


def abstention_scores(outcomes: Sequence[tuple[bool, bool]]) -> AbstentionScores:
    """`outcomes` holds (answerable, abstained) for every question."""
    true_positive = sum(1 for answerable, abstained in outcomes if abstained and not answerable)
    abstained = sum(1 for _, did_abstain in outcomes if did_abstain)
    unanswerable = sum(1 for answerable, _ in outcomes if not answerable)
    answerable_total = len(outcomes) - unanswerable
    false_abstentions = abstained - true_positive
    return AbstentionScores(
        precision=proportion(true_positive, abstained),
        recall=proportion(true_positive, unanswerable),
        false_abstention=proportion(false_abstentions, answerable_total),
    )
