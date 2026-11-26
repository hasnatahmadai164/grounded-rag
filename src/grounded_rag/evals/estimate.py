from collections.abc import Sequence
from dataclasses import dataclass, field

from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.evals.ablations import Ablation
from grounded_rag.llm.pricing import Pricing

_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class Assumptions:
    """Rough sizes behind the estimate. They are guesses about typical prompts, not measurements,
    and the dry run says so. Replace them with figures from a report once one exists."""

    question_tokens: int = 40
    chunk_tokens: int = 330
    rerank_candidates: int = 20
    rerank_passage_tokens: int = 175
    prompt_overhead_tokens: int = 250
    grade_output_tokens: int = 60
    rerank_output_tokens: int = 60
    rewrite_output_tokens: int = 40
    generate_output_tokens: int = 220
    judge_support_input_tokens: int = 450
    judge_correctness_input_tokens: int = 700
    judge_output_tokens: int = 50
    answerable_retry_rate: float = 0.15
    unanswerable_false_answer_rate: float = 0.2
    unanswerable_retry_rate: float = 1.0
    answerable_answer_rate: float = 0.9


@dataclass(slots=True)
class PurposeEstimate:
    calls: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0

    def add(self, calls: float, input_tokens: float, output_tokens: float) -> None:
        self.calls += calls
        self.input_tokens += calls * input_tokens
        self.output_tokens += calls * output_tokens


@dataclass(slots=True)
class Estimate:
    expected: dict[str, PurposeEstimate] = field(default_factory=dict)
    ceiling: dict[str, PurposeEstimate] = field(default_factory=dict)
    corpus_embedding_tokens: float = 0.0

    def _cost(self, table: dict[str, PurposeEstimate], pricing: Pricing) -> float:
        chat = sum(
            pricing.chat_usage(int(p.input_tokens), int(p.output_tokens)).cost_usd
            for name, p in table.items()
            if name != "embedding"
        )
        embedded = (
            table.get("embedding", PurposeEstimate()).input_tokens + self.corpus_embedding_tokens
        )
        return chat + pricing.embedding_usage(int(embedded)).cost_usd

    def expected_usd(self, pricing: Pricing) -> float:
        return self._cost(self.expected, pricing)

    def ceiling_usd(self, pricing: Pricing) -> float:
        return self._cost(self.ceiling, pricing)

    def calls(self, table: dict[str, PurposeEstimate]) -> float:
        return sum(p.calls for name, p in table.items() if name != "embedding")


def _add(table: dict[str, PurposeEstimate], purpose: str, calls: float, i: int, o: int) -> None:
    table.setdefault(purpose, PurposeEstimate()).add(calls, i, o)


def _retrieval_calls(
    table: dict[str, PurposeEstimate], ablation: Ablation, attempts: float, a: Assumptions
) -> None:
    if ablation.agent.mode is not RetrievalMode.LEXICAL:
        _add(table, "embedding", attempts, a.question_tokens, 0)
    if ablation.agent.rerank:
        tokens = a.rerank_candidates * a.rerank_passage_tokens + a.prompt_overhead_tokens
        _add(table, "rerank", attempts, tokens, a.rerank_output_tokens)


def _question_cost(
    table: dict[str, PurposeEstimate],
    ablation: Ablation,
    *,
    weight: float,
    retry_rate: float,
    answer_rate: float,
    correctness_judged: float,
    a: Assumptions,
) -> None:
    """Add `weight` questions to `table`. `retry_rate` of them rewrite to the configured limit."""
    context = ablation.agent.top_k * a.chunk_tokens + a.prompt_overhead_tokens
    rewrites = ablation.agent.max_rewrites
    attempts = 1 + retry_rate * rewrites
    _retrieval_calls(table, ablation, weight * attempts, a)
    _add(table, "grade", weight * attempts, context, a.grade_output_tokens)
    if rewrites:
        _add(
            table,
            "rewrite",
            weight * retry_rate * rewrites,
            a.question_tokens + 150,
            a.rewrite_output_tokens,
        )
    answered = weight * answer_rate
    _add(table, "generate", answered, context + a.question_tokens, a.generate_output_tokens)
    _add(table, "judge_support", answered, a.judge_support_input_tokens, a.judge_output_tokens)
    _add(
        table,
        "judge_correctness",
        weight * correctness_judged,
        a.judge_correctness_input_tokens,
        a.judge_output_tokens,
    )


def estimate_run(
    *,
    answerable: int,
    unanswerable: int,
    probes: int,
    ablations: Sequence[Ablation],
    corpus_chars: int,
    assumptions: Assumptions | None = None,
) -> Estimate:
    """Expected and worst-case number of live calls and tokens, assuming nothing is cached."""
    a = assumptions or Assumptions()
    estimate = Estimate(corpus_embedding_tokens=corpus_chars / _CHARS_PER_TOKEN * 1.1)
    for ablation in ablations:
        for table, retry_ans, retry_un, ans_rate, un_rate in (
            (
                estimate.expected,
                a.answerable_retry_rate,
                a.unanswerable_retry_rate,
                a.answerable_answer_rate,
                a.unanswerable_false_answer_rate,
            ),
            (estimate.ceiling, 1.0, 1.0, 1.0, 1.0),
        ):
            _question_cost(
                table,
                ablation,
                weight=answerable,
                retry_rate=retry_ans,
                answer_rate=ans_rate,
                correctness_judged=ans_rate,
                a=a,
            )
            _question_cost(
                table,
                ablation,
                weight=unanswerable + probes,
                retry_rate=retry_un,
                answer_rate=un_rate,
                correctness_judged=0.0,
                a=a,
            )
    return estimate


def render_estimate(estimate: Estimate, pricing: Pricing) -> str:
    lines = [f"{'purpose':<20}{'calls':>9}{'input tok':>13}{'output tok':>12}{'usd':>10}"]
    for purpose, item in sorted(estimate.expected.items()):
        usd = (
            pricing.embedding_usage(int(item.input_tokens)).cost_usd
            if purpose == "embedding"
            else pricing.chat_usage(int(item.input_tokens), int(item.output_tokens)).cost_usd
        )
        lines.append(
            f"{purpose:<20}{item.calls:>9.0f}{item.input_tokens:>13.0f}{item.output_tokens:>12.0f}{usd:>10.4f}"
        )
    lines.append(
        f"{'corpus embedding':<20}{'':>9}{estimate.corpus_embedding_tokens:>13.0f}{'':>12}"
        f"{pricing.embedding_usage(int(estimate.corpus_embedding_tokens)).cost_usd:>10.4f}"
    )
    lines += [
        "",
        f"Expected: about {estimate.calls(estimate.expected):.0f} model calls, "
        f"${estimate.expected_usd(pricing):.2f}",
        f"Ceiling (every question rewrites to the limit and answers): about "
        f"{estimate.calls(estimate.ceiling):.0f} calls, ${estimate.ceiling_usd(pricing):.2f}",
        "",
        "These figures assume no call is cached and use fixed guesses for prompt sizes and "
        "retry rates (see Assumptions in grounded_rag/evals/estimate.py). Calls already in "
        "the cache are free, and the run stops at --max-cost.",
    ]
    return "\n".join(lines)
