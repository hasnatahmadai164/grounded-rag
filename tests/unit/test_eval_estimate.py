import pytest

from grounded_rag.agent.config import AgentConfig
from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.evals.ablations import Ablation
from grounded_rag.evals.estimate import estimate_run, render_estimate
from grounded_rag.llm.pricing import Pricing

PRICING = Pricing(chat_input=0.15, chat_output=0.60, embedding=0.02)


def ablation(name: str, *, mode=RetrievalMode.HYBRID, rerank=False, rewrites=0) -> Ablation:
    return Ablation(name, "", AgentConfig(mode=mode, rerank=rerank, max_rewrites=rewrites))


def estimate(*ablations: Ablation, answerable=100, unanswerable=20, probes=0, chars=0):
    return estimate_run(
        answerable=answerable,
        unanswerable=unanswerable,
        probes=probes,
        ablations=list(ablations),
        corpus_chars=chars,
    )


def test_a_plain_configuration_makes_one_embedding_one_grade_and_a_judged_answer_per_question() -> (
    None
):
    result = estimate(
        ablation("dense_only", mode=RetrievalMode.DENSE), answerable=10, unanswerable=0
    )

    assert result.ceiling["embedding"].calls == 10
    assert result.ceiling["grade"].calls == 10
    assert result.ceiling["generate"].calls == 10
    assert result.ceiling["judge_correctness"].calls == 10
    assert "rerank" not in result.ceiling
    assert "rewrite" not in result.ceiling


def test_lexical_search_needs_no_embedding_calls() -> None:
    result = estimate(ablation("lexical_only", mode=RetrievalMode.LEXICAL))
    assert "embedding" not in result.expected


def test_reranking_adds_one_call_per_retrieval() -> None:
    plain = estimate(ablation("a"))
    reranked = estimate(ablation("b", rerank=True))

    assert reranked.ceiling["rerank"].calls == plain.ceiling["grade"].calls
    assert reranked.expected_usd(PRICING) > plain.expected_usd(PRICING)


def test_retries_multiply_retrieval_grading_and_rewrite_calls_at_the_ceiling() -> None:
    result = estimate(ablation("r", rerank=True, rewrites=2), answerable=0, unanswerable=10)

    assert result.ceiling["rewrite"].calls == 20
    assert result.ceiling["grade"].calls == 30
    assert result.ceiling["rerank"].calls == 30


def test_the_ceiling_is_never_below_the_expectation() -> None:
    result = estimate(ablation("r", rerank=True, rewrites=2), probes=50)
    assert result.ceiling_usd(PRICING) >= result.expected_usd(PRICING)
    assert result.calls(result.ceiling) >= result.calls(result.expected)


def test_cost_grows_with_the_number_of_questions_and_configurations() -> None:
    small = estimate(ablation("a"), answerable=10)
    larger = estimate(ablation("a"), answerable=100)
    more_configs = estimate(ablation("a"), ablation("b"), answerable=10)

    assert larger.expected_usd(PRICING) > small.expected_usd(PRICING)
    assert more_configs.expected_usd(PRICING) == pytest.approx(2 * small.expected_usd(PRICING))


def test_probes_cost_as_unanswerable_questions() -> None:
    with_probes = estimate(ablation("a"), answerable=0, unanswerable=0, probes=30)
    as_questions = estimate(ablation("a"), answerable=0, unanswerable=30)

    assert with_probes.expected_usd(PRICING) == pytest.approx(as_questions.expected_usd(PRICING))


def test_the_corpus_embedding_cost_follows_its_size() -> None:
    without = estimate(ablation("a"), chars=0).expected_usd(PRICING)
    with_corpus = estimate(ablation("a"), chars=40_000_000).expected_usd(PRICING)

    assert with_corpus - without == pytest.approx(40_000_000 / 4 * 1.1 * 0.02 / 1_000_000)


def test_the_rendered_estimate_states_that_it_is_an_assumption_based_figure() -> None:
    text = render_estimate(estimate(ablation("a")), PRICING)

    assert "Expected: about" in text
    assert "Ceiling" in text
    assert "no call is cached" in text
    assert "--max-cost" in text
