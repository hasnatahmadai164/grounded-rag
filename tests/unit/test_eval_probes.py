from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.service import AnswerResult
from grounded_rag.agent.state import AgentState
from grounded_rag.domain.answer import AbstainReason, Answer, Citation
from grounded_rag.domain.documents import Chunk, RetrievedChunk
from grounded_rag.domain.principal import Principal
from grounded_rag.domain.retrieval import Hit, RetrievalResult, RetrievalTrace, StageTrace
from grounded_rag.domain.usage import Usage
from grounded_rag.evals.dataset import (
    ACCESS_LABELS,
    EvalContract,
    EvalDataset,
    EvalQuestion,
    GoldSpan,
)
from grounded_rag.evals.probes import (
    ADVERSARIAL_TEMPLATES,
    Probe,
    build_probes,
    copies_text,
    find_leaks,
)

LONG_CLAUSE = (
    "This Agreement shall be governed by and construed in accordance with the laws of the "
    "State of Delaware without regard to its conflict of laws principles"
)


def make_dataset(contracts: int = 12) -> EvalDataset:
    items, questions = [], []
    for n in range(contracts):
        contract = EvalContract(
            id=f"c{n}",
            title=f"C{n}",
            display_title=f"C{n}",
            file=f"contracts/c{n}.txt",
            sha256="0" * 64,
            access_label=ACCESS_LABELS[n % 3],
            split="test",
        )
        items.append(contract)
        questions.append(
            EvalQuestion(
                id=f"c{n}:law",
                contract_id=contract.id,
                category="Governing Law",
                question=f"Which law governs C{n}?",
                answerable=n != 5,
                gold_spans=[GoldSpan(start=0, end=10, text=LONG_CLAUSE)] if n != 5 else [],
                split="test",
            )
        )
    return EvalDataset(source={}, contracts=items, questions=questions)


def probe(**overrides: object) -> Probe:
    values: dict[str, object] = {
        "id": "p1",
        "kind": "direct",
        "asker_label": "legal",
        "question": "Which law governs C0?",
        "contract_id": "c0",
        "gold_texts": (LONG_CLAUSE,),
    }
    return Probe(**(values | overrides))  # type: ignore[arg-type]


def test_every_direct_probe_asks_as_someone_who_cannot_read_the_document() -> None:
    dataset = make_dataset()

    probes = [p for p in build_probes(dataset, "test") if p.kind == "direct"]

    assert len(probes) == 11
    for item in probes:
        contract = dataset.contract(item.contract_id)
        assert item.asker_label != contract.access_label
        assert item.asker_label in ACCESS_LABELS


def test_contracts_without_an_answerable_question_get_no_probe() -> None:
    ids = {p.contract_id for p in build_probes(make_dataset(), "test")}
    assert "c5" not in ids


def test_adversarial_probes_wrap_the_same_question_in_each_template() -> None:
    probes = build_probes(make_dataset(), "test")
    adversarial = [p for p in probes if p.kind == "adversarial"]

    assert len(adversarial) == 8 * len(ADVERSARIAL_TEMPLATES)
    assert all("Which law governs" in p.question for p in adversarial)
    assert len({p.question for p in adversarial}) == len(adversarial)


def test_probes_are_deterministic_and_limited_to_the_requested_split() -> None:
    dataset = make_dataset()

    assert build_probes(dataset, "test") == build_probes(dataset, "test")
    assert build_probes(dataset, "dev") == []


def test_copying_a_long_run_of_a_forbidden_clause_is_detected() -> None:
    answer = "Sure: it says that " + LONG_CLAUSE[:80] + " and more."
    assert copies_text(answer, [LONG_CLAUSE])


def test_unrelated_answers_and_short_passages_are_not_flagged() -> None:
    assert not copies_text("The documents do not contain enough evidence.", [LONG_CLAUSE])
    assert not copies_text("Delaware", ["Delaware"])


def chunk(doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        Chunk(f"{doc_id}-00000", doc_id, 0, "text", 0, 4, None, "T"), "T", frozenset({"x"})
    )


def result_with(
    *,
    chunks: tuple[RetrievedChunk, ...] = (),
    context: tuple[RetrievedChunk, ...] = (),
    stage_ids: tuple[str, ...] = (),
    answer: Answer | None = None,
) -> AnswerResult:
    retrieval = RetrievalResult(
        chunks=list(chunks),
        hits=[],
        trace=RetrievalTrace((StageTrace("dense", tuple(Hit(i, 1.0) for i in stage_ids), 0.0),)),
        usage=Usage(),
        candidates={c.chunk.id: c for c in chunks},
    )
    state = AgentState("q", Principal("p", frozenset({"legal"})), "c", 6, ["q"])
    state.retrievals = [retrieval]
    state.context = list(context)
    final = answer or Answer.abstention(AbstainReason.NO_RELEVANT_CONTEXT)
    return AnswerResult(final, Usage(), {}, 0, 1.0, 0.0, state, AgentConfig())


def test_a_clean_result_has_no_findings() -> None:
    result = result_with(chunks=(chunk("allowed"),), context=(chunk("allowed"),))
    assert find_leaks(probe(), "forbidden", result) == []


def test_a_forbidden_chunk_in_retrieval_is_found() -> None:
    result = result_with(chunks=(chunk("forbidden"),))
    assert [f.location for f in find_leaks(probe(), "forbidden", result)] == ["retrieval"]


def test_a_forbidden_chunk_id_in_a_stage_trace_is_found() -> None:
    result = result_with(stage_ids=("forbidden-00003",))
    assert [f.location for f in find_leaks(probe(), "forbidden", result)] == ["retrieval"]


def test_a_forbidden_chunk_in_the_model_context_is_found() -> None:
    result = result_with(context=(chunk("forbidden"),))
    assert [f.location for f in find_leaks(probe(), "forbidden", result)] == ["context"]


def test_a_citation_of_the_forbidden_document_is_found() -> None:
    citation = Citation(1, "forbidden-00001", "forbidden", "T", None, "quote text", 0, 10)
    answer = Answer("It says so. [1]", False, None, (citation,))
    result = result_with(answer=answer)
    assert [f.location for f in find_leaks(probe(), "forbidden", result)] == ["citation"]


def test_an_answer_that_reproduces_the_forbidden_clause_is_found() -> None:
    answer = Answer(LONG_CLAUSE, False, None)
    result = result_with(answer=answer)
    assert [f.location for f in find_leaks(probe(), "forbidden", result)] == ["answer"]


def test_an_identical_doc_id_prefix_in_another_document_is_not_a_false_positive() -> None:
    result = result_with(stage_ids=("forbidden2-00001",), chunks=(chunk("forbidden2"),))
    assert find_leaks(probe(), "forbidden", result) == []
