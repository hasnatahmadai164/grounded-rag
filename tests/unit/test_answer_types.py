from grounded_rag.agent.schemas import GeneratedAnswer, GradeVerdict
from grounded_rag.domain.answer import ABSTENTION_TEXT, AbstainReason, Answer
from grounded_rag.llm.client import strict_json_schema


def test_abstention_is_a_normal_answer_with_a_reason_code() -> None:
    answer = Answer.abstention(AbstainReason.INSUFFICIENT_EVIDENCE)

    assert answer.abstained is True
    assert answer.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE
    assert answer.text == ABSTENTION_TEXT
    assert answer.citations == ()


def test_reason_codes_are_stable_strings() -> None:
    assert {reason.value for reason in AbstainReason} == {
        "no_relevant_context",
        "insufficient_evidence",
        "unsupported_citations",
        "budget_exceeded",
    }


def test_generation_schema_is_usable_as_a_strict_structured_output() -> None:
    schema = strict_json_schema(GeneratedAnswer)

    assert schema["required"] == ["answerable", "claims"]
    assert schema["$defs"]["GeneratedCitation"]["required"] == ["passage_id", "quote"]


def test_grade_verdict_requires_all_three_fields() -> None:
    assert strict_json_schema(GradeVerdict)["required"] == ["sufficient", "reason", "missing"]
