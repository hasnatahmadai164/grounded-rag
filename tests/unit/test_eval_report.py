import json
from datetime import UTC, datetime
from pathlib import Path

from grounded_rag.agent.config import AgentConfig
from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.evals.ablations import Ablation
from grounded_rag.evals.report import (
    SUMMARY_END,
    SUMMARY_START,
    build_report,
    render_markdown,
    report_meta,
    run_directory,
    summarize_config,
    summary_table,
    write_report,
)
from grounded_rag.evals.runner import QuestionRecord, RunOutput, SecurityRecord

HYBRID = Ablation("hybrid", "Hybrid.", AgentConfig(mode=RetrievalMode.HYBRID, rerank=False))
DENSE = Ablation("dense_only", "Dense.", AgentConfig(mode=RetrievalMode.DENSE, rerank=False))


def record(config: str = "hybrid", **overrides: object) -> QuestionRecord:
    values: dict[str, object] = {
        "config": config,
        "question_id": "q",
        "category": "Governing Law",
        "contract_id": "c",
        "answerable": True,
        "abstained": False,
        "abstain_reason": None,
        "answer": "text",
        "hit_at": {1: True, 3: True, 5: True},
        "recall_at": {1: 1.0, 3: 1.0, 5: 1.0},
        "reciprocal_rank": 1.0,
        "context_hit": True,
        "correctness": "correct",
        "supported": True,
        "citations": 1,
        "first_pass_verified_rate": 1.0,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "embedding_tokens": 20,
        "cost_usd": 0.001,
        "latency_ms": 2000.0,
        "llm_calls": 3,
    }
    return QuestionRecord(**(values | overrides))  # type: ignore[arg-type]


META = report_meta(
    run_id="2026-01-02-abc1234",
    git_sha="abc1234",
    chat_model="chat-x",
    judge_model="judge-y",
    embedding_model="embed-z",
    concurrency=4,
    cache_hits=10,
    cache_misses=5,
)


def test_answerable_questions_drive_retrieval_and_correctness_rates() -> None:
    records = [
        record(question_id="a"),
        record(
            question_id="b",
            hit_at={1: False, 3: False, 5: False},
            reciprocal_rank=0.0,
            correctness="incorrect",
        ),
        record(
            question_id="c",
            abstained=True,
            abstain_reason="insufficient_evidence",
            correctness=None,
            supported=None,
            hit_at={1: False, 3: False, 5: True},
            reciprocal_rank=0.25,
        ),
        record(
            question_id="d",
            answerable=False,
            abstained=True,
            hit_at={},
            recall_at={},
            reciprocal_rank=None,
            context_hit=None,
            correctness=None,
            supported=None,
        ),
    ]

    summary = summarize_config(HYBRID, records)

    assert (summary["answerable"], summary["unanswerable"]) == (3, 1)
    assert summary["retrieval"]["hit_at"]["5"]["successes"] == 2
    assert summary["retrieval"]["hit_at"]["5"]["total"] == 3
    assert summary["retrieval"]["mrr"] == (1.0 + 0.0 + 0.25) / 3
    assert summary["answers"]["correct"]["successes"] == 1
    assert summary["answers"]["correct"]["total"] == 3
    assert summary["answers"]["false_abstention"]["successes"] == 1
    assert summary["answers"]["abstention_recall"]["value"] == 1.0


def test_abstention_precision_counts_only_abstentions_on_unanswerable_questions() -> None:
    records = [
        record(question_id="a", answerable=False, abstained=True, hit_at={}, correctness=None),
        record(question_id="b", abstained=True, abstain_reason="x", correctness=None),
        record(question_id="c", answerable=False, hit_at={}, correctness=None),
    ]

    answers = summarize_config(HYBRID, records)["answers"]

    assert answers["abstention_precision"]["value"] == 0.5
    assert answers["abstention_recall"]["value"] == 0.5


def test_faithfulness_separates_first_pass_verification_from_repairs() -> None:
    records = [
        record(question_id="a"),
        record(question_id="b", first_pass_verified_rate=0.5, repaired=True),
        record(question_id="c", abstained=True, first_pass_verified_rate=None, supported=None),
    ]

    faithfulness = summarize_config(HYBRID, records)["faithfulness"]

    assert faithfulness["mean_first_pass_citation_verification"] == 0.75
    assert faithfulness["all_citations_verified_first_pass"]["successes"] == 1
    assert faithfulness["needed_repair"]["successes"] == 1
    assert faithfulness["judge_supported"]["total"] == 2


def test_operations_report_percentiles_and_per_query_means() -> None:
    records = [record(question_id=str(n), latency_ms=float(n * 100)) for n in range(1, 101)]

    operations = summarize_config(HYBRID, records)["operations"]

    assert operations["latency_ms_p50"] == 5000.0
    assert operations["latency_ms_p95"] == 9500.0
    assert operations["tokens_per_query"] == 1120
    assert operations["cost_usd_per_query"] == 0.001
    assert operations["model_calls_per_query"] == 3.0


def test_an_empty_configuration_summarises_without_dividing_by_zero() -> None:
    summary = summarize_config(HYBRID, [])

    assert summary["retrieval"]["mrr"] is None
    assert summary["operations"]["latency_ms_p50"] is None
    assert summary["answers"]["correct"]["value"] is None


def output_with_leak(leak: bool) -> RunOutput:
    records = [record("hybrid"), record("dense_only")]
    security = [
        SecurityRecord(config="hybrid", probe_id="p1", kind="direct", abstained=True, leaked=False),
        SecurityRecord(
            config="dense_only",
            probe_id="p1",
            kind="direct",
            abstained=False,
            leaked=leak,
            findings=["citation: c-1"] if leak else [],
        ),
    ]
    return RunOutput(split="test", questions=1, probes=1, records=records, security=security)


def test_security_counts_probes_and_leaks_per_configuration() -> None:
    report = build_report(output_with_leak(True), [HYBRID, DENSE], META)

    assert report["security"]["probes"] == 2
    assert report["security"]["leaks"] == 1
    assert report["security"]["by_config"]["dense_only"] == {"probes": 1, "leaks": 1}
    assert report["security"]["findings"][0]["probe"] == "p1"


def test_the_markdown_report_carries_the_provenance_needed_to_trust_it() -> None:
    text = render_markdown(build_report(output_with_leak(False), [HYBRID, DENSE], META))

    assert "judge-y" in text
    assert "judge-v1" in text
    assert "chat-x" in text
    assert "abc1234" in text
    assert "10 replayed, 5 live" in text
    assert "## Retrieval" in text and "## Security" in text
    assert "0 leaked (this must be 0)" in text


def test_a_leak_is_called_out_in_the_markdown() -> None:
    text = render_markdown(build_report(output_with_leak(True), [HYBRID, DENSE], META))
    assert "LEAK in dense_only" in text


def test_the_summary_block_is_delimited_for_injection_into_the_readme() -> None:
    table = summary_table(build_report(output_with_leak(False), [HYBRID, DENSE], META))

    lines = table.splitlines()
    assert lines[0] == SUMMARY_START
    assert lines[-1] == SUMMARY_END
    assert "| hybrid |" in table
    assert "0 / 1" in table


def test_no_report_text_uses_a_dash_character_that_the_project_avoids() -> None:
    text = render_markdown(build_report(output_with_leak(False), [HYBRID], META))
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_reports_are_written_as_json_markdown_and_per_question_records(tmp_path: Path) -> None:
    output = output_with_leak(False)
    report = build_report(output, [HYBRID, DENSE], META)

    paths = write_report(tmp_path / "run", report, output.records)

    assert {p.name for p in paths} == {"report-test.json", "report-test.md", "records-test.jsonl"}
    assert json.loads(paths[0].read_text(encoding="utf-8"))["meta"]["split"] == "test"
    lines = paths[2].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["config"] == "hybrid"


def test_the_run_directory_is_named_by_date_and_revision(tmp_path: Path) -> None:
    directory = run_directory(tmp_path, tmp_path, datetime(2026, 3, 9, tzinfo=UTC))
    assert directory == tmp_path / "2026-03-09-nogit"
