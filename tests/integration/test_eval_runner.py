from pathlib import Path

import pytest

from grounded_rag.agent.config import AgentConfig
from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.errors import ConfigurationError
from grounded_rag.evals import runner as runner_module
from grounded_rag.evals.ablations import Ablation
from grounded_rag.evals.probes import LeakFinding
from grounded_rag.evals.runner import EvalRunner, sample
from tests.fakes import InMemoryVectorStore
from tests.integration.eval_world import build_world, eval_environment

pytestmark = pytest.mark.db

HYBRID = Ablation(
    "hybrid", "", AgentConfig(mode=RetrievalMode.HYBRID, rerank=False, max_rewrites=0)
)
DENSE = Ablation(
    "dense_only", "", AgentConfig(mode=RetrievalMode.DENSE, rerank=False, max_rewrites=0)
)


async def test_the_corpus_holds_every_contract_with_its_roles_and_readable_title(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    async with eval_environment(migrated_database_url) as env:
        runner = EvalRunner(env, dataset, tmp_path)

        doc_ids = await runner.prepare_corpus()

        documents = [await env.container.documents.get(d) for d in doc_ids.values()]
        assert len(documents) == 4
        for document, contract in zip(documents, dataset.contracts, strict=True):
            assert document is not None
            assert document.title == contract.display_title
            assert document.allowed_roles == contract.allowed_roles


async def test_a_missing_contract_file_points_at_the_download_script(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    (tmp_path / dataset.contracts[0].file).unlink()
    async with eval_environment(migrated_database_url) as env:
        with pytest.raises(ConfigurationError, match="fetch_cuad"):
            await EvalRunner(env, dataset, tmp_path).prepare_corpus()


async def test_a_contract_that_differs_from_the_manifest_is_rejected(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    (tmp_path / dataset.contracts[1].file).write_text("changed", encoding="utf-8")
    async with eval_environment(migrated_database_url) as env:
        with pytest.raises(ConfigurationError, match="differs from the dataset manifest"):
            await EvalRunner(env, dataset, tmp_path).prepare_corpus()


async def test_every_question_gets_a_record_per_configuration(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    async with eval_environment(migrated_database_url) as env:
        output = await EvalRunner(env, dataset, tmp_path).run(
            "test", [HYBRID, DENSE], security=False
        )

    assert len(output.records) == 2 * len(dataset.questions)
    assert {r.config for r in output.records} == {"hybrid", "dense_only"}
    assert output.probes == 0


async def test_answerable_questions_get_retrieval_metrics_and_a_judged_correctness(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    async with eval_environment(migrated_database_url) as env:
        output = await EvalRunner(env, dataset, tmp_path).run("test", [HYBRID], security=False)

    law = [r for r in output.records if r.category == "Governing Law"]
    assert len(law) == 4
    for record in law:
        assert record.abstained is False
        assert record.hit_at[5] is True
        assert record.reciprocal_rank is not None and record.reciprocal_rank > 0
        assert record.context_hit is True
        assert record.correctness == "correct"
        assert record.supported is True
        assert record.first_pass_verified_rate == 1.0
        assert record.citations == 1
        assert record.cost_usd > 0
        assert record.llm_calls == 2


async def test_unanswerable_questions_are_scored_on_abstention_only(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    async with eval_environment(migrated_database_url) as env:
        output = await EvalRunner(env, dataset, tmp_path).run("test", [HYBRID], security=False)

    audit = [r for r in output.records if r.category == "Audit Rights"]
    assert all(r.abstained and r.abstain_reason == "insufficient_evidence" for r in audit)
    assert all(r.hit_at == {} and r.correctness is None and r.supported is None for r in audit)


async def test_the_judge_is_only_called_for_answers_that_need_grading(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    async with eval_environment(migrated_database_url) as env:
        await EvalRunner(env, dataset, tmp_path).run("test", [HYBRID], security=False)

        assert env.judge_meter.by_purpose["judge_correctness"].prompt_tokens == 4 * 300
        assert env.judge_meter.by_purpose["judge_support"].prompt_tokens == 4 * 300


async def test_probes_run_as_principals_without_access_and_find_no_leaks(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    async with eval_environment(migrated_database_url) as env:
        output = await EvalRunner(env, dataset, tmp_path).run("test", [HYBRID])

    assert output.probes == 4 + 4 * 4
    assert len(output.security) == output.probes
    assert not any(record.leaked for record in output.security)


async def test_a_leak_found_by_the_detector_is_recorded(
    migrated_database_url, engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = build_world(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "find_leaks",
        lambda probe, doc_id, result: [LeakFinding(probe.id, "citation", "x")],
    )
    async with eval_environment(migrated_database_url) as env:
        output = await EvalRunner(env, dataset, tmp_path).run("test", [HYBRID], limit=3)

    assert output.security and all(r.leaked for r in output.security)
    assert output.security[0].findings == ["citation: x"]


async def test_limit_selects_a_repeatable_subset(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)
    questions = dataset.questions_in("test")

    assert sample(questions, 3) == sample(questions, 3)
    assert len(sample(questions, 3)) == 3
    assert sample(questions, None) == questions
    assert sample(questions, 100) == questions


async def test_concurrency_does_not_change_the_results(
    migrated_database_url, engine, tmp_path: Path
) -> None:
    dataset = build_world(tmp_path)

    vectors = InMemoryVectorStore()

    async def outcomes(concurrency: int):
        async with eval_environment(migrated_database_url, vectors=vectors) as env:
            output = await EvalRunner(env, dataset, tmp_path, concurrency=concurrency).run(
                "test", [HYBRID], security=False
            )
        return sorted(
            (r.question_id, r.abstained, r.correctness, r.hit_at.get(1)) for r in output.records
        )

    assert await outcomes(1) == await outcomes(8)
