from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grounded_rag import cli_eval
from grounded_rag.cli import app
from grounded_rag.config import Settings
from grounded_rag.evals import runner as runner_module
from grounded_rag.evals.dataset import write_dataset
from grounded_rag.evals.environment import EvalEnvironment
from grounded_rag.evals.probes import LeakFinding
from tests.fakes import InMemoryVectorStore
from tests.integration.eval_world import build_world, eval_environment

pytestmark = pytest.mark.db

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[2] / "evals" / "configs"


@pytest.fixture
def prepared(migrated_database_url: str, engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    write_dataset(build_world(tmp_path / "data"), tmp_path / "datasets")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    vectors = InMemoryVectorStore()

    @asynccontextmanager
    async def fake_environment(
        settings: Settings, *, max_cost_usd: float | None
    ) -> AsyncIterator[EvalEnvironment]:
        async with eval_environment(migrated_database_url, vectors=vectors) as env:
            yield env

    monkeypatch.setattr(cli_eval, "open_eval_environment", fake_environment)
    return [
        "eval",
        "run",
        "--split",
        "test",
        "--configs",
        "hybrid,dense_only",
        "--datasets",
        str(tmp_path / "datasets"),
        "--data",
        str(tmp_path / "data"),
        "--configs-dir",
        str(CONFIGS),
        "--reports",
        str(tmp_path / "reports"),
    ]


def test_a_run_writes_a_report_folder_named_by_date_and_revision(prepared, tmp_path: Path) -> None:
    result = runner.invoke(app, prepared)

    assert result.exit_code == 0, result.output
    folders = list((tmp_path / "reports").iterdir())
    assert len(folders) == 1
    assert {p.name for p in folders[0].iterdir()} == {
        "report-test.json",
        "report-test.md",
        "records-test.jsonl",
    }
    assert "| hybrid |" in result.output
    assert "| dense_only |" in result.output
    assert "Live spend this run" in result.output


def test_the_report_command_prints_the_latest_report(prepared, tmp_path: Path) -> None:
    runner.invoke(app, prepared)

    shown = runner.invoke(
        app, ["eval", "report", "--split", "test", "--reports", str(tmp_path / "reports")]
    )

    assert shown.exit_code == 0
    assert "## Retrieval" in shown.output
    assert "judge-v1" in shown.output


def test_a_leak_fails_the_run_with_a_distinct_exit_code(
    prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner_module,
        "find_leaks",
        lambda probe, doc_id, result: [LeakFinding(probe.id, "citation", "x")],
    )

    result = runner.invoke(app, prepared)

    assert result.exit_code == 3
    assert "SECURITY FAILURE" in result.output


def test_running_the_test_split_prints_a_reminder_not_to_tune(prepared) -> None:
    result = runner.invoke(app, prepared)
    assert "Do not tune prompts or settings" in result.output
