import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grounded_rag.cli import app
from grounded_rag.evals.dataset import write_dataset
from tests.integration.eval_world import build_world

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[2] / "evals" / "configs"


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_build_dataset_writes_questions_for_both_splits(tmp_path: Path) -> None:
    source = tmp_path / "cuad"
    (source / "contracts").mkdir(parents=True)
    contracts = []
    for n in range(4):
        file = f"contracts/c{n}.txt"
        (source / file).write_text(f"Contract {n} text.", encoding="utf-8")
        contracts.append(
            {
                "title": f"C{n}_2020.pdf",
                "file": file,
                "questions": [
                    {
                        "category": "Governing Law",
                        "answerable": True,
                        "answers": [{"text": "Contract", "start": 0, "end": 8}],
                    },
                    {"category": "Audit Rights", "answerable": False, "answers": []},
                ],
            }
        )
    (source / "subset.json").write_text(
        json.dumps({"source": {"license": "CC BY 4.0"}, "contracts": contracts}), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "build-dataset",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "out"),
            "--dev-contracts",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "4 contracts and 8 questions" in result.output
    assert "dev: 2 questions" in result.output
    assert "test: 6 questions" in result.output
    assert (tmp_path / "out" / "contracts.json").exists()


def test_build_dataset_without_the_download_explains_what_to_do(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "build-dataset", "--source", str(tmp_path)])

    assert result.exit_code == 1
    assert "fetch_cuad.py" in result.output


def test_a_dry_run_prints_the_plan_and_estimate_without_needing_any_keys(tmp_path: Path) -> None:
    write_dataset(build_world(tmp_path / "data"), tmp_path / "datasets")

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--dry-run",
            "--split",
            "test",
            "--datasets",
            str(tmp_path / "datasets"),
            "--data",
            str(tmp_path / "data"),
            "--configs-dir",
            str(CONFIGS),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "split=test, 8 questions (4 answerable)" in result.output
    assert "5 configurations" in result.output
    assert "Expected: about" in result.output
    assert "Ceiling" in result.output


def test_a_dry_run_can_be_limited_to_named_configurations(tmp_path: Path) -> None:
    write_dataset(build_world(tmp_path / "data"), tmp_path / "datasets")

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--dry-run",
            "--split",
            "test",
            "--configs",
            "hybrid",
            "--limit",
            "4",
            "--datasets",
            str(tmp_path / "datasets"),
            "--data",
            str(tmp_path / "data"),
            "--configs-dir",
            str(CONFIGS),
        ],
    )

    assert "1 configurations (hybrid)" in result.output
    assert "4 questions" in result.output


def test_a_real_run_refuses_to_start_without_provider_keys(tmp_path: Path) -> None:
    write_dataset(build_world(tmp_path / "data"), tmp_path / "datasets")

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--split",
            "test",
            "--datasets",
            str(tmp_path / "datasets"),
            "--data",
            str(tmp_path / "data"),
            "--configs-dir",
            str(CONFIGS),
        ],
    )

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output


def test_an_unknown_split_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "run", "--dry-run", "--split", "validation"])
    assert result.exit_code == 1
    assert "dev or test" in result.output


def test_reading_a_report_when_none_exist_says_so(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "report", "--reports", str(tmp_path / "none")])

    assert result.exit_code == 1
    assert "eval run" in result.output
