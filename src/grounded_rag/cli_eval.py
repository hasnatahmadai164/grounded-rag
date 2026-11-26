import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from grounded_rag.cli_common import run_command
from grounded_rag.config import Settings
from grounded_rag.container import agent_config_from_settings
from grounded_rag.errors import ConfigurationError
from grounded_rag.evals.ablations import load_ablations
from grounded_rag.evals.dataset import Split, build_dataset, load_dataset, write_dataset
from grounded_rag.evals.environment import open_eval_environment
from grounded_rag.evals.estimate import estimate_run, render_estimate
from grounded_rag.evals.probes import build_probes
from grounded_rag.evals.report import (
    build_report,
    git_short_sha,
    render_markdown,
    report_meta,
    run_directory,
    summary_table,
    write_report,
)
from grounded_rag.evals.runner import EvalRunner, sample
from grounded_rag.llm.pricing import Pricing
from grounded_rag.logging import configure_logging

eval_app = typer.Typer(help="Build the evaluation set, run ablations and read reports.")

DEFAULT_DATA = Path("data/cuad")
DEFAULT_DATASETS = Path("evals/datasets")
DEFAULT_CONFIGS = Path("evals/configs")
DEFAULT_REPORTS = Path("evals/reports")
LEAK_EXIT_CODE = 3


@eval_app.command("build-dataset")
def build_dataset_command(
    source: Annotated[
        Path, typer.Option(help="Folder written by scripts/fetch_cuad.py.")
    ] = DEFAULT_DATA,
    output: Annotated[
        Path, typer.Option(help="Where to write the question files.")
    ] = DEFAULT_DATASETS,
    dev_contracts: Annotated[int, typer.Option(help="Contracts reserved for tuning.")] = 10,
) -> None:
    subset_path = source / "subset.json"
    if not subset_path.exists():
        typer.echo(
            f"error: {subset_path} not found. Run `python scripts/fetch_cuad.py` first.", err=True
        )
        raise typer.Exit(code=1)
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    dataset = build_dataset(subset, source, dev_count=dev_contracts)
    write_dataset(dataset, output)
    answerable = sum(1 for q in dataset.questions if q.answerable)
    typer.echo(
        f"Wrote {len(dataset.contracts)} contracts and {len(dataset.questions)} questions "
        f"({answerable} answerable, {len(dataset.questions) - answerable} not) to {output}."
    )
    for name in ("dev", "test"):
        typer.echo(f"  {name}: {len(dataset.questions_in(parse_split(name)))} questions")


def parse_split(value: str) -> Split:
    if value == "dev":
        return "dev"
    if value == "test":
        return "test"
    raise ConfigurationError("--split must be dev or test.")


def _names(configs: str | None) -> list[str] | None:
    return [name.strip() for name in configs.split(",") if name.strip()] if configs else None


@eval_app.command("run")
@run_command
async def run(
    split: Annotated[str, typer.Option(help="dev for tuning, test for the final numbers.")] = "dev",
    configs: Annotated[
        str | None, typer.Option(help="Comma-separated configuration names.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Use a repeatable sample of this many questions.")
    ] = None,
    max_cost: Annotated[
        float | None, typer.Option(help="Stop after this many USD of live calls.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Print the estimated calls and cost, then exit.")
    ] = False,
    concurrency: Annotated[int, typer.Option(help="Questions in flight at once.")] = 4,
    security: Annotated[bool, typer.Option(help="Run the access-control probes.")] = True,
    data: Annotated[
        Path, typer.Option(help="Folder written by scripts/fetch_cuad.py.")
    ] = DEFAULT_DATA,
    datasets: Annotated[
        Path, typer.Option(help="Folder with the question files.")
    ] = DEFAULT_DATASETS,
    configs_dir: Annotated[
        Path, typer.Option(help="Folder with the configuration files.")
    ] = DEFAULT_CONFIGS,
    reports: Annotated[Path, typer.Option(help="Where reports are written.")] = DEFAULT_REPORTS,
) -> None:
    which = parse_split(split)
    settings = Settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json, stream=sys.stderr)
    dataset = load_dataset(datasets)
    ablations = load_ablations(configs_dir, _names(configs), agent_config_from_settings(settings))
    questions = sample(dataset.questions_in(which), limit)
    probes = build_probes(dataset, which) if security else []
    if limit is not None:
        probes = probes[:limit]
    answerable = sum(1 for q in questions if q.answerable)

    if dry_run:
        corpus_chars = sum(
            (data / c.file).stat().st_size for c in dataset.contracts if (data / c.file).exists()
        )
        typer.echo(
            f"Evaluation plan: split={split}, {len(questions)} questions "
            f"({answerable} answerable), {len(probes)} security probes, "
            f"{len(ablations)} configurations "
            f"({', '.join(a.name for a in ablations)}).\n"
        )
        estimate = estimate_run(
            answerable=answerable,
            unanswerable=len(questions) - answerable,
            probes=len(probes),
            ablations=ablations,
            corpus_chars=corpus_chars,
        )
        typer.echo(render_estimate(estimate, Pricing.from_settings(settings)))
        return

    for name, value in (
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("PINECONE_API_KEY", settings.pinecone_api_key),
    ):
        if value is None:
            raise ConfigurationError(f"{name} is not set. See .env.example.")
    if split == "test":
        typer.echo("Running the test split. Do not tune prompts or settings against these results.")

    async with open_eval_environment(settings, max_cost_usd=max_cost) as env:
        runner = EvalRunner(env, dataset, data, concurrency=concurrency)
        output = await runner.run(which, ablations, limit=limit, security=security)
        directory = run_directory(reports, Path.cwd())
        cache = env.cache_stats
        meta = report_meta(
            run_id=directory.name,
            git_sha=git_short_sha(Path.cwd()),
            chat_model=env.chat_model,
            judge_model=env.judge_model,
            embedding_model=env.embedding_model,
            concurrency=concurrency,
            cache_hits=cache.hits if cache else 0,
            cache_misses=cache.misses if cache else 0,
        )
        report = build_report(output, ablations, meta)
        paths = write_report(directory, report, output.records)
        spend = env.guard.spent_usd if env.guard else 0.0

    typer.echo(summary_table(report))
    for path in paths:
        typer.echo(f"Wrote {path}")
    typer.echo(f"Live spend this run: ${spend:.4f} (judge calls included).")
    leaks = report["security"]["leaks"]
    if leaks:
        typer.echo(f"SECURITY FAILURE: {leaks} probe(s) leaked. See the report.", err=True)
        raise typer.Exit(code=LEAK_EXIT_CODE)


@eval_app.command("report")
def report_command(
    split: Annotated[str, typer.Option(help="dev or test.")] = "test",
    run_id: Annotated[
        str | None, typer.Option("--run", help="Run folder name; default is the newest.")
    ] = None,
    reports: Annotated[Path, typer.Option(help="Where reports are written.")] = DEFAULT_REPORTS,
) -> None:
    if not reports.is_dir():
        typer.echo(
            f"error: no reports under {reports}. Run `grounded-rag eval run` first.", err=True
        )
        raise typer.Exit(code=1)
    runs = sorted(
        (d for d in reports.iterdir() if (d / f"report-{split}.json").exists()), reverse=True
    )
    chosen = next((d for d in runs if run_id in (None, d.name)), None)
    if chosen is None:
        typer.echo(f"error: no {split} report found.", err=True)
        raise typer.Exit(code=1)
    report = json.loads((chosen / f"report-{split}.json").read_text(encoding="utf-8"))
    typer.echo(render_markdown(report))
