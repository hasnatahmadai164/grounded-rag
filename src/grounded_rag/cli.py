import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer

from grounded_rag.agent.service import AnswerResult
from grounded_rag.cli_common import run_command
from grounded_rag.cli_eval import eval_app
from grounded_rag.config import Settings
from grounded_rag.container import Container, open_container, pinecone_from_settings
from grounded_rag.domain.principal import parse_roles
from grounded_rag.errors import InvalidRequestError, NotFoundError
from grounded_rag.identity import generate_api_key
from grounded_rag.ingestion.loaders import PDF_SUFFIXES, TEXT_SUFFIXES
from grounded_rag.logging import configure_logging

app = typer.Typer(
    help="Operate grounded-rag: ingest documents, manage keys, run evaluations.",
    add_completion=False,
)
keys_app = typer.Typer(help="Create, list and revoke API keys.")
index_app = typer.Typer(help="Manage the vector index.")
app.add_typer(keys_app, name="keys")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")


async def with_container[T](action: Callable[[Container], Awaitable[T]]) -> T:
    settings = Settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json, stream=sys.stderr)
    async with open_container(settings) as container:
        return await action(container)


@keys_app.command("create")
@run_command
async def keys_create(
    name: Annotated[str, typer.Option(help="Principal name the key authenticates as.")],
    roles: Annotated[str, typer.Option(help="Comma-separated roles, for example legal,finance.")],
) -> None:
    parsed_roles = parse_roles(roles)
    generated = generate_api_key()

    async def action(container: Container) -> None:
        record = await container.api_keys.create(
            name=name,
            key_hash=generated.key_hash,
            key_prefix=generated.key_prefix,
            roles=parsed_roles,
        )
        roles_text = ", ".join(sorted(record.roles))
        typer.echo(f"Created key {record.id} for '{record.name}' with roles: {roles_text}")
        typer.echo(f"API key (shown once, store it now): {generated.secret}")

    await with_container(action)


@keys_app.command("list")
@run_command
async def keys_list() -> None:
    async def action(container: Container) -> None:
        records = await container.api_keys.list()
        if not records:
            typer.echo("No API keys.")
            return
        typer.echo(f"{'ID':<5}{'NAME':<24}{'PREFIX':<10}{'STATUS':<9}ROLES")
        for record in records:
            status = "active" if record.active else "revoked"
            roles = ",".join(sorted(record.roles))
            typer.echo(f"{record.id:<5}{record.name:<24}{record.key_prefix:<10}{status:<9}{roles}")

    await with_container(action)


@keys_app.command("revoke")
@run_command
async def keys_revoke(
    key_id: Annotated[int, typer.Argument(help="Id shown by `keys list`.")],
) -> None:
    async def action(container: Container) -> None:
        if await container.api_keys.revoke(key_id):
            typer.echo(f"Revoked key {key_id}.")
        else:
            typer.echo(f"No active key with id {key_id}.", err=True)
            raise typer.Exit(code=1)

    await with_container(action)


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise InvalidRequestError(f"No such file or directory: {path}")
    supported = TEXT_SUFFIXES | PDF_SUFFIXES
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported)


@index_app.command("create")
@run_command
async def index_create() -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json, stream=sys.stderr)
    store = pinecone_from_settings(settings)
    try:
        created = await store.ensure_index(
            dimension=settings.embedding_dimensions,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    finally:
        await store.aclose()
    verb = "Created" if created else "Found existing"
    typer.echo(f"{verb} index '{settings.pinecone_index_name}' and it is ready.")


@app.command("ingest")
@run_command
async def ingest(
    path: Annotated[Path, typer.Argument(help="A PDF or text file, or a directory of them.")],
    roles: Annotated[str, typer.Option(help="Roles allowed to read the document.")],
    corpus: Annotated[str | None, typer.Option(help="Corpus (Pinecone namespace).")] = None,
    title: Annotated[str | None, typer.Option(help="Title for a single file.")] = None,
    replace: Annotated[
        bool, typer.Option(help="Delete older versions from the same path.")
    ] = False,
) -> None:
    allowed_roles = parse_roles(roles)
    files = discover_files(path)
    if not files:
        raise InvalidRequestError(f"No supported documents (.pdf, .txt, .md) under {path}")
    if title and len(files) > 1:
        raise InvalidRequestError("--title can only be used with a single file.")

    async def action(container: Container) -> None:
        total_cost = 0.0
        for file in files:
            result = await container.ingestion.ingest(
                file,
                roles=allowed_roles,
                corpus=corpus or container.settings.default_corpus,
                title=title,
                replace=replace,
            )
            total_cost += result.usage.cost_usd
            typer.echo(
                f"{result.outcome:<14}{result.doc_id}  "
                f"{result.chunk_count:>4} chunks  {result.title}"
            )
            for removed in result.removed_doc_ids:
                typer.echo(f"{'removed':<14}{removed}")
        typer.echo(f"Embedding cost: ${total_cost:.6f}")

    await with_container(action)


@app.command("delete")
@run_command
async def delete(doc_id: Annotated[str, typer.Argument(help="Document id from `ingest`.")]) -> None:
    async def action(container: Container) -> None:
        if await container.ingestion.delete(doc_id):
            typer.echo(f"Deleted {doc_id}.")
        else:
            typer.echo(f"No document with id {doc_id}.")

    await with_container(action)


def render_result(result: AnswerResult, *, debug: bool) -> None:
    answer = result.answer
    if answer.abstained:
        reason = answer.abstain_reason.value if answer.abstain_reason else "unknown"
        typer.echo(f"{answer.text}\n(abstained: {reason})")
    else:
        typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nCitations")
        for citation in answer.citations:
            where = f"{citation.document_title}" + (
                f", {citation.section}" if citation.section else ""
            )
            typer.echo(
                f"[{citation.index}] {where} (chars {citation.start_char}-{citation.end_char})"
            )
            typer.echo(f'    "{citation.quote}"')
    usage = result.usage
    typer.echo(
        f"\ntokens: {usage.prompt_tokens} prompt, {usage.completion_tokens} completion, "
        f"{usage.embedding_tokens} embedding | cost: ${usage.cost_usd:.6f} | "
        f"latency: {result.latency_ms:.0f} ms | model calls: {result.llm_calls}"
    )
    if debug:
        typer.echo("\nTrace")
        for event in result.state.events:
            typer.echo(f"  {event.node}: {event.detail}")
        for number, retrieval in enumerate(result.state.retrievals, start=1):
            for stage in retrieval.trace.stages:
                top = ", ".join(f"{hit.chunk_id[-12:]}={hit.score:.3f}" for hit in stage.hits[:5])
                typer.echo(f"  retrieval {number} {stage.name}: {top}")


@app.command("query")
@run_command
async def query(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    principal: Annotated[
        str, typer.Option("--as", help="Name of an API key whose roles the query runs with.")
    ],
    corpus: Annotated[str | None, typer.Option(help="Corpus to search.")] = None,
    top_k: Annotated[int | None, typer.Option(help="Passages sent to the model.")] = None,
    debug: Annotated[bool, typer.Option(help="Print the retrieval trace.")] = False,
) -> None:
    async def action(container: Container) -> None:
        record = await container.api_keys.find_active_by_name(principal)
        if record is None:
            raise NotFoundError(
                f"No active API key named '{principal}'. Create one with `keys create`."
            )
        result = await container.answers.answer(
            question,
            record.to_principal(),
            corpus=corpus or container.settings.default_corpus,
            top_k=top_k,
        )
        render_result(result, debug=debug)

    await with_container(action)
