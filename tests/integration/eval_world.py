import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from grounded_rag.config import Settings
from grounded_rag.container import open_container
from grounded_rag.evals.dataset import (
    EvalContract,
    EvalDataset,
    EvalQuestion,
    GoldSpan,
    access_label,
)
from grounded_rag.evals.environment import EvalEnvironment
from grounded_rag.evals.judge import Judge
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from tests.fakes import FakeEmbedder, HeuristicChat, InMemoryVectorStore

CLAUSES = {
    "Governing Law": "This Agreement shall be governed by the laws of the State of {place}.",
    "Audit Rights": "Customer may audit the books and records of Supplier upon ten days notice.",
}


def contract_text(n: int) -> str:
    return (
        f"MASTER SERVICES AGREEMENT {n}\n\n"
        f"1. Parties\n\nThis agreement number {n} is between Alpha{n} Corp and Beta{n} LLC. "
        + "Background text about the relationship. " * 8
        + "\n\n2. Governing Law\n\n"
        + CLAUSES["Governing Law"].format(place=f"Place{n}")
        + "\n\n3. Payment\n\nInvoices are payable within thirty days. "
        + "More filler about payment mechanics. " * 8
    )


def build_world(directory: Path, count: int = 4, split: str = "test") -> EvalDataset:
    (directory / "contracts").mkdir(parents=True, exist_ok=True)
    contracts, questions = [], []
    for n in range(count):
        text = contract_text(n)
        file = f"contracts/contract{n}.txt"
        (directory / file).write_text(text, encoding="utf-8")
        contract = EvalContract(
            id=f"contract{n}",
            title=f"contract{n}.pdf",
            display_title=f"Services Agreement {n}",
            file=file,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            access_label=access_label(f"contract{n}"),
            split=split,  # type: ignore[arg-type]
        )
        contracts.append(contract)
        clause = CLAUSES["Governing Law"].format(place=f"Place{n}")
        start = text.index(clause)
        questions.append(
            EvalQuestion(
                id=f"contract{n}:governing-law",
                contract_id=contract.id,
                category="Governing Law",
                question=f'Which law governs the "Services Agreement {n}"?',
                answerable=True,
                gold_spans=[GoldSpan(start=start, end=start + len(clause), text=clause)],
                split=split,  # type: ignore[arg-type]
            )
        )
        questions.append(
            EvalQuestion(
                id=f"contract{n}:audit-rights",
                contract_id=contract.id,
                category="Audit Rights",
                question=(
                    f'What audit rights are granted under the "Services Agreement {n}"? '
                    "never-mentioned"
                ),
                answerable=False,
                gold_spans=[],
                split=split,  # type: ignore[arg-type]
            )
        )
    return EvalDataset(source={"license": "CC BY 4.0"}, contracts=contracts, questions=questions)


@asynccontextmanager
async def eval_environment(
    database_url: str,
    chat: HeuristicChat | None = None,
    vectors: InMemoryVectorStore | None = None,
) -> AsyncIterator[EvalEnvironment]:
    chat = chat or HeuristicChat(sufficient=lambda question: "never-mentioned" not in question)
    settings = Settings(_env_file=None, database_url=database_url, eval_corpus="eval-test")
    async with open_container(
        settings, embedder=FakeEmbedder(), vectors=vectors or InMemoryVectorStore(), chat=chat
    ) as container:
        judge_meter = UsageMeter()
        judge = Judge(MeteredLLM(chat, judge_meter, Budget(10**12, 1e12)))
        yield EvalEnvironment(
            container=container,
            judge=judge,
            judge_meter=judge_meter,
            corpus="eval-test",
            chat_model="heuristic",
            judge_model="heuristic",
            embedding_model="fake-embedding",
        )
