import asyncio
import hashlib
from collections.abc import Coroutine, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from grounded_rag.agent.service import AnswerResult
from grounded_rag.errors import ConfigurationError, DependencyUnavailableError
from grounded_rag.evals.ablations import Ablation
from grounded_rag.evals.dataset import (
    ADMIN_PRINCIPAL,
    EvalDataset,
    EvalQuestion,
    Split,
    principal_for,
)
from grounded_rag.evals.environment import EvalEnvironment
from grounded_rag.evals.metrics import Span, hit_at_k, recall_at_k, reciprocal_rank
from grounded_rag.evals.probes import Probe, build_probes, find_leaks
from grounded_rag.logging import get_logger

log = get_logger(__name__)

RETRIEVAL_KS = (1, 3, 5)
_MAX_RECORDED_ANSWER_CHARS = 800
_MAX_REFERENCE_SPANS = 3
_NOT_A_HIT = Span(-2, -1)
_VECTOR_WAIT_SECONDS = 120.0
_PROGRESS_EVERY = 25


class QuestionRecord(BaseModel):
    config: str
    question_id: str
    category: str
    contract_id: str
    answerable: bool
    abstained: bool
    abstain_reason: str | None
    answer: str
    hit_at: dict[int, bool] = {}
    recall_at: dict[int, float] = {}
    reciprocal_rank: float | None = None
    context_hit: bool | None = None
    correctness: Literal["correct", "partially_correct", "incorrect"] | None = None
    supported: bool | None = None
    citations: int = 0
    first_pass_verified_rate: float | None = None
    repaired: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0
    rewrites: int = 0


class SecurityRecord(BaseModel):
    config: str
    probe_id: str
    kind: Literal["direct", "adversarial"]
    abstained: bool
    leaked: bool
    findings: list[str] = []


class RunOutput(BaseModel):
    split: str
    questions: int
    probes: int
    records: list[QuestionRecord]
    security: list[SecurityRecord]


def sample(items: Sequence[EvalQuestion], limit: int | None) -> list[EvalQuestion]:
    """A repeatable subset: ordered by a hash of the id, so it does not favour any contract."""
    if limit is None or limit >= len(items):
        return list(items)
    ordered = sorted(items, key=lambda q: hashlib.sha256(q.id.encode()).hexdigest())
    return ordered[:limit]


def retrieved_spans(result: AnswerResult, doc_id: str, attempt: int = 0) -> list[Span]:
    """Retrieved chunks as spans of the gold document; other documents rank but never hit."""
    chunks = result.state.retrievals[attempt].chunks
    return [
        Span(item.chunk.start_char, item.chunk.end_char)
        if item.chunk.doc_id == doc_id
        else _NOT_A_HIT
        for item in chunks
    ]


class EvalRunner:
    def __init__(
        self,
        env: EvalEnvironment,
        dataset: EvalDataset,
        data_dir: Path,
        *,
        concurrency: int = 4,
    ) -> None:
        self._env = env
        self._dataset = dataset
        self._data_dir = data_dir
        self._concurrency = concurrency
        self._doc_ids: dict[str, str] = {}

    async def prepare_corpus(self) -> dict[str, str]:
        """Ingest every contract, so the corpus is the same whichever split is evaluated."""
        ingestion = self._env.container.ingestion
        for contract in self._dataset.contracts:
            path = self._data_dir / contract.file
            if not path.exists():
                raise ConfigurationError(
                    f"Missing {path}. Run `python scripts/fetch_cuad.py` to download the contracts."
                )
            if hashlib.sha256(path.read_bytes()).hexdigest() != contract.sha256:
                raise ConfigurationError(
                    f"{path.name} differs from the dataset manifest. Re-run scripts/fetch_cuad.py "
                    "with the same arguments, or rebuild the dataset."
                )
            result = await ingestion.ingest(
                path,
                roles=contract.allowed_roles,
                corpus=self._env.corpus,
                title=contract.display_title,
            )
            self._doc_ids[contract.id] = result.doc_id
        await self._wait_for_vectors()
        return dict(self._doc_ids)

    async def _wait_for_vectors(self) -> None:
        """Vector indexes are eventually consistent, so wait until the new vectors are visible."""
        container = self._env.container
        expected = 0
        for doc_id in self._doc_ids.values():
            document = await container.documents.get(doc_id)
            expected += document.chunk_count if document else 0
        waited = 0.0
        while await container.vectors.count(self._env.corpus) < expected:
            if waited >= _VECTOR_WAIT_SECONDS:
                raise DependencyUnavailableError(
                    f"The vector index did not show {expected} vectors within "
                    f"{_VECTOR_WAIT_SECONDS:.0f} seconds."
                )
            await asyncio.sleep(2.0)
            waited += 2.0

    async def run(
        self,
        split: Split,
        ablations: Sequence[Ablation],
        *,
        limit: int | None = None,
        security: bool = True,
    ) -> RunOutput:
        questions = sample(self._dataset.questions_in(split), limit)
        probes = build_probes(self._dataset, split) if security else []
        if limit is not None:
            probes = probes[:limit]
        if not self._doc_ids:
            await self.prepare_corpus()

        records: list[QuestionRecord] = []
        findings: list[SecurityRecord] = []
        for ablation in ablations:
            log.info("eval_config_started", config=ablation.name, questions=len(questions))
            records.extend(await self._bounded(self._record(ablation, q) for q in questions))
            findings.extend(await self._bounded(self._probe(ablation, p) for p in probes))
        return RunOutput(
            split=split,
            questions=len(questions),
            probes=len(probes),
            records=records,
            security=findings,
        )

    async def _bounded[T](self, jobs: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def limited(job: Coroutine[Any, Any, T]) -> T:
            async with semaphore:
                return await job

        return list(await asyncio.gather(*(limited(job) for job in jobs)))

    async def _record(self, ablation: Ablation, question: EvalQuestion) -> QuestionRecord:
        container, judge = self._env.container, self._env.judge
        doc_id = self._doc_ids[question.contract_id]
        result = await container.answers.answer(
            question.question, ADMIN_PRINCIPAL, corpus=self._env.corpus, config=ablation.agent
        )
        answer, usage = result.answer, result.usage
        record = QuestionRecord(
            config=ablation.name,
            question_id=question.id,
            category=question.category,
            contract_id=question.contract_id,
            answerable=question.answerable,
            abstained=answer.abstained,
            abstain_reason=answer.abstain_reason.value if answer.abstain_reason else None,
            answer=answer.text[:_MAX_RECORDED_ANSWER_CHARS],
            citations=len(answer.citations),
            repaired=result.state.repairs_used > 0,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            embedding_tokens=usage.embedding_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=result.reported_latency_ms,
            llm_calls=result.llm_calls,
            rewrites=result.state.rewrites_used,
        )
        if result.state.reports:
            record.first_pass_verified_rate = result.state.reports[0].verified_rate
        if question.answerable and result.state.retrievals:
            gold = [Span(s.start, s.end) for s in question.gold_spans]
            first = retrieved_spans(result, doc_id)
            record.hit_at = {k: hit_at_k(first, gold, k) for k in RETRIEVAL_KS}
            record.recall_at = {k: recall_at_k(first, gold, k) for k in RETRIEVAL_KS}
            record.reciprocal_rank = reciprocal_rank(first, gold, max(RETRIEVAL_KS))
            context = [
                Span(i.chunk.start_char, i.chunk.end_char)
                if i.chunk.doc_id == doc_id
                else _NOT_A_HIT
                for i in result.state.context
            ]
            record.context_hit = hit_at_k(context, gold, len(context))
        if not answer.abstained:
            quotes = [c.quote for c in answer.citations]
            record.supported = (await judge.support(answer.text, quotes)).supported
            if question.answerable:
                references = [s.text for s in question.gold_spans][:_MAX_REFERENCE_SPANS]
                verdict = await judge.correctness(question.question, references, answer.text)
                record.correctness = verdict.verdict
        return record

    async def _probe(self, ablation: Ablation, probe: Probe) -> SecurityRecord:
        container = self._env.container
        result = await container.answers.answer(
            probe.question,
            principal_for(probe.asker_label),
            corpus=self._env.corpus,
            config=ablation.agent,
        )
        leaks = find_leaks(probe, self._doc_ids[probe.contract_id], result)
        return SecurityRecord(
            config=ablation.name,
            probe_id=probe.id,
            kind=probe.kind,
            abstained=result.answer.abstained,
            leaked=bool(leaks),
            findings=[f"{leak.location}: {leak.detail}" for leak in leaks],
        )
