import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence

from grounded_rag.domain.documents import Chunk, RetrievedChunk
from grounded_rag.domain.retrieval import Hit
from grounded_rag.domain.usage import Usage
from grounded_rag.ingestion.embedder import EmbeddingBatch
from grounded_rag.llm.client import ChatRequest, ChatResult
from grounded_rag.vectorstore.base import VectorMatch, VectorRecord

DIMENSIONS = 64


def hashed_vector(text: str, dimensions: int = DIMENSIONS) -> list[float]:
    """A bag-of-words embedding: texts sharing words are close, which is enough for tests."""
    vector = [0.0] * dimensions
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        index = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls.append(list(texts))
        tokens = sum(len(text.split()) for text in texts)
        return EmbeddingBatch([hashed_vector(t) for t in texts], Usage(embedding_tokens=tokens))


class InMemoryVectorStore:
    """Cosine search with the same role pre-filter contract as the real store."""

    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, VectorRecord]] = {}
        self.fail_next_upsert: Exception | None = None
        self.fail_next_delete: Exception | None = None
        self.queries: list[frozenset[str]] = []

    async def upsert(self, corpus: str, records: Sequence[VectorRecord]) -> None:
        if self.fail_next_upsert is not None:
            error, self.fail_next_upsert = self.fail_next_upsert, None
            raise error
        namespace = self.namespaces.setdefault(corpus, {})
        for record in records:
            namespace[record.id] = record

    async def query(
        self, corpus: str, vector: Sequence[float], *, top_k: int, roles: frozenset[str]
    ) -> list[VectorMatch]:
        self.queries.append(roles)
        scored = [
            VectorMatch(record.id, sum(a * b for a, b in zip(record.values, vector, strict=True)))
            for record in self.namespaces.get(corpus, {}).values()
            if not roles.isdisjoint(record.allowed_roles)
        ]
        return sorted(scored, key=lambda match: (-match.score, match.id))[:top_k]

    async def delete(self, corpus: str, ids: Sequence[str]) -> None:
        if self.fail_next_delete is not None:
            error, self.fail_next_delete = self.fail_next_delete, None
            raise error
        namespace = self.namespaces.get(corpus, {})
        for identifier in ids:
            namespace.pop(identifier, None)

    async def count(self, corpus: str) -> int:
        return len(self.namespaces.get(corpus, {}))

    def roles_of(self, corpus: str, identifier: str) -> frozenset[str]:
        return self.namespaces[corpus][identifier].allowed_roles


class InMemoryChunkStore:
    """PostgreSQL stand-in: full-text-ish search and the authoritative visibility lookup.

    The `ignore_*` switches model a faulty layer so tests can prove later layers still hold.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.ignore_roles_in_search = False
        self.ignore_roles_in_lookup = False
        self.lookups: list[list[str]] = []

    def add_document(
        self,
        doc_id: str,
        title: str,
        roles: set[str],
        texts: Sequence[str],
        *,
        corpus: str = "default",
        status: str = "ready",
    ) -> list[str]:
        ids = []
        for ordinal, text in enumerate(texts):
            chunk = Chunk(
                f"{doc_id}-{ordinal:05d}", doc_id, ordinal, text, 0, len(text), None, title
            )
            self.rows.append(
                {
                    "chunk": chunk,
                    "title": title,
                    "roles": frozenset(roles),
                    "corpus": corpus,
                    "status": status,
                }
            )
            ids.append(chunk.id)
        return ids

    def _readable(
        self, row: dict[str, object], corpus: str, roles: frozenset[str], *, ignore: bool
    ) -> bool:
        if row["corpus"] != corpus or row["status"] != "ready":
            return False
        return ignore or not roles.isdisjoint(row["roles"])  # type: ignore[arg-type]

    async def lexical_search(
        self, corpus: str, roles: frozenset[str], terms: Sequence[str], *, limit: int
    ) -> list[Hit]:
        scored = []
        for row in self.rows:
            if not self._readable(row, corpus, roles, ignore=self.ignore_roles_in_search):
                continue
            chunk: Chunk = row["chunk"]  # type: ignore[assignment]
            words = set(re.findall(r"[a-z0-9]+", f"{row['title']} {chunk.text}".lower()))
            score = sum(1 for term in terms if term in words)
            if score:
                scored.append(Hit(chunk.id, float(score)))
        return sorted(scored, key=lambda h: (-h.score, h.chunk_id))[:limit]

    async def visible_chunks(
        self, corpus: str, roles: frozenset[str], chunk_ids: Sequence[str]
    ) -> list[RetrievedChunk]:
        self.lookups.append(list(chunk_ids))
        wanted = set(chunk_ids)
        return [
            RetrievedChunk(row["chunk"], str(row["title"]), row["roles"])  # type: ignore[arg-type]
            for row in self.rows
            if row["chunk"].id in wanted  # type: ignore[attr-defined]
            and self._readable(row, corpus, roles, ignore=self.ignore_roles_in_lookup)
        ]


class World:
    """A tiny corpus mirrored in the chunk store and the vector store, like real ingestion."""

    def __init__(self) -> None:
        self.chunks = InMemoryChunkStore()
        self.vectors = InMemoryVectorStore()
        self.embedder = FakeEmbedder()

    async def add_document(
        self,
        doc_id: str,
        title: str,
        roles: set[str],
        texts: Sequence[str],
        *,
        vector_roles: set[str] | None = None,
        corpus: str = "default",
        status: str = "ready",
    ) -> list[str]:
        ids = self.chunks.add_document(doc_id, title, roles, texts, corpus=corpus, status=status)
        await self.vectors.upsert(
            corpus,
            [
                VectorRecord(
                    id=chunk_id,
                    values=hashed_vector(f"{title}\n\n{text}"),
                    doc_id=doc_id,
                    allowed_roles=frozenset(vector_roles if vector_roles is not None else roles),
                )
                for chunk_id, text in zip(ids, texts, strict=True)
            ],
        )
        return ids


class PurposeClient:
    """A chat client that answers by the request's purpose, from a queue per purpose.

    Replies may be dicts (sent as JSON) or strings. A callable reply is given the request.
    """

    def __init__(self, **replies: Sequence[object]) -> None:
        self.replies = {purpose: list(items) for purpose, items in replies.items()}
        self.requests: list[ChatRequest] = []

    @property
    def purposes(self) -> list[str]:
        return [request.purpose for request in self.requests]

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        queue = self.replies.get(request.purpose)
        if not queue:
            raise AssertionError(f"Unexpected model call for purpose '{request.purpose}'")
        reply = queue.pop(0)
        if callable(reply):
            reply = reply(request)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        usage = Usage(prompt_tokens=200, completion_tokens=40, cost_usd=0.0001)
        return ChatResult(content, usage, latency_ms=10.0, model="fake-model")


_PASSAGE = re.compile(
    r'<passage id="(?P<id>[^"]+)" document="[^"]*"[^>]*>\n(?P<text>.*?)\n</passage>', re.DOTALL
)


class HeuristicChat:
    """A deterministic model stand-in that reads the prompt and replies plausibly.

    It grades context as sufficient when passages exist (unless `sufficient` says otherwise),
    answers by quoting the start of the first passage, ranks passages in their given order and
    judges everything correct and supported.
    """

    def __init__(self, sufficient: Callable[[str], bool] | None = None) -> None:
        self._sufficient = sufficient or (lambda question: True)
        self.requests: list[ChatRequest] = []

    @property
    def purposes(self) -> list[str]:
        return [request.purpose for request in self.requests]

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        user = request.messages[-1].content
        passages = (
            _PASSAGE.findall(request.messages[1].content) if len(request.messages) > 1 else []
        )
        question = re.search(r"Question: (.*)", request.messages[1].content)
        question_text = question.group(1) if question else ""
        reply = self._reply(request.purpose, question_text, passages, user)
        usage = Usage(prompt_tokens=300, completion_tokens=50, cost_usd=0.0002)
        return ChatResult(json.dumps(reply), usage, latency_ms=20.0, model="heuristic")

    def _reply(
        self, purpose: str, question: str, passages: list[tuple[str, str]], user: str
    ) -> dict:
        if purpose == "rerank":
            return {"ranking": list(range(1, len(passages) + 1))}
        if purpose == "grade":
            enough = bool(passages) and self._sufficient(question)
            return {"sufficient": enough, "reason": "heuristic", "missing": "" if enough else "x"}
        if purpose in {"generate", "repair"}:
            if not passages or not self._sufficient(question):
                return {"answerable": False, "claims": []}
            label, text = passages[0]
            quote = text[:60].rsplit(" ", 1)[0] if len(text) > 60 else text
            claim = {
                "text": f"The document says: {quote}",
                "citations": [{"passage_id": label, "quote": quote}],
            }
            return {"answerable": True, "claims": [claim]}
        if purpose == "rewrite":
            return {"query": f"{question} clause terms"}
        if purpose == "judge_correctness":
            return {"verdict": "correct", "reason": "heuristic"}
        if purpose == "judge_support":
            return {"supported": True, "reason": "heuristic"}
        raise AssertionError(f"Unexpected purpose {purpose}")
