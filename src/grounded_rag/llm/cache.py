import asyncio
import base64
import hashlib
import json
import os
import time
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_rag.ingestion.embedder import Embedder, EmbeddingBatch
from grounded_rag.llm.client import ChatClient, ChatRequest, ChatResult
from grounded_rag.llm.pricing import Pricing


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0


class DiskCache:
    """Content-addressed JSON files. Entries are written atomically and never modified."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.stats = CacheStats()

    @staticmethod
    def key(namespace: str, payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {"namespace": namespace, "payload": payload}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._directory / key[:2] / f"{key}.json"

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = await asyncio.to_thread(self._read, self._path(key))
        if entry is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return entry

    async def put(self, key: str, entry: Mapping[str, Any]) -> None:
        await asyncio.to_thread(self._write, self._path(key), entry)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _write(path: Path, entry: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entry), encoding="utf-8", newline="\n")
        os.replace(temporary, path)


class CachedChatClient:
    """Replays stored responses for identical requests.

    `fingerprint` holds everything configured outside the request that changes the output
    (model, temperature, default token limit). Cost is recomputed from stored token counts,
    and the original call latency is kept so reports can credit it back.
    """

    def __init__(
        self,
        inner: ChatClient,
        cache: DiskCache,
        pricing: Pricing,
        fingerprint: Mapping[str, Any],
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._pricing = pricing
        self._fingerprint = dict(fingerprint)

    async def complete(self, request: ChatRequest) -> ChatResult:
        key = DiskCache.key(
            "chat",
            {
                "fingerprint": self._fingerprint,
                "messages": [[m.role, m.content] for m in request.messages],
                "schema_name": request.schema_name,
                "json_schema": request.json_schema,
                "max_output_tokens": request.max_output_tokens,
            },
        )
        started = time.perf_counter()
        entry = await self._cache.get(key)
        if entry is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return ChatResult(
                content=entry["content"],
                usage=self._pricing.chat_usage(entry["prompt_tokens"], entry["completion_tokens"]),
                latency_ms=elapsed_ms,
                model=entry["model"],
                cached=True,
                replay_credit_ms=max(entry["latency_ms"] - elapsed_ms, 0.0),
            )
        result = await self._inner.complete(request)
        await self._cache.put(
            key,
            {
                "content": result.content,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "latency_ms": result.latency_ms,
                "model": result.model,
            },
        )
        return result


def _encode_vectors(vectors: Sequence[Sequence[float]]) -> list[str]:
    return [base64.b64encode(array("f", vector).tobytes()).decode("ascii") for vector in vectors]


def _decode_vectors(encoded: Sequence[str]) -> list[list[float]]:
    vectors = []
    for item in encoded:
        values: array[float] = array("f")
        values.frombytes(base64.b64decode(item))
        vectors.append(values.tolist())
    return vectors


class CachedEmbedder:
    """Caches whole `embed` calls. Vectors are stored as float32, which is what the index holds,
    and are returned in that form on a miss too so a first run and a replay agree exactly."""

    def __init__(
        self,
        inner: Embedder,
        cache: DiskCache,
        pricing: Pricing,
        fingerprint: Mapping[str, Any],
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._pricing = pricing
        self._fingerprint = dict(fingerprint)

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        key = DiskCache.key("embedding", {"fingerprint": self._fingerprint, "texts": list(texts)})
        started = time.perf_counter()
        entry = await self._cache.get(key)
        if entry is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return EmbeddingBatch(
                vectors=_decode_vectors(entry["vectors"]),
                usage=self._pricing.embedding_usage(entry["tokens"]),
                latency_ms=elapsed_ms,
                replay_credit_ms=max(entry["latency_ms"] - elapsed_ms, 0.0),
            )
        batch = await self._inner.embed(texts)
        encoded = _encode_vectors(batch.vectors)
        await self._cache.put(
            key,
            {
                "vectors": encoded,
                "tokens": batch.usage.embedding_tokens,
                "latency_ms": batch.latency_ms,
            },
        )
        return EmbeddingBatch(_decode_vectors(encoded), batch.usage, batch.latency_ms)
