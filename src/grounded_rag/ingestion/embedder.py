from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from grounded_rag.domain.usage import Usage
from grounded_rag.errors import InvalidRequestError, LLMOutputError
from grounded_rag.llm.openai_http import OpenAIHttp
from grounded_rag.llm.pricing import Pricing


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    usage: Usage
    latency_ms: float = 0.0
    replay_credit_ms: float = 0.0


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


class OpenAIEmbedder:
    def __init__(
        self,
        http: OpenAIHttp,
        *,
        model: str,
        dimensions: int,
        pricing: Pricing,
        batch_size: int = 96,
    ) -> None:
        self._http = http
        self._model = model
        self._dimensions = dimensions
        self._pricing = pricing
        self._batch_size = batch_size

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if any(not text.strip() for text in texts):
            raise InvalidRequestError("Cannot embed empty text.")
        vectors: list[list[float]] = []
        tokens = 0
        latency_ms = 0.0
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            result = await self._http.post(
                "/embeddings",
                {"model": self._model, "input": batch, "dimensions": self._dimensions},
            )
            vectors.extend(self._vectors_from(result.body, expected=len(batch)))
            tokens += int(result.body["usage"]["prompt_tokens"])
            latency_ms += result.latency_ms
        return EmbeddingBatch(vectors, self._pricing.embedding_usage(tokens), latency_ms)

    def _vectors_from(self, body: dict[str, object], *, expected: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise LLMOutputError("Embedding response did not match the number of inputs.")
        ordered = sorted(data, key=lambda item: item["index"])
        vectors = [item["embedding"] for item in ordered]
        if any(len(vector) != self._dimensions for vector in vectors):
            raise LLMOutputError(
                f"Embedding dimension differs from EMBEDDING_DIMENSIONS={self._dimensions}."
            )
        return vectors
