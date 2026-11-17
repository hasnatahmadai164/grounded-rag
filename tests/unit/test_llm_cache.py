from pathlib import Path

import pytest

from grounded_rag.domain.usage import Usage
from grounded_rag.llm.cache import CachedChatClient, CachedEmbedder, DiskCache
from grounded_rag.llm.client import ChatMessage, ChatRequest, ChatResult
from grounded_rag.llm.pricing import Pricing
from tests.fakes import FakeEmbedder

PRICING = Pricing(chat_input=1.0, chat_output=2.0, embedding=0.5)


class CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.calls += 1
        usage = PRICING.chat_usage(1000, 100)
        return ChatResult(f"answer {self.calls}", usage, latency_ms=900.0, model="m-1")


def request(text: str = "question", **overrides: object) -> ChatRequest:
    return ChatRequest(messages=(ChatMessage("user", text),), **overrides)  # type: ignore[arg-type]


def cached(
    inner: CountingClient, tmp_path: Path, pricing: Pricing = PRICING, **fingerprint: object
):
    cache = DiskCache(tmp_path)
    fingerprint = {"model": "m-1", "temperature": 0.0} | fingerprint
    return CachedChatClient(inner, cache, pricing, fingerprint), cache


async def test_identical_requests_are_answered_from_disk(tmp_path: Path) -> None:
    inner = CountingClient()
    client, _ = cached(inner, tmp_path)

    first = await client.complete(request())
    second = await client.complete(request())

    assert inner.calls == 1
    assert second.content == first.content
    assert second.cached is True
    assert first.cached is False


async def test_a_second_process_reads_what_the_first_one_stored(tmp_path: Path) -> None:
    await cached(CountingClient(), tmp_path)[0].complete(request())
    other = CountingClient()

    replay = await cached(other, tmp_path)[0].complete(request())

    assert other.calls == 0
    assert replay.content == "answer 1"


async def test_replay_keeps_token_counts_and_credits_the_original_latency(tmp_path: Path) -> None:
    client, _ = cached(CountingClient(), tmp_path)
    await client.complete(request())

    replay = await client.complete(request())

    assert (replay.usage.prompt_tokens, replay.usage.completion_tokens) == (1000, 100)
    assert replay.latency_ms < 100
    assert replay.replay_credit_ms == pytest.approx(900.0, abs=100)


async def test_replayed_cost_follows_the_current_prices(tmp_path: Path) -> None:
    await cached(CountingClient(), tmp_path)[0].complete(request())
    cheaper = Pricing(chat_input=0.5, chat_output=1.0, embedding=0.5)

    replay = await cached(CountingClient(), tmp_path, cheaper)[0].complete(request())

    assert replay.usage.cost_usd == pytest.approx((1000 * 0.5 + 100 * 1.0) / 1_000_000)


@pytest.mark.parametrize(
    "different",
    [
        request("another question"),
        request(max_output_tokens=50),
        request(schema_name="S", json_schema={"type": "object"}),
    ],
)
async def test_any_change_to_the_request_misses_the_cache(
    tmp_path: Path, different: ChatRequest
) -> None:
    inner = CountingClient()
    client, _ = cached(inner, tmp_path)
    await client.complete(request())

    await client.complete(different)

    assert inner.calls == 2


async def test_a_different_model_or_temperature_misses_the_cache(tmp_path: Path) -> None:
    inner = CountingClient()
    await cached(inner, tmp_path, model="m-1")[0].complete(request())
    await cached(inner, tmp_path, model="m-2")[0].complete(request())
    await cached(inner, tmp_path, temperature=0.7)[0].complete(request())

    assert inner.calls == 3


async def test_the_purpose_label_does_not_affect_the_key(tmp_path: Path) -> None:
    inner = CountingClient()
    client, _ = cached(inner, tmp_path)
    await client.complete(request(purpose="grade"))
    await client.complete(request(purpose="generate"))

    assert inner.calls == 1


async def test_a_corrupt_entry_is_treated_as_a_miss_and_rewritten(tmp_path: Path) -> None:
    inner = CountingClient()
    client, cache = cached(inner, tmp_path)
    await client.complete(request())
    for path in tmp_path.rglob("*.json"):
        path.write_text("{truncated", encoding="utf-8")

    await client.complete(request())
    replay = await client.complete(request())

    assert inner.calls == 2
    assert replay.cached is True
    assert cache.stats.hits == 1


async def test_hit_and_miss_counts_are_tracked(tmp_path: Path) -> None:
    client, cache = cached(CountingClient(), tmp_path)
    await client.complete(request("a"))
    await client.complete(request("a"))
    await client.complete(request("b"))

    assert (cache.stats.hits, cache.stats.misses) == (1, 2)


async def test_failed_calls_are_not_stored(tmp_path: Path) -> None:
    class Failing:
        async def complete(self, request: ChatRequest) -> ChatResult:
            raise RuntimeError("boom")

    client = CachedChatClient(Failing(), DiskCache(tmp_path), PRICING, {})
    with pytest.raises(RuntimeError):
        await client.complete(request())

    assert list(tmp_path.rglob("*.json")) == []


async def test_cached_embeddings_match_the_first_run_exactly(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, DiskCache(tmp_path), PRICING, {"model": "e-1"})

    first = await embedder.embed(["alpha beta", "gamma"])
    second = await embedder.embed(["alpha beta", "gamma"])

    assert len(inner.calls) == 1
    assert second.vectors == first.vectors
    assert second.usage.embedding_tokens == first.usage.embedding_tokens == 3
    assert second.usage.cost_usd == pytest.approx(3 * 0.5 / 1_000_000)


async def test_embedding_cache_distinguishes_inputs_and_models(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    cache = DiskCache(tmp_path)
    await CachedEmbedder(inner, cache, PRICING, {"model": "e-1"}).embed(["alpha"])
    await CachedEmbedder(inner, cache, PRICING, {"model": "e-1"}).embed(["beta"])
    await CachedEmbedder(inner, cache, PRICING, {"model": "e-2"}).embed(["alpha"])

    assert len(inner.calls) == 3


def test_usage_of_a_free_call_is_zero() -> None:
    assert Usage().total_tokens == 0
