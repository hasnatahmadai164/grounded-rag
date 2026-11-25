from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from grounded_rag.config import Settings
from grounded_rag.container import Container, open_container
from grounded_rag.evals.guard import GuardedChat, GuardedEmbedder, SpendGuard
from grounded_rag.evals.judge import Judge
from grounded_rag.ingestion.embedder import OpenAIEmbedder
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from grounded_rag.llm.cache import CachedChatClient, CachedEmbedder, CacheStats, DiskCache
from grounded_rag.llm.client import OpenAIChatClient
from grounded_rag.llm.openai_http import OpenAIHttp
from grounded_rag.llm.pricing import Pricing

_UNLIMITED = Budget(max_tokens=10**12, max_cost_usd=1e12)


@dataclass(slots=True)
class EvalEnvironment:
    """Everything an evaluation run needs, with provider calls cached and spend-limited."""

    container: Container
    judge: Judge
    judge_meter: UsageMeter
    corpus: str
    chat_model: str
    judge_model: str
    embedding_model: str
    guard: SpendGuard | None = None
    cache_stats: CacheStats | None = None


@asynccontextmanager
async def open_eval_environment(
    settings: Settings, *, max_cost_usd: float | None
) -> AsyncIterator[EvalEnvironment]:
    pricing = Pricing.from_settings(settings)
    guard = SpendGuard(max_usd=max_cost_usd)
    cache = DiskCache(settings.eval_cache_dir)
    judge_model = settings.judge_model or settings.chat_model
    temperature = settings.llm_temperature if settings.llm_send_temperature else None

    async with AsyncExitStack() as stack:
        http = OpenAIHttp(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        stack.push_async_callback(http.aclose)

        def cached_chat(model: str) -> CachedChatClient:
            live = OpenAIChatClient(
                http,
                model=model,
                pricing=pricing,
                temperature=temperature,
                default_max_output_tokens=settings.llm_max_output_tokens,
            )
            fingerprint = {
                "model": model,
                "temperature": temperature,
                "max_output_tokens": settings.llm_max_output_tokens,
            }
            return CachedChatClient(GuardedChat(live, guard), cache, pricing, fingerprint)

        embedder = CachedEmbedder(
            GuardedEmbedder(
                OpenAIEmbedder(
                    http,
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                    pricing=pricing,
                    batch_size=settings.embedding_batch_size,
                ),
                guard,
            ),
            cache,
            pricing,
            {"model": settings.embedding_model, "dimensions": settings.embedding_dimensions},
        )
        container = await stack.enter_async_context(
            open_container(settings, embedder=embedder, chat=cached_chat(settings.chat_model))
        )
        judge_meter = UsageMeter()
        judge = Judge(MeteredLLM(cached_chat(judge_model), judge_meter, _UNLIMITED))
        yield EvalEnvironment(
            container=container,
            judge=judge,
            judge_meter=judge_meter,
            corpus=settings.eval_corpus,
            chat_model=settings.chat_model,
            judge_model=judge_model,
            embedding_model=settings.embedding_model,
            guard=guard,
            cache_stats=cache.stats,
        )
