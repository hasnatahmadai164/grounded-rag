from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from grounded_rag.agent.config import AgentConfig
from grounded_rag.agent.service import AnswerService
from grounded_rag.config import Settings
from grounded_rag.db.repositories import ApiKeyRepository, ChunkRepository, DocumentRepository
from grounded_rag.db.session import create_engine, create_session_factory
from grounded_rag.ingestion.chunking import ChunkingConfig
from grounded_rag.ingestion.embedder import Embedder, OpenAIEmbedder
from grounded_rag.ingestion.pipeline import IngestionPipeline
from grounded_rag.llm.budget import Budget
from grounded_rag.llm.client import ChatClient, OpenAIChatClient
from grounded_rag.llm.openai_http import OpenAIHttp
from grounded_rag.llm.pricing import Pricing
from grounded_rag.retrieval.dense import DenseRetriever
from grounded_rag.retrieval.service import RetrievalConfig, RetrievalService
from grounded_rag.vectorstore.base import VectorStore
from grounded_rag.vectorstore.pinecone_store import PineconeStore


@dataclass(slots=True)
class Container:
    """The composition root: the only place that decides which implementation backs what."""

    settings: Settings
    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]
    documents: DocumentRepository
    api_keys: ApiKeyRepository
    embedder: Embedder
    vectors: VectorStore
    ingestion: IngestionPipeline
    chunks: ChunkRepository
    retrieval: RetrievalService
    chat: ChatClient
    answers: AnswerService


def pinecone_from_settings(settings: Settings) -> PineconeStore:
    return PineconeStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        index_host=settings.pinecone_index_host,
        api_version=settings.pinecone_api_version,
        timeout_seconds=settings.pinecone_timeout_seconds,
        max_retries=settings.pinecone_max_retries,
    )


def agent_config_from_settings(settings: Settings) -> AgentConfig:
    return AgentConfig(
        rerank=settings.rerank_enabled,
        max_rewrites=settings.agent_max_rewrites,
        max_repairs=settings.agent_max_repairs,
        top_k=settings.default_top_k,
        max_context_chunks=settings.agent_max_context_chunks,
    )


def chunking_from_settings(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(
        target_chars=settings.chunk_target_chars,
        max_chars=settings.chunk_max_chars,
        min_chars=settings.chunk_min_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )


@asynccontextmanager
async def open_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vectors: VectorStore | None = None,
    chat: ChatClient | None = None,
) -> AsyncIterator[Container]:
    """Build every service from settings. Explicit arguments replace the real providers."""
    async with AsyncExitStack() as stack:
        engine = create_engine(settings)
        stack.push_async_callback(engine.dispose)
        sessions = create_session_factory(engine)
        documents = DocumentRepository(sessions)

        pricing = Pricing.from_settings(settings)
        if embedder is None or chat is None:
            http = OpenAIHttp(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
            stack.push_async_callback(http.aclose)
            embedder = embedder or OpenAIEmbedder(
                http,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                pricing=pricing,
                batch_size=settings.embedding_batch_size,
            )
            chat = chat or OpenAIChatClient(
                http,
                model=settings.chat_model,
                pricing=pricing,
                temperature=settings.llm_temperature if settings.llm_send_temperature else None,
                default_max_output_tokens=settings.llm_max_output_tokens,
            )
        if vectors is None:
            pinecone = pinecone_from_settings(settings)
            stack.push_async_callback(pinecone.aclose)
            vectors = pinecone

        chunks = ChunkRepository(sessions)
        retrieval = RetrievalService(
            DenseRetriever(embedder, vectors),
            chunks,
            RetrievalConfig(
                dense_top_k=settings.retrieval_dense_top_k,
                lexical_top_k=settings.retrieval_lexical_top_k,
                rrf_k=settings.retrieval_rrf_k,
                candidates=settings.retrieval_candidates,
            ),
        )
        yield Container(
            settings=settings,
            engine=engine,
            sessions=sessions,
            documents=documents,
            api_keys=ApiKeyRepository(sessions),
            embedder=embedder,
            vectors=vectors,
            ingestion=IngestionPipeline(
                documents, embedder, vectors, chunking_from_settings(settings)
            ),
            chunks=chunks,
            retrieval=retrieval,
            chat=chat,
            answers=AnswerService(
                retrieval,
                chat,
                Budget(settings.request_token_budget, settings.request_cost_budget_usd),
                agent_config_from_settings(settings),
            ),
        )
