import asyncio
from collections.abc import Awaitable, Callable

from fastapi import APIRouter
from sqlalchemy import text

from grounded_rag.api.deps import ContainerDep
from grounded_rag.errors import DependencyUnavailableError
from grounded_rag.logging import get_logger

router = APIRouter(tags=["health"])
log = get_logger(__name__)

_CHECK_TIMEOUT_SECONDS = 5.0


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up and serving requests."""
    return {"status": "ok"}


async def _run_check(name: str, check: Callable[[], Awaitable[object]]) -> str:
    try:
        async with asyncio.timeout(_CHECK_TIMEOUT_SECONDS):
            await check()
    except Exception as error:
        log.warning(
            "readiness_check_failed", dependency=name, error=type(error).__name__, exc_info=error
        )
        return "unavailable"
    return "ok"


@router.get("/readyz")
async def readyz(container: ContainerDep) -> dict[str, object]:
    """Readiness: PostgreSQL answers a query and the vector index is reachable."""

    async def postgres() -> None:
        async with container.sessions() as session:
            await session.execute(text("SELECT 1"))

    async def pinecone() -> None:
        await container.vectors.count(container.settings.default_corpus)

    names = ["postgres", "pinecone"]
    outcomes = await asyncio.gather(
        _run_check("postgres", postgres), _run_check("pinecone", pinecone)
    )
    checks = dict(zip(names, outcomes, strict=True))
    if any(state != "ok" for state in checks.values()):
        raise DependencyUnavailableError("The service is not ready.", details={"checks": checks})
    return {"status": "ready", "checks": checks}
