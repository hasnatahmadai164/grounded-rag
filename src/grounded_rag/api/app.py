from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from grounded_rag import __version__
from grounded_rag.api.errors import register_error_handlers
from grounded_rag.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from grounded_rag.api.routes import documents, health, query
from grounded_rag.config import Settings
from grounded_rag.container import Container, open_container
from grounded_rag.logging import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the application. A ready-made `container` is used as is; otherwise one is opened
    from `settings` at startup and closed at shutdown."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(level=settings.log_level, json_output=settings.log_json)
        if container is not None:
            yield
            return
        async with open_container(settings) as opened:
            app.state.container = opened
            yield

    app = FastAPI(title="grounded-rag", version=__version__, lifespan=lifespan)
    if container is not None:
        app.state.container = container
    register_error_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app
