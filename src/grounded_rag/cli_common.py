import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any

import typer

from grounded_rag.errors import GroundedRagError


def run_command[**P, T](handler: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """Run an async command, report service errors on stderr and exit non-zero."""

    @wraps(handler)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return asyncio.run(handler(*args, **kwargs))
        except GroundedRagError as error:
            typer.echo(f"error [{error.code}]: {error.message}", err=True)
            raise typer.Exit(code=1) from error

    return wrapper
