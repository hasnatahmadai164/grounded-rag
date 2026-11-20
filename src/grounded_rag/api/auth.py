from typing import Annotated

import structlog
from fastapi import Depends, Header

from grounded_rag.api.deps import ContainerDep
from grounded_rag.domain.principal import Principal
from grounded_rag.errors import InvalidApiKeyError, MissingApiKeyError
from grounded_rag.identity import hash_api_key


async def authenticate(
    container: ContainerDep,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """Resolve the caller from `X-API-Key`. Only the key's hash is ever compared or stored."""
    if not api_key:
        raise MissingApiKeyError("Send your API key in the X-API-Key header.")
    record = await container.api_keys.find_active_by_hash(hash_api_key(api_key))
    if record is None:
        raise InvalidApiKeyError("The API key is not valid.")
    structlog.contextvars.bind_contextvars(principal=record.name)
    return record.to_principal()


PrincipalDep = Annotated[Principal, Depends(authenticate)]
