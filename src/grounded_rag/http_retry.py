import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from grounded_rag.errors import GroundedRagError
from grounded_rag.logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 30.0

Sleep = Callable[[float], Awaitable[None]]


def retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(0.5 * 2.0**attempt, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.25)


def error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        detail = body["error"]["message"] if "error" in body else body["message"]
    except (ValueError, KeyError, TypeError):
        detail = response.reason_phrase
    return str(detail)


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    provider: str,
    error: type[GroundedRagError],
    timeout_error: type[GroundedRagError],
    max_retries: int,
    json: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    accept_status: frozenset[int] = frozenset(),
    sleep: Sleep = asyncio.sleep,
) -> httpx.Response:
    """Send a request, retrying timeouts, connection errors, 429 and 5xx with backoff.

    Any other status of 400 or above raises `error`, unless listed in `accept_status`.
    Messages carry the status and the provider's explanation, never the request.
    """
    for attempt in range(max_retries + 1):
        final_attempt = attempt == max_retries
        try:
            response = await client.request(method, path, json=json, headers=headers)
        except httpx.TimeoutException as exc:
            if final_attempt:
                raise timeout_error(f"{provider} request to {path} timed out.") from exc
            await _back_off(sleep, provider, path, attempt, None, "timeout")
            continue
        except httpx.TransportError as exc:
            if final_attempt:
                raise error(f"Could not reach {provider}: {type(exc).__name__}.") from exc
            await _back_off(sleep, provider, path, attempt, None, type(exc).__name__)
            continue

        if response.status_code in RETRYABLE_STATUS and not final_attempt:
            retry_after = response.headers.get("retry-after")
            await _back_off(sleep, provider, path, attempt, retry_after, response.status_code)
            continue
        if response.status_code >= 400 and response.status_code not in accept_status:
            raise error(f"{provider} returned {response.status_code}: {error_detail(response)}")
        return response
    raise AssertionError("unreachable: the final attempt always returns or raises")


async def _back_off(
    sleep: Sleep, provider: str, path: str, attempt: int, retry_after: str | None, reason: object
) -> None:
    delay = retry_delay(attempt, retry_after)
    log.warning(
        "provider_retry",
        provider=provider,
        path=path,
        attempt=attempt + 1,
        reason=reason,
        delay_s=delay,
    )
    await sleep(delay)
