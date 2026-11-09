import hashlib
import secrets
from dataclasses import dataclass

_KEY_PREFIX = "grk_"


@dataclass(frozen=True, slots=True)
class GeneratedApiKey:
    """The plaintext `secret` exists only at creation time; the database keeps `key_hash`."""

    secret: str
    key_hash: str
    key_prefix: str


def hash_api_key(secret: str) -> str:
    """SHA-256 suffices: keys carry 256 bits of entropy, so there is nothing to brute-force."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def generate_api_key() -> GeneratedApiKey:
    secret = _KEY_PREFIX + secrets.token_urlsafe(32)
    return GeneratedApiKey(secret=secret, key_hash=hash_api_key(secret), key_prefix=secret[:8])
