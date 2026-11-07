import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime

from grounded_rag.errors import InvalidRequestError

_ROLE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller: a name and the roles that decide what it may read."""

    name: str
    roles: frozenset[str]

    def can_see(self, allowed_roles: Collection[str]) -> bool:
        return not self.roles.isdisjoint(allowed_roles)


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    id: int
    name: str
    key_prefix: str
    roles: frozenset[str]
    created_at: datetime
    revoked_at: datetime | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def to_principal(self) -> Principal:
        return Principal(name=self.name, roles=self.roles)


def normalize_roles(raw: Iterable[str]) -> frozenset[str]:
    roles = {item.strip().lower() for item in raw if item.strip()}
    invalid = sorted(role for role in roles if not _ROLE_PATTERN.match(role))
    if invalid:
        raise InvalidRequestError(
            f"Invalid role name(s): {', '.join(invalid)}. "
            "Roles use lowercase letters, digits and . _ : - (at most 64 characters).",
            details={"roles": invalid},
        )
    return frozenset(roles)


def parse_roles(csv: str) -> frozenset[str]:
    return normalize_roles(csv.split(","))
