import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from grounded_rag.db.repositories import ApiKeyRepository
from grounded_rag.identity import generate_api_key

pytestmark = pytest.mark.db


@pytest.fixture
def repository(session_factory: async_sessionmaker[AsyncSession]) -> ApiKeyRepository:
    return ApiKeyRepository(session_factory)


async def create_key(repository: ApiKeyRepository, name: str = "analyst", roles=("legal",)):
    key = generate_api_key()
    record = await repository.create(
        name=name, key_hash=key.key_hash, key_prefix=key.key_prefix, roles=frozenset(roles)
    )
    return key, record


async def test_created_key_is_found_by_its_hash_with_its_roles(
    repository: ApiKeyRepository,
) -> None:
    key, record = await create_key(repository, roles=("legal", "finance"))

    found = await repository.find_active_by_hash(key.key_hash)

    assert found is not None
    assert found.id == record.id
    assert found.to_principal().roles == frozenset({"legal", "finance"})


async def test_unknown_hash_is_not_found(repository: ApiKeyRepository) -> None:
    assert await repository.find_active_by_hash("0" * 64) is None


async def test_revoked_key_no_longer_authenticates(repository: ApiKeyRepository) -> None:
    key, record = await create_key(repository)

    assert await repository.revoke(record.id) is True

    assert await repository.find_active_by_hash(key.key_hash) is None


async def test_revoking_twice_reports_that_nothing_changed(repository: ApiKeyRepository) -> None:
    _, record = await create_key(repository)
    await repository.revoke(record.id)

    assert await repository.revoke(record.id) is False


async def test_listing_includes_revoked_keys_in_creation_order(
    repository: ApiKeyRepository,
) -> None:
    _, first = await create_key(repository, name="first")
    await create_key(repository, name="second")
    await repository.revoke(first.id)

    listed = await repository.list()

    assert [(item.name, item.active) for item in listed] == [("first", False), ("second", True)]


async def test_the_same_hash_cannot_be_stored_twice(repository: ApiKeyRepository) -> None:
    key, _ = await create_key(repository)
    with pytest.raises(IntegrityError):
        await repository.create(
            name="clone", key_hash=key.key_hash, key_prefix=key.key_prefix, roles=frozenset()
        )


async def test_principal_lookup_by_name_ignores_revoked_keys(repository: ApiKeyRepository) -> None:
    _, old = await create_key(repository, name="analyst", roles=("legal",))
    await repository.revoke(old.id)
    await create_key(repository, name="analyst", roles=("finance",))

    found = await repository.find_active_by_name("analyst")

    assert found is not None
    assert found.roles == frozenset({"finance"})
