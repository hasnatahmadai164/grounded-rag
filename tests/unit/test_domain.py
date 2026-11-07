import pytest

from grounded_rag.domain.documents import content_sha256, make_chunk_id, make_document_id
from grounded_rag.domain.principal import Principal, normalize_roles, parse_roles
from grounded_rag.errors import InvalidRequestError


def test_principal_sees_document_when_any_role_matches() -> None:
    principal = Principal("analyst", frozenset({"legal", "finance"}))
    assert principal.can_see(["finance", "ops"])


def test_principal_does_not_see_document_without_a_shared_role() -> None:
    assert not Principal("analyst", frozenset({"legal"})).can_see(["finance"])


def test_principal_without_roles_sees_nothing() -> None:
    assert not Principal("nobody", frozenset()).can_see(["legal"])


def test_document_with_no_allowed_roles_is_invisible_to_everyone() -> None:
    assert not Principal("admin", frozenset({"admin"})).can_see([])


def test_roles_are_trimmed_lowercased_and_deduplicated() -> None:
    assert parse_roles(" Legal, finance ,LEGAL,,") == frozenset({"legal", "finance"})


@pytest.mark.parametrize("bad", ["has space", "-leading", "semi;colon", "x" * 65])
def test_invalid_role_names_are_rejected(bad: str) -> None:
    with pytest.raises(InvalidRequestError):
        normalize_roles([bad])


def test_document_id_is_stable_and_scoped_to_the_corpus() -> None:
    digest = content_sha256("same text")
    assert make_document_id("a", digest) == make_document_id("a", digest)
    assert make_document_id("a", digest) != make_document_id("b", digest)


def test_chunk_ids_sort_in_document_order() -> None:
    ids = [make_chunk_id("doc", n) for n in (2, 10, 1)]
    assert sorted(ids) == [make_chunk_id("doc", n) for n in (1, 2, 10)]
