import re
from collections.abc import Sequence
from typing import Protocol

from grounded_rag.domain.retrieval import Hit

_WORD = re.compile(r"[a-z0-9]+")
_MAX_TERMS = 24


class LexicalIndex(Protocol):
    async def lexical_search(
        self, corpus: str, roles: frozenset[str], terms: Sequence[str], *, limit: int
    ) -> list[Hit]: ...


def query_terms(question: str) -> list[str]:
    """Distinct lowercase words of a question, in order of appearance.

    Restricting terms to `[a-z0-9]` is what makes it safe to join them into a tsquery.
    Stop words are removed by the database's text search configuration.
    """
    seen: dict[str, None] = {}
    for word in _WORD.findall(question.lower()):
        seen.setdefault(word, None)
    return list(seen)[:_MAX_TERMS]


async def lexical_search(
    index: LexicalIndex, corpus: str, roles: frozenset[str], question: str, *, limit: int
) -> list[Hit]:
    return await index.lexical_search(corpus, roles, query_terms(question), limit=limit)
