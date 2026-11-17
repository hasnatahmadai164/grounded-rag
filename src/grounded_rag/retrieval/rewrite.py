from collections.abc import Sequence

from pydantic import BaseModel, Field

from grounded_rag.llm.budget import MeteredLLM
from grounded_rag.llm.prompts import rewrite_messages

_MAX_QUERY_CHARS = 300


class RewriteOutput(BaseModel):
    query: str = Field(description="A single search query.")


def clean_query(raw: str, fallback: str) -> str:
    """One line, bounded length. Falls back to `fallback` when the model returned nothing usable."""
    query = " ".join(raw.split())[:_MAX_QUERY_CHARS].strip()
    return query or fallback


class QueryRewriter:
    def __init__(self, llm: MeteredLLM) -> None:
        self._llm = llm

    async def rewrite(self, question: str, previous_queries: Sequence[str], feedback: str) -> str:
        output = await self._llm.structured(
            rewrite_messages(question, previous_queries, feedback),
            RewriteOutput,
            purpose="rewrite",
            max_output_tokens=128,
        )
        return clean_query(output.query, fallback=question)
