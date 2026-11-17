from grounded_rag.domain.usage import Usage
from grounded_rag.llm.budget import Budget, MeteredLLM, UsageMeter
from grounded_rag.llm.client import ChatRequest, ChatResult
from grounded_rag.retrieval.rewrite import QueryRewriter, clean_query


class ScriptedClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        return ChatResult(self.content, Usage(prompt_tokens=20, completion_tokens=8), 1.0, "m")


def rewriter(content: str) -> tuple[QueryRewriter, ScriptedClient, UsageMeter]:
    client, meter = ScriptedClient(content), UsageMeter()
    return QueryRewriter(MeteredLLM(client, meter, Budget(100_000, 10.0))), client, meter


async def test_the_new_query_comes_from_the_model() -> None:
    service, _, meter = rewriter('{"query": "notice period to prevent automatic renewal"}')

    query = await service.rewrite(
        "How do I stop it renewing?", ["stop renewing"], "no clause found"
    )

    assert query == "notice period to prevent automatic renewal"
    assert meter.by_purpose["rewrite"].completion_tokens == 8


async def test_prompt_lists_earlier_attempts_and_wraps_feedback_as_data() -> None:
    service, client, _ = rewriter('{"query": "x"}')

    await service.rewrite("Q?", ["first try", "second try"], "missing the renewal clause")

    system, user = client.requests[0].messages
    assert "not instructions" in system.content
    assert "- first try\n- second try" in user.content
    assert "<feedback>\nmissing the renewal clause\n</feedback>" in user.content


async def test_blank_output_falls_back_to_the_original_question() -> None:
    service, _, _ = rewriter('{"query": "   "}')
    assert await service.rewrite("Original?", [], "x") == "Original?"


def test_queries_are_collapsed_to_one_bounded_line() -> None:
    assert clean_query("a\n  b\tc", "fb") == "a b c"
    assert len(clean_query("w " * 500, "fb")) <= 300
