import pytest

from grounded_rag.config import Settings
from grounded_rag.domain.usage import Usage
from grounded_rag.llm.pricing import Pricing

PRICING = Pricing(chat_input=0.15, chat_output=0.60, embedding=0.02)


def test_chat_cost_charges_input_and_output_separately() -> None:
    usage = PRICING.chat_usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert usage.cost_usd == pytest.approx(0.15 + 0.30)


def test_embedding_cost_uses_the_embedding_rate() -> None:
    assert PRICING.embedding_usage(2_000_000).cost_usd == pytest.approx(0.04)


def test_zero_tokens_cost_nothing() -> None:
    assert PRICING.chat_usage(0, 0).cost_usd == 0.0


def test_usage_values_add_up() -> None:
    total = PRICING.chat_usage(100, 20) + PRICING.embedding_usage(50)
    assert (total.prompt_tokens, total.completion_tokens, total.embedding_tokens) == (100, 20, 50)
    assert total.total_tokens == 170


def test_pricing_follows_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_INPUT_USD_PER_MTOK", "1.0")
    pricing = Pricing.from_settings(Settings(_env_file=None))
    assert pricing.chat_input == 1.0


def test_empty_usage_is_the_additive_identity() -> None:
    usage = PRICING.chat_usage(10, 5)
    assert usage + Usage() == usage
