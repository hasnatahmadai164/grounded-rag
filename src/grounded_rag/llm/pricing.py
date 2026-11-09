from dataclasses import dataclass

from grounded_rag.config import Settings
from grounded_rag.domain.usage import Usage

_TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD per million tokens. Values come from settings so a price change is not a code change."""

    chat_input: float
    chat_output: float
    embedding: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pricing":
        return cls(
            chat_input=settings.chat_input_usd_per_mtok,
            chat_output=settings.chat_output_usd_per_mtok,
            embedding=settings.embedding_usd_per_mtok,
        )

    def chat_usage(self, prompt_tokens: int, completion_tokens: int) -> Usage:
        cost = (
            prompt_tokens * self.chat_input + completion_tokens * self.chat_output
        ) / _TOKENS_PER_MILLION
        return Usage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=cost
        )

    def embedding_usage(self, tokens: int) -> Usage:
        return Usage(
            embedding_tokens=tokens, cost_usd=tokens * self.embedding / _TOKENS_PER_MILLION
        )
