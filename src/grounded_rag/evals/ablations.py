import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from grounded_rag.agent.config import AgentConfig
from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.errors import ConfigurationError

# The order configurations appear in reports: each row adds one component to the previous one.
CANONICAL_ORDER = (
    "dense_only",
    "lexical_only",
    "hybrid",
    "hybrid_rerank",
    "hybrid_rerank_retry",
)

_ALLOWED_KEYS = frozenset(
    {
        "name",
        "description",
        "retrieval_mode",
        "rerank",
        "max_rewrites",
        "max_repairs",
        "top_k",
        "max_context_chunks",
    }
)


@dataclass(frozen=True, slots=True)
class Ablation:
    name: str
    description: str
    agent: AgentConfig

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "retrieval_mode": self.agent.mode.value,
            "rerank": self.agent.rerank,
            "max_rewrites": self.agent.max_rewrites,
            "max_repairs": self.agent.max_repairs,
            "top_k": self.agent.top_k,
        }


def load_ablation(path: Path, base: AgentConfig | None = None) -> Ablation:
    """Read one named configuration. Settings the file omits come from `base`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"{path.name} is not valid TOML: {error}") from error
    unknown = sorted(set(data) - _ALLOWED_KEYS)
    if unknown:
        raise ConfigurationError(f"{path.name} has unknown settings: {', '.join(unknown)}")
    if "name" not in data:
        raise ConfigurationError(f"{path.name} needs a name.")
    overrides: dict[str, Any] = {k: v for k, v in data.items() if k not in {"name", "description"}}
    if "retrieval_mode" in overrides:
        try:
            overrides["mode"] = RetrievalMode(overrides.pop("retrieval_mode"))
        except ValueError as error:
            raise ConfigurationError(f"{path.name}: unknown retrieval_mode.") from error
    agent = replace(base or AgentConfig(), **overrides)
    return Ablation(
        name=str(data["name"]), description=str(data.get("description", "")), agent=agent
    )


def load_ablations(
    directory: Path, names: list[str] | None = None, base: AgentConfig | None = None
) -> list[Ablation]:
    found = {a.name: a for a in (load_ablation(p, base) for p in sorted(directory.glob("*.toml")))}
    wanted = (
        names
        if names
        else [n for n in CANONICAL_ORDER if n in found] + sorted(set(found) - set(CANONICAL_ORDER))
    )
    missing = [name for name in wanted if name not in found]
    if missing:
        raise ConfigurationError(
            f"Unknown configuration(s) {', '.join(missing)}. Available: {', '.join(sorted(found))}."
        )
    return [found[name] for name in wanted]
