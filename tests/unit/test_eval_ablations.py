from pathlib import Path

import pytest

from grounded_rag.agent.config import AgentConfig
from grounded_rag.domain.retrieval import RetrievalMode
from grounded_rag.errors import ConfigurationError
from grounded_rag.evals.ablations import CANONICAL_ORDER, load_ablation, load_ablations

CONFIGS = Path(__file__).resolve().parents[2] / "evals" / "configs"


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_checked_in_configurations_are_the_five_ablations_in_order() -> None:
    names = [a.name for a in load_ablations(CONFIGS)]

    assert names == list(CANONICAL_ORDER)


def test_each_configuration_adds_one_component_to_the_previous_one() -> None:
    by_name = {a.name: a.agent for a in load_ablations(CONFIGS)}

    assert by_name["dense_only"].mode is RetrievalMode.DENSE
    assert by_name["lexical_only"].mode is RetrievalMode.LEXICAL
    assert (by_name["hybrid"].mode, by_name["hybrid"].rerank) == (RetrievalMode.HYBRID, False)
    assert by_name["hybrid_rerank"].rerank is True
    assert by_name["hybrid_rerank"].max_rewrites == 0
    assert by_name["hybrid_rerank_retry"].max_rewrites == 2


def test_only_the_retry_configuration_may_rewrite_queries() -> None:
    rewrites = {a.name: a.agent.max_rewrites for a in load_ablations(CONFIGS)}

    assert [name for name, count in rewrites.items() if count > 0] == ["hybrid_rerank_retry"]


def test_omitted_settings_come_from_the_base_configuration(tmp_path: Path) -> None:
    path = write(tmp_path, "x", 'name = "x"\nretrieval_mode = "dense"\n')

    ablation = load_ablation(path, AgentConfig(top_k=9, max_repairs=3))

    assert (ablation.agent.top_k, ablation.agent.max_repairs) == (9, 3)
    assert ablation.agent.mode is RetrievalMode.DENSE


def test_a_subset_can_be_selected_by_name(tmp_path: Path) -> None:
    assert [a.name for a in load_ablations(CONFIGS, ["hybrid", "dense_only"])] == [
        "hybrid",
        "dense_only",
    ]


def test_an_unknown_name_lists_what_is_available() -> None:
    with pytest.raises(ConfigurationError, match=r"nope.*Available: dense_only"):
        load_ablations(CONFIGS, ["nope"])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('name = "x"\ncolour = "red"\n', "unknown settings: colour"),
        ('description = "no name"\n', "needs a name"),
        ('name = "x"\nretrieval_mode = "sparse"\n', "unknown retrieval_mode"),
        ("name = ", "not valid TOML"),
    ],
)
def test_malformed_configuration_files_are_rejected_with_a_reason(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_ablation(write(tmp_path, "bad", body))


def test_the_description_is_available_for_reports() -> None:
    ablation = load_ablation(CONFIGS / "hybrid_rerank_retry.toml")

    assert ablation.describe()["max_rewrites"] == 2
    assert "rewrites" in ablation.describe()["description"]
