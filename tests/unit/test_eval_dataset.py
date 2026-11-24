import json
from pathlib import Path

import pytest

from grounded_rag.errors import ConfigurationError
from grounded_rag.evals.dataset import (
    ACCESS_LABELS,
    QUESTION_TEMPLATES,
    access_label,
    assign_splits,
    build_dataset,
    display_title,
    load_dataset,
    write_dataset,
)

SPAN = {"text": "governed by the laws of Delaware", "start": 10, "end": 42}


def make_subset(tmp_path: Path, count: int = 6) -> dict:
    (tmp_path / "contracts").mkdir()
    contracts = []
    for n in range(count):
        file = f"contracts/Co{n}_2020-SUPPLY AGREEMENT_{n:08x}.txt"
        (tmp_path / file).write_text(
            f"Contract number {n}. governed by the laws of Delaware.", encoding="utf-8"
        )
        contracts.append(
            {
                "title": f"Co{n}_2020-SUPPLY AGREEMENT.pdf",
                "file": file,
                "questions": [
                    {"category": "Governing Law", "answerable": True, "answers": [SPAN]},
                    {"category": "Audit Rights", "answerable": False, "answers": []},
                    {"category": "Parties", "answerable": True, "answers": [SPAN]},
                ],
            }
        )
    return {"source": {"license": "CC BY 4.0"}, "contracts": contracts}


def test_display_titles_are_readable() -> None:
    assert display_title("LIME_ENERGY_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.pdf") == (
        "LIME ENERGY 09 09 1999 EX 10 DISTRIBUTOR AGREEMENT"
    )


def test_access_labels_are_deterministic_and_spread_across_all_labels() -> None:
    labels = [access_label(f"contract-{n}") for n in range(60)]

    assert labels == [access_label(f"contract-{n}") for n in range(60)]
    assert set(labels) == set(ACCESS_LABELS)


def test_the_dev_set_has_a_fixed_size_and_does_not_depend_on_listing_order() -> None:
    ids = [f"c{n}" for n in range(20)]

    forward = assign_splits(ids, 5)
    backward = assign_splits(list(reversed(ids)), 5)

    assert forward == backward
    assert sum(1 for split in forward.values() if split == "dev") == 5


def test_questions_are_built_from_templates_for_selected_categories_only(tmp_path: Path) -> None:
    dataset = build_dataset(make_subset(tmp_path), tmp_path, dev_count=2)

    categories = {q.category for q in dataset.questions}
    assert categories == {"Governing Law", "Audit Rights"}
    first = dataset.questions[0]
    assert (
        first.question == 'Which law governs the "Co0 2020 SUPPLY AGREEMENT"?'
        or "Co" in first.question
    )
    assert "{title}" not in first.question


def test_absent_clauses_become_unanswerable_questions_without_gold_spans(tmp_path: Path) -> None:
    dataset = build_dataset(make_subset(tmp_path), tmp_path)

    audit = next(q for q in dataset.questions if q.category == "Audit Rights")
    law = next(q for q in dataset.questions if q.category == "Governing Law")
    assert (audit.answerable, audit.gold_spans) == (False, [])
    assert law.answerable is True
    assert law.gold_spans[0].text == SPAN["text"]


def test_each_contract_has_a_label_a_split_and_a_checksum_of_its_text(tmp_path: Path) -> None:
    dataset = build_dataset(make_subset(tmp_path), tmp_path, dev_count=2)

    assert sum(1 for c in dataset.contracts if c.split == "dev") == 2
    contract = dataset.contracts[0]
    assert contract.access_label in ACCESS_LABELS
    assert contract.allowed_roles == {contract.access_label, "admin"}
    assert len(contract.sha256) == 64


def test_every_question_belongs_to_the_split_of_its_contract(tmp_path: Path) -> None:
    dataset = build_dataset(make_subset(tmp_path), tmp_path, dev_count=2)

    for question in dataset.questions:
        assert question.split == dataset.contract(question.contract_id).split


def test_a_dataset_survives_a_round_trip_through_the_checked_in_files(tmp_path: Path) -> None:
    dataset = build_dataset(make_subset(tmp_path), tmp_path, dev_count=2)
    output = tmp_path / "datasets"

    write_dataset(dataset, output)
    loaded = load_dataset(output)

    assert loaded.contracts == dataset.contracts
    assert loaded.source == dataset.source
    assert sorted(loaded.questions, key=lambda x: x.id) == sorted(
        dataset.questions, key=lambda x: x.id
    )
    assert (
        json.loads((output / "contracts.json").read_text(encoding="utf-8"))["source"]["license"]
        == "CC BY 4.0"
    )
    assert (output / "dev.jsonl").read_text(encoding="utf-8").endswith("\n")


def test_loading_a_missing_dataset_explains_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="build-dataset"):
        load_dataset(tmp_path / "nothing")


def test_there_is_a_template_for_each_of_the_fourteen_categories() -> None:
    assert len(QUESTION_TEMPLATES) == 14
    assert all("{title}" in template for template in QUESTION_TEMPLATES.values())
