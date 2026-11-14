import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_cuad.py"


@pytest.fixture(scope="module")
def fetch() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_cuad", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entry(title: str, context: str, qas: list[dict]) -> dict:
    return {"title": title, "paragraphs": [{"context": context, "qas": qas}]}


def qa(title: str, category: str, answers: list[dict], impossible: bool = False) -> dict:
    return {
        "id": f"{title}__{category}",
        "question": f'Highlight the parts related to "{category}"',
        "answers": answers,
        "is_impossible": impossible,
    }


def make_zip(path: Path, data: list[dict]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("CUAD_v1/CUAD_v1.json", json.dumps({"version": "x", "data": data}))


def test_answer_offsets_are_relocated_after_line_ending_normalisation(fetch: ModuleType) -> None:
    context = "Intro line.\r\nThis Agreement is governed by Delaware law.\r\nEnd."
    start = context.index("This Agreement")
    answer = {"text": "This Agreement is governed by Delaware law.", "answer_start": start}

    span = fetch.relocate_answer(context, fetch.normalize(context), answer)

    normalized = fetch.normalize(context)
    assert normalized[span["start"] : span["end"]] == answer["text"]
    assert span["start"] == start - 1


def test_answer_with_wrong_offset_is_found_by_text(fetch: ModuleType) -> None:
    context = "Alpha. Governing law is Delaware. Omega."
    answer = {"text": "Governing law is Delaware.", "answer_start": 0}

    span = fetch.relocate_answer(context, context, answer)

    assert context[span["start"] : span["end"]] == answer["text"]


def test_answer_missing_from_the_text_is_dropped(fetch: ModuleType) -> None:
    assert fetch.relocate_answer("abc", "abc", {"text": "zzz", "answer_start": 0}) is None


def test_selection_is_repeatable_and_respects_length_limits(fetch: ModuleType) -> None:
    data = [entry(f"C{n}", "x" * (1000 * n), []) for n in range(1, 11)]

    first = fetch.select_contracts(data, 3, 7, 2000, 8000)
    second = fetch.select_contracts(data, 3, 7, 2000, 8000)

    assert [e["title"] for e in first] == [e["title"] for e in second]
    assert all(2000 <= len(e["paragraphs"][0]["context"]) <= 8000 for e in first)


def test_end_to_end_extraction_writes_contracts_and_annotations(
    fetch: ModuleType, tmp_path: Path
) -> None:
    context = (
        "Preamble text.\nThis Agreement is governed by the laws of Delaware.\n" + "Filler. " * 50
    )
    title = "ACME_2020-SERVICES AGREEMENT"
    start = context.index("This Agreement")
    sentence = "This Agreement is governed by the laws of Delaware."
    qas = [
        qa(title, "Governing Law", [{"text": sentence, "answer_start": start}]),
        qa(title, "Audit Rights", [], impossible=True),
        qa(title, "Parties", [{"text": "ACME", "answer_start": 0}]),
    ]
    archive = tmp_path / "cuad.zip"
    make_zip(archive, [entry(title, context, qas)])
    output = tmp_path / "out"

    code = fetch.main(
        ["--zip", str(archive), "--output", str(output), "--contracts", "1", "--min-chars", "10"]
    )

    subset = json.loads((output / "subset.json").read_text(encoding="utf-8"))
    contract = subset["contracts"][0]
    assert code == 0
    assert subset["source"]["license"] == "CC BY 4.0"
    assert [(q["category"], q["answerable"]) for q in contract["questions"]] == [
        ("Governing Law", True),
        ("Audit Rights", False),
    ]
    text = (output / contract["file"]).read_text(encoding="utf-8")
    span = contract["questions"][0]["answers"][0]
    assert text[span["start"] : span["end"]] == sentence


def test_a_present_clause_whose_span_cannot_be_located_is_left_out(
    fetch: ModuleType, tmp_path: Path
) -> None:
    context = "Some contract text without the quoted words. " * 5
    title = "ACME"
    qas = [
        qa(title, "Governing Law", [{"text": "words that are not in the text", "answer_start": 0}])
    ]
    archive = tmp_path / "cuad.zip"
    make_zip(archive, [entry(title, context, qas)])

    fetch.main(
        [
            "--zip",
            str(archive),
            "--output",
            str(tmp_path / "out"),
            "--contracts",
            "1",
            "--min-chars",
            "10",
        ]
    )

    subset = json.loads((tmp_path / "out" / "subset.json").read_text(encoding="utf-8"))
    assert subset["contracts"][0]["questions"] == []
    assert subset["dropped_answers"] == 1
