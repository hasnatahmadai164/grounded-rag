import pytest

from grounded_rag.errors import ConfigurationError
from grounded_rag.ingestion.chunking import (
    ChunkingConfig,
    chunk_text,
    detect_heading,
    make_context_header,
)

SMALL = ChunkingConfig(target_chars=200, max_chars=300, min_chars=50, overlap_chars=30)


def texts(text: str, config: ChunkingConfig = SMALL) -> list[str]:
    return [text[span.start : span.end] for span in chunk_text(text, config)]


def test_short_document_becomes_a_single_chunk() -> None:
    text = "This Agreement is governed by the laws of Delaware."
    assert texts(text) == [text]


def test_empty_and_blank_documents_produce_no_chunks() -> None:
    assert chunk_text("", SMALL) == []
    assert chunk_text("  \n\n  \n", SMALL) == []


def test_paragraphs_are_packed_up_to_the_target_size() -> None:
    paragraph = "x" * 80
    text = "\n\n".join([paragraph] * 5)

    chunks = texts(text)

    assert chunks == [paragraph + "\n\n" + paragraph, paragraph + "\n\n" + paragraph, paragraph]


def test_a_new_numbered_heading_starts_a_new_chunk_once_the_previous_one_is_large_enough() -> None:
    first = "1. Definitions\n\n" + "Alpha " * 15
    second = "2. Payment Terms\n\n" + "Beta " * 15
    text = first + "\n\n" + second

    spans = chunk_text(text, SMALL)

    assert [span.section for span in spans] == ["1. Definitions", "2. Payment Terms"]
    assert text[spans[1].start : spans[1].end].startswith("2. Payment Terms")


def test_a_heading_after_a_tiny_chunk_stays_with_it() -> None:
    text = "Preamble.\n\n2. Payment Terms\n\n" + "Beta " * 10

    spans = chunk_text(text, SMALL)

    assert len(spans) == 1
    assert spans[0].section is None


def test_heading_is_never_separated_from_the_paragraph_below_it() -> None:
    body = "Gamma " * 30
    text = "x" * 150 + "\n\nARTICLE III\n\n" + body

    chunks = texts(
        text, ChunkingConfig(target_chars=200, max_chars=400, min_chars=20, overlap_chars=0)
    )

    assert any(chunk.startswith("ARTICLE III") and "Gamma" in chunk for chunk in chunks)


def test_markdown_headings_provide_section_names() -> None:
    text = "# Overview\n\nSome intro text.\n\n## Termination\n\n" + "Clause text. " * 20

    sections = [span.section for span in chunk_text(text, SMALL)]

    assert "Termination" in sections


def test_oversized_paragraph_is_cut_at_sentence_boundaries_with_overlap() -> None:
    sentences = [f"Sentence number {n} describes an obligation." for n in range(30)]
    text = " ".join(sentences)

    chunks = texts(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= SMALL.max_chars for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)
    opening_words = " ".join(chunks[1].split()[:2])
    assert opening_words in chunks[0][-SMALL.overlap_chars - 15 :]


def test_text_without_any_whitespace_is_hard_split() -> None:
    text = "a" * 1000

    chunks = texts(
        text, ChunkingConfig(target_chars=200, max_chars=300, min_chars=0, overlap_chars=0)
    )

    assert "".join(chunks) == text
    assert max(len(chunk) for chunk in chunks) <= 300


def test_chunks_are_trimmed_and_keep_exact_offsets() -> None:
    text = "\n\n   First paragraph here.   \n\n\n\n  Second paragraph here.  \n\n"

    for span in chunk_text(
        text, ChunkingConfig(target_chars=30, max_chars=60, min_chars=0, overlap_chars=0)
    ):
        piece = text[span.start : span.end]
        assert piece == piece.strip()
        assert piece


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1. Definitions", "1. Definitions"),
        ("3.2 Term and Termination", "3.2 Term and Termination"),
        ("ARTICLE IV - GENERAL", "ARTICLE IV - GENERAL"),
        ("MASTER SERVICES AGREEMENT", "MASTER SERVICES AGREEMENT"),
        ("## Governing Law", "Governing Law"),
        ("1. The Company shall pay the fees.", None),
        ("The parties agree as follows:", None),
        ("", None),
        ("x" * 200, None),
        ("12345", None),
    ],
)
def test_heading_detection(line: str, expected: str | None) -> None:
    assert detect_heading(line) == expected


def test_context_header_joins_title_and_section() -> None:
    assert make_context_header("Supply Agreement", "4. Term") == "Supply Agreement > 4. Term"
    assert make_context_header("Supply Agreement", None) == "Supply Agreement"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_chars": 100, "max_chars": 50},
        {"target_chars": 100, "overlap_chars": 100},
        {"target_chars": 100, "max_chars": 200, "min_chars": 150},
    ],
)
def test_inconsistent_size_limits_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ConfigurationError):
        ChunkingConfig(**kwargs)
