from pathlib import Path

import pytest

from grounded_rag.errors import EmptyDocumentError, UnsupportedDocumentError
from grounded_rag.ingestion.loaders import load_document, normalize_text
from tests.helpers import make_pdf


def test_windows_and_old_mac_line_endings_become_newlines() -> None:
    assert normalize_text("a\r\nb\rc\n") == "a\nb\nc\n"


def test_byte_order_mark_and_nul_characters_are_removed() -> None:
    assert normalize_text(chr(0xFEFF) + "hello\x00 world") == "hello world"


def test_text_file_is_loaded_with_stem_as_default_title(tmp_path: Path) -> None:
    path = tmp_path / "master services agreement.txt"
    path.write_text("Governing law: Delaware.\r\nSecond line.", encoding="utf-8")

    loaded = load_document(path)

    assert loaded.text == "Governing law: Delaware.\nSecond line."
    assert loaded.title == "master services agreement"
    assert loaded.source.endswith("master services agreement.txt")


def test_explicit_title_overrides_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("content", encoding="utf-8")
    assert load_document(path, title="Supply Agreement").title == "Supply Agreement"


def test_bytes_that_are_not_utf8_fall_back_to_windows_1252(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes("Café terms".encode("cp1252"))
    assert load_document(path).text == "Café terms"


def test_empty_text_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_unknown_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path)


def test_pdf_pages_are_joined_with_a_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "contract.pdf"
    path.write_bytes(make_pdf([["Page one clause."], ["Page two clause."]]))

    loaded = load_document(path)

    assert "Page one clause." in loaded.text
    assert loaded.text.index("Page one clause.") < loaded.text.index("Page two clause.")
    assert "\n\n" in loaded.text


def test_pdf_with_multiple_lines_keeps_line_breaks(tmp_path: Path) -> None:
    path = tmp_path / "lines.pdf"
    path.write_bytes(make_pdf([["First line", "Second line"]]))
    assert load_document(path).text.split() == ["First", "line", "Second", "line"]


def test_pdf_without_text_is_reported_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf([[]]))
    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_corrupt_pdf_is_reported_as_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a real pdf")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path)
