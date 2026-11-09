from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from grounded_rag.errors import EmptyDocumentError, UnsupportedDocumentError

BYTE_ORDER_MARK = chr(0xFEFF)
TEXT_SUFFIXES = frozenset({".txt", ".text", ".md"})
PDF_SUFFIXES = frozenset({".pdf"})


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    text: str
    title: str
    source: str


def normalize_text(raw: str) -> str:
    """Make line endings uniform and drop characters that break storage or offsets.

    Nothing else is touched, so offsets in the stored text stay predictable for callers
    that hold the original text.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\x00", "").replace(BYTE_ORDER_MARK, "")


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _extract_pdf(data: bytes, name: str) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnsupportedDocumentError(f"{name} is password protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PyPdfError as error:
        raise UnsupportedDocumentError(f"{name} is not a readable PDF: {error}") from error
    return "\n\n".join(pages)


def load_document(path: Path, *, title: str | None = None) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        raw = _decode(path.read_bytes())
    elif suffix in PDF_SUFFIXES:
        raw = _extract_pdf(path.read_bytes(), path.name)
    else:
        raise UnsupportedDocumentError(
            f"Unsupported file type '{suffix or path.name}'. Supported: .pdf, .txt, .md."
        )
    text = normalize_text(raw)
    if not text.strip():
        raise EmptyDocumentError(
            f"{path.name} has no extractable text. Scanned PDFs need OCR, which is out of scope."
        )
    return LoadedDocument(text=text, title=title or path.stem, source=path.as_posix())
