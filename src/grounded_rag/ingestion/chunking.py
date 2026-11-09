import re
from collections.abc import Iterator
from dataclasses import dataclass

from grounded_rag.errors import ConfigurationError

_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>\S.*)$")
_KEYWORD_HEADING = re.compile(
    r"^(?:ARTICLE|SECTION|SCHEDULE|EXHIBIT|APPENDIX|PART)\s+[0-9IVXLC]+\b[^.]{0,80}$"
)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z][^.!?]{2,80}$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;:])\s+|\n")
_WHITESPACE = re.compile(r"\s+")
_MAX_HEADING_CHARS = 90


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Size limits in characters. A chunk only exceeds `target_chars` when one unit is larger."""

    target_chars: int = 1200
    max_chars: int = 1800
    min_chars: int = 200
    overlap_chars: int = 150

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_chars < self.target_chars <= self.max_chars:
            raise ConfigurationError(
                "Chunking needs 0 <= overlap < target <= max, got "
                f"overlap={self.overlap_chars}, target={self.target_chars}, max={self.max_chars}."
            )
        if self.min_chars < 0 or self.min_chars > self.target_chars:
            raise ConfigurationError("Chunking min_chars must be between 0 and target_chars.")


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    start: int
    end: int
    section: str | None


@dataclass(frozen=True, slots=True)
class _Unit:
    """A paragraph, joined with the heading above it so a heading is never left alone."""

    start: int
    end: int
    section: str | None
    begins_section: bool


def detect_heading(line: str) -> str | None:
    """Return the heading title when a line looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None
    markdown = _MARKDOWN_HEADING.match(stripped)
    if markdown:
        return markdown.group("title").strip()
    if _KEYWORD_HEADING.match(stripped) or _NUMBERED_HEADING.match(stripped):
        return stripped
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) >= 3 and stripped.isupper() and not stripped.endswith((".", ",", ";")):
        return stripped
    return None


def _lines(text: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) of each line, excluding the newline."""
    position = 0
    for line in text.split("\n"):
        yield position, position + len(line)
        position += len(line) + 1


def _blocks(text: str) -> list[list[tuple[int, int]]]:
    """Group consecutive non-blank lines into blocks of trimmed line spans."""
    blocks: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for start, end in _lines(text):
        line = text[start:end]
        if line.strip():
            current.append((start + len(line) - len(line.lstrip()), start + len(line.rstrip())))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _units(text: str) -> Iterator[_Unit]:
    section: str | None = None
    pending_start: int | None = None
    for block in _blocks(text):
        first_start, first_end = block[0]
        heading = detect_heading(text[first_start:first_end])
        block_start, block_end = first_start, block[-1][1]
        if heading is not None and len(block) == 1:
            section = heading
            if pending_start is None:
                pending_start = block_start
            continue
        if heading is not None:
            section = heading
        begins_section = heading is not None or pending_start is not None
        start = pending_start if pending_start is not None else block_start
        pending_start = None
        yield _Unit(start, block_end, section, begins_section)
    if pending_start is not None:
        yield _Unit(pending_start, _last_non_blank_end(text), section, True)


def _last_non_blank_end(text: str) -> int:
    return len(text.rstrip())


def _find_cut(text: str, position: int, limit: int, floor: int) -> int:
    last_break = None
    for match in _SENTENCE_BREAK.finditer(text, floor, limit):
        last_break = match
    if last_break is not None:
        return last_break.start()
    for index in range(limit - 1, position, -1):
        if text[index].isspace():
            return index
    return limit


def _split_oversize(
    text: str, start: int, end: int, config: ChunkingConfig
) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    position = start
    while position < end:
        limit = min(position + config.target_chars, end)
        cut = (
            end
            if limit == end
            else _find_cut(text, position, limit, position + config.target_chars // 2)
        )
        piece_end = len(text[:cut].rstrip())
        pieces.append((position, max(piece_end, position + 1)))
        if cut >= end:
            break
        position = _next_position(text, position, cut, end, config.overlap_chars)
    return pieces


def _next_position(text: str, position: int, cut: int, end: int, overlap: int) -> int:
    candidate = cut
    if overlap:
        boundary = _WHITESPACE.search(text, cut - overlap)
        if boundary is not None and position < boundary.end() < cut:
            candidate = boundary.end()
    while candidate < end and text[candidate].isspace():
        candidate += 1
    return candidate


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[ChunkSpan]:
    """Split text into spans along paragraph and heading boundaries.

    Spans are trimmed, ordered, and never cross a heading unless the preceding span is
    still smaller than `min_chars`. Paragraphs larger than `max_chars` are cut at sentence
    boundaries with `overlap_chars` of shared context.
    """
    config = config or ChunkingConfig()
    spans: list[ChunkSpan] = []
    current_start: int | None = None
    current_end = 0
    current_section: str | None = None

    def flush() -> None:
        nonlocal current_start
        if current_start is not None:
            spans.append(ChunkSpan(current_start, current_end, current_section))
            current_start = None

    for unit in _units(text):
        if unit.end - unit.start > config.max_chars:
            flush()
            spans.extend(
                ChunkSpan(start, end, unit.section)
                for start, end in _split_oversize(text, unit.start, unit.end, config)
            )
            continue
        if current_start is not None:
            section_break = unit.begins_section and current_end - current_start >= config.min_chars
            too_large = unit.end - current_start > config.target_chars
            if section_break or too_large:
                flush()
        if current_start is None:
            current_start, current_end, current_section = unit.start, unit.end, unit.section
        else:
            current_end = unit.end
    flush()
    return spans


def make_context_header(title: str, section: str | None) -> str:
    """Document and section names prepended to a chunk's text for embedding and lexical search."""
    return f"{title} > {section}" if section else title
