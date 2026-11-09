import itertools
import random
import re

import pytest

from grounded_rag.ingestion.chunking import ChunkingConfig, chunk_text

CONFIGS = [
    ChunkingConfig(target_chars=120, max_chars=200, min_chars=30, overlap_chars=20),
    ChunkingConfig(target_chars=400, max_chars=600, min_chars=100, overlap_chars=0),
    ChunkingConfig(),
]
WORDS = [
    "agreement",
    "party",
    "shall",
    "term",
    "fee",
    "notice",
    "law",
    "renew",
    "audit",
    "liable",
    "assign",
    "Delaware",
]


def random_document(seed: int) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    for index in range(rng.randint(1, 25)):
        kind = rng.random()
        if kind < 0.15:
            parts.append(f"{index + 1}. {rng.choice(WORDS).title()} {rng.choice(WORDS).title()}")
        elif kind < 0.2:
            parts.append(rng.choice(WORDS).upper() + " " + rng.choice(WORDS).upper())
        else:
            sentence = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 120))) + "."
            parts.append(sentence)
    separator = rng.choice(["\n\n", "\n\n\n", "\n \n", "\n"])
    return separator.join(parts)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("config", CONFIGS, ids=["small", "no-overlap", "default"])
def test_spans_are_ordered_trimmed_and_bounded(seed: int, config: ChunkingConfig) -> None:
    text = random_document(seed)

    spans = chunk_text(text, config)

    for earlier, later in itertools.pairwise(spans):
        assert later.start > earlier.start
        assert later.end > earlier.end
    for span in spans:
        piece = text[span.start : span.end]
        assert piece
        assert piece == piece.strip()
        assert len(piece) <= config.max_chars


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("config", CONFIGS, ids=["small", "no-overlap", "default"])
def test_no_non_whitespace_character_is_lost(seed: int, config: ChunkingConfig) -> None:
    text = random_document(seed)
    covered = [False] * len(text)
    for span in chunk_text(text, config):
        for position in range(span.start, span.end):
            covered[position] = True

    missing = [i for i, char in enumerate(text) if not char.isspace() and not covered[i]]

    assert missing == []


@pytest.mark.parametrize("seed", range(40))
def test_without_overlap_oversized_paragraphs_do_not_repeat_text(seed: int) -> None:
    config = ChunkingConfig(target_chars=150, max_chars=200, min_chars=0, overlap_chars=0)
    text = random_document(seed)

    spans = chunk_text(text, config)

    for earlier, later in itertools.pairwise(spans):
        assert later.start >= earlier.end


def test_repeated_paragraphs_become_separate_chunks_with_their_own_offsets() -> None:
    paragraph = "The Supplier shall maintain insurance coverage at all times."
    text = f"{paragraph}\n\n{paragraph}"
    config = ChunkingConfig(target_chars=70, max_chars=100, min_chars=0, overlap_chars=0)

    spans = chunk_text(text, config)

    assert [text[s.start : s.end] for s in spans] == [paragraph, paragraph]
    assert spans[0].start != spans[1].start


def test_crlf_free_text_with_windows_style_blank_lines_is_handled() -> None:
    text = "First clause.\n \t \nSecond clause."
    config = ChunkingConfig(target_chars=15, max_chars=30, min_chars=0, overlap_chars=0)

    assert [text[s.start : s.end] for s in chunk_text(text, config)] == [
        "First clause.",
        "Second clause.",
    ]


def test_very_long_unbroken_contract_text_stays_within_limits() -> None:
    sentence = "The Licensee shall not sublicense the Software without prior consent. "
    text = sentence * 2000

    spans = chunk_text(text)

    assert len(spans) > 50
    assert all(span.end - span.start <= 1800 for span in spans)
    assert re.sub(r"\s+", " ", text[spans[0].start : spans[0].end]).startswith("The Licensee")
