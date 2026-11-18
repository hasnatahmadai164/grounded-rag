import pytest

from grounded_rag.agent.schemas import GeneratedCitation, GeneratedClaim
from grounded_rag.agent.verify import (
    CitationProblem,
    locate_in_text,
    normalize_quote,
    verify_claims,
)
from grounded_rag.domain.documents import Chunk, RetrievedChunk

RSQUO, LDQUO, RDQUO = chr(0x2019), chr(0x201C), chr(0x201D)
TEXT = (
    "13.1 This Agreement shall be governed by, and construed in accordance with, the laws of\n"
    "the State of  Delaware, without regard to conflict-of-law principles. "
    f"Either party{RSQUO}s {LDQUO}Notice{RDQUO} must be in writing."
)


def chunk(chunk_id: str = "c1", text: str = TEXT, start: int = 1000) -> RetrievedChunk:
    return RetrievedChunk(
        Chunk(chunk_id, "doc", 0, text, start, start + len(text), None, "Doc"),
        "Contract",
        frozenset({"legal"}),
    )


CONTEXT = {
    "c1": chunk("c1"),
    "c2": chunk("c2", "Payment is due within thirty (30) days of invoice."),
}


def claim(*quotes: tuple[str, str], text: str = "A statement.") -> GeneratedClaim:
    return GeneratedClaim(
        text=text, citations=[GeneratedCitation(passage_id=c, quote=q) for c, q in quotes]
    )


def only_check(*quotes: tuple[str, str]):
    return verify_claims([claim(*quotes)], CONTEXT).checks[0]


def test_an_exact_quote_is_verified() -> None:
    assert only_check(("c1", "the laws of\nthe State of  Delaware")).ok


@pytest.mark.parametrize(
    "quote",
    [
        "THE STATE OF DELAWARE",
        "the   state\tof\ndelaware",
        "  the State of Delaware, without regard  ",
        'Either party\'s "Notice" must be in writing',
    ],
)
def test_case_spacing_and_typography_do_not_matter(quote: str) -> None:
    assert only_check(("c1", quote)).ok


def test_a_quote_that_is_not_in_the_chunk_fails() -> None:
    check = only_check(("c1", "the laws of the State of California"))
    assert check.problem is CitationProblem.QUOTE_NOT_FOUND


def test_a_real_quote_cited_to_the_wrong_chunk_fails() -> None:
    check = only_check(("c2", "the State of Delaware"))
    assert check.problem is CitationProblem.QUOTE_NOT_FOUND


def test_a_passage_id_that_was_never_shown_fails() -> None:
    assert (
        only_check(("secret-chunk", "anything long enough")).problem
        is CitationProblem.UNKNOWN_PASSAGE
    )


@pytest.mark.parametrize("quote", ["", "   ", "the", "Delaware"[:7]])
def test_trivially_short_quotes_are_rejected(quote: str) -> None:
    assert only_check(("c1", quote)).problem is CitationProblem.QUOTE_TOO_SHORT


def test_a_quote_stitched_from_two_places_fails() -> None:
    assert not only_check(("c1", "This Agreement shall be governed by the State of Delaware")).ok


def test_offsets_point_at_the_quote_in_the_original_text() -> None:
    quote = "the laws of the state of delaware"
    start, end = locate_in_text(quote, TEXT)  # type: ignore[misc]
    assert TEXT[start:end] == "the laws of\nthe State of  Delaware"


def test_offsets_survive_curly_quotes() -> None:
    start, end = locate_in_text('Either party\'s "Notice"', TEXT)  # type: ignore[misc]
    assert TEXT[start:end] == f"Either party{RSQUO}s {LDQUO}Notice{RDQUO}"


def test_a_missing_quote_has_no_offsets() -> None:
    assert locate_in_text("nothing like this", TEXT) is None
    assert locate_in_text("", TEXT) is None


def test_normalisation_collapses_whitespace_and_lowercases() -> None:
    assert normalize_quote("  Hello \n\t WORLD ") == "hello world"


def test_a_claim_with_one_good_and_one_bad_citation_keeps_only_the_good_one() -> None:
    report = verify_claims(
        [claim(("c1", "the State of Delaware"), ("c1", "made up quotation text"))], CONTEXT
    )

    assert not report.all_verified
    survivors = report.supported_claims()
    assert [len(c.citations) for c in survivors] == [1]
    assert survivors[0].citations[0].quote == "the State of Delaware"


def test_a_claim_without_a_genuine_citation_is_dropped() -> None:
    report = verify_claims(
        [
            claim(("c1", "invented words here"), text="Bad."),
            claim(("c2", "within thirty (30) days"), text="Good."),
        ],
        CONTEXT,
    )

    assert [c.text for c in report.supported_claims()] == ["Good."]


def test_a_claim_with_no_citations_is_unsupported() -> None:
    assert (
        verify_claims([GeneratedClaim(text="Unsourced.", citations=[])], CONTEXT).supported_claims()
        == []
    )


def test_a_fully_verified_answer_reports_so() -> None:
    report = verify_claims([claim(("c1", "the State of Delaware"))], CONTEXT)
    assert report.all_verified
    assert report.verified_rate == 1.0


def test_verified_rate_counts_every_citation() -> None:
    report = verify_claims(
        [
            claim(
                ("c1", "the State of Delaware"),
                ("c1", "invented words here"),
                ("zz", "unknown chunk quote"),
            )
        ],
        CONTEXT,
    )
    assert report.verified_rate == pytest.approx(1 / 3)
    assert len(report.failures) == 2


def test_an_empty_answer_has_nothing_verified() -> None:
    report = verify_claims([], CONTEXT)
    assert not report.all_verified
    assert report.verified_rate == 0.0
