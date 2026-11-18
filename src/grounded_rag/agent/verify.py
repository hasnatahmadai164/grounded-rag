import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from grounded_rag.agent.schemas import GeneratedCitation, GeneratedClaim
from grounded_rag.domain.documents import RetrievedChunk

MIN_QUOTE_CHARS = 8

# Typographic variants that differ between how a model writes a quote and how a PDF stores it:
# curly single and double quotes, en and em dashes, and the no-break space.
_FOLD = str.maketrans(
    {
        0x2018: "'",
        0x2019: "'",
        0x201C: '"',
        0x201D: '"',
        0x2013: "-",
        0x2014: "-",
        0x00A0: " ",
    }
)


class CitationProblem(StrEnum):
    UNKNOWN_PASSAGE = "unknown_passage"
    QUOTE_TOO_SHORT = "quote_too_short"
    QUOTE_NOT_FOUND = "quote_not_found"


@dataclass(frozen=True, slots=True)
class CitationCheck:
    citation: GeneratedCitation
    problem: CitationProblem | None

    @property
    def ok(self) -> bool:
        return self.problem is None


@dataclass(frozen=True, slots=True)
class ClaimCheck:
    claim: GeneratedClaim
    checks: tuple[CitationCheck, ...]

    @property
    def valid_citations(self) -> list[GeneratedCitation]:
        return [check.citation for check in self.checks if check.ok]

    @property
    def supported(self) -> bool:
        """A claim stands only if at least one of its citations is genuine."""
        return bool(self.claim.text.strip()) and bool(self.valid_citations)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    claims: tuple[ClaimCheck, ...]

    @property
    def checks(self) -> list[CitationCheck]:
        return [check for claim in self.claims for check in claim.checks]

    @property
    def failures(self) -> list[CitationCheck]:
        return [check for check in self.checks if not check.ok]

    @property
    def all_verified(self) -> bool:
        return bool(self.claims) and all(c.supported for c in self.claims) and not self.failures

    @property
    def verified_rate(self) -> float:
        checks = self.checks
        return sum(1 for check in checks if check.ok) / len(checks) if checks else 0.0

    def supported_claims(self) -> list[GeneratedClaim]:
        """Claims that survive, each reduced to its genuine citations."""
        return [
            GeneratedClaim(text=claim.claim.text.strip(), citations=claim.valid_citations)
            for claim in self.claims
            if claim.supported
        ]


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, fold typography and collapse whitespace, remembering where each kept
    character came from so a match can be traced back to the original text."""
    kept: list[str] = []
    origin: list[int] = []
    for position, character in enumerate(text):
        folded = unicodedata.normalize("NFKC", character).translate(_FOLD).casefold()
        for piece in folded:
            if piece.isspace():
                if kept and kept[-1] != " ":
                    kept.append(" ")
                    origin.append(position)
            else:
                kept.append(piece)
                origin.append(position)
    if kept and kept[-1] == " ":
        kept.pop()
        origin.pop()
    return "".join(kept), origin


def normalize_quote(text: str) -> str:
    return normalize_with_map(text)[0]


def locate_in_text(quote: str, text: str) -> tuple[int, int] | None:
    """Offsets of `quote` inside `text`, ignoring case, spacing and typographic variants."""
    needle = normalize_quote(quote)
    if not needle:
        return None
    haystack, origin = normalize_with_map(text)
    start = haystack.find(needle)
    if start < 0:
        return None
    return origin[start], origin[start + len(needle) - 1] + 1


def check_citation(
    citation: GeneratedCitation, context: Mapping[str, RetrievedChunk]
) -> CitationCheck:
    chunk = context.get(citation.passage_id)
    if chunk is None:
        return CitationCheck(citation, CitationProblem.UNKNOWN_PASSAGE)
    if len(normalize_quote(citation.quote)) < MIN_QUOTE_CHARS:
        return CitationCheck(citation, CitationProblem.QUOTE_TOO_SHORT)
    if locate_in_text(citation.quote, chunk.chunk.text) is None:
        return CitationCheck(citation, CitationProblem.QUOTE_NOT_FOUND)
    return CitationCheck(citation, None)


def verify_claims(
    claims: Sequence[GeneratedClaim], context: Mapping[str, RetrievedChunk]
) -> VerificationReport:
    """Check every quote against the exact chunk it names.

    Only chunks that were actually shown to the model are in `context`, so an invented or
    unauthorised chunk id fails as `unknown_chunk`.
    """
    return VerificationReport(
        tuple(
            ClaimCheck(claim, tuple(check_citation(c, context) for c in claim.citations))
            for claim in claims
        )
    )
