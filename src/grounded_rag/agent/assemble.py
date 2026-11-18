from collections.abc import Mapping, Sequence

from grounded_rag.agent.schemas import GeneratedClaim
from grounded_rag.agent.verify import locate_in_text
from grounded_rag.domain.answer import Answer, Citation
from grounded_rag.domain.documents import RetrievedChunk


def build_answer(claims: Sequence[GeneratedClaim], context: Mapping[str, RetrievedChunk]) -> Answer:
    """Turn verified claims into answer text with numbered citations.

    Every citation quote is taken from the stored chunk, not from the model, and carries
    its offsets in the document text. Claims must already have passed verification.
    """
    citations: list[Citation] = []
    numbers: dict[tuple[str, int, int], int] = {}
    sentences: list[str] = []
    for claim in claims:
        markers: list[int] = []
        for cited in claim.citations:
            item = context[cited.passage_id]
            located = locate_in_text(cited.quote, item.chunk.text)
            if located is None:
                raise ValueError(f"Unverified quote reached answer assembly: {cited.passage_id}")
            start, end = located
            key = (item.chunk.id, start, end)
            if key not in numbers:
                numbers[key] = len(citations) + 1
                citations.append(
                    Citation(
                        index=numbers[key],
                        chunk_id=item.chunk.id,
                        doc_id=item.chunk.doc_id,
                        document_title=item.document_title,
                        section=item.chunk.section,
                        quote=item.chunk.text[start:end],
                        start_char=item.chunk.start_char + start,
                        end_char=item.chunk.start_char + end,
                    )
                )
            if numbers[key] not in markers:
                markers.append(numbers[key])
        sentences.append(f"{claim.text.strip()} {''.join(f'[{n}]' for n in markers)}")
    return Answer(
        text=" ".join(sentences), abstained=False, abstain_reason=None, citations=tuple(citations)
    )
