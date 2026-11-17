from pydantic import BaseModel, Field


class GradeVerdict(BaseModel):
    """The grader's judgement of whether the retrieved passages can answer the question."""

    sufficient: bool
    reason: str = Field(description="One short sentence explaining the verdict.")
    missing: str = Field(description="What information is absent, or an empty string.")


class GeneratedCitation(BaseModel):
    passage_id: str = Field(description="The id attribute of the passage the quote comes from.")
    quote: str = Field(description="Text copied exactly from that passage.")


class GeneratedClaim(BaseModel):
    text: str = Field(description="One statement that answers part of the question.")
    citations: list[GeneratedCitation]


class GeneratedAnswer(BaseModel):
    answerable: bool = Field(description="False when the passages do not contain the answer.")
    claims: list[GeneratedClaim]
