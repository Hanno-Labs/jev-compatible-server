"""The wire contract shared with Jev clients."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Content = str | dict[str, Any] | list[Any]
QuestionType = Literal["choice", "score", "noul"]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Content | None] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=255)


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Content
    false: Content


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


class DecisionRequest(BaseModel):
    """A Jev-compatible request: one state and any number of named questions."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    state: Any
    questions: dict[str, Question] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_question_names(self) -> DecisionRequest:
        for name in self.questions:
            if not name or len(name) > 128:
                raise ValueError("question names must be 1-128 characters")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    probabilities: dict[str, float]
    confidence: float
    legend: list[Content] | None = None


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class UnsupportedAnswer(BaseModel):
    """Per-question capability result for partially compatible models."""

    type: Literal["unsupported"]
    question_type: QuestionType
    reason: Literal["question_type_not_supported"] = "question_type_not_supported"
    supported_types: list[QuestionType]


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer | UnsupportedAnswer


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
