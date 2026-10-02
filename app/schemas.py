"""Pydantic request and response schemas."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

NonEmptyText = Annotated[str, Field(min_length=1)]


class DecideRequest(BaseModel):
    """A decision context and two to twenty labelled choices."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "state": "Production API is returning HTTP 500 errors.",
        "question": "What should the engineering team do?",
        "options": {"investigate": "Investigate logs.", "rollback": "Rollback the deployment."},
    }})

    state: NonEmptyText
    question: NonEmptyText
    options: dict[str, NonEmptyText] = Field(min_length=2, max_length=20)

    @field_validator("state", "question", mode="before")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) < 2 or len(value) > 20:
            raise ValueError("options must contain between 2 and 20 entries")
        normalized: dict[str, str] = {}
        for label, description in value.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("option labels must not be empty")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("option descriptions must not be empty")
            clean_label = label.strip()
            if clean_label in normalized:
                raise ValueError("option labels must be unique")
            normalized[clean_label] = description.strip()
        return normalized


class DecideResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "model": "pai-1-0.5b", "choice": "rollback", "confidence": 0.9635,
        "probabilities": {"rollback": 0.9635, "investigate": 0.0365}, "inference_ms": 42.7,
    }})
    model: str
    choice: str
    confidence: float
    probabilities: dict[str, float]
    inference_ms: float


class QuestionSpec(BaseModel):
    """One typed question evaluated against the shared state."""

    type: str
    instructions: NonEmptyText
    criteria: dict[str, NonEmptyText] | list[NonEmptyText] | None = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question type must not be empty")
        normalized = value.strip().lower()
        if normalized not in {"noul", "bool", "boolean", "choice", "score"}:
            raise ValueError("question type must be noul, bool, choice, or score")
        return normalized

    @field_validator("instructions", mode="before")
    @classmethod
    def normalize_instructions(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("instructions must not be empty")
        return value.strip()

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, value: dict[str, str] | list[str] | None) -> dict[str, str] | list[str] | None:
        if value is not None and len(value) < 2:
            raise ValueError("criteria must contain at least two entries")
        if isinstance(value, dict) and len(value) > 20:
            raise ValueError("criteria must contain at most twenty entries")
        if isinstance(value, list) and len(value) > 20:
            raise ValueError("criteria must contain at most twenty entries")
        return value


class EvaluateRequest(BaseModel):
    """A shared state with multiple typed questions."""

    model: str = "clef"
    state: NonEmptyText
    questions: dict[str, QuestionSpec] = Field(min_length=1, max_length=20)

    @field_validator("state", mode="before")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("state must not be empty")
        return value.strip()


class QuestionResult(BaseModel):
    type: str
    choice: str
    confidence: float
    probabilities: dict[str, float]


class EvaluateResponse(BaseModel):
    model: str
    results: dict[str, QuestionResult]
    inference_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str