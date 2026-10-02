"""TypeSafe-compatible classification endpoint (POST /v1/systemone).

This module implements the wire protocol expected by ``langchain-typesafe``:

* request: ``{"state": str|object|array, "model": str,
  "questions": {id: Noul|Choice|Score}}``
* response: ``{"model": str, "answers": {id: ...}, "usage": {...}}``
  plus ``x-typesafe-request-id`` header.

It reuses the local :class:`~app.model.Pai1` decision head, so
``ModelRouterMiddleware`` / ``AutoModeMiddleware`` / ``TypeSafeClassifier``
can point at Pai-1 via ``TYPESAFE_BASE_URL=http://127.0.0.1:8000``.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

QuestionContent = Annotated[Any, Field(description="Text or structured JSON content")]
StateValue = Any


def to_text(value: Any) -> str:
    """Render instructions/criteria values as text for the backbone."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def state_to_text(state: Any) -> str:
    """Render TypeSafe state (str, object, array, serialized messages) as text."""
    if isinstance(state, str):
        return state
    return to_text(state)


class NoulCriteriaModel(BaseModel):
    model_config = {"populate_by_name": True}
    true: Any = None
    false: Any = None


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: Any
    criteria: NoulCriteriaModel | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: Any
    criteria: dict[str, Any]


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: Any
    criteria: list[Any] = Field(min_length=2)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: Any
    model: str = "pai-1"
    questions: dict[str, Question] = Field(min_length=1, max_length=20)

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, value: Any) -> Any:
        if value is None or isinstance(value, (int, float, bool)):
            raise ValueError("state must be a string, object, array, or message sequence")
        if isinstance(value, str) and not value.strip():
            raise ValueError("state must not be empty")
        return value

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model must not be empty")
        return value.strip()


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[int, Any]
    probabilities: dict[int, float]
    confidence: float = Field(ge=0.0, le=1.0)


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class ClassifierResponse(BaseModel):
    model: str
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None


REQUEST_ID_HEADER = "x-typesafe-request-id"


def new_request_id() -> str:
    return uuid.uuid4().hex


def classify_questions(
    pai_model,
    payload: SystemOneRequest,
) -> tuple[dict[str, Any], int | None]:
    """Run all questions through Pai1.predict and build TypeSafe answers.

    Returns (answers, input_tokens). ``input_tokens`` is a best-effort count
    from the local tokenizer, or ``None`` when it cannot be determined.
    """
    state_text = state_to_text(payload.state)
    answers: dict[str, Any] = {}
    input_tokens: int | None = None

    try:
        tok = pai_model.tokenizer(state_text, truncation=False, add_special_tokens=False)
        ids = tok.get("input_ids", [])
        input_tokens = len(ids) if isinstance(ids, list) else None
    except Exception:
        input_tokens = None

    for qid, question in payload.questions.items():
        instructions_text = to_text(question.instructions)
        # Keep token usage roughly honest without re-tokenizing everything twice.
        if input_tokens is not None:
            try:
                extra = pai_model.tokenizer(instructions_text, truncation=False, add_special_tokens=False)
                extra_ids = extra.get("input_ids", [])
                if isinstance(extra_ids, list):
                    input_tokens += len(extra_ids)
            except Exception:
                pass

        if isinstance(question, (NoulQuestion,)) or getattr(question, "type", None) == "noul":
            criteria = getattr(question, "criteria", None)
            true_text = to_text(getattr(criteria, "true", None)) if criteria else ""
            false_text = to_text(getattr(criteria, "false", None)) if criteria else ""
            options = {
                "yes": true_text.strip() or "The answer to this question is yes.",
                "no": false_text.strip() or "The answer to this question is no.",
            }
            pred = pai_model.predict(state_text, instructions_text, options)
            p_yes = float(pred["probabilities"].get("yes", 0.0))
            answers[qid] = {"type": "noul", "noul": min(1.0, max(0.0, p_yes))}
        elif getattr(question, "type", None) == "choice":
            options = {label: to_text(desc) or label for label, desc in question.criteria.items()}
            if len(options) < 1:
                raise ValueError(f"question '{qid}' requires at least one criterion")
            pred = pai_model.predict(state_text, instructions_text, options)
            answers[qid] = {
                "type": "choice",
                "choice": pred["choice"],
                "probabilities": {k: float(v) for k, v in pred["probabilities"].items()},
                "confidence": float(pred["confidence"]),
            }
        else:  # score
            options = {str(i): to_text(c) or str(c) for i, c in enumerate(question.criteria)}
            pred = pai_model.predict(state_text, instructions_text, options)
            probs = {int(k): float(v) for k, v in pred["probabilities"].items()}
            expected = sum(level * p for level, p in probs.items())
            confidence = float(max(probs.values())) if probs else 0.0
            legend = {i: c for i, c in enumerate(question.criteria)}
            answers[qid] = {
                "type": "score",
                "score": float(expected),
                "legend": legend,
                "probabilities": probs,
                "confidence": min(1.0, max(0.0, confidence)),
            }

    return answers, input_tokens
