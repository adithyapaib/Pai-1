"""Unit tests for pydantic schemas: normalization and boundary rules."""

import pytest
from pydantic import ValidationError

from app.schemas import DecideRequest, EvaluateRequest, QuestionSpec


def test_decide_request_strips_text():
    req = DecideRequest(
        state="  hello  ",
        question=" what? ",
        options={" a ": "  First  ", "b": "Second"},
    )
    assert req.state == "hello"
    assert req.question == "what?"
    assert req.options == {"a": "First", "b": "Second"}


def test_decide_request_rejects_duplicate_labels_after_strip():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", question="q", options={"a": "x", " a ": "y", "b": "z"})


@pytest.mark.parametrize("n", [2, 20])
def test_decide_request_option_count_boundaries_ok(n):
    options = {f"k{i}": f"v{i}" for i in range(n)}
    req = DecideRequest(state="s", question="q", options=options)
    assert len(req.options) == n


@pytest.mark.parametrize("n", [0, 1, 21])
def test_decide_request_option_count_boundaries_rejected(n):
    options = {f"k{i}": f"v{i}" for i in range(n)}
    with pytest.raises(ValidationError):
        DecideRequest(state="s", question="q", options=options)


def test_question_spec_normalizes_type_case_and_whitespace():
    assert QuestionSpec(type=" NOUL ", instructions="x").type == "noul"
    assert QuestionSpec(type="BOOL", instructions="x").type == "bool"
    assert QuestionSpec(type="Boolean", instructions="x").type == "boolean"
    assert QuestionSpec(type="Choice", instructions="x").type == "choice"
    assert QuestionSpec(type=" SCORE ", instructions="x").type == "score"


def test_question_spec_rejects_unknown_type():
    with pytest.raises(ValidationError):
        QuestionSpec(type="essay", instructions="x")


def test_question_spec_strips_instructions():
    assert QuestionSpec(type="noul", instructions="  hi  ").instructions == "hi"
    with pytest.raises(ValidationError):
        QuestionSpec(type="noul", instructions="   ")


@pytest.mark.parametrize(
    "criteria",
    [
        {"a": "A"},  # dict with 1 entry
        ["only"],  # list with 1 entry
        {f"k{i}": f"v{i}" for i in range(21)},  # dict too long
        [f"v{i}" for i in range(21)],  # list too long
    ],
)
def test_question_spec_rejects_bad_criteria_lengths(criteria):
    with pytest.raises(ValidationError):
        QuestionSpec(type="choice", instructions="x", criteria=criteria)


def test_evaluate_request_limits_question_count():
    ok = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(20)}
    assert len(EvaluateRequest(state="s", questions=ok).questions) == 20
    with pytest.raises(ValidationError):
        EvaluateRequest(state="s", questions={})
    too_many = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(21)}
    with pytest.raises(ValidationError):
        EvaluateRequest(state="s", questions=too_many)


def test_evaluate_request_strips_state():
    req = EvaluateRequest(state="  hello  ", questions={"q": {"type": "noul", "instructions": "x"}})
    assert req.state == "hello"
    with pytest.raises(ValidationError):
        EvaluateRequest(state="   ", questions={"q": {"type": "noul", "instructions": "x"}})
