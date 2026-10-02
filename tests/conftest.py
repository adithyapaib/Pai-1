"""Shared fixtures for mocked API tests.

These tests never load the real Qwen backbone. They inject a lightweight
FakeModel into ``app.state.model`` and use ``TestClient`` *without* a
lifespan context, so ``lifespan()`` (which would construct ``Pai1``)
is never executed.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main


class FakeModel:
    """Minimal stand-in matching the real Pai1.predict contract."""

    device = "cpu"
    model_name = "pai-1-0.5b"
    hidden_size = 896
    parameter_count = 1414657

    def __init__(self, backbone: str = "test-backbone"):
        # Instance attribute so tests can customise per-fixture if needed.
        self.settings = type("Settings", (), {"backbone": backbone})()
        self.calls: list[tuple[str, str, dict]] = []

    def predict(self, state, question, options):
        self.calls.append((state, question, dict(options)))
        labels = list(options.keys())
        assert len(labels) >= 2, "FakeModel expects at least two options"
        first_prob = 0.6
        rest = (1.0 - first_prob) / (len(labels) - 1)
        probabilities = {label: (first_prob if i == 0 else rest) for i, label in enumerate(labels)}
        return {"choice": labels[0], "confidence": first_prob, "probabilities": probabilities}


class FailingModel(FakeModel):
    def predict(self, state, question, options):
        raise RuntimeError("boom")


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def client(fake_model: FakeModel):
    """TestClient with FakeModel installed, lifespan disabled."""
    previous = getattr(main.app.state, "model", None)
    main.app.state.model = fake_model
    test_client = TestClient(main.app)
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                delattr(main.app.state, "model")
            except (AttributeError, KeyError):
                pass
        else:
            main.app.state.model = previous


@pytest.fixture
def client_no_raise(fake_model: FakeModel):
    """Same as client but converts handler 500s to responses instead of raising."""
    previous = getattr(main.app.state, "model", None)
    main.app.state.model = fake_model
    test_client = TestClient(main.app, raise_server_exceptions=False)
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                delattr(main.app.state, "model")
            except (AttributeError, KeyError):
                pass
        else:
            main.app.state.model = previous


@pytest.fixture
def client_no_model():
    """TestClient with no model installed (degraded health path)."""
    previous = getattr(main.app.state, "model", None)
    try:
        delattr(main.app.state, "model")
    except (AttributeError, KeyError):
        pass
    test_client = TestClient(main.app)
    try:
        yield test_client
    finally:
        if previous is not None:
            main.app.state.model = previous


@pytest.fixture
def client_failing_model():
    previous = getattr(main.app.state, "model", None)
    main.app.state.model = FailingModel()
    test_client = TestClient(main.app, raise_server_exceptions=False)
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                delattr(main.app.state, "model")
            except (AttributeError, KeyError):
                pass
        else:
            main.app.state.model = previous


@pytest.fixture
def decide_payload():
    return {
        "state": "Production API is returning HTTP 500 errors.",
        "question": "What should the engineering team do?",
        "options": {
            "investigate": "Investigate logs and identify the root cause.",
            "rollback": "Rollback the latest deployment.",
        },
    }


@pytest.fixture
def evaluate_payload():
    return {
        "model": "clef",
        "state": "Checkout has been failing for every customer for the last hour.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Payments, invoices, and refunds",
                    "technical": "Outages, errors, and configuration",
                    "sales": "Plans and upgrades",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is the customer impact?",
                "criteria": ["No impact", "Minor", "Major", "Critical"],
            },
        },
    }
