"""Tests for POST /v1/decide: happy path, contract, and validation."""

import pytest


def test_decide_happy_path_contract(client, decide_payload):
    response = client.post("/v1/decide", json=decide_payload)
    assert response.status_code == 200
    body = response.json()

    # Required response fields.
    assert body["model"] == "pai-1-0.5b"
    assert body["choice"] in decide_payload["options"]
    assert isinstance(body["inference_ms"], (int, float))
    assert body["inference_ms"] >= 0

    # Confidence must match the winning probability.
    assert body["confidence"] == pytest.approx(body["probabilities"][body["choice"]])

    # Probabilities cover exactly the requested labels, sum to 1,
    # and are sorted highest-first (API guarantee documented in README).
    assert set(body["probabilities"]) == set(decide_payload["options"])
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    values = list(body["probabilities"].values())
    assert values == sorted(values, reverse=True)


def test_decide_forwards_state_question_options_to_model(client, fake_model, decide_payload):
    client.post("/v1/decide", json=decide_payload)
    assert len(fake_model.calls) == 1
    state, question, options = fake_model.calls[0]
    assert state == decide_payload["state"]
    assert question == decide_payload["question"]
    assert options == decide_payload["options"]


def test_decide_strips_whitespace(client, fake_model):
    payload = {
        "state": "  An incident  ",
        "question": "  What next? ",
        "options": {" a ": "  Investigate  ", "b": "Rollback"},
    }
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 200
    # Labels and text are stripped by DecideRequest validators.
    _, _, options = fake_model.calls[0]
    assert set(options) == {"a", "b"}


@pytest.mark.parametrize(
    "payload",
    [
        # Blank / missing fields.
        {"state": "", "question": "What next?", "options": {"a": "A", "b": "B"}},
        {"state": "   ", "question": "What next?", "options": {"a": "A", "b": "B"}},
        {"state": "Incident", "question": "", "options": {"a": "A", "b": "B"}},
        {"state": "Incident", "question": "   ", "options": {"a": "A", "b": "B"}},
        # Too few options.
        {"state": "Incident", "question": "What next?", "options": {}},
        {"state": "Incident", "question": "What next?", "options": {"only": "One option"}},
        # Blank label / description.
        {"state": "Incident", "question": "What next?", "options": {"": "A", "b": "B"}},
        {"state": "Incident", "question": "What next?", "options": {"a": "", "b": "B"}},
        {"state": "Incident", "question": "What next?", "options": {"a": "   ", "b": "B"}},
        # Wrong types are also rejected by pydantic.
        {"state": "Incident", "question": "What next?", "options": "not-a-dict"},
        {"state": None, "question": "What next?", "options": {"a": "A", "b": "B"}},
    ],
)
def test_decide_rejects_invalid_payloads_with_422(client, payload):
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 422


def test_decide_rejects_more_than_20_options(client):
    payload = {
        "state": "Incident",
        "question": "What next?",
        "options": {f"opt{i}": f"Option {i}" for i in range(21)},
    }
    assert client.post("/v1/decide", json=payload).status_code == 422


def test_decide_rejects_duplicate_labels_after_stripping(client):
    payload = {
        "state": "Incident",
        "question": "What next?",
        # "a" and " a " normalize to the same label.
        "options": {"a": "First", " a ": "Second", "b": "Third"},
    }
    response = client.post("/v1/decide", json=payload)
    # Pydantic treats dict keys as already unique, so the request itself
    # passes JSON parsing but our validator must catch the normalized dup.
    # Either 422 (validator) is correct; 200 would mean the bug regressed.
    assert response.status_code == 422


def test_decide_model_failure_returns_500_without_leak(client_failing_model, decide_payload):
    response = client_failing_model.post("/v1/decide", json=decide_payload)
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "Model inference failed"
    # Generic handler must not leak internal tracebacks.
    assert "boom" not in body.get("detail", "")
