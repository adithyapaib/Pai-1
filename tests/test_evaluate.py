"""Tests for POST /v1/evaluate: typed questions, mapping, and validation."""

import pytest


def test_evaluate_all_typed_questions(client, evaluate_payload):
    response = client.post("/v1/evaluate", json=evaluate_payload)
    assert response.status_code == 200
    body = response.json()

    assert body["model"] == "pai-1-0.5b"
    assert set(body["results"]) == {"urgent", "team", "severity"}
    assert isinstance(body["inference_ms"], (int, float))
    assert body["inference_ms"] >= 0

    # noul / boolean produces yes|no.
    assert body["results"]["urgent"]["type"] == "noul"
    assert body["results"]["urgent"]["choice"] in {"yes", "no"}

    # choice echoes one of the criteria keys.
    assert body["results"]["team"]["choice"] in {"billing", "technical", "sales"}

    # score maps internal "0"/"1"/... indices back to human labels,
    # for both the choice and the probability keys.
    severity = body["results"]["severity"]
    assert severity["choice"] in ["No impact", "Minor", "Major", "Critical"]
    assert set(severity["probabilities"]) == {"No impact", "Minor", "Major", "Critical"}
    assert sum(severity["probabilities"].values()) == pytest.approx(1.0)
    assert severity["confidence"] == pytest.approx(severity["probabilities"][severity["choice"]])


def test_evaluate_boolean_aliases_accepted(client):
    for alias in ("bool", "boolean", "noul", "NOUL", " Bool "):
        payload = {
            "state": "Server is down.",
            "questions": {"q": {"type": alias, "instructions": "Is it urgent?"}},
        }
        response = client.post("/v1/evaluate", json=payload)
        assert response.status_code == 200, alias
        assert response.json()["results"]["q"]["choice"] in {"yes", "no"}


def test_evaluate_score_uses_criteria_labels_not_indices(client, fake_model):
    payload = {
        "state": "Checkout failing.",
        "questions": {
            "sev": {
                "type": "score",
                "instructions": "How severe?",
                "criteria": ["Low", "High"],
            }
        },
    }
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()["results"]["sev"]
    assert body["choice"] in {"Low", "High"}
    # FakeModel saw stringified indices as option keys.
    _, _, options = fake_model.calls[0]
    assert set(options) == {"0", "1"}


def test_evaluate_single_shared_state_forwarded(client, fake_model, evaluate_payload):
    client.post("/v1/evaluate", json=evaluate_payload)
    assert len(fake_model.calls) == 3
    for state, _, _ in fake_model.calls:
        assert state == evaluate_payload["state"]


@pytest.mark.parametrize(
    "questions",
    [
        {},  # min_length=1
        {f"q{i}": {"type": "noul", "instructions": "Urgent?"} for i in range(21)},  # max 20
    ],
)
def test_evaluate_rejects_wrong_question_count(client, questions):
    response = client.post("/v1/evaluate", json={"state": "s", "questions": questions})
    assert response.status_code == 422


def test_evaluate_rejects_unknown_type(client):
    payload = {"state": "s", "questions": {"q": {"type": "essay", "instructions": "Write more"}}}
    assert client.post("/v1/evaluate", json=payload).status_code == 422


def test_evaluate_rejects_blank_instructions_and_state(client):
    assert client.post(
        "/v1/evaluate",
        json={"state": "   ", "questions": {"q": {"type": "noul", "instructions": "Ok?"}}},
    ).status_code == 422
    assert client.post(
        "/v1/evaluate",
        json={"state": "ok", "questions": {"q": {"type": "noul", "instructions": "   "}}},
    ).status_code == 422


def test_evaluate_rejects_short_criteria(client):
    # criteria with fewer than 2 entries is rejected by QuestionSpec.
    payload = {
        "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"only": "one"}}},
    }
    assert client.post("/v1/evaluate", json=payload).status_code == 422

    payload = {
        "state": "s",
        "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["Only"]}},
    }
    assert client.post("/v1/evaluate", json=payload).status_code == 422


def test_evaluate_choice_without_dict_criteria_fails_safe(client_no_raise):
    # QuestionSpec allows criteria=None, but /v1/evaluate requires a dict
    # for choice. Current app converts this to 500 via the generic handler.
    # NOTE: needs raise_server_exceptions=False so the handler response
    # is returned instead of re-raised by TestClient.
    payload = {
        "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "Pick"}},
    }
    response = client_no_raise.post("/v1/evaluate", json=payload)
    assert response.status_code == 500
    assert response.json()["error"] == "Model inference failed"


def test_evaluate_score_without_list_criteria_fails_safe(client_no_raise):
    payload = {
        "state": "s",
        "questions": {
            "q": {"type": "score", "instructions": "Rate", "criteria": {"a": "A", "b": "B"}},
        },
    }
    response = client_no_raise.post("/v1/evaluate", json=payload)
    assert response.status_code == 500


def test_evaluate_model_failure_returns_500(client_failing_model, evaluate_payload):
    response = client_failing_model.post("/v1/evaluate", json=evaluate_payload)
    assert response.status_code == 500
    assert response.json()["error"] == "Model inference failed"
