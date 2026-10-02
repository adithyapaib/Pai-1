"""Tests for POST /v1/systemone (TypeSafe / LangChain compatible)."""

import pytest


def _router_payload():
    # Mirrors what ModelRouterMiddleware sends: state = latest human message
    # (serialized to role/content JSON), one Choice question built from
    # ModelChoice entries.
    return {
        "state": [{"role": "user", "content": "The checkout page crashes on pay."}],
        "model": "pai-1",
        "questions": {
            "model_route": {
                "type": "choice",
                "instructions": "Choose the least costly model that can complete the task.",
                "criteria": {
                    "fast": "Direct lookups, extraction, and localized changes.",
                    "powerful": "Architecture and high-stakes decisions.",
                },
            }
        },
    }


def test_systemone_choice_router_shape(client):
    r = client.post("/v1/systemone", json=_router_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "pai-1-0.5b"
    ans = body["answers"]["model_route"]
    assert ans["type"] == "choice"
    assert ans["choice"] in {"fast", "powerful"}
    assert set(ans["probabilities"]) == {"fast", "powerful"}
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-6
    assert body["usage"]["output_tokens"] == 0 or body["usage"]["output_tokens"] is None
    assert body["request_id"]
    assert r.headers.get("x-typesafe-request-id") == body["request_id"]


def test_systemone_noul_and_score(client):
    payload = {
        "state": "Checkout has been failing for every customer for the last hour.",
        "model": "pai-1",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
            "severity": {
                "type": "score",
                "instructions": "How severe is the customer impact?",
                "criteria": ["No impact", "Minor", "Major", "Critical"],
            },
        },
    }
    r = client.post("/v1/systemone", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["answers"]["urgent"]["noul"] <= 1.0
    score = body["answers"]["severity"]
    assert score["type"] == "score"
    # Expected value over 0..3, legend preserved, probs sum to 1.
    assert 0.0 <= score["score"] <= 3.0
    assert {int(k) for k in score["probabilities"]} == {0, 1, 2, 3}
    assert sum(score["probabilities"].values()) == pytest.approx(1.0)
    assert {int(k) for k in score["legend"]} == {0, 1, 2, 3}


def test_systemone_accepts_object_state_and_json_criteria(client):
    payload = {
        "state": {"conversation": [{"role": "user", "content": "Refund please"}], "tier": "enterprise"},
        "model": "pai-1",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": {"task": "route", "lang": "en"},
                "criteria": {"billing": {"topic": "payments"}, "technical": "bugs"},
            }
        },
    }
    r = client.post("/v1/systemone", json=payload)
    assert r.status_code == 200
    assert r.json()["answers"]["dept"]["choice"] in {"billing", "technical"}


def test_systemone_ignores_bearer_auth(client):
    r = client.post(
        "/v1/systemone",
        json=_router_payload(),
        headers={"Authorization": "Bearer local-test-key"},
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"state": "", "model": "pai-1", "questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"state": 123, "model": "pai-1", "questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"state": "s", "model": "pai-1", "questions": {}},
        {"state": "s", "model": "pai-1", "questions": {"q": {"type": "essay", "instructions": "x"}}},
    ],
)
def test_systemone_rejects_bad_requests(client, payload):
    assert client.post("/v1/systemone", json=payload).status_code == 422
