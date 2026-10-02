"""Health, root, and model-metadata endpoint tests."""


def test_root_returns_name_and_docs(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body == {"name": "Pai-1 API", "docs": "/docs"}


def test_health_ok_when_model_loaded(client, fake_model):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == str(fake_model.device)


def test_health_degraded_when_no_model(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["device"] is None


def test_model_info_reports_loaded_checkpoint(client, fake_model):
    response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == fake_model.model_name
    assert body["backbone"] == fake_model.settings.backbone
    assert body["hidden_size"] == fake_model.hidden_size
    assert body["decision_head_parameters"] == fake_model.parameter_count
    assert body["device"] == str(fake_model.device)
