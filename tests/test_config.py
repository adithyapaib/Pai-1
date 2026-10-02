"""Unit tests for Settings.from_environment()."""

from pathlib import Path

from app.config import Settings

NEW_VARS = (
    "PAI1_MODEL_PATH",
    "PAI1_BACKBONE",
    "PAI1_DEVICE",
    "PAI1_MAX_LENGTH",
    "PAI1_CORS_ORIGINS",
)
LEGACY_VARS = (
    "PAICLEF_MODEL_PATH",
    "PAICLEF_BACKBONE",
    "PAICLEF_DEVICE",
    "PAICLEF_MAX_LENGTH",
    "PAICLEF_CORS_ORIGINS",
)


def test_settings_defaults(monkeypatch):
    for var in (*NEW_VARS, *LEGACY_VARS):
        monkeypatch.delenv(var, raising=False)
    settings = Settings.from_environment()
    assert settings.model_path == Path("models/pai-1-0.5b")
    assert settings.backbone == "Qwen/Qwen2.5-0.5B-Instruct"
    assert settings.device == "auto"
    assert settings.max_length == 2048
    assert settings.cors_origins == ["*"]


def test_settings_reads_new_environment(monkeypatch):
    for var in LEGACY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PAI1_MODEL_PATH", "/tmp/m")
    monkeypatch.setenv("PAI1_BACKBONE", "org/model")
    monkeypatch.setenv("PAI1_DEVICE", "cpu")
    monkeypatch.setenv("PAI1_MAX_LENGTH", "512")
    monkeypatch.setenv("PAI1_CORS_ORIGINS", "https://a.example, https://b.example ,, ")
    settings = Settings.from_environment()
    assert settings.model_path == Path("/tmp/m")
    assert settings.backbone == "org/model"
    assert settings.device == "cpu"
    assert settings.max_length == 512
    # Empty entries are dropped and whitespace is stripped.
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_settings_falls_back_to_legacy_vars(monkeypatch):
    for var in NEW_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PAICLEF_MODEL_PATH", "/tmp/legacy")
    monkeypatch.setenv("PAICLEF_DEVICE", "cpu")
    settings = Settings.from_environment()
    assert settings.model_path == Path("/tmp/legacy")
    assert settings.device == "cpu"


def test_settings_prefers_new_vars_over_legacy(monkeypatch):
    monkeypatch.setenv("PAI1_MODEL_PATH", "/tmp/new")
    monkeypatch.setenv("PAICLEF_MODEL_PATH", "/tmp/legacy")
    assert Settings.from_environment().model_path == Path("/tmp/new")
