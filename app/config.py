"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    backbone: str
    device: str
    max_length: int
    cors_origins: list[str]

    @classmethod
    def from_environment(cls) -> "Settings":
        def _get(*names: str, default: str) -> str:
            for name in names:
                value = os.getenv(name)
                if value is not None and value != "":
                    return value
            return default

        origins = _get("PAI1_CORS_ORIGINS", "PAICLEF_CORS_ORIGINS", default="*")
        return cls(
            model_path=Path(_get("PAI1_MODEL_PATH", "PAICLEF_MODEL_PATH", default="models/pai-1-0.5b")),
            backbone=_get("PAI1_BACKBONE", "PAICLEF_BACKBONE", default="Qwen/Qwen2.5-0.5B-Instruct"),
            device=_get("PAI1_DEVICE", "PAICLEF_DEVICE", default="auto"),
            max_length=int(_get("PAI1_MAX_LENGTH", "PAICLEF_MAX_LENGTH", default="2048")),
            cors_origins=[item.strip() for item in origins.split(",") if item.strip()],
        )