from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    agent_api_key: str | None = None
    agent_base_url: str | None = None
    agent_model: str | None = None
    agent_timeout_seconds: float = Field(default=90, gt=0)
    bge_model: str = "BAAI/bge-m3"
    bge_device: str | None = None
    procedural_memory_mode: Literal["off", "write", "flat", "rsi"] = "off"
    procedural_memory_context_tokens: int = Field(default=512, gt=0, le=4096)
    procedural_cluster_threshold: float = Field(default=0.75, ge=-1, le=1)
    procedural_recall_threshold: float = Field(default=0.45, ge=-1, le=1)
    memory_related_threshold: float = Field(default=0.75, ge=0, le=1)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf8", extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
