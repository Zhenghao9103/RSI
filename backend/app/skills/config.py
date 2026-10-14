"""Settings belonging only to the explicitly invoked skill pool."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SkillPoolSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SKILL_POOL_", extra="ignore")

    root: Path = Path(__file__).resolve().parents[3] / "storage" / "skill_pool"
    auto_accumulation: bool = False
    cluster_threshold: float = Field(default=0.8, ge=-1, le=1)
    generation_min_confidence: float = Field(default=0.8, ge=0, le=1)
    cluster_limit: int = Field(default=3, gt=0)
    reference_limit: int = Field(default=3, gt=0)
    repeat_threshold: int = Field(default=3, gt=1)
    independent_hit_threshold: int = Field(default=3, gt=0)
    reference_update_threshold: int = Field(default=3, gt=0)
    maintenance_interval: int = Field(default=10, gt=0)
    cooldown_hours: float = Field(default=24, ge=0)
    discovery_threshold: float = Field(default=0.65, ge=-1, le=1)
    discovery_limit: int = Field(default=3, gt=0)
    divergence_similarity: float = Field(default=0.75, ge=-1, le=1)
    llm_max_tokens: int = Field(default=4096, gt=0)
