"""Typed configuration for the smart-router plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Local Ollama configuration for the Gemma classifier."""

    model: str = "gemma4:31b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 30
    temperature: float = 0.0
    max_output_tokens: int = 512
    confidence_threshold: float = 0.70


class AliasRouteConfig(BaseModel):
    """Mapping from a logical alias to a concrete model slug."""

    model_slug: str
    requires_tools: bool = True
    requires_vision: bool = False
    fallback_slug: str | None = None


class RouteTableConfig(BaseModel):
    """Per-task-class route configuration."""

    primary: str
    escalation: str


class SensitivityConfig(BaseModel):
    """Sensitivity-based provider restrictions."""

    enabled: bool = False
    allowed_providers: list[str] | None = None
    """When enabled, only these providers may handle confidential/restricted tasks."""

    warn_providers: list[str] = Field(
        default_factory=lambda: ["deepseek", "zai", "moonshot"]
    )
    """Providers that trigger a warning for confidential/restricted content."""


class OpenRouterConfig(BaseModel):
    """OpenRouter-specific configuration."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    provider_preferences: dict[str, Any] | None = None
    """OpenRouter provider-level routing preferences (ZDR, allowlist, etc.)."""

    strict_identity: bool = False
    """When True, reject responses identifying an unexpected concrete model."""


class StateConfig(BaseModel):
    """Task-state backend configuration."""

    backend: str = "sqlite"
    path: str = "~/.hermes/smart-router/state.db"


class SmartRouterConfig(BaseModel):
    """Top-level configuration for the smart-router plugin."""

    mode: str = "shadow"
    """Operating mode: shadow, active, or fixed."""

    fixed_alias: str = "luna"
    """Alias to use in fixed mode."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    aliases: dict[str, AliasRouteConfig] = Field(default_factory=dict)
    """Logical alias -> concrete model mapping. Overrides defaults."""

    routes: dict[str, RouteTableConfig] = Field(default_factory=dict)
    """Task class -> (primary, escalation) alias mapping."""

    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)

    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)

    state: StateConfig = Field(default_factory=StateConfig)

    task_ttl_seconds: int = 3600
    """How long a task pin remains valid without activity."""

    @classmethod
    def from_file(cls, path: str | Path) -> SmartRouterConfig:
        """Load configuration from a YAML file."""
        path = Path(path).expanduser()
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("smart_router", {}))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmartRouterConfig:
        """Load configuration from a dictionary."""
        return cls(**data)
