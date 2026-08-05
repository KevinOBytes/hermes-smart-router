"""Dependency and configuration health checks."""

from __future__ import annotations

import logging

import httpx

from hermes_smart_router.catalog import CatalogValidator
from hermes_smart_router.config import SmartRouterConfig
from hermes_smart_router.models import DEFAULT_ALIAS_MAPPINGS, AliasMapping, HealthStatus
from hermes_smart_router.state import TaskState

logger = logging.getLogger(__name__)


class HealthChecker:
    """Runs health checks on all plugin dependencies."""

    def __init__(
        self,
        config: SmartRouterConfig,
        state: TaskState,
        catalog: CatalogValidator | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._catalog = catalog

    async def check_all(self) -> list[HealthStatus]:
        """Run all health checks.

        Returns:
            List of HealthStatus results.
        """
        results: list[HealthStatus] = []

        # Configuration validity
        results.append(self._check_config())

        # SQLite access
        results.append(self._check_state())

        # Ollama reachability
        results.extend(await self._check_ollama())

        # OpenRouter auth
        results.append(await self._check_openrouter_auth())

        # Alias mappings
        if self._catalog is not None:
            results.extend(await self._check_aliases())

        return results

    def _check_config(self) -> HealthStatus:
        """Check configuration validity."""
        if self._config.mode not in ("shadow", "active", "fixed"):
            return HealthStatus(
                name="config",
                healthy=False,
                detail=f"Invalid mode: {self._config.mode}",
            )
        return HealthStatus(name="config", healthy=True, detail="OK")

    def _check_state(self) -> HealthStatus:
        """Check SQLite state store."""
        try:
            self._state.get_task("__health_check__")
            return HealthStatus(name="state", healthy=True, detail="OK")
        except Exception as e:
            return HealthStatus(
                name="state",
                healthy=False,
                detail=f"State store error: {e}",
            )

    async def _check_ollama(self) -> list[HealthStatus]:
        """Check Ollama reachability and Gemma availability."""
        results: list[HealthStatus] = []
        ollama_url = self._config.ollama.base_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                results.append(
                    HealthStatus(
                        name="ollama",
                        healthy=True,
                        detail=f"Reachable at {ollama_url}",
                    )
                )
                # Check Gemma
                gemma_model = self._config.ollama.model
                if any(gemma_model in m for m in models):
                    results.append(
                        HealthStatus(
                            name=f"ollama:{gemma_model}",
                            healthy=True,
                            detail=f"Model '{gemma_model}' available",
                        )
                    )
                else:
                    results.append(
                        HealthStatus(
                            name=f"ollama:{gemma_model}",
                            healthy=False,
                            detail=f"Model '{gemma_model}' not found in Ollama",
                        )
                    )
        except httpx.ConnectError:
            results.append(
                HealthStatus(
                    name="ollama",
                    healthy=False,
                    detail=f"Connection refused at {ollama_url}",
                )
            )
        except Exception as e:
            results.append(
                HealthStatus(
                    name="ollama",
                    healthy=False,
                    detail=str(e),
                )
            )

        return results

    async def _check_openrouter_auth(self) -> HealthStatus:
        """Check OpenRouter authentication."""
        api_key = self._get_api_key()
        if not api_key:
            return HealthStatus(
                name="openrouter",
                healthy=False,
                detail=f"API key not found in env '{self._config.openrouter.api_key_env}'",
            )

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self._config.openrouter.base_url.rstrip('/')}/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    return HealthStatus(
                        name="openrouter",
                        healthy=True,
                        detail="Authenticated",
                    )
                return HealthStatus(
                    name="openrouter",
                    healthy=False,
                    detail=f"Auth failed: HTTP {resp.status_code}",
                )
        except Exception as e:
            return HealthStatus(
                name="openrouter",
                healthy=False,
                detail=str(e),
            )

    async def _check_aliases(self) -> list[HealthStatus]:
        """Check all enabled alias mappings."""
        if self._catalog is None:
            return []

        results: list[HealthStatus] = []
        mappings = self._build_alias_mappings()

        for alias, mapping in mappings.items():
            results.append(self._catalog.validate_mapping(mapping))

        return results

    def _build_alias_mappings(self) -> dict[str, AliasMapping]:
        """Build alias mappings from config overrides on top of defaults."""
        mappings = dict(DEFAULT_ALIAS_MAPPINGS)
        for alias, route_cfg in self._config.aliases.items():
            mappings[alias] = AliasMapping(
                alias=alias,
                model_slug=route_cfg.model_slug,
                requires_tools=route_cfg.requires_tools,
                requires_vision=route_cfg.requires_vision,
                fallback_slug=route_cfg.fallback_slug,
            )
        return mappings

    @staticmethod
    def _get_api_key() -> str | None:
        """Get the OpenRouter API key from environment."""
        import os
        return os.environ.get("OPENROUTER_API_KEY")
