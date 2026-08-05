"""Live catalog validation and alias resolution.

Queries OpenRouter's model catalog at startup to validate every enabled
alias mapping. Missing models, unavailable models, or models without
required capabilities appear in unhealthy status.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from hermes_smart_router.models import AliasMapping, HealthStatus

logger = logging.getLogger(__name__)


class CatalogValidator:
    """Validates alias mappings against the live OpenRouter model catalog."""

    def __init__(
        self,
        catalog_url: str = "https://openrouter.ai/api/v1/models",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._catalog_url = catalog_url
        self._http = http_client or httpx.AsyncClient(timeout=10)
        self._catalog: dict[str, dict[str, Any]] = {}
        self._loaded = False

    async def load(self) -> None:
        """Load the live model catalog from OpenRouter."""
        try:
            response = await self._http.get(self._catalog_url)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            self._catalog = {m["id"]: m for m in models if isinstance(m, dict) and "id" in m}
            self._loaded = True
            logger.info("Loaded %d models from OpenRouter catalog", len(self._catalog))
        except Exception as exc:
            logger.warning("Failed to load OpenRouter catalog: %s", exc)
            self._loaded = False

    def validate_mapping(self, mapping: AliasMapping) -> HealthStatus:
        """Validate a single alias mapping against the catalog.

        Args:
            mapping: The alias mapping to validate.

        Returns:
            HealthStatus indicating whether the mapping is valid.
        """
        if not self._loaded:
            return HealthStatus(
                name=f"alias:{mapping.alias}",
                healthy=False,
                detail="Catalog not loaded",
            )

        # Check primary slug
        primary_ok = mapping.model_slug in self._catalog
        if not primary_ok and mapping.fallback_slug:
            primary_ok = mapping.fallback_slug in self._catalog

        if not primary_ok:
            return HealthStatus(
                name=f"alias:{mapping.alias}",
                healthy=False,
                detail=f"Model slug '{mapping.model_slug}' not found in OpenRouter catalog",
            )

        # Check required capabilities
        model_info = self._catalog.get(mapping.model_slug, {})
        if mapping.requires_tools:
            params = model_info.get("supported_parameters", [])
            if "tools" not in params and "tool_choice" not in params:
                return HealthStatus(
                    name=f"alias:{mapping.alias}",
                    healthy=False,
                    detail=f"Model '{mapping.model_slug}' does not support tool calling",
                )

        return HealthStatus(
            name=f"alias:{mapping.alias}",
            healthy=True,
            detail=f"Mapped to '{mapping.model_slug}'",
        )

    def resolve(self, alias: str, mappings: dict[str, AliasMapping]) -> str | None:
        """Resolve a logical alias to a concrete model slug.

        Args:
            alias: The logical alias name.
            mappings: The configured alias mappings.

        Returns:
            Concrete model slug, or None if the alias is not configured.
        """
        mapping = mappings.get(alias)
        if mapping is None:
            return None

        slug = mapping.model_slug

        # Check fallback if primary is not in catalog
        if self._loaded and slug not in self._catalog and mapping.fallback_slug:
            if mapping.fallback_slug in self._catalog:
                logger.info(
                    "Alias '%s': primary '%s' not found, using fallback '%s'",
                    alias, slug, mapping.fallback_slug,
                )
                return mapping.fallback_slug

        return slug

    async def close(self) -> None:
        await self._http.aclose()
