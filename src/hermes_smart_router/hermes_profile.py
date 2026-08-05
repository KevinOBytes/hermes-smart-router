"""Hermes-native provider profile with active per-request routing.

This module implements active routing WITHOUT modifying Hermes core.
It uses two sanctioned ProviderProfile hooks:

1. ``prepare_messages()`` — receives the full message list before the
   request is built; we extract the latest user message for classification.
2. ``build_api_kwargs_extras()`` — its top-level kwargs are merged into
   the API request AFTER ``model`` is set, so returning ``{"model": slug}``
   swaps the virtual ``tko/smart-router`` model for the routed concrete
   OpenRouter slug.

Reasoning-config and extra_body quirks (Anthropic adaptive-thinking 400s,
OpenRouter sticky session keys, etc.) are delegated to the registered
OpenRouter profile so this plugin never duplicates that logic.

Routing pipeline per request:
  deterministic classifier (regex, instant)
    → Gemma via local Ollama (sync HTTP, config timeout)
      → fallback alias (luna)

The selected route is pinned per Hermes session (TTL from config) so only
the first request of a session pays classification latency.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hermes_smart_router.deterministic import classify_deterministic
from hermes_smart_router.models import (
    DEFAULT_ALIAS_MAPPINGS,
    DEFAULT_ROUTE_TABLE,
    ClassifierResult,
    TaskClass,
)

if TYPE_CHECKING:  # pragma: no cover
    from providers.base import ProviderProfile  # type: ignore[import-not-found]
else:
    from providers.base import ProviderProfile  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

_FALLBACK_ALIAS = "luna"

_CLASSIFICATION_PROMPT = (
    "Classify this request into exactly one category.\n\n"
    "Categories: structured_simple, agentic_execution, software_engineering, "
    "security_engineering, knowledge_reasoning, writing_communication, "
    "computer_use, visual_frontend\n\n"
    "Return ONLY valid JSON:\n"
    '{"task_class": "<category>", "risk": "low|moderate|high|critical", '
    '"sensitivity": "public|internal|confidential|restricted", '
    '"confidence": 0.0-1.0}\n\n'
    "Request:"
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _hermes_home() -> Path:
    """Resolve $HERMES_HOME without a hard dependency on hermes internals."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


class _ConfigCache:
    """Lazy mtime-cached view of the smart_router section of config.yaml."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._mtime: float = 0.0
        self._path = _hermes_home() / "config.yaml"

    def get(self) -> dict[str, Any]:
        try:
            mtime = self._path.stat().st_mtime
            if mtime != self._mtime:
                import yaml

                with self._path.open() as fh:
                    full = yaml.safe_load(fh) or {}
                self._data = full.get("smart_router") or {}
                self._mtime = mtime
        except Exception as exc:  # config unreadable → defaults
            logger.debug("smart-router config read failed: %s", exc)
        return self._data


class SmartRouterProfile(ProviderProfile):  # type: ignore[misc]
    """Virtual-model profile that swaps in a routed concrete model per request."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cfg = _ConfigCache()
        self._lock = threading.Lock()
        # session_id -> (concrete_slug, alias, pinned_at)
        self._pins: dict[str, tuple[str, str, float]] = {}
        self._last_user_text: str = ""

    # ── Model catalog: virtual model first, then the live catalog ────

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """List the virtual model first, then the live OpenRouter catalog."""
        catalog: list[str] = []
        try:
            live = super().fetch_models(
                api_key=api_key, base_url=base_url, timeout=timeout
            )
            if live:
                catalog = [m for m in live if m != "tko/smart-router"]
        except Exception as exc:
            logger.debug("smart-router catalog fetch failed: %s", exc)
        return ["tko/smart-router", *catalog]

    # ── Hook 1: capture the latest user message ─────────────────────

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text = ""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                text = "\n".join(p for p in parts if p)
            break
        self._last_user_text = text or ""
        return messages

    # ── Hook 2: override the model in top-level api kwargs ──────────

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict[str, Any] | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        routed_slug, alias = self._route(session_id)

        # Delegate reasoning handling to the OpenRouter profile with the
        # ROUTED model name so Anthropic adaptive-thinking rules apply to
        # the model actually being called, not the virtual name.
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        delegate = self._openrouter_profile()
        if delegate is not None:
            try:
                extra_body, top_level = delegate.build_api_kwargs_extras(
                    reasoning_config=reasoning_config,
                    supports_reasoning=supports_reasoning,
                    model=routed_slug,
                    session_id=session_id,
                    **context,
                )
            except Exception as exc:
                logger.debug("openrouter delegate extras failed: %s", exc)

        top_level["model"] = routed_slug
        logger.info(
            "smart-router: routed session=%s alias=%s model=%s",
            (session_id or "-")[:12],
            alias,
            routed_slug,
        )
        return extra_body, top_level

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        delegate = self._openrouter_profile()
        if delegate is not None:
            try:
                body: dict[str, Any] = delegate.build_extra_body(
                    session_id=session_id, **context
                )
                return body
            except Exception as exc:
                logger.debug("openrouter delegate extra_body failed: %s", exc)
        return {}

    # ── Routing core ─────────────────────────────────────────────────

    def _route(self, session_id: str | None) -> tuple[str, str]:
        """Return (concrete_slug, alias) for this request."""
        cfg = self._cfg.get()
        mode = str(cfg.get("mode", "active")).lower()
        ttl = int(cfg.get("task_ttl_seconds", 3600))

        # Fixed mode: always the configured alias.
        if mode == "fixed":
            alias = str(cfg.get("fixed_alias", _FALLBACK_ALIAS))
            return self._resolve_alias(alias, cfg), alias

        # Session pin: reuse a prior classification within TTL.
        if session_id:
            with self._lock:
                pin = self._pins.get(session_id)
                if pin and (time.time() - pin[2]) < ttl:
                    return pin[0], pin[1]

        # Classify: deterministic → Gemma → fallback.
        alias = self._classify_alias(cfg)
        slug = self._resolve_alias(alias, cfg)

        if session_id:
            with self._lock:
                if len(self._pins) > 512:  # bound memory
                    cutoff = time.time() - ttl
                    self._pins = {
                        k: v for k, v in self._pins.items() if v[2] >= cutoff
                    }
                self._pins[session_id] = (slug, alias, time.time())
        return slug, alias

    def _classify_alias(self, cfg: dict[str, Any]) -> str:
        text = self._last_user_text
        if not text:
            return _FALLBACK_ALIAS

        result = classify_deterministic(text, None)
        if result is None:
            result = self._classify_gemma(text, cfg)
        if result is None:
            return _FALLBACK_ALIAS

        threshold = float(
            (cfg.get("ollama") or {}).get("confidence_threshold", 0.70)
        )
        if result.confidence < threshold:
            return _FALLBACK_ALIAS

        route = DEFAULT_ROUTE_TABLE.get(result.task_class)
        if route is None:
            return _FALLBACK_ALIAS
        primary_alias = route[0]

        # High/critical risk starts directly on the escalation alias.
        if result.risk.value in ("high", "critical"):
            return route[1]
        return primary_alias

    def _classify_gemma(
        self, text: str, cfg: dict[str, Any]
    ) -> ClassifierResult | None:
        """Synchronous Gemma classification via local Ollama."""
        ollama = cfg.get("ollama") or {}
        base_url = str(ollama.get("base_url", "http://127.0.0.1:11434"))
        model = str(ollama.get("model", "gemma4:31b"))
        timeout = float(ollama.get("timeout_seconds", 30))
        max_tokens = int(ollama.get("max_output_tokens", 512))
        temperature = float(ollama.get("temperature", 0.0))

        try:
            import httpx

            resp = httpx.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{_CLASSIFICATION_PROMPT} {text[:4000]}",
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = str(resp.json().get("response", "")).strip()
        except Exception as exc:
            logger.debug("gemma classification unavailable: %s", exc)
            return None

        if not raw:
            return None
        fence = _JSON_FENCE_RE.search(raw)
        if fence:
            raw = fence.group(1)
        try:
            data = json.loads(raw)
            return ClassifierResult.model_validate(data)
        except Exception as exc:
            logger.debug("gemma returned unparseable classification: %s", exc)
            return None

    def _resolve_alias(self, alias: str, cfg: dict[str, Any]) -> str:
        """Alias → concrete OpenRouter slug (config overrides > defaults)."""
        overrides = cfg.get("aliases") or {}
        if isinstance(overrides, dict) and alias in overrides:
            slug = (overrides[alias] or {}).get("model_slug")
            if slug:
                return str(slug)
        mapping = DEFAULT_ALIAS_MAPPINGS.get(alias)
        if mapping is not None:
            return mapping.model_slug
        fallback = DEFAULT_ALIAS_MAPPINGS[_FALLBACK_ALIAS]
        return fallback.model_slug

    @staticmethod
    def _openrouter_profile() -> Any | None:
        try:
            from providers import get_provider_profile  # type: ignore[import-not-found]

            profile = get_provider_profile("openrouter")
            # Never delegate to ourselves on misconfiguration.
            if profile is not None and profile.name == "openrouter":
                return profile
        except Exception:
            pass
        return None


__all__ = ["SmartRouterProfile", "TaskClass"]
