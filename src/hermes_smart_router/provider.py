"""Request lifecycle orchestration for the smart-router provider.

Orchestrates classification, routing, model pinning, escalation, and
OpenRouter transport for each task.
"""

from __future__ import annotations

import logging
from typing import Any

from hermes_smart_router.catalog import CatalogValidator
from hermes_smart_router.classifier import Classifier
from hermes_smart_router.config import SmartRouterConfig
from hermes_smart_router.deterministic import classify_deterministic
from hermes_smart_router.escalation import EscalationDetector
from hermes_smart_router.health import HealthChecker
from hermes_smart_router.models import (
    DEFAULT_ALIAS_MAPPINGS,
    AliasMapping,
    HealthStatus,
    ModelPin,
    ReasonCode,
    RouteSelection,
)
from hermes_smart_router.openrouter import OpenRouterTransport
from hermes_smart_router.policy import RoutePolicy
from hermes_smart_router.state import TaskState
from hermes_smart_router.telemetry import TelemetryCollector, Timer

logger = logging.getLogger(__name__)


class SmartRouterProvider:
    """Main provider orchestration for the smart-router plugin.

    Handles the complete lifecycle: classification → routing → pinning →
    transport → escalation.
    """

    def __init__(
        self,
        config: SmartRouterConfig,
        state: TaskState,
        classifier: Classifier,
        policy: RoutePolicy,
        catalog: CatalogValidator,
        transport: OpenRouterTransport,
        escalation: EscalationDetector,
        telemetry: TelemetryCollector,
    ) -> None:
        self._config = config
        self._state = state
        self._classifier = classifier
        self._policy = policy
        self._catalog = catalog
        self._transport = transport
        self._escalation = escalation
        self._telemetry = telemetry

        # Build alias mappings from config overrides
        self._alias_mappings: dict[str, AliasMapping] = dict(DEFAULT_ALIAS_MAPPINGS)
        for alias, route_cfg in config.aliases.items():
            self._alias_mappings[alias] = AliasMapping(
                alias=alias,
                model_slug=route_cfg.model_slug,
                requires_tools=route_cfg.requires_tools,
                requires_vision=route_cfg.requires_vision,
                fallback_slug=route_cfg.fallback_slug,
            )

    async def classify_and_route(
        self,
        task_id: str,
        session_id: str | None,
        user_request: str,
        tool_names: list[str] | None = None,
    ) -> RouteSelection:
        """Classify a task and select a route.

        Args:
            task_id: Unique task identifier.
            session_id: Optional session identifier.
            user_request: The initial user request text.
            tool_names: Available tool names.

        Returns:
            RouteSelection with primary and escalation aliases.
        """
        with Timer() as timer:
            # 1. Try deterministic classification first
            deterministic_result = classify_deterministic(user_request, tool_names)
            if deterministic_result is not None:
                route = self._policy.evaluate(
                    deterministic_result,
                    reason_code=ReasonCode.DETERMINISTIC,
                )
                self._telemetry.classification(
                    task_id=task_id,
                    session_id=session_id,
                    task_class=route.task_class.value,
                    confidence=deterministic_result.confidence,
                    latency_ms=timer.elapsed_ms,
                )
                self._telemetry.route_selected(
                    task_id=task_id,
                    session_id=session_id,
                    primary_alias=route.primary_alias,
                    escalation_alias=route.escalation_alias,
                    reason_code=route.reason_code.value,
                )
                self._persist_route(task_id, session_id, route)
                return route

            # 2. Try Gemma classifier
            classifier_result = await self._classifier.classify(user_request)

            if classifier_result is not None:
                route = self._policy.evaluate(classifier_result)
                self._telemetry.classification(
                    task_id=task_id,
                    session_id=session_id,
                    task_class=route.task_class.value,
                    confidence=classifier_result.confidence,
                    latency_ms=timer.elapsed_ms,
                )
            else:
                # 3. Fallback to luna
                route = self._policy.evaluate(
                    None,
                    reason_code=ReasonCode.CLASSIFIER_FALLBACK,
                )
                self._telemetry.classification(
                    task_id=task_id,
                    session_id=session_id,
                    task_class="fallback",
                    confidence=0.0,
                    latency_ms=timer.elapsed_ms,
                )

            self._telemetry.route_selected(
                task_id=task_id,
                session_id=session_id,
                primary_alias=route.primary_alias,
                escalation_alias=route.escalation_alias,
                reason_code=route.reason_code.value,
            )
            self._persist_route(task_id, session_id, route)
            return route

    def _persist_route(
        self,
        task_id: str,
        session_id: str | None,
        route: RouteSelection,
    ) -> None:
        """Persist the route selection to state."""
        # Resolve alias to concrete model
        concrete = self._resolve_alias(route.primary_alias)
        self._state.create_task(
            task_id=task_id,
            session_id=session_id,
            task_class=route.task_class.value,
            primary_alias=route.primary_alias,
            escalation_alias=route.escalation_alias,
            concrete_model=concrete,
            reason_code=route.reason_code.value,
        )

    def _resolve_alias(self, alias: str) -> str | None:
        """Resolve a logical alias to a concrete model slug."""
        mapping = self._alias_mappings.get(alias)
        if mapping is None:
            return None
        return self._catalog.resolve(alias, self._alias_mappings) or mapping.model_slug

    def get_pinned_model(self, task_id: str) -> ModelPin | None:
        """Get the currently pinned model for a task.

        Args:
            task_id: The task identifier.

        Returns:
            ModelPin if a model is pinned, None otherwise.
        """
        task = self._state.get_task(task_id)
        if task is None:
            return None

        # Check TTL
        if self._state.is_expired(task_id, self._config.task_ttl_seconds):
            return None

        concrete = task.get("concrete_model")
        if not concrete:
            return None

        return ModelPin(
            alias=task.get("primary_alias", ""),
            concrete_model=concrete,
        )

    async def check_escalation(
        self,
        task_id: str,
        recent_tool_results: list[dict[str, Any]],
    ) -> bool:
        """Check if the current task should be escalated.

        Args:
            task_id: The task identifier.
            recent_tool_results: Recent tool call results.

        Returns:
            True if escalation is needed.
        """
        decision = await self._escalation.check(task_id, recent_tool_results)
        if decision.should_escalate:
            task = self._state.get_task(task_id)
            if task is None:
                return False

            current_alias = task.get("primary_alias", "")
            escalation_alias = task.get("escalation_alias", "opus")

            if current_alias == escalation_alias:
                # Already on escalation alias, don't re-escalate
                return False

            # Resolve escalation alias
            concrete = self._resolve_alias(escalation_alias)
            if concrete:
                self._state.update_model(
                    task_id,
                    concrete_model=concrete,
                    alias=escalation_alias,
                )
                self._telemetry.escalation(
                    task_id=task_id,
                    session_id=task.get("session_id"),
                    from_alias=current_alias,
                    to_alias=escalation_alias,
                    reason_code=decision.reason_code.value if decision.reason_code else None,
                )
                return True

        return False

    async def health(self) -> list[HealthStatus]:
        """Run health checks."""
        checker = HealthChecker(self._config, self._state, self._catalog)
        return await checker.check_all()

    async def close(self) -> None:
        """Close all resources."""
        await self._classifier.close()
        await self._transport.close()
        await self._catalog.close()
        self._state.close()
