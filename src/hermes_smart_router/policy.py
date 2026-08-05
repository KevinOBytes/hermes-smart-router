"""Classification-to-route policy mapping.

Maps a ClassifierResult to a RouteSelection using the configured route table.
"""

from __future__ import annotations

from hermes_smart_router.config import SmartRouterConfig
from hermes_smart_router.models import (
    DEFAULT_ROUTE_TABLE,
    ClassifierResult,
    OperatingMode,
    ReasonCode,
    RiskLevel,
    RouteSelection,
    Sensitivity,
    TaskClass,
)


class RoutePolicy:
    """Maps classification results to route selections."""

    def __init__(self, config: SmartRouterConfig) -> None:
        self._config = config
        self._route_table: dict[TaskClass, tuple[str, str]] = {}
        self._build_route_table()

    def _build_route_table(self) -> None:
        """Build route table from config overrides on top of defaults."""
        self._route_table = dict(DEFAULT_ROUTE_TABLE)
        for task_class_str, route_cfg in self._config.routes.items():
            try:
                tc = TaskClass(task_class_str)
                self._route_table[tc] = (route_cfg.primary, route_cfg.escalation)
            except ValueError:
                pass

    def evaluate(
        self,
        classifier_result: ClassifierResult | None,
        reason_code: ReasonCode = ReasonCode.CLASSIFIER,
    ) -> RouteSelection:
        """Evaluate a classifier result into a route selection.

        Args:
            classifier_result: The classifier result, or None for fallback.
            reason_code: The reason code for this selection.

        Returns:
            A RouteSelection with primary and escalation aliases.
        """
        if classifier_result is None:
            # Fallback: route to luna (cheapest general-purpose)
            risk_val = classifier_result.risk if classifier_result else RiskLevel.LOW
            sens_val = classifier_result.sensitivity if classifier_result else Sensitivity.PUBLIC
            return RouteSelection(
                primary_alias="luna",
                escalation_alias="glm",
                reason_code=ReasonCode.CLASSIFIER_FALLBACK,
                task_class=TaskClass.KNOWLEDGE_REASONING,
                risk=risk_val,
                sensitivity=sens_val,
                confidence=0.0,
            )

        task_class = classifier_result.task_class
        primary, escalation = self._route_table.get(
            task_class, ("luna", "glm")
        )

        # Fixed mode override
        if self._config.mode == OperatingMode.FIXED.value:
            return RouteSelection(
                primary_alias=self._config.fixed_alias,
                escalation_alias=self._config.fixed_alias,
                reason_code=ReasonCode.FIXED_MODE,
                task_class=task_class,
                risk=classifier_result.risk,
                sensitivity=classifier_result.sensitivity,
                confidence=classifier_result.confidence,
                classifier_raw=classifier_result,
            )

        # High-risk or destructive: start on escalation alias
        if (
            classifier_result.risk.value in ("high", "critical")
            or classifier_result.destructive_potential
        ):
            return RouteSelection(
                primary_alias=escalation,
                escalation_alias=escalation,
                reason_code=ReasonCode.ESCALATION_HIGH_RISK,
                task_class=task_class,
                risk=classifier_result.risk,
                sensitivity=classifier_result.sensitivity,
                confidence=classifier_result.confidence,
                classifier_raw=classifier_result,
            )

        return RouteSelection(
            primary_alias=primary,
            escalation_alias=escalation,
            reason_code=reason_code,
            task_class=task_class,
            risk=classifier_result.risk,
            sensitivity=classifier_result.sensitivity,
            confidence=classifier_result.confidence,
            classifier_raw=classifier_result,
        )

    def get_escalation_for_alias(self, alias: str) -> str:
        """Get the escalation alias for a given primary alias.

        Scans the route table for the alias and returns its escalation partner.
        Defaults to 'opus' if not found.
        """
        for _tc, (primary, escalation) in self._route_table.items():
            if primary == alias:
                return escalation
        return "opus"
