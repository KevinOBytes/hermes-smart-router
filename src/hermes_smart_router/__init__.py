"""Hermes Smart Router — task-aware model routing plugin for Hermes Agent."""

__version__ = "0.1.0"

from hermes_smart_router.config import SmartRouterConfig
from hermes_smart_router.models import (
    DEFAULT_ALIAS_MAPPINGS,
    DEFAULT_ROUTE_TABLE,
    AliasMapping,
    ClassifierResult,
    EscalationDecision,
    EventType,
    HealthStatus,
    ModelPin,
    OperatingMode,
    ReasonCode,
    RiskLevel,
    RouteSelection,
    Sensitivity,
    TaskClass,
    TelemetryEvent,
)

__all__ = [
    "AliasMapping",
    "ClassifierResult",
    "DEFAULT_ALIAS_MAPPINGS",
    "DEFAULT_ROUTE_TABLE",
    "EscalationDecision",
    "EventType",
    "HealthStatus",
    "ModelPin",
    "OperatingMode",
    "ReasonCode",
    "RiskLevel",
    "RouteSelection",
    "Sensitivity",
    "SmartRouterConfig",
    "TaskClass",
    "TelemetryEvent",
]
