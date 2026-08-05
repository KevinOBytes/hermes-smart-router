"""Content-free structured events for observability."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from hermes_smart_router.models import EventType, ReasonCode, TelemetryEvent

logger = logging.getLogger("hermes_smart_router.telemetry")


class TelemetryCollector:
    """Collects and emits content-free structured events.

    No prompts, tool arguments, tool outputs, model text, or credentials
    are ever included in events.
    """

    def __init__(self, sink: Callable[[TelemetryEvent], None] | None = None) -> None:
        self._sink = sink or self._default_sink

    @staticmethod
    def _default_sink(event: TelemetryEvent) -> None:
        level = logging.INFO if event.event_type not in (
            EventType.PROVIDER_FAILURE,
            EventType.SENSITIVITY_BLOCK,
        ) else logging.WARNING
        logger.log(level, "%s: %s", event.event_type.value, event.detail)

    def emit(self, event: TelemetryEvent) -> None:
        self._sink(event)

    def classification(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        task_class: str | None = None,
        confidence: float = 0.0,
        latency_ms: float | None = None,
    ) -> None:
        self.emit(TelemetryEvent(
            event_type=EventType.CLASSIFICATION,
            task_id=task_id,
            session_id=session_id,
            primary_alias=task_class,
            reason_code=None,
            latency_ms=latency_ms,
            detail=f"class={task_class} conf={confidence:.2f}",
        ))

    def route_selected(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        primary_alias: str | None = None,
        escalation_alias: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        self.emit(TelemetryEvent(
            event_type=EventType.ROUTE_SELECTED,
            task_id=task_id,
            session_id=session_id,
            primary_alias=primary_alias,
            escalation_alias=escalation_alias,
            reason_code=ReasonCode(reason_code) if reason_code else None,
            detail=f"route={primary_alias} esc={escalation_alias} reason={reason_code}",
        ))

    def escalation(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        from_alias: str | None = None,
        to_alias: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        self.emit(TelemetryEvent(
            event_type=EventType.ESCALATION,
            task_id=task_id,
            session_id=session_id,
            primary_alias=from_alias,
            escalation_alias=to_alias,
            reason_code=ReasonCode(reason_code) if reason_code else None,
            detail=f"escalated {from_alias} -> {to_alias} reason={reason_code}",
        ))

    def sensitivity_warning(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        primary_alias: str | None = None,
        escalation_alias: str | None = None,
        sensitivity: str | None = None,
    ) -> None:
        self.emit(TelemetryEvent(
            event_type=EventType.SENSITIVITY_WARNING,
            task_id=task_id,
            session_id=session_id,
            primary_alias=primary_alias,
            escalation_alias=escalation_alias,
            detail=f"sensitivity={sensitivity} primary={primary_alias} esc={escalation_alias}",
        ))

    def provider_failure(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        concrete_model: str | None = None,
        detail: str = "",
    ) -> None:
        self.emit(TelemetryEvent(
            event_type=EventType.PROVIDER_FAILURE,
            task_id=task_id,
            session_id=session_id,
            concrete_model=concrete_model,
            detail=f"model={concrete_model} {detail}",
        ))


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.monotonic() - self.start) * 1000
