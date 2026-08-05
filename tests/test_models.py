"""Tests for domain models and schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hermes_smart_router.models import (
    ClassifierResult,
    EscalationDecision,
    EventType,
    ModelPin,
    ReasonCode,
    RiskLevel,
    RouteSelection,
    Sensitivity,
    TaskClass,
    TelemetryEvent,
)


class TestClassifierResult:
    """Tests for the ClassifierResult schema."""

    def test_valid_minimal(self) -> None:
        result = ClassifierResult(
            task_class="software_engineering",
            risk="moderate",
            sensitivity="internal",
            requires_tools=True,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=0.85,
        )
        assert result.task_class == TaskClass.SOFTWARE_ENGINEERING
        assert result.risk == RiskLevel.MODERATE
        assert result.sensitivity == Sensitivity.INTERNAL
        assert result.confidence == 0.85

    def test_all_task_classes(self) -> None:
        for tc in TaskClass:
            result = ClassifierResult(
                task_class=tc.value,
                risk="low",
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
            )
            assert result.task_class == tc

    def test_all_risk_levels(self) -> None:
        for rl in RiskLevel:
            result = ClassifierResult(
                task_class="knowledge_reasoning",
                risk=rl.value,
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
            )
            assert result.risk == rl

    def test_all_sensitivities(self) -> None:
        for s in Sensitivity:
            result = ClassifierResult(
                task_class="knowledge_reasoning",
                risk="low",
                sensitivity=s.value,
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
            )
            assert result.sensitivity == s

    def test_invalid_task_class(self) -> None:
        with pytest.raises(ValidationError):
            ClassifierResult(
                task_class="invalid_class",
                risk="low",
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
            )

    def test_invalid_risk(self) -> None:
        with pytest.raises(ValidationError):
            ClassifierResult(
                task_class="knowledge_reasoning",
                risk="extreme",
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
            )

    def test_invalid_sensitivity(self) -> None:
        with pytest.raises(ValidationError):
            ClassifierResult(
                task_class="knowledge_reasoning",
                risk="low",
                sensitivity="top_secret",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
            )

    def test_confidence_out_of_range_high(self) -> None:
        with pytest.raises(ValidationError):
            ClassifierResult(
                task_class="knowledge_reasoning",
                risk="low",
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=1.5,
            )

    def test_confidence_out_of_range_low(self) -> None:
        with pytest.raises(ValidationError):
            ClassifierResult(
                task_class="knowledge_reasoning",
                risk="low",
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=-0.1,
            )

    def test_confidence_boundary(self) -> None:
        result = ClassifierResult(
            task_class="knowledge_reasoning",
            risk="low",
            sensitivity="public",
            requires_tools=False,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=0.0,
        )
        assert result.confidence == 0.0

        result = ClassifierResult(
            task_class="knowledge_reasoning",
            risk="low",
            sensitivity="public",
            requires_tools=False,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=1.0,
        )
        assert result.confidence == 1.0

    def test_prompt_injection_attempt(self) -> None:
        """Task text cannot override policy - unknown fields are rejected."""
        with pytest.raises(ValidationError):
            ClassifierResult(
                task_class="software_engineering",
                risk="low",
                sensitivity="public",
                requires_tools=False,
                requires_vision=False,
                long_context=False,
                destructive_potential=False,
                confidence=0.9,
                provider="openai",  # type: ignore[call-arg]
                model="gpt-4",  # type: ignore[call-arg]
            )


class TestRouteSelection:
    def test_valid(self) -> None:
        route = RouteSelection(
            primary_alias="glm",
            escalation_alias="opus",
            reason_code=ReasonCode.CLASSIFIER,
            task_class=TaskClass.SOFTWARE_ENGINEERING,
            risk=RiskLevel.MODERATE,
            sensitivity=Sensitivity.INTERNAL,
            confidence=0.85,
        )
        assert route.primary_alias == "glm"
        assert route.escalation_alias == "opus"

    def test_high_risk_route(self) -> None:
        route = RouteSelection(
            primary_alias="opus",
            escalation_alias="opus",
            reason_code=ReasonCode.ESCALATION_HIGH_RISK,
            task_class=TaskClass.SECURITY_ENGINEERING,
            risk=RiskLevel.CRITICAL,
            sensitivity=Sensitivity.RESTRICTED,
            confidence=0.95,
        )
        assert route.primary_alias == route.escalation_alias


class TestModelPin:
    def test_valid(self) -> None:
        pin = ModelPin(
            alias="opus",
            concrete_model="anthropic/claude-opus-5",
        )
        assert pin.concrete_model == "anthropic/claude-opus-5"
        assert pin.provider == "openrouter"


class TestEscalationDecision:
    def test_no_escalation(self) -> None:
        d = EscalationDecision(should_escalate=False)
        assert not d.should_escalate
        assert d.reason_code is None

    def test_escalation_with_reason(self) -> None:
        d = EscalationDecision(
            should_escalate=True,
            reason_code=ReasonCode.ESCALATION_TOOL_LOOP,
            detail="3 failed tool calls",
        )
        assert d.should_escalate
        assert d.reason_code == ReasonCode.ESCALATION_TOOL_LOOP


class TestTelemetryEvent:
    def test_content_free(self) -> None:
        """Telemetry events must not contain prompts or model content."""
        event = TelemetryEvent(
            event_type=EventType.CLASSIFICATION,
            task_id="task-123",
            primary_alias="glm",
            detail="class=software_engineering conf=0.85",
        )
        assert "prompt" not in event.model_dump()
        assert "content" not in event.detail
        assert "password" not in event.detail
