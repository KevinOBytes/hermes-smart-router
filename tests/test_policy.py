"""Tests for the route policy."""

from __future__ import annotations

from hermes_smart_router.config import SmartRouterConfig
from hermes_smart_router.models import (
    ClassifierResult,
    ReasonCode,
    RiskLevel,
    Sensitivity,
    TaskClass,
)
from hermes_smart_router.policy import RoutePolicy


def _make_result(
    task_class: str = "knowledge_reasoning",
    risk: str = "low",
    sensitivity: str = "public",
    confidence: float = 0.85,
    destructive: bool = False,
) -> ClassifierResult:
    return ClassifierResult(
        task_class=task_class,
        risk=risk,
        sensitivity=sensitivity,
        requires_tools=False,
        requires_vision=False,
        long_context=False,
        destructive_potential=destructive,
        confidence=confidence,
    )


class TestRoutePolicy:
    """Tests for the route policy."""

    def setup_method(self) -> None:
        self.config = SmartRouterConfig(mode="active")
        self.policy = RoutePolicy(self.config)

    def test_structured_simple_routes_to_luna(self) -> None:
        result = _make_result(task_class="structured_simple")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "luna"
        assert route.escalation_alias == "glm"

    def test_agentic_execution_routes_to_deepseek_flash(self) -> None:
        result = _make_result(task_class="agentic_execution")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "deepseek_flash"
        assert route.escalation_alias == "sol"

    def test_software_engineering_routes_to_glm(self) -> None:
        result = _make_result(task_class="software_engineering")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "glm"
        assert route.escalation_alias == "opus"

    def test_security_engineering_routes_to_sol(self) -> None:
        result = _make_result(task_class="security_engineering")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "sol"
        assert route.escalation_alias == "fable"

    def test_knowledge_reasoning_routes_to_glm(self) -> None:
        result = _make_result(task_class="knowledge_reasoning")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "glm"
        assert route.escalation_alias == "kimi_k3"

    def test_writing_communication_routes_to_sonnet(self) -> None:
        result = _make_result(task_class="writing_communication")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "sonnet"
        assert route.escalation_alias == "opus"

    def test_computer_use_routes_to_sonnet(self) -> None:
        result = _make_result(task_class="computer_use")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "sonnet"
        assert route.escalation_alias == "opus"

    def test_visual_frontend_routes_to_kimi_k3(self) -> None:
        result = _make_result(task_class="visual_frontend")
        route = self.policy.evaluate(result)
        assert route.primary_alias == "kimi_k3"
        assert route.escalation_alias == "opus"

    def test_classifier_fallback_routes_to_luna(self) -> None:
        route = self.policy.evaluate(None, reason_code=ReasonCode.CLASSIFIER_FALLBACK)
        assert route.primary_alias == "luna"
        assert route.reason_code == ReasonCode.CLASSIFIER_FALLBACK

    def test_high_risk_starts_on_escalation(self) -> None:
        result = _make_result(
            task_class="software_engineering",
            risk="critical",
        )
        route = self.policy.evaluate(result)
        assert route.primary_alias == "opus"  # escalation alias
        assert route.reason_code == ReasonCode.ESCALATION_HIGH_RISK

    def test_destructive_starts_on_escalation(self) -> None:
        result = _make_result(
            task_class="agentic_execution",
            destructive=True,
        )
        route = self.policy.evaluate(result)
        assert route.primary_alias == "sol"  # escalation alias
        assert route.reason_code == ReasonCode.ESCALATION_HIGH_RISK

    def test_fixed_mode_uses_fixed_alias(self) -> None:
        config = SmartRouterConfig(mode="fixed", fixed_alias="luna")
        policy = RoutePolicy(config)
        result = _make_result(task_class="software_engineering")
        route = policy.evaluate(result)
        assert route.primary_alias == "luna"
        assert route.reason_code == ReasonCode.FIXED_MODE

    def test_shadow_mode_records_but_uses_baseline(self) -> None:
        config = SmartRouterConfig(mode="shadow")
        policy = RoutePolicy(config)
        result = _make_result(task_class="software_engineering")
        route = policy.evaluate(result)
        assert route.primary_alias == "glm"
        assert route.reason_code == ReasonCode.CLASSIFIER

    def test_route_contains_classification_metadata(self) -> None:
        result = _make_result(
            task_class="security_engineering",
            risk="high",
            sensitivity="confidential",
            confidence=0.93,
        )
        route = self.policy.evaluate(result)
        assert route.task_class == TaskClass.SECURITY_ENGINEERING
        assert route.risk == RiskLevel.HIGH
        assert route.sensitivity == Sensitivity.CONFIDENTIAL
        assert route.confidence == 0.93
        assert route.classifier_raw is not None

    def test_get_escalation_for_alias(self) -> None:
        assert self.policy.get_escalation_for_alias("glm") == "opus"
        assert self.policy.get_escalation_for_alias("sonnet") == "opus"
        assert self.policy.get_escalation_for_alias("luna") == "glm"
        assert self.policy.get_escalation_for_alias("unknown") == "opus"
