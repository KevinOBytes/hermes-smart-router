"""Tests for deterministic classification rules."""

from __future__ import annotations

from hermes_smart_router.deterministic import classify_deterministic
from hermes_smart_router.models import TaskClass


class TestDeterministicClassification:
    """Tests for the deterministic classifier."""

    def test_security_vulnerability_research(self) -> None:
        result = classify_deterministic(
            "Analyze this CVE-2026-12345 exploit and write a detection rule"
        )
        assert result is not None
        assert result.task_class == TaskClass.SECURITY_ENGINEERING
        assert result.confidence >= 0.85

    def test_security_malware_analysis(self) -> None:
        result = classify_deterministic(
            "Reverse engineer this malware sample and identify the C2 infrastructure"
        )
        assert result is not None
        assert result.task_class == TaskClass.SECURITY_ENGINEERING

    def test_security_detection_engineering(self) -> None:
        result = classify_deterministic(
            "Write a YARA rule to detect this backdoor and a Sigma rule for the network IOC"
        )
        assert result is not None
        assert result.task_class == TaskClass.SECURITY_ENGINEERING

    def test_software_engineering_refactor(self) -> None:
        """'Refactor the authentication module' is a software engineering task."""
        result = classify_deterministic(
            "Refactor the authentication module to use async/await patterns"
        )
        assert result is not None
        assert result.task_class == TaskClass.SOFTWARE_ENGINEERING

    def test_software_engineering_build_fix(self) -> None:
        result = classify_deterministic(
            "Fix the failing test suite and update the CI pipeline configuration"
        )
        assert result is not None
        assert result.task_class == TaskClass.SOFTWARE_ENGINEERING

    def test_software_engineering_create_module(self) -> None:
        result = classify_deterministic(
            "Create a new API endpoint for user management"
        )
        assert result is not None
        assert result.task_class == TaskClass.SOFTWARE_ENGINEERING

    def test_agentic_execution_deploy(self) -> None:
        result = classify_deterministic(
            "Deploy the updated container to production using kubectl"
        )
        assert result is not None
        assert result.task_class == TaskClass.AGENTIC_EXECUTION

    def test_agentic_execution_install(self) -> None:
        result = classify_deterministic(
            "Run this bash script to install dependencies and configure the service"
        )
        assert result is not None
        assert result.task_class == TaskClass.AGENTIC_EXECUTION

    def test_agentic_execution_run_command(self) -> None:
        result = classify_deterministic(
            "Run a shell command to check disk usage"
        )
        assert result is not None
        assert result.task_class == TaskClass.AGENTIC_EXECUTION

    def test_writing_communication_report(self) -> None:
        result = classify_deterministic(
            "Write a security incident report for the board of directors"
        )
        assert result is not None
        assert result.task_class == TaskClass.WRITING_COMMUNICATION

    def test_writing_communication_doc(self) -> None:
        result = classify_deterministic(
            "Draft a README for the new API with usage examples"
        )
        assert result is not None
        assert result.task_class == TaskClass.WRITING_COMMUNICATION

    def test_writing_communication_email(self) -> None:
        result = classify_deterministic(
            "Draft an email to the team about the deployment schedule"
        )
        assert result is not None
        assert result.task_class == TaskClass.WRITING_COMMUNICATION

    def test_visual_frontend_ui(self) -> None:
        """'Build a React dashboard with Chart.js' is software engineering
        (building a component), not visual frontend."""
        result = classify_deterministic(
            "Build a React dashboard with Chart.js visualizations for the metrics"
        )
        assert result is not None
        assert result.task_class == TaskClass.SOFTWARE_ENGINEERING

    def test_visual_frontend_diagram(self) -> None:
        result = classify_deterministic(
            "Create an SVG network topology diagram showing the DMZ architecture"
        )
        assert result is not None
        assert result.task_class == TaskClass.VISUAL_FRONTEND

    def test_visual_frontend_css(self) -> None:
        result = classify_deterministic(
            "Design a CSS layout for the landing page"
        )
        assert result is not None
        assert result.task_class == TaskClass.VISUAL_FRONTEND

    def test_structured_simple_extract(self) -> None:
        result = classify_deterministic(
            "Extract all email addresses from this CSV and validate the format"
        )
        assert result is not None
        assert result.task_class == TaskClass.STRUCTURED_SIMPLE

    def test_structured_simple_transform(self) -> None:
        result = classify_deterministic(
            "Convert this YAML configuration to JSON format"
        )
        assert result is not None
        assert result.task_class == TaskClass.STRUCTURED_SIMPLE

    def test_structured_simple_parse(self) -> None:
        result = classify_deterministic(
            "Parse this JSON data and extract the relevant fields"
        )
        assert result is not None
        assert result.task_class == TaskClass.STRUCTURED_SIMPLE

    def test_computer_use_with_tools(self) -> None:
        result = classify_deterministic(
            "Open the browser and navigate to the admin panel",
            tool_names=["computer_use", "web_search"],
        )
        assert result is not None
        assert result.task_class == TaskClass.COMPUTER_USE

    def test_ambiguous_defer_to_gemma(self) -> None:
        """Ambiguous requests should return None to defer to Gemma."""
        result = classify_deterministic(
            "What do you think about the future of AI?"
        )
        assert result is None

    def test_short_ambiguous(self) -> None:
        result = classify_deterministic("Hello")
        assert result is None

    def test_empty_request(self) -> None:
        result = classify_deterministic("")
        assert result is None

    def test_whitespace_only(self) -> None:
        result = classify_deterministic("   ")
        assert result is None

    def test_security_takes_priority_over_software(self) -> None:
        """Security patterns should take priority over general software patterns."""
        result = classify_deterministic(
            "Fix the authentication bypass vulnerability in the login module"
        )
        assert result is not None
        assert result.task_class == TaskClass.SECURITY_ENGINEERING

    def test_destructive_security(self) -> None:
        result = classify_deterministic(
            "Analyze this rootkit sample and build a detection signature"
        )
        assert result is not None
        assert result.task_class == TaskClass.SECURITY_ENGINEERING
        assert result.destructive_potential is True

    def test_destructive_agentic(self) -> None:
        result = classify_deterministic(
            "Run rm -rf on the temp directory and format the USB drive"
        )
        assert result is not None
        assert result.task_class == TaskClass.AGENTIC_EXECUTION
        assert result.destructive_potential is True
