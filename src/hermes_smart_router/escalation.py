"""Runtime escalation evidence detection.

Uses deterministic evidence from tool results and stored counters.
Never escalates based on the worker model saying it is uncertain.
"""

from __future__ import annotations

import re
from typing import Any

from hermes_smart_router.models import EscalationDecision, ReasonCode
from hermes_smart_router.state import TaskState


class EscalationDetector:
    """Detects when a task should be escalated to a more capable model.

    Uses observable evidence from tool results and stored counters.
    """

    # Thresholds for escalation triggers
    MAX_FAILED_TOOL_CALLS = 2
    MAX_FAILED_COMMANDS = 2
    MAX_BUILD_FAILURES = 2
    MAX_SCHEMA_FAILURES = 2
    MAX_TOOL_LOOP_DEPTH = 5

    def __init__(self, state: TaskState) -> None:
        self._state = state

    async def check(
        self,
        task_id: str,
        recent_tool_results: list[dict[str, Any]],
    ) -> EscalationDecision:
        """Check if the current task should be escalated.

        Args:
            task_id: The task identifier.
            recent_tool_results: Recent tool call results for analysis.

        Returns:
            EscalationDecision with should_escalate flag and reason.
        """
        # Check stored counters first
        failed_tools = self._state.get_counter(task_id, "failed_tool_calls")
        failed_commands = self._state.get_counter(task_id, "failed_commands")
        build_failures = self._state.get_counter(task_id, "build_failures")
        schema_failures = self._state.get_counter(task_id, "schema_failures")

        if failed_tools >= self.MAX_FAILED_TOOL_CALLS:
            return EscalationDecision(
                should_escalate=True,
                reason_code=ReasonCode.ESCALATION_TOOL_LOOP,
                detail=f"Failed tool calls: {failed_tools}",
            )

        if failed_commands >= self.MAX_FAILED_COMMANDS:
            return EscalationDecision(
                should_escalate=True,
                reason_code=ReasonCode.ESCALATION_COMMAND_FAILURE,
                detail=f"Failed commands: {failed_commands}",
            )

        if build_failures >= self.MAX_BUILD_FAILURES:
            return EscalationDecision(
                should_escalate=True,
                reason_code=ReasonCode.ESCALATION_BUILD_FAILURE,
                detail=f"Build failures: {build_failures}",
            )

        if schema_failures >= self.MAX_SCHEMA_FAILURES:
            return EscalationDecision(
                should_escalate=True,
                reason_code=ReasonCode.ESCALATION_SCHEMA_FAILURE,
                detail=f"Schema failures: {schema_failures}",
            )

        # Check recent tool results for loops
        if self._detect_tool_loop(recent_tool_results):
            return EscalationDecision(
                should_escalate=True,
                reason_code=ReasonCode.ESCALATION_TOOL_LOOP,
                detail="Detected repeated equivalent tool calls",
            )

        return EscalationDecision(should_escalate=False)

    def record_tool_result(
        self,
        task_id: str,
        tool_name: str,
        exit_code: int | None,
        output: str,
    ) -> None:
        """Record a tool result and update escalation counters.

        Args:
            task_id: The task identifier.
            tool_name: Name of the tool that was called.
            exit_code: Exit code (None for non-command tools).
            output: Tool output text.
        """
        # Failed tool call (tool returned error or empty result)
        if exit_code is not None and exit_code != 0:
            self._state.increment_counter(task_id, "failed_commands")

        # Build/test failure patterns
        if re.search(r"\b(?:FAILED|ERROR|failure|failed|traceback|Error)\b", output, re.I):
            if re.search(r"\b(?:build|test|compile|pytest|ruff|mypy)\b", tool_name, re.I):
                self._state.increment_counter(task_id, "build_failures")

        # Schema validation failure
        if re.search(r"\b(?:validation.error|schema.error|invalid.response|parse.error)\b", output, re.I):
            self._state.increment_counter(task_id, "schema_failures")

    def record_failed_tool_call(self, task_id: str) -> None:
        """Record a failed tool call (tool returned error or no useful result)."""
        self._state.increment_counter(task_id, "failed_tool_calls")

    @staticmethod
    def _detect_tool_loop(
        recent_results: list[dict[str, Any]],
        max_similar: int = 3,
    ) -> bool:
        """Detect if the same or equivalent tool is being called repeatedly.

        Args:
            recent_results: Recent tool call results.
            max_similar: Maximum allowed similar calls before loop detection.

        Returns:
            True if a tool loop is detected.
        """
        if len(recent_results) < max_similar:
            return False

        # Get the last N tool names
        tool_names = [
            r.get("tool_name", r.get("name", ""))
            for r in recent_results[-max_similar:]
        ]

        # If the same tool appears max_similar times in a row
        if len(set(tool_names)) == 1 and len(tool_names) >= max_similar:
            return True

        return False
