"""Local Ollama/Gemma classifier for task routing.

Calls a local Ollama instance running gemma4:12b to classify tasks.
Returns a validated ClassifierResult or None on failure.
"""

from __future__ import annotations

import json
import logging

import httpx
from pydantic import ValidationError

from hermes_smart_router.config import OllamaConfig
from hermes_smart_router.models import ClassifierResult, RiskLevel, Sensitivity, TaskClass

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """Classify this request into exactly one category.

Categories: structured_simple, agentic_execution, software_engineering, security_engineering, knowledge_reasoning, writing_communication, computer_use, visual_frontend

Return ONLY valid JSON:
{"task_class": "<category>", "risk": "low|moderate|high|critical", "sensitivity": "public|internal|confidential|restricted", "confidence": 0.0-1.0}

Request:"""


class Classifier:
    """Local Ollama classifier using Gemma."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or OllamaConfig()
        self._http = http_client or httpx.AsyncClient(timeout=self._config.timeout_seconds)

    async def classify(self, user_request: str) -> ClassifierResult | None:
        """Classify a user request using local Gemma.

        Args:
            user_request: The initial user request text (not accumulated history).

        Returns:
            ClassifierResult on success, None on any failure (timeout, connection,
            invalid JSON, schema violation, or low confidence).
        """
        if not user_request or not user_request.strip():
            return None

        try:
            response = await self._http.post(
                f"{self._config.base_url.rstrip('/')}/api/generate",
                json={
                    "model": self._config.model,
                    "prompt": _CLASSIFICATION_PROMPT + "\n\n" + user_request,
                    "stream": False,
                    "options": {
                        "temperature": self._config.temperature,
                        "num_predict": self._config.max_output_tokens,
                    },
                },
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data.get("response", "")
        except httpx.TimeoutException:
            logger.warning("Classifier timeout after %ss", self._config.timeout_seconds)
            return None
        except httpx.ConnectError:
            logger.warning("Classifier connection refused to %s", self._config.base_url)
            return None
        except httpx.HTTPStatusError as e:
            logger.warning("Classifier HTTP error: %s", e)
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Classifier invalid response: %s", e)
            return None

        return self._parse_response(raw_text)

    def _parse_response(self, raw_text: str) -> ClassifierResult | None:
        """Parse and validate the raw Gemma response.

        Tries to extract JSON from the response, handling markdown code blocks.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Find the first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Classifier returned invalid JSON: %.200s", text)
            return None

        # Validate required fields
        required = {"task_class", "risk", "sensitivity", "confidence"}
        if not required.issubset(parsed.keys()):
            missing = required - set(parsed.keys())
            logger.warning("Classifier missing fields: %s", missing)
            return None

        # Validate enum values
        valid_classes = {e.value for e in TaskClass}
        if parsed.get("task_class") not in valid_classes:
            logger.warning("Classifier unknown task_class: %s", parsed.get("task_class"))
            return None

        valid_risks = {e.value for e in RiskLevel}
        if parsed.get("risk") not in valid_risks:
            logger.warning("Classifier unknown risk: %s", parsed.get("risk"))
            return None

        valid_sensitivities = {e.value for e in Sensitivity}
        if parsed.get("sensitivity") not in valid_sensitivities:
            logger.warning("Classifier unknown sensitivity: %s", parsed.get("sensitivity"))
            return None

        try:
            result = ClassifierResult(**parsed)
        except ValidationError as e:
            logger.warning("Classifier schema violation: %s", e)
            return None

        # Check confidence threshold
        if result.confidence < self._config.confidence_threshold:
            logger.info(
                "Classifier confidence %.2f below threshold %.2f",
                result.confidence,
                self._config.confidence_threshold,
            )
            return None

        return result

    async def close(self) -> None:
        await self._http.aclose()
