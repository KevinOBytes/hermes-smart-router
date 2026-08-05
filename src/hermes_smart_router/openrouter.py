"""OpenRouter transport layer.

Preserves all Hermes-required OpenAI-compatible semantics including
streaming, tool calls, structured output, and multimodal messages.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from hermes_smart_router.config import OpenRouterConfig

logger = logging.getLogger(__name__)


class OpenRouterTransport:
    """HTTPX-based transport for OpenRouter API calls.

    Handles streaming and non-streaming requests, tool definitions,
    structured output, and response parsing.
    """

    def __init__(
        self,
        config: OpenRouterConfig,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._base_url = config.base_url.rstrip("/")
        self._http = http_client or httpx.AsyncClient(timeout=60)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a non-streaming chat completion request.

        Args:
            model: The concrete model slug.
            messages: The message list.
            **kwargs: Additional parameters (temperature, max_tokens, tools, etc.).

        Returns:
            The parsed response dict.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        # Add provider preferences if configured
        if self._config.provider_preferences:
            body.setdefault("extra_body", {})
            body["extra_body"]["provider"] = self._config.provider_preferences

        try:
            response = await self._http.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=body,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            logger.error(
                "OpenRouter HTTP %s: %s",
                e.response.status_code,
                e.response.text[:500],
            )
            raise
        except httpx.TimeoutException:
            logger.error("OpenRouter timeout for model %s", model)
            raise

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request.

        Args:
            model: The concrete model slug.
            messages: The message list.
            **kwargs: Additional parameters.

        Yields:
            Parsed SSE event dicts.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        if self._config.provider_preferences:
            body.setdefault("extra_body", {})
            body["extra_body"]["provider"] = self._config.provider_preferences

        async with self._http.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    def extract_model_id(self, response: dict[str, Any]) -> str | None:
        """Extract the actual model ID from an OpenRouter response.

        OpenRouter includes the model in the response under 'model'.
        """
        return response.get("model")

    def check_identity(
        self,
        expected_model: str,
        actual_model: str | None,
    ) -> bool:
        """Check if the actual response model matches expectations.

        When strict_identity is enabled, reject unexpected models.
        """
        if not self._config.strict_identity:
            return True
        if actual_model is None:
            return True
        return actual_model == expected_model

    async def close(self) -> None:
        await self._http.aclose()
