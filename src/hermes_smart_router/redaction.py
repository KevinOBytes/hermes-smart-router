"""Secret and sensitive-content redaction utilities."""

from __future__ import annotations

import re
from re import Pattern

# Match the full value after the colon — entire line content
BEARER_PATTERN: Pattern[str] = re.compile(
    r"(?i)(Authorization|Bearer)\s*:\s*.*"
)
API_KEY_PATTERN: Pattern[str] = re.compile(
    r"(?i)(api[_-]?key|apikey|api_token|token)\s*[:=]\s*['\"]?\S+['\"]?"
)
COOKIE_PATTERN: Pattern[str] = re.compile(
    r"(?i)(cookie|set-cookie)\s*:\s*.*"
)

_DEFAULT_PATTERNS: list[Pattern[str]] = [
    BEARER_PATTERN,
    API_KEY_PATTERN,
    COOKIE_PATTERN,
]


def redact_credentials(
    text: str,
    additional_patterns: list[Pattern[str]] | None = None,
    replacement: str = "[REDACTED]",
) -> str:
    patterns = _DEFAULT_PATTERNS + (additional_patterns or [])
    result = text
    for pattern in patterns:
        result = pattern.sub(
            lambda m: m.group(0).split(":")[0] + ": " + replacement
            if ":" in m.group(0)
            else replacement,
            result,
        )
    return result


def redact_headers(
    headers: dict[str, str],
    sensitive_keys: set[str] | None = None,
) -> dict[str, str]:
    if sensitive_keys is None:
        sensitive_keys = {"authorization", "x-api-key", "cookie", "set-cookie"}
    result: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in sensitive_keys:
            result[key] = "[REDACTED]"
        else:
            result[key] = value
    return result


def redact_url(url: str) -> str:
    return re.sub(
        r"(://)([^:]+):([^@]+)@",
        r"\1[REDACTED]:[REDACTED]@",
        url,
    )
