"""Tests for redaction utilities."""

from __future__ import annotations

from hermes_smart_router.redaction import (
    redact_credentials,
    redact_headers,
    redact_url,
)


class TestRedaction:
    def test_redact_bearer_token(self) -> None:
        text = 'Authorization: Bearer sk-1234567890abcdef'
        result = redact_credentials(text)
        assert 'sk-1234567890abcdef' not in result
        assert '[REDACTED]' in result

    def test_redact_api_key(self) -> None:
        text = 'api_key = "sk-proj-1234567890"'
        result = redact_credentials(text)
        assert 'sk-proj-1234567890' not in result
        assert '[REDACTED]' in result

    def test_redact_multiple_credentials(self) -> None:
        text = (
            'Authorization: Bearer token123\n'
            'X-API-Key: my-secret-key\n'
            'Cookie: session=abc123'
        )
        result = redact_credentials(text)
        assert 'token123' not in result
        assert 'my-secret-key' not in result
        assert 'abc123' not in result
        assert result.count('[REDACTED]') >= 3

    def test_no_false_positives(self) -> None:
        text = 'This is normal text with no credentials'
        result = redact_credentials(text)
        assert result == text

    def test_redact_headers(self) -> None:
        headers = {
            'Authorization': 'Bearer token123',
            'Content-Type': 'application/json',
            'X-API-Key': 'secret-key',
        }
        result = redact_headers(headers)
        assert result['Authorization'] == '[REDACTED]'
        assert result['Content-Type'] == 'application/json'
        assert result['X-API-Key'] == '[REDACTED]'

    def test_redact_url(self) -> None:
        url = 'https://user:password@api.example.com/v1/chat'
        result = redact_url(url)
        assert 'user:password' not in result
        assert '[REDACTED]:[REDACTED]' in result

    def test_redact_url_no_creds(self) -> None:
        url = 'https://api.example.com/v1/chat'
        result = redact_url(url)
        assert result == url

    def test_redact_empty_string(self) -> None:
        result = redact_credentials('')
        assert result == ''
