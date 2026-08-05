"""Hermes Agent model-provider plugin registration.

This module registers the smart-router as a Hermes model-provider plugin.
It presents one stable virtual model, tko/smart-router, that routes tasks
to concrete OpenRouter models based on local Gemma classification.

The plugin is discovered automatically by Hermes when installed under
$HERMES_HOME/plugins/model-providers/smart-router/ or when the
hermes-smart-router package is installed and the entry point is registered.
"""

from __future__ import annotations

import logging

from providers import register_provider  # type: ignore[import-not-found]
from providers.base import ProviderProfile  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

# The smart-router presents as a single virtual model.
# All routing logic lives in the provider module.
SMART_ROUTER_PROFILE = ProviderProfile(
    name="smart-router",
    aliases=("tko/smart-router",),
    display_name="TKO Smart Router",
    description="Task-aware model router: classifies tasks via local Gemma, routes to optimal OpenRouter model",
    signup_url="",
    env_vars=("OPENROUTER_API_KEY",),
    base_url="https://openrouter.ai/api/v1",
    supports_health_check=True,
    fallback_models=("tko/smart-router",),
)

register_provider(SMART_ROUTER_PROFILE)
