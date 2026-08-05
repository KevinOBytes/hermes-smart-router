"""Hermes Agent model-provider plugin registration.

This module registers the smart-router as a Hermes model-provider plugin.
It presents one stable virtual model, tko/smart-router, that routes tasks
to concrete OpenRouter models based on local BERT classification.

The plugin is discovered automatically by Hermes when installed under
$HERMES_HOME/plugins/model-providers/smart-router/ or when the
hermes-smart-router package is installed and the entry point is registered.

Active routing works WITHOUT modifying Hermes core: the profile's
``build_api_kwargs_extras`` hook returns a top-level ``model`` override,
which the ChatCompletions transport merges into the request after the
virtual model name — swapping in the routed concrete OpenRouter slug.
"""

from __future__ import annotations

import logging

from providers import register_provider  # type: ignore[import-not-found]

from hermes_smart_router.hermes_profile import SmartRouterProfile

logger = logging.getLogger(__name__)

# The smart-router presents as a single virtual model.
# Routing happens per-request inside SmartRouterProfile.
SMART_ROUTER_PROFILE = SmartRouterProfile(
    name="smart-router",
    aliases=("tko/smart-router",),
    display_name="TKO Smart Router",
    description="Task-aware model router: classifies tasks via local BERT, routes to optimal OpenRouter model",
    signup_url="https://openrouter.ai/keys",
    env_vars=("OPENROUTER_API_KEY",),
    base_url="https://openrouter.ai/api/v1",
    models_url="https://openrouter.ai/api/v1/models",
    supports_health_check=True,
    supports_vision=True,
    # Auxiliary tasks (title generation, compression, etc.) must NOT use the
    # virtual model name — route them to the cheap structured-output model.
    default_aux_model="openai/gpt-5.6-luna",
    fallback_models=("tko/smart-router",),
)

register_provider(SMART_ROUTER_PROFILE)
