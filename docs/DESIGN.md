Objective

Build a production-ready, standalone Hermes model-provider plugin that presents one stable virtual model, tko/smart-router, while selecting and escalating concrete OpenRouter models according to task type and deterministic runtime evidence. A local BERT classifier (distilbert + MLX) classifies new tasks. Hermes users never manually change models.

Architecture

The package registers a Hermes model-provider called smart-router. On the first inference request for a new task, it asks the local BERT classifier (distilbert + MLX, `~/.smart-router-proxy/classifier-model`) for a classification. The policy maps that classification to a logical model alias and resolves the alias to a configured OpenRouter model identifier.

The chosen model remains pinned for the complete Hermes tool loop. On subsequent calls, the plugin inspects tool results and state for deterministic escalation signals. Escalation changes the pinned model while preserving the complete message, tool-call, and tool-result history.

The plugin owns model selection. OpenRouter may perform provider-level routing for the selected model, subject to the configured provider allowlist, but OpenRouter automatic cross-model routing is disabled.

Task classes and default routes

Task class

Primary alias

Escalation alias

structured_simple

luna

glm

agentic_execution

deepseek_flash

sol

software_engineering

glm

opus

security_engineering

sol

fable

knowledge_reasoning

glm

kimi_k3

writing_communication

sonnet

opus

computer_use

sonnet

opus

visual_frontend

kimi_k3

opus

Logical aliases must be configuration values. No application module may depend directly on a concrete OpenRouter model slug.

Default logical aliases:

Alias

Intended model

luna

OpenRouter-provided GPT-5.6 Luna

deepseek_flash

DeepSeek V4 Flash 0731 or current configured equivalent

glm

GLM-5.2

sol

GPT-5.6 Sol

sonnet

Claude Sonnet 5

opus

Claude Opus 5

fable

Claude Fable 5; fall back to opus only when explicitly configured

kimi_k3

Kimi K3

kimi_code

Kimi K2.7 Code, available for future worker-specific policy

The installer must not assume those slugs exist. Startup validation queries the OpenRouter model catalog and reports missing or unavailable mappings as an unhealthy configuration.

Classification contract

The BERT classifier returns a `ClassifierResult` with these typed fields:

{
  "task_class": "software_engineering",
  "risk": "high",
  "sensitivity": "confidential",
  "requires_tools": true,
  "requires_vision": false,
  "long_context": true,
  "destructive_potential": false,
  "confidence": 0.93
}

The plugin validates the response using an enum-constrained typed schema. The classifier never returns a provider or model name. Prompt text cannot select a provider or bypass policy.

Classification uses the initial root user request and minimal task metadata, not the entire accumulated transcript. The default classifier confidence threshold is 0.45.

If the classifier is unavailable, returns None (below confidence threshold), or the label has no TaskClass mapping, route to the OpenRouter-provided luna alias. Record the failure reason without recording prompt content.

Deterministic policy

Rules are applied on top of the BERT classification:

Obvious structured extraction requests may be deterministically classified as structured_simple.

Explicit shell execution or deployment work maps to agentic_execution.

Repository modification, debugging, tests, or build work maps to software_engineering unless security purpose dominates.

Vulnerability research, malware analysis, detection engineering, exploit analysis, authentication or authorization review maps to security_engineering.

GUI-control tools force computer_use.

Image-led interface or presentation production maps to visual_frontend.

High-risk tasks may start on the escalation alias without first using the primary.

Destructive or irreversible actions require the configured high-risk model and must not bypass Hermes approval controls.

Sensitivity-based provider restrictions are implemented, documented, and tested but disabled by default. When disabled, confidential or restricted classifications routed to DeepSeek, Z.ai, or Moonshot emit a structured warning. When enabled, the configured allowlist overrides normal routing.

Escalation

Escalation uses observable evidence, not the worker model's self-reported confidence. Default triggers:

Two equivalent failed tool calls.

Two nonzero command exits attributable to the same attempted operation.

Two repeated test or build failures without material progress.

Two response-schema validation failures.

Detected tool-call loop.

Explicit task validator failure.

Context overflow that requires a larger configured context route.

High-risk or destructive policy selecting the escalation model at task start.

Infrastructure fallback and capability escalation remain separate. OpenRouter provider failover may handle rate limits or provider outages for the same concrete model. A cross-model change is made only by this plugin and is recorded as an escalation event.

Task identity and stickiness

The plugin must use a Hermes-provided session/task identifier where available. If the active Hermes provider interface lacks a task identifier, implement an isolated adapter that derives a stable task key from documented session metadata. Do not base identity solely on prompt hashing.

The route remains pinned until task completion, explicit cancellation, new-task detection, session termination, or TTL expiry. SQLite is the default state backend. The state interface must permit a future Redis implementation without changing routing code.

Security and privacy

Bind Ollama to loopback or an explicitly configured Unix/local endpoint.

Reject unknown classification values and model aliases.

Never allow classifier output to supply arbitrary model identifiers.

Do not log prompts, tool arguments, credentials, tool results, or generated text by default.

Redact authorization headers and secrets in all exception paths.

Support OpenRouter ZDR/provider allowlist configuration.

Verify the actual response model/provider when OpenRouter exposes them.

Fail closed on unexpected model identity when strict identity enforcement is enabled.

Preserve Hermes destructive-operation approval behavior.

Keep project plugins disabled unless Hermes explicitly enables them.

Operating modes

active: route and escalate normally.

shadow: call the configured baseline model while recording the route that would have been selected.

fixed: always call one configured alias while retaining metrics and validation.

New installations default to shadow mode.

Observability

Produce structured events for classification, route selection, classifier fallback, provider request, provider failure, escalation, task completion, and policy warnings. Include task/session correlation, aliases, concrete model identifiers, latency, token usage, estimated cost when supplied, and reasons. Exclude content by default.

Provide health checks for configuration validity, BERT classifier availability, OpenRouter authentication, model-catalog resolution, database access, and shadow-baseline validity.

Testing and acceptance

The package requires unit tests for schemas, deterministic rules, classifier failures, route resolution, sensitivity policy, stickiness, escalation, redaction, and catalog validation. Integration tests use mocked Ollama and OpenRouter servers. An optional live test requires explicit environment variables and is skipped by default.

Acceptance requires:

Hermes is configured with only tko/smart-router as its main model.

Eight representative tasks select the expected primary aliases.

All calls in an uneventful tool loop remain pinned.

Defined failure sequences escalate exactly once and preserve full history.

Ollama failure routes through the luna alias.

Shadow mode never changes the baseline model.

Unknown or missing model mappings make health status unhealthy.

Sensitive-content warnings contain metadata but no prompt content.

Streaming, non-streaming, tools, tool choice, structured output, and usage accounting remain compatible with Hermes and OpenRouter.

The full test suite, linting, and type checking pass.
