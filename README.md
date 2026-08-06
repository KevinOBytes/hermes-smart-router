# Hermes Smart Router

A task-aware model routing plugin for [Hermes Agent](https://hermes-agent.nousresearch.com). Exposes one stable virtual model — `tko/smart-router` — that classifies each task using a local BERT classifier (distilbert + MLX), selects the optimal OpenRouter model, and pins it through the complete tool loop with automatic escalation when the selected model is failing.

No single model provides the best combination of terminal execution, repository coding, security reasoning, writing, computer use, visual production, latency, and price. Manually changing models is disruptive and unreliable for persistent agents.

The smart-router classifies each user request before routing — no manual model selection, no mid-stream switching, no wasted tokens on the wrong model for the job.

## Architecture

```
User Request
     │
     ▼
┌────────────┐
│  BERT      │────>  TaskClass + Confidence
│  Classifier│      (distilbert + MLX, ~5-15ms)
└────────────┘
     │
     ▼
┌────────────┐     ┌──────────────┐
│  Route     │────>│  OpenRouter  │
│  Policy    │     │  Transport   │
└────────────┘     └──────────────┘
     │
     ▼
Concrete Model (pinned per session)
```

## How It Works

1. **BERT classifier** (local distilbert + MLX, ~66M params, ~5-15ms per call) classifies the user request into one of eight categories with risk, sensitivity, and confidence scores.
2. **Route policy** maps the classification to an alias (e.g., `opus` for security, `sonnet` for coding, `kimi_k3` for structured).
3. **Transport** resolves the alias to a concrete OpenRouter slug and proxies the request.
4. **Session pinning** caches the route per-session so subsequent requests use the same model without re-classifying.
5. **Escalation** detects repeated tool failures and upgrades to a stronger model automatically.

If the classifier is unavailable or confidence is below threshold, the request falls back to the `luna` alias (`openai/gpt-5.6-luna`).

## Installation

```bash
pip install hermes-smart-router
```

Then add the plugin to your Hermes config.yaml:

```yaml
providers:
  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: https://openrouter.ai/api/v1

model_providers:
  smart-router:
    module: hermes_smart_router

models:
  default: tko/smart-router
```

The BERT classifier model must be trained separately. See [prompt-classifier](https://github.com/KevinOBytes/prompt-classifier) for training instructions.

> **Prefer the standalone proxy.** For a persistent, always-on router that
> serves any OpenAI-compatible client, use
> [smart-router-proxy](https://github.com/KevinOBytes/smart-router-proxy)
> instead — it exposes the same routing logic as an HTTP service (e.g. as a
> Hermes custom provider at `http://127.0.0.1:8199/v1`) and shares this
> repo's task taxonomy. This package is the in-process model-provider plugin
> variant for embedding routing directly inside Hermes.

## Configuration

```yaml
# ~/.hermes/config.yaml
smart_router:
  mode: active            # active | shadow | fixed
  task_ttl_seconds: 3600  # session pin TTL
  fixed_alias: sonnet     # only used in fixed mode
```

The plugin reads from the `smart_router` section of Hermes config.yaml. Configuration is live-reloaded on every request by checking file mtime, so no restart is needed for changes.

## Route Table

| Task Class | Primary Alias | Escalation Alias |
|---|---|---|
| structured_simple | kimi_k3 | sonnet |
| agentic_execution | sonnet | opus |
| software_engineering | sonnet | opus |
| security_engineering | opus | opus |
| writing_communication | fable | opus |
| visual_frontend | sonnet | opus |
| computer_use | sonnet | sonnet |
| research_analysis | deepseek_flash | opus |

## Alias Mappings

| Alias | OpenRouter Slug |
|---|---|
| luna | openai/gpt-5.6-luna |
| sonnet | anthropic/claude-sonnet-5 |
| opus | anthropic/claude-opus-5 |
| fable | anthropic/claude-sonnet-5-fable |
| deepseek_flash | deepseek/deepseek-v4-flash-0731 |
| kimi_k3 | moonshot/kimi-k3 |
| sol | openai/gpt-5.6-sol |
| glm | z-ai/glm-5.2 |

## Health Checks

```bash
hermes health
```

Checks:
- Configuration validity
- State store (SQLite)
- BERT classifier availability
- OpenRouter authentication
- Alias mapping validity

## Development

```bash
git clone https://github.com/KevinOBytes/hermes-smart-router.git
cd hermes-smart-router
pip install -e ".[dev]"
pytest
```

## Security

- **Classifier cannot select models**: returns only enum-constrained classification fields.
- **No prompt injection surface**: BERT model is a frozen distilbert backbone with a trained classification head — no prompt, no injection vector.
- **No secrets in transit**: classification is local, no data leaves your machine.
- **Session isolation**: route selections are per-conversation, never shared between contexts.

## License

Apache 2.0
