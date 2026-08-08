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
# optional — local BERT classifier for per-task model selection:
pip install 'hermes-smart-router[classifier]'
```

Without the `[classifier]` extra the router still works, but every request
falls back to the `luna` alias (no per-task classification).

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

The BERT classifier model must be trained separately, or download the
pre-built weights from the
[GitHub Release](https://github.com/KevinOBytes/hermes-smart-router/releases/tag/classifier-v1.0.0):

```bash
# Download the archive
curl -L -o classifier.tar.zst \
  https://github.com/KevinOBytes/hermes-smart-router/releases/download/classifier-v1.0.0/hermes-smart-router-classifier-v1.0.0-darwin-arm64.tar.zst

# Verify the SHA-256
shasum -a 256 classifier.tar.zst
# 22916dce44764f08b5eeb9a70beb3a991c4ef9ce33292ed4f4fa83b6e3c0fcf7  classifier.tar.zst

# Extract to the configured model path (default ~/.smart-router-proxy/classifier-model)
mkdir -p ~/.smart-router-proxy/classifier-model
zstd -dc classifier.tar.zst | tar -xf - -C ~/.smart-router-proxy/classifier-model
```

See [prompt-classifier](https://github.com/KevinOBytes/prompt-classifier) for training instructions if you'd rather build your own.

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
| structured_simple | luna | glm |
| agentic_execution | deepseek_flash | sol |
| software_engineering | glm | opus |
| security_engineering | sol | fable |
| knowledge_reasoning | glm | kimi_k3 |
| writing_communication | sonnet | opus |
| computer_use | sonnet | opus |
| visual_frontend | kimi_k3 | opus |

## Alias Mappings

| Alias | OpenRouter Slug |
|---|---|
| luna | openai/gpt-5.6-luna |
| deepseek_flash | deepseek/deepseek-v4-flash-0731 |
| glm | z-ai/glm-5.2 |
| sol | openai/gpt-5.6-sol |
| sonnet | anthropic/claude-sonnet-5 |
| opus | anthropic/claude-opus-5 |
| fable | anthropic/claude-fable-5 |
| kimi_k3 | moonshotai/kimi-k3 |
| kimi_code | moonshotai/kimi-k2.7-code |

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
