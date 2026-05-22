# DeepSeek Provider

DeepSeek LLM provider for OptiResearch Agent.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | (required) | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API base URL |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Model name |
| `DEEPSEEK_THINKING_TYPE` | `enabled` | Thinking mode |
| `DEEPSEEK_REASONING_EFFORT` | `high` | Reasoning effort level |
| `DEEPSEEK_TIMEOUT` | `120` | Request timeout (seconds) |
| `DEEPSEEK_MAX_TOKENS` | (unset) | Max output tokens |
| `DEEPSEEK_TEMPERATURE` | `0.2` | Sampling temperature |

## Configuration

```bash
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

## Provider Selection

The provider registry auto-discovers available providers:

1. `OPTIRESEARCH_LLM_PROVIDER` env var override
2. DeepSeek (if `DEEPSEEK_API_KEY` is set)
3. OpenAI (if `OPENAI_API_KEY` is set)
4. Anthropic (if `ANTHROPIC_API_KEY` is set)
5. Local provider
6. Mock provider (always available fallback)

Explicit selection: `--provider deepseek` or `--llm-provider deepseek` in CLI.

## Error Codes

| Code | Description |
|------|-------------|
| `DEEPSEEK_API_KEY_MISSING` | API key not configured |
| `DEEPSEEK_HTTP_ERROR` | HTTP request failed |
| `DEEPSEEK_TIMEOUT` | Request timed out |
| `DEEPSEEK_RESPONSE_PARSE_ERROR` | Could not parse API response |

## Structured Output

The base `LLMProvider` class provides `structured_complete()` with automatic JSON repair:
1. Attempts to parse LLM output as JSON
2. Strips markdown code fences if present
3. On failure, sends a repair request to fix the JSON
4. Falls back to `LLMProviderError` if repair also fails

## Trace Redaction

API keys are redacted from all trace files:
- `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` values replaced with `[REDACTED]`
- Authorization headers stripped from logged data
- `config_summary()` never includes the API key value

## Rate Limiting

DeepSeek API has rate limits. The autonomous loop uses `max_iterations` to cap LLM calls. Fallback to rule-based StrategyEngine triggers on any provider error.
