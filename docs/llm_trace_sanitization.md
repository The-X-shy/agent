# LLM Trace Sanitization

All planner traces and LLM call records are sanitized to prevent credential leakage.

## What is Redacted

| Data | Redaction |
|------|-----------|
| `DEEPSEEK_API_KEY` value | `[REDACTED]` |
| `OPENAI_API_KEY` value | `[REDACTED]` |
| `ANTHROPIC_API_KEY` value | `[REDACTED]` |
| `Authorization` header | `[REDACTED]` |
| `x-api-key` header | `[REDACTED]` |
| `api_key` field in context | Removed entirely |
| `DEEPSEEK_API_KEY` field in context | Removed entirely |

## Sanitizer Functions

Located in `optiresearch/agents/planner_trace.py`:

- **`redact_api_keys(data)`** — Recursively replaces API key values in strings, dicts, and lists
- **`redact_authorization_headers(data)`** — Redacts Authorization and x-api-key keys from dicts
- **`redact_env_values(data)`** — Alias for `redact_api_keys`

## Where Sanitization is Applied

1. **`record_context()`** — Pops `api_key`/`DEEPSEEK_API_KEY` fields, then runs `redact_api_keys()` on the full context dict before writing `context_summary.json`
2. **`record_response()`** — Runs `redact_api_keys()` on the raw LLM response before writing `raw_response.json`
3. **`llm/audit.py`** — `redact_secrets()` and `record_llm_call()` both redact keys before writing to disk

## Verification

```bash
# Check no API key in trace files
grep -r "sk-" workspace/planner_traces/ && echo "LEAK DETECTED" || echo "Clean"

# Run sanitization tests
python -m pytest tests/test_planner_trace_sanitization.py -v
```

## Security Notes

- Raw LLM responses are saved for debugging but NEVER include HTTP headers
- The `DeepSeekProvider.complete()` method only stores the response JSON payload in `LLMResponse.raw`, not the request headers
- `config_summary()` on all providers never returns the API key
- The `check_llm_provider()` function never prints or logs the API key
