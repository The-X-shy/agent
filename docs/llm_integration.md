# LLM Integration

Phase 8 adds an optional LLM provider layer. If no provider is configured, agents use rule-based fallback.

## Providers

- `mock`: deterministic local provider for tests.
- `deepseek`: HTTP provider for `https://api.deepseek.com/chat/completions`.
- `openai`: optional SDK placeholder.
- `anthropic`: optional placeholder.
- `local`: optional local HTTP placeholder.

## DeepSeek

macOS / Linux:

```bash
export DEEPSEEK_API_KEY="你的 key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_THINKING_TYPE="enabled"
export DEEPSEEK_REASONING_EFFORT="high"
```

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"
$env:DEEPSEEK_THINKING_TYPE="enabled"
$env:DEEPSEEK_REASONING_EFFORT="high"
```

Commands:

```bash
python -m optiresearch.cli check-llm --provider deepseek
python -m optiresearch.cli test-llm --provider deepseek --prompt "Hello"
python -m optiresearch.cli run-mvp --use-llm --llm-provider deepseek --objective "Design a mock EDOF-HSI encoder"
```

## Safety Rules

- LLM output must pass Pydantic schema validation.
- LLM does not execute shell commands.
- LLM cannot bypass `SkillExecutor` allowlist.
- LLM cannot decide final claim status.
- LLM text is not evidence.
- Prompt and response audit data redacts API keys.
- Phase 9 HSI claims still require reconstruction metrics; LLM summaries cannot replace `reconstruction_metrics.json`.
