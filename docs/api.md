# API

Start:

```bash
uvicorn optiresearch.api.app:app --reload
```

## Endpoints

- `GET /health`
- `POST /v1/runs/mvp`
- `POST /v1/traces`
- `GET /v1/traces/{trace_id}`
- `POST /v1/memory/query`
- `GET /v1/artifacts`
- `GET /v1/claims/{claim_id}`
- `GET /v1/claims/{claim_id}/explain`
- `POST /v1/skills/resolve`
- `POST /v1/benchmarks/opti-memory/run`

## Example

```bash
curl -X POST http://127.0.0.1:8000/v1/runs/mvp \
  -H "Content-Type: application/json" \
  -d '{"objective":"Design a mock EDOF-HSI optical encoder","workspace_id":"opti_lab"}'
```

Memory query:

```bash
curl -X POST http://127.0.0.1:8000/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{"role":"CriticalReviewer","intent":"evidence claim","query":"depth stability","scope":{}}'
```
