# DeepLens Source Integration

DeepLens source execution is controlled by `DEEPLENS_REPO_PATH`.

```bash
export DEEPLENS_REPO_PATH=/mnt/d/external/DeepLens
python -m optiresearch.cli probe-deeplens-source
python -m optiresearch.cli inspect-deeplens-source
python -m optiresearch.cli run-deeplens-source-smoke
```

The WSL worker wrapper must set the same variable:

```bash
#!/usr/bin/env bash
export DEEPLENS_REPO_PATH=/mnt/d/external/DeepLens
cd /mnt/d/agent
exec /mnt/d/agent/.venv/bin/python "$@"
```

DeepLens-backed claims require successful source import and artifact output. Fallback or mock output must remain labeled as fallback/mock and cannot be promoted to DeepLens-backed evidence.
