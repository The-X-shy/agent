# WSL Lens File Resolution

How DeepLens lens files are resolved on WSL (Windows Subsystem for Linux) remote workers.

## Problem

Phase 58 GeoLens diagnostics on WSL failed with `lens_file_not_found` because the lens file resolver only checked hardcoded macOS paths.

## Solution

The Phase 59 `lens_file_resolver.py` provides cross-platform resolution:

1. Checks `$DEEPLENS_REPO_PATH` environment variable first
2. Falls back to known WSL paths:
   - `/mnt/d/agent/external/DeepLens/datasets/lenses/`
   - `/mnt/d/DeepLens/datasets/lenses/`
   - `/mnt/d/external/DeepLens/datasets/lenses/`
3. Checks installed `deeplens` package
4. Limited safe root search

## WSL Worker Configuration

Add to worker capabilities:

```json
{
  "capabilities": {
    "known_lens_roots": ["/mnt/d/DeepLens", "/mnt/d/agent/external/DeepLens"]
  }
}
```

Set on the remote worker:

```bash
export DEEPLENS_REPO_PATH=/mnt/d/DeepLens
```

## Verification

```bash
# Local resolution
python -m optiresearch.cli resolve-lens-file --lens-file auto:cooke

# Remote resolution on WSL
python -m optiresearch.cli run-remote-resolve-lens-file \
  --worker-id windows_wsl \
  --lens-file auto:cooke \
  --backend-id deeplens_geolens_geometric
```
