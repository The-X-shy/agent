# Lens File Resolver

Cross-platform DeepLens lens file resolver that translates logical lens identifiers (`auto:cooke`) into real filesystem paths.

## Usage

### CLI

```bash
python -m optiresearch.cli resolve-lens-file --lens-file auto:cooke --backend-id deeplens_geolens_geometric
```

### Python API

```python
from optiresearch.optics.lens_file_resolver import resolve_lens_file

result = resolve_lens_file("auto:cooke", backend_id="deeplens_geolens_geometric")
if result.exists:
    print(f"Resolved: {result.resolved_path} (source: {result.source})")
else:
    print(f"Not found: {result.error_code}")
```

## Resolution Priority

1. `$OPTIRESEARCH_COOKE_LENS_FILE` environment variable
2. `$DEEPLENS_REPO_PATH/datasets/lenses/<name>`
3. `/mnt/d/agent/external/DeepLens/datasets/lenses/<name>` (WSL project-relative)
4. `/mnt/d/DeepLens/datasets/lenses/<name>` (WSL standalone)
5. `/Users/lilin/Desktop/external/DeepLens/datasets/lenses/<name>` (macOS)
6. Installed `deeplens` package adjacent `datasets/lenses/<name>`
7. Safe root repository search (limited depth)

## Result Schema

| Field | Type | Description |
|-------|------|-------------|
| `requested_lens_file` | str | Original lens file identifier |
| `resolved_path` | str or None | Resolved filesystem path |
| `exists` | bool | Whether the resolved path exists |
| `source` | str | Resolution source label |
| `checked_paths` | list[str] | All paths checked |
| `alternatives` | list[str] | Alternative paths found |
| `error_code` | str or None | Error code if unresolved |
| `warnings` | list[str] | Warning messages |

## Safety

- Only scans allowlisted safe roots
- No full filesystem scan
- No arbitrary shell execution
- Max search depth limited
