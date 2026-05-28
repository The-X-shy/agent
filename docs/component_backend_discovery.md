# Component Backend Discovery

Phase 62 introduces `discover_deeplens_components()` — a lightweight probe that
checks which DeepLens surface component classes are importable and instantiatable
without running any optimization loop.

## Usage

```bash
# Local discovery
python -m optiresearch.cli discover-deeplens-components

# Remote discovery (WSL, via SSH)
python -m optiresearch.cli run-remote-discover-deeplens-components \
    --worker-id windows_wsl
```

## Checked Components

- **Fresnel** (`deeplens.diffractive_surface.fresnel`)
- **Binary2Phase** (`deeplens.phase_surface.binary2`)
- **Diffractive candidates** (all 14 diffractive/phase surface classes)

## Output

`DiscoveryManifest` with:
- `deeplens_available` — is the `deeplens` package importable?
- `deeplens_version` — installed version string
- `available_components` / `unavailable_components` — classification
- `diffractive_candidates_found` — all importable diffractive-like classes
- `import_paths_checked` — full module paths probed
- `constructor_signatures` — `inspect.signature(cls.__init__)` for each found class
