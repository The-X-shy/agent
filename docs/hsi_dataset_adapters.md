# HSI Dataset Adapters

Phase 11 adds a common adapter interface for synthetic and local-path HSI datasets.

## Adapters

| Adapter | Dataset family | Default availability | Download policy |
|---|---|---:|---|
| `synthetic` | `synthetic` | yes | generated locally |
| `local_npz` | `local_npz` | only when path exists | no download |
| `cave` | `cave` | only when `OPTIRESEARCH_CAVE_PATH` exists | no download |
| `icvl` | `icvl` | only when `OPTIRESEARCH_ICVL_PATH` exists | no download |

## Local NPZ Structure

Supported input formats:

- Format A: `train.npz`, `val.npz`, `test.npz`, each with `hsi: [N, B, H, W]`.
- Format B: `dataset.npz` with `hsi: [N, B, H, W]` and optional `split`.
- Format C: single cube file with `cube: [B, H, W]` or `[H, W, B]`, patched into train/val/test.

Each dataset directory must contain:

```text
train.npz
val.npz
test.npz
```

Required array:

- `hsi`: `[N, B, H, W]`

Optional arrays:

- `depth_indices`
- `wavelengths_nm`
- `masks`

Prepared output:

```text
workspace/hsi/datasets/<dataset_id>/
├── train.npz
├── val.npz
├── test.npz
└── dataset_manifest.json
```

## Commands

```bash
python -m optiresearch.cli list-hsi-datasets
python -m optiresearch.cli prepare-hsi-dataset --dataset synthetic
python -m optiresearch.cli prepare-hsi-dataset --dataset local_npz --path /path/to/hsi_npz --crop-size 32 --patch-stride 32 --normalization per_band
python -m optiresearch.cli prepare-hsi-dataset --dataset cave
python -m optiresearch.cli prepare-hsi-dataset --dataset icvl
```

## Evidence Boundary

Public datasets are local-path only. No automatic download is performed.

A public/local dataset combined with `mock_deeplens` is still not a real camera experiment. It is a dataset/reconstruction evaluation with a mock optical encoder.
