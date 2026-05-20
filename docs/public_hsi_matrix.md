# Public HSI Matrix

Phase 12 adds a matrix protocol for local/public HSI datasets.

## Command

```bash
python -m optiresearch.cli run-public-hsi-matrix --dataset local_npz --path /path/to/local_npz --backend mock_deeplens
python -m optiresearch.cli run-public-hsi-matrix --dataset cave --backend mock_deeplens
python -m optiresearch.cli run-public-hsi-matrix --dataset icvl --backend mock_deeplens
```

Output:

```text
workspace/hsi/public_matrix/<matrix_id>/
├── public_hsi_matrix_results.json
├── public_hsi_matrix_results.md
└── public_hsi_matrix_summary.json
```

## Scope

Each condition records dataset family, backend, realization level, encoder, reconstructor, forward mode, metrics, evidence level, and caveat.

If a dataset path is missing, the command returns a structured skip. It does not traceback.

Public/local data with `mock_deeplens` has evidence level `public_hsi_mock`. It is not real camera validation.

## Phase 13 Status

The public HSI matrix is included in the final benchmark registry (Group D) and paper table export (Table 8). See `docs/final_benchmark.md`.

