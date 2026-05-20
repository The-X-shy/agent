# HSI Pipeline

Phase 9 established the synthetic HSI reconstruction evaluation loop. Phase 10 added optical-sensitive rendering that makes encoder PSF differences affect reconstruction metrics.
Phase 11 adds local-path dataset adapters and a dataset/reconstructor matrix.

Current scope:

- synthetic HSI dataset with multiple spectral patterns (mixed_materials default);
- PSF-cube forward model with 4 modes (depth_spectral_coded default);
- optical feature extraction from PSF cubes;
- single-shot monochrome measurement (evaluation proxy, not physical sensor model);
- numpy optical-conditioned linear reconstruction baseline;
- optional TinyCNN reconstructor (torch required);
- optional UNetTiny reconstructor (torch required);
- `synthetic`, `local_npz`, `cave`, and `icvl` dataset adapters;
- optical scalar feature maps for stronger networks;
- matrix-level ClaimEvidence and DesignRule compilation;
- public/local HSI matrix protocol;
- DeepLens wavelength-aware PSF compatibility checks;
- metrics-backed ClaimEvidence with reconstruction ranking.

This proves end-to-end evaluability with encoder sensitivity. It does not prove final optical performance.

Commands:

```bash
python -m optiresearch.cli run-hsi-reconstruction --backend mock_deeplens --encoder controlled_chromatic_edof --objective "Evaluate synthetic HSI reconstruction"
python -m optiresearch.cli run-hsi-baselines --backend mock_deeplens
python -m optiresearch.cli list-hsi-datasets
python -m optiresearch.cli prepare-hsi-dataset --dataset synthetic
python -m optiresearch.cli run-hsi-matrix --datasets synthetic --backends mock_deeplens --reconstructors optical_conditioned_linear,tiny_cnn --forward-modes depth_spectral_coded --objective "Compare encoder ranking across reconstructors"
python -m optiresearch.cli export-phase9-report
python -m optiresearch.cli export-phase10-report
python -m optiresearch.cli export-phase11-report
python -m optiresearch.cli run-public-hsi-matrix --dataset synthetic --backend mock_deeplens
python -m optiresearch.cli export-phase12-report
```

See `docs/optical_sensitive_hsi.md` for Phase 10 details.

Public datasets are local-path only. Synthetic/public dataset results with a mock optical encoder are not real camera validation.
