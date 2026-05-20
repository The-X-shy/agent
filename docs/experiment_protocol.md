# Experiment Protocol

Phase 3 adds a deterministic mock experiment protocol for comparing optical encoder families.

## Encoder Families

- `conventional`
- `achromatic`
- `edof`
- `chromatic_coded`
- `controlled_chromatic_edof`

Each encoder uses the same `ExperimentSpec` shape and the same mock sweep:

- 31 wavelength bands
- 9 depth planes
- 32 x 32 PSF size
- seed 42

## Metrics

The mock backend reports:

- `psf_depth_similarity`
- `spectral_separability`
- `mock_mtf_mean`
- `mock_energy_efficiency`

The baseline runner computes a joint tradeoff score from those metrics:

```text
0.35 * depth_similarity
+ 0.35 * spectral_separability
+ 0.15 * mock_mtf_mean
+ 0.15 * energy_efficiency
```

## Command

```bash
python -m optiresearch.cli run-baselines --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
```

Outputs:

- `workspace/baselines/baseline_comparison.json`
- `workspace/baselines/baseline_comparison.md`
