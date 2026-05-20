# Final Benchmark Registry

The FinalBenchmarkRegistry freezes the paper-ready benchmark into 5 groups.

## Groups

| Group | Description |
|---|---|
| A. System | Memory ablation, skill routing, claim evidence rate, LLM fallback audit |
| B. Optical Backend | Mock encoder baseline, DeepLens smoke/proxy/semi-native, wavelength-aware PSF |
| C. HSI Synthetic | Optical-sensitive HSI baseline, reconstructor matrix, encoder ranking |
| D. Public/Local HSI | Local NPZ/CAVE/ICVL adapters, public HSI matrix, structured skip |
| E. Evidence | Claim whitelist/blacklist, design rule status, evidence distribution |

## CLI

```bash
python -m optiresearch.cli list-final-benchmarks
python -m optiresearch.cli collect-final-benchmark
```

## Output

`workspace/final_benchmark/`
- `final_benchmark_summary.json`
- `final_benchmark_summary.md`
- `artifact_inventory.json`
