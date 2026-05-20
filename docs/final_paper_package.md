# Final Paper Package

Complete reproducibility package for the OptiResearch Agent paper.

## Contents

- `README.md` — Package overview
- `final_benchmark_summary.md` — 5-group benchmark registry
- `paper_tables/` — 10 paper-ready tables
- `claim_boundary.md` — Supported/qualified/unsupported claims
- `evidence_distribution.md` — Evidence level distribution
- `paper_experiment_protocol_v0.1_freeze.md` — Frozen protocol
- `phase_reports/` — Phase 10-12 reports
- `artifact_inventory.json` — Complete artifact list
- `reproducibility_manifest.json` — Environment and availability

## CLI

```bash
python -m optiresearch.cli export-final-paper-package
```

Output: `workspace/final_paper_package/`
