# Quickstart

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Initialize

```bash
python -m optiresearch.cli init-db
```

## Run MVP

```bash
python -m optiresearch.cli run-mvp --objective "Design a mock depth-invariant and spectrally discriminative EDOF-HSI optical encoder"
python -m optiresearch.cli run-mvp --backend mock_deeplens --objective "Design a mock EDOF-HSI encoder"
python -m optiresearch.cli run-mvp --backend deeplens --objective "Design a minimal DeepLens PSF smoke run"
```

Inspect artifacts from that run:

```bash
python -m optiresearch.cli inspect-artifacts --run-id <run_id>
```

Explain a claim:

```bash
python -m optiresearch.cli explain-claim --claim-id <claim_id>
```

## Query Memory

```bash
python -m optiresearch.cli query-memory --intent evidence --query "depth stability"
```

## Plans And Skills

```bash
python -m optiresearch.cli list-plans
python -m optiresearch.cli match-plan --intent "evaluate edof hsi"
python -m optiresearch.cli list-skills-memory
python -m optiresearch.cli recommend-skills --intent "simulate psf"
```

## Benchmark

```bash
python -m optiresearch.cli run-benchmark --name opti-memory
```

Run a specific ablation mode:

```bash
python -m optiresearch.cli run-benchmark --name opti-memory --mode full_rmos
```

## Baselines And Rules

```bash
python -m optiresearch.cli run-baselines --backend mock_deeplens --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
MPLCONFIGDIR=workspace/mplconfig .venv-deeplens/bin/python -m optiresearch.cli run-baselines --backend deeplens --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
python -m optiresearch.cli explain-rule --rule-id <rule_id>
```

## DeepLens Backend

```bash
python -m optiresearch.cli check-deeplens
python -m optiresearch.cli deeplens-capabilities
python -m optiresearch.cli run-deeplens-smoke --objective "Design a minimal DeepLens PSF smoke run"
MPLCONFIGDIR=workspace/mplconfig .venv-deeplens/bin/python -m optiresearch.cli run-baselines --backend deeplens --objective "Design depth-invariant and spectrally discriminative EDOF-HSI encoder"
```

If DeepLens is not installed, both commands return structured output instead of a traceback.

The real backend target is `https://github.com/vccimaging/DeepLens`. Install it in a Python 3.12+ environment:

```bash
python -m pip install "deeplens-core @ git+https://github.com/vccimaging/DeepLens.git"
```

## Remote WSL Worker

```bash
python -m optiresearch.cli add-remote-worker \
  --worker-id windows_wsl \
  --host wslbox \
  --port 22 \
  --username ysl \
  --remote-project-dir /mnt/d/agent \
  --remote-workspace-dir /mnt/d/agent/workspace \
  --python-executable /mnt/d/agent/run_agent_python.sh

python -m optiresearch.cli check-remote-worker --worker-id windows_wsl
python -m optiresearch.cli run-remote-deeplens-source-smoke --worker-id windows_wsl
python -m optiresearch.cli run-remote-codesign \
  --worker-id windows_wsl \
  --objective "Run strict DeepLens-backed co-design on WSL D drive worker" \
  --psf-source deeplens_parameterized \
  --backend deeplens \
  --fallback-policy fail \
  --max-iterations 2
```

Remote artifacts are copied back to `workspace/remote_jobs/<job_id>/` and then ingested into the local artifact, memory, and claim stores.

## Paper Exports

```bash
python -m optiresearch.cli export-paper-summary
python -m optiresearch.cli export-evidence-tables
python -m optiresearch.cli compare-backends --left mock_deeplens --right deeplens
python -m optiresearch.cli export-phase6-report
python -m optiresearch.cli export-phase7-report
python -m optiresearch.cli export-phase8-report
python -m optiresearch.cli probe-deeplens-api
python -m optiresearch.cli llm-providers
python -m optiresearch.cli check-llm
python -m optiresearch.cli test-llm --provider mock --prompt "Hello"
python -m optiresearch.cli run-hsi-reconstruction --backend mock_deeplens --encoder controlled_chromatic_edof --objective "Evaluate synthetic HSI reconstruction with controlled chromatic EDOF encoder"
python -m optiresearch.cli run-hsi-baselines --backend mock_deeplens
python -m optiresearch.cli export-phase9-report
python -m optiresearch.cli run-hsi-reconstruction --backend mock_deeplens --encoder controlled_chromatic_edof --forward-mode depth_spectral_coded --reconstructor optical_conditioned_linear --dataset-pattern mixed_materials --objective "Evaluate optical-sensitive synthetic HSI reconstruction"
python -m optiresearch.cli run-hsi-baselines --backend mock_deeplens --forward-mode depth_spectral_coded --reconstructor optical_conditioned_linear --dataset-pattern mixed_materials
python -m optiresearch.cli export-phase10-report
python -m optiresearch.cli list-hsi-datasets
python -m optiresearch.cli prepare-hsi-dataset --dataset synthetic
python -m optiresearch.cli run-hsi-matrix --datasets synthetic --backends mock_deeplens --reconstructors optical_conditioned_linear,tiny_cnn --forward-modes depth_spectral_coded --objective "Compare encoder ranking across reconstructors"
python -m optiresearch.cli export-phase11-report
python -m optiresearch.cli run-public-hsi-matrix --dataset synthetic --backend mock_deeplens
python -m optiresearch.cli freeze-paper-protocol
python -m optiresearch.cli export-phase12-report
```

Outputs:

- `workspace/reports/phase3_experiment_summary.md`
- `workspace/reports/evidence_claims.md`
- `workspace/reports/evidence_rules.md`
- `workspace/reports/backend_alignment_mock_vs_deeplens.md`
- `workspace/reports/phase6_real_deeplens_report.md`
- `workspace/reports/phase7_deeplens_encoder_proxy_report.md`
- `workspace/reports/phase8_deeplens_semi_native_report.md`
- `workspace/reports/deeplens_api_probe.md`
- `workspace/reports/phase9_hsi_reconstruction_report.md`
- `workspace/reports/phase10_optical_sensitive_hsi_report.md`
- `workspace/reports/phase11_hsi_network_dataset_report.md`
- `workspace/reports/paper_experiment_protocol_v0.1_freeze.md`
- `workspace/reports/phase12_public_hsi_deeplens_protocol_report.md`

## Start API

```bash
uvicorn optiresearch.api.app:app --reload
```

## Phase 13: Paper Evidence Package

```bash
# List all final benchmarks
python -m optiresearch.cli list-final-benchmarks

# Export complete paper evidence package
python -m optiresearch.cli export-final-paper-package

# Export Phase 13 report
python -m optiresearch.cli export-phase13-report
```

## Test

```bash
python -m pytest
```
