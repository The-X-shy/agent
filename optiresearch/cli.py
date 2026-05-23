"""Command line interface for OptiResearch Agent MVP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.deeplens_api_probe import export_deeplens_api_probe
from optiresearch.benchmarks.opti_memory_bench.runner import OptiMemoryBenchRunner
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.design_rule import DesignRuleManager
from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.plan_template import PlanTemplateManager
from optiresearch.memory.router import MemoryRouter
from optiresearch.memory.skill_memory import SkillMemoryManager
from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.runtime.baselines import run_baseline_batch
from optiresearch.runtime.hsi_pipeline import run_hsi_reconstruction_flow
from optiresearch.runtime.hsi_baselines import run_hsi_encoder_baselines
from optiresearch.runtime.hsi_matrix import run_hsi_matrix
from optiresearch.runtime.hsi_public_matrix import run_public_hsi_matrix
from optiresearch.hsi.public_datasets import get_hsi_dataset_adapter, list_hsi_dataset_adapters
from optiresearch.reports.backend_alignment import export_backend_alignment_report
from optiresearch.reports.paper import export_evidence_tables, export_phase3_experiment_summary
from optiresearch.reports.phase6 import export_phase6_report
from optiresearch.reports.phase7 import export_phase7_report
from optiresearch.reports.phase8 import export_phase8_report
from optiresearch.reports.phase9 import export_phase9_report
from optiresearch.reports.phase10 import export_phase10_report
from optiresearch.reports.phase11 import export_phase11_report
from optiresearch.reports.phase12 import export_phase12_report
from optiresearch.reports.phase13 import export_phase13_report
from optiresearch.reports.phase16 import export_phase16_report
from optiresearch.adapters.deeplens_source_inspector import export_source_inspection
from optiresearch.reports.paper_tables import export_paper_tables
from optiresearch.reports.claim_boundary import generate_claim_whitelist_blacklist
from optiresearch.reports.evidence_distribution import compute_evidence_distribution
from optiresearch.reports.warnings_audit import WarningsAudit
from optiresearch.reports.final_package import export_final_paper_package
from optiresearch.runtime.final_benchmark import FinalBenchmarkRegistry
from optiresearch.runtime.autonomous_loop import run_autonomous_research_loop
from optiresearch.schemas.autonomous import AutonomousLoopConfig
from optiresearch.runtime.codesign_loop import run_codesign_loop
from optiresearch.schemas.optimization import OptimizationSpec, build_default_optimization_spec
from optiresearch.reports.protocol_freeze import freeze_paper_protocol
from optiresearch.llm.registry import get_llm_provider, list_llm_providers
from optiresearch.llm.base import LLMProviderError
from optiresearch.skills.artifact_inspector.inspector import ArtifactInspector
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore
from optiresearch.schemas.remote import RemoteWorkerSpec
from optiresearch.remote.worker_registry import RemoteWorkerRegistry
from optiresearch.runtime.remote_jobs import (
    check_remote_worker,
    export_remote_job_outputs,
    run_remote_codesign,
    run_remote_deeplens_lensfile_optimization_probe,
    run_remote_deeplens_source_smoke,
    run_remote_deeplens_surface_optimization_probe,
    run_remote_hsi_reconstruction,
    run_remote_native_hsi_codesign,
    run_remote_native_hsi_reconstruction_codesign,
    run_remote_native_optimization_probe,
    run_remote_deeplens_waveoptics_probe,
    run_remote_native_waveoptics_hsi_codesign,
    run_remote_stable_native_lens_hsi_codesign,
    run_remote_stable_native_lens_hsi_ablation,
    run_remote_deeplens_native_geolens_hsi_codesign,
)
from optiresearch.reports.remote_execution import export_remote_execution_report
from optiresearch.reports.native_geolens_hsi_report import export_native_geolens_hsi_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="optiresearch")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db", help="Initialize SQLite database and artifact workspace.")

    run = sub.add_parser("run-mvp", help="Run the full MVP flow.")
    run.add_argument("--objective", required=True)
    run.add_argument("--workspace-id", default="default")
    run.add_argument("--backend", choices=["mock_deeplens", "deeplens"], default="mock_deeplens")
    run.add_argument("--use-llm", action="store_true")
    run.add_argument("--llm-provider")
    run.add_argument("--llm-model")

    query = sub.add_parser("query-memory", help="Query memory.")
    query.add_argument("--intent", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--role", default="System")

    sub.add_parser("list-artifacts", help="List registered artifacts.")
    sub.add_parser("list-traces", help="List Meta-Trace entries.")
    inspect = sub.add_parser("inspect-artifacts", help="Inspect registered artifacts.")
    inspect.add_argument("--run-id", required=True)
    explain = sub.add_parser("explain-claim", help="Explain a claim evidence chain.")
    explain.add_argument("--claim-id", required=True)
    sub.add_parser("list-plans", help="List plan templates.")
    match = sub.add_parser("match-plan", help="Match plan templates by intent.")
    match.add_argument("--intent", required=True)
    sub.add_parser("list-skills-memory", help="List compiled skill memories.")
    recommend = sub.add_parser("recommend-skills", help="Recommend skills by intent.")
    recommend.add_argument("--intent", required=True)
    bench = sub.add_parser("run-benchmark", help="Run a benchmark.")
    bench.add_argument("--name", default="opti-memory")
    bench.add_argument("--mode", default="full_rmos")
    baselines = sub.add_parser("run-baselines", help="Run mock encoder baseline batch.")
    baselines.add_argument("--objective", required=True)
    baselines.add_argument("--workspace-id", default="default")
    baselines.add_argument("--backend", choices=["mock_deeplens", "deeplens"], default="mock_deeplens")
    baselines.add_argument("--encoder", choices=["all", "conventional", "achromatic", "edof", "chromatic_coded", "controlled_chromatic_edof"], default="all")
    baselines.add_argument("--output-dir")
    baselines.add_argument("--realization", choices=["auto", "adapter_proxy", "semi_native", "native"], default="auto")
    baselines.add_argument("--use-llm-plan", action="store_true")
    explain_rule = sub.add_parser("explain-rule", help="Explain a compiled design rule.")
    explain_rule.add_argument("--rule-id", required=True)
    sub.add_parser("export-paper-summary", help="Export paper experiment summary markdown.")
    sub.add_parser("export-evidence-tables", help="Export claim and rule evidence markdown tables.")
    sub.add_parser("check-deeplens", help="Probe the real DeepLens backend environment.")
    sub.add_parser("deeplens-capabilities", help="Print DeepLens backend capabilities.")
    smoke = sub.add_parser("run-deeplens-smoke", help="Run a minimal real DeepLens smoke simulation.")
    smoke.add_argument("--objective", required=True)
    compare = sub.add_parser("compare-backends", help="Compare two backend baseline reports.")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    sub.add_parser("export-phase6-report", help="Export Phase 6 real DeepLens report.")
    sub.add_parser("export-phase7-report", help="Export Phase 7 DeepLens encoder proxy report.")
    sub.add_parser("export-phase8-report", help="Export Phase 8 semi-native report.")
    sub.add_parser("probe-deeplens-api", help="Probe installed DeepLens API surface.")
    providers = sub.add_parser("llm-providers", help="List LLM providers.")
    check_llm = sub.add_parser("check-llm", help="Check selected LLM provider.")
    check_llm.add_argument("--provider")
    test_llm = sub.add_parser("test-llm", help="Run a small LLM completion.")
    test_llm.add_argument("--provider")
    test_llm.add_argument("--prompt", required=True)
    sub.add_parser("list-llm-calls", help="List LLM traces.")
    hsi_run = sub.add_parser("run-hsi-reconstruction", help="Run synthetic HSI reconstruction flow.")
    hsi_run.add_argument("--objective", required=True)
    hsi_run.add_argument("--backend", choices=["mock_deeplens", "deeplens"], default="mock_deeplens")
    hsi_run.add_argument("--encoder", default="controlled_chromatic_edof")
    hsi_run.add_argument("--workspace-id", default="default")
    hsi_run.add_argument("--use-llm", action="store_true")
    hsi_run.add_argument("--llm-provider")
    hsi_run.add_argument("--realization", default="auto")
    hsi_run.add_argument("--forward-mode", choices=["simple_sum", "psf_weighted", "coded_aperture_proxy", "depth_spectral_coded"], default="depth_spectral_coded")
    hsi_run.add_argument("--reconstructor", choices=["linear_baseline", "optical_conditioned_linear", "tiny_cnn", "unet_tiny"], default="optical_conditioned_linear")
    hsi_run.add_argument("--dataset", choices=["synthetic", "local_npz", "cave", "icvl"], default="synthetic")
    hsi_run.add_argument("--dataset-path")
    hsi_run.add_argument("--dataset-pattern", choices=["smooth_low_rank", "mixed_materials", "sparse_peaks", "edge_spectral_contrast"], default="mixed_materials")
    hsi_run.add_argument("--tiny-cnn-epochs", type=int, default=5)
    hsi_run.add_argument("--tiny-cnn-hidden", type=int, default=32)
    hsi_run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    hsi_run.add_argument("--use-optical-feature-maps", action="store_true")
    hsi_run.add_argument("--remote-job-id")
    hsi_baselines = sub.add_parser("run-hsi-baselines", help="Run HSI reconstruction baselines.")
    hsi_baselines.add_argument("--backend", choices=["mock_deeplens", "deeplens"], default="mock_deeplens")
    hsi_baselines.add_argument("--objective", default="Evaluate synthetic HSI reconstruction")
    hsi_baselines.add_argument("--forward-mode", choices=["simple_sum", "psf_weighted", "coded_aperture_proxy", "depth_spectral_coded"], default="depth_spectral_coded")
    hsi_baselines.add_argument("--reconstructor", choices=["linear_baseline", "optical_conditioned_linear", "tiny_cnn", "unet_tiny"], default="optical_conditioned_linear")
    hsi_baselines.add_argument("--dataset-pattern", choices=["smooth_low_rank", "mixed_materials", "sparse_peaks", "edge_spectral_contrast"], default="mixed_materials")
    sub.add_parser("list-hsi-datasets", help="List configured HSI dataset adapters.")
    prepare_hsi = sub.add_parser("prepare-hsi-dataset", help="Prepare a synthetic or local-path HSI dataset.")
    prepare_hsi.add_argument("--dataset", choices=["synthetic", "local_npz", "cave", "icvl"], required=True)
    prepare_hsi.add_argument("--path")
    prepare_hsi.add_argument("--crop-size", type=int, default=32)
    prepare_hsi.add_argument("--patch-stride", type=int, default=32)
    prepare_hsi.add_argument("--normalization", choices=["none", "minmax", "per_band", "global"], default="per_band")
    hsi_matrix = sub.add_parser("run-hsi-matrix", help="Run HSI dataset/reconstructor matrix.")
    hsi_matrix.add_argument("--datasets", default="synthetic")
    hsi_matrix.add_argument("--backends", default="mock_deeplens")
    hsi_matrix.add_argument("--encoders", default="conventional,achromatic,edof,chromatic_coded,controlled_chromatic_edof")
    hsi_matrix.add_argument("--reconstructors", default="optical_conditioned_linear,tiny_cnn")
    hsi_matrix.add_argument("--forward-modes", default="depth_spectral_coded")
    hsi_matrix.add_argument("--objective", required=True)
    hsi_matrix.add_argument("--workspace-id", default="default")
    hsi_matrix.add_argument("--dataset-path")
    hsi_matrix.add_argument("--tiny-cnn-epochs", type=int, default=5)
    hsi_matrix.add_argument("--tiny-cnn-hidden", type=int, default=32)
    hsi_matrix.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    hsi_matrix.add_argument("--use-optical-feature-maps", action="store_true")
    hsi_matrix.add_argument("--remote-job-id")
    public_matrix = sub.add_parser("run-public-hsi-matrix", help="Run public/local HSI matrix with structured skips.")
    public_matrix.add_argument("--dataset", choices=["synthetic", "local_npz", "cave", "icvl"], required=True)
    public_matrix.add_argument("--path")
    public_matrix.add_argument("--backend", choices=["mock_deeplens", "deeplens"], default="mock_deeplens")
    public_matrix.add_argument("--encoders", default="conventional,achromatic,edof,chromatic_coded,controlled_chromatic_edof")
    public_matrix.add_argument("--reconstructors", default="optical_conditioned_linear")
    public_matrix.add_argument("--forward-modes", default="depth_spectral_coded")
    public_matrix.add_argument("--realization", default="auto")
    public_matrix.add_argument("--workspace-id", default="default")
    sub.add_parser("freeze-paper-protocol", help="Freeze paper experiment protocol v0.1.")
    sub.add_parser("export-phase9-report", help="Export Phase 9 HSI reconstruction report.")
    sub.add_parser("export-phase10-report", help="Export Phase 10 optical-sensitive HSI report.")
    sub.add_parser("export-phase11-report", help="Export Phase 11 HSI network/dataset report.")
    sub.add_parser("export-phase12-report", help="Export Phase 12 public HSI and protocol report.")
    sub.add_parser("list-final-benchmarks", help="List all final benchmark items.")
    sub.add_parser("collect-final-benchmark", help="Collect and export final benchmark summary.")
    sub.add_parser("export-paper-tables", help="Export paper-ready tables (MD/CSV/JSON).")
    sub.add_parser("export-claim-boundary", help="Export claim whitelist/blacklist.")
    sub.add_parser("export-evidence-distribution", help="Export evidence level distribution.")
    sub.add_parser("export-warnings-audit", help="Export pytest warnings audit.")
    sub.add_parser("export-final-paper-package", help="Export final paper reproducibility package.")
    sub.add_parser("export-phase13-report", help="Export Phase 13 final benchmark report.")
    aloop = sub.add_parser("run-autonomous-loop", help="Run LLM-driven autonomous research loop.")
    aloop.add_argument("--objective", required=True)
    aloop.add_argument("--llm-provider", default="mock")
    aloop.add_argument("--max-iterations", type=int, default=3)
    aloop.add_argument("--backend", choices=["mock_deeplens", "deeplens"], default="mock_deeplens")
    aloop.add_argument("--dataset", default="synthetic")
    aloop.add_argument("--allowed-reconstructors", default="optical_conditioned_linear,tiny_cnn")
    aloop.add_argument("--allowed-encoders", default="conventional,achromatic,edof,chromatic_coded,controlled_chromatic_edof")
    aloop.add_argument("--execution-mode", choices=["local", "remote"], default="local")
    aloop.add_argument("--worker-id")
    aloop.add_argument("--remote-job-id")
    aloop_report = sub.add_parser("export-autonomous-loop-report", help="Export autonomous loop report.")
    aloop_report.add_argument("--loop-id", required=True)
    codesign = sub.add_parser("run-codesign-loop", help="Run optical-HSI co-design optimization loop.")
    codesign.add_argument("--objective", required=True)
    codesign.add_argument("--llm-provider", default="mock")
    codesign.add_argument("--max-iterations", type=int, default=5)
    codesign.add_argument("--backend", default="mock_deeplens")
    codesign.add_argument("--encoder", default="controlled_chromatic_edof")
    codesign.add_argument("--reconstructor", default="optical_conditioned_linear")
    codesign.add_argument("--forward-mode", default="depth_spectral_coded")
    codesign.add_argument("--dataset", default="synthetic")
    codesign.add_argument("--psf-source", choices=["parameterized_mock", "deeplens_parameterized"], default="parameterized_mock")
    codesign.add_argument("--fallback-policy", choices=["fail", "fallback_to_mock", "partial_deeplens_proxy"], default="fallback_to_mock")
    codesign.add_argument("--strict-deeplens", action="store_true")
    codesign.add_argument("--remote-job-id")
    compare_psf = sub.add_parser("compare-psf-sources", help="Compare parameterized_mock vs deeplens_parameterized PSF sources.")
    compare_psf.add_argument("--left", default="parameterized_mock")
    compare_psf.add_argument("--right", default="deeplens_parameterized")
    compare_psf.add_argument("--objective", default="PSF source comparison")
    sub.add_parser("export-phase16-report", help="Export Phase 16 DeepLens-backed co-design report.")
    sub.add_parser("export-phase19-report", help="Export Phase 19 native optimization probe report.")
    probe_source = sub.add_parser("probe-deeplens-source", help="Probe DeepLens source checkout capabilities.")
    probe_source.add_argument("--remote-job-id")
    inspect_source = sub.add_parser("inspect-deeplens-source", help="Inspect DeepLens source structure.")
    inspect_source.add_argument("--remote-job-id")
    inspect_native = sub.add_parser("inspect-deeplens-native-optimization", help="Inspect DeepLens native optimization capabilities.")
    inspect_native.add_argument("--remote-job-id")
    sub.add_parser("scan-deeplens-optimization-paths", help="Scan DeepLens source for native optimization paths.")
    source_smoke = sub.add_parser("run-deeplens-source-smoke", help="Run minimal PSF smoke with source DeepLens.")
    source_smoke.add_argument("--remote-job-id")
    native_probe = sub.add_parser("run-native-optimization-probe", help="Run native differentiable optimization probe on a DeepLens lens class.")
    native_probe.add_argument("--lens-class", required=True, choices=["ParaxialLens", "GeoLens", "DiffractiveLens", "HybridLens", "PSFNetLens"])
    native_probe.add_argument("--objective", required=True, choices=["minimize_psf_width", "maximize_center_intensity", "match_target_psf", "hsi_reconstruction_loss"])
    native_probe.add_argument("--max-steps", type=int, default=2)
    native_probe.add_argument("--learning-rate", type=float, default=1e-3)
    native_probe.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    native_probe.add_argument("--strict-native", action="store_true")
    native_probe.add_argument("--allow-adapter-proxy", action="store_true")
    native_probe.add_argument("--remote-job-id")
    surface_probe = sub.add_parser("run-deeplens-surface-optimization-probe", help="Run native optimization probe on a DeepLens surface class.")
    surface_probe.add_argument("--surface", required=True)
    surface_probe.add_argument("--objective", required=True, choices=["minimize_phase_variance", "match_target_phase", "parameter_sanity_check"])
    surface_probe.add_argument("--max-steps", type=int, default=3)
    surface_probe.add_argument("--learning-rate", type=float, default=1e-3)
    surface_probe.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    surface_probe.add_argument("--remote-job-id")
    lensfile_probe = sub.add_parser("run-deeplens-lensfile-optimization-probe", help="Run native optimization probe on DeepLens example lens files.")
    lensfile_probe.add_argument("--lens-class", required=True, choices=["GeoLens", "HybridLens", "DiffractiveLens"])
    lensfile_probe.add_argument("--max-files", type=int, default=5)
    lensfile_probe.add_argument("--max-steps", type=int, default=2)
    lensfile_probe.add_argument("--learning-rate", type=float, default=1e-3)
    lensfile_probe.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    lensfile_probe.add_argument("--remote-job-id")
    hsi_codesign = sub.add_parser("run-native-hsi-codesign", help="Run native optical-HSI co-design loop.")
    hsi_codesign.add_argument("--optical-component", required=True, choices=["Fresnel", "Binary2Phase", "GeoLensCooke"])
    hsi_codesign.add_argument("--objective", required=True, choices=["minimize_hsi_proxy_loss", "maximize_reconstruction_score", "minimize_spectral_mse", "minimize_measurement_consistency_loss"])
    hsi_codesign.add_argument("--max-steps", type=int, default=3)
    hsi_codesign.add_argument("--learning-rate", type=float, default=1e-3)
    hsi_codesign.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    hsi_codesign.add_argument("--bands", type=int, default=31)
    hsi_codesign.add_argument("--image-size", type=int, default=32)
    hsi_codesign.add_argument("--psf-size", type=int, default=16)
    hsi_codesign.add_argument("--remote-job-id")
    sub.add_parser("export-phase19b-report", help="Export Phase 19B native optimization path report.")
    sub.add_parser("export-phase20-report", help="Export Phase 20 native HSI co-design report.")
    sub.add_parser("export-phase21-report", help="Export Phase 21 native HSI reconstruction co-design report.")
    sub.add_parser("export-phase22-report", help="Export Phase 22 full wave-optics native HSI co-design report.")
    sub.add_parser("export-phase23-report", help="Export Phase 23 stable native lens HSI co-design report.")
    diagnose = sub.add_parser("diagnose-native-lens-hsi-codesign", help="Diagnose Phase 22 native lens HSI co-design instability.")
    diagnose.add_argument("--run-dir", required=True)
    stable_hsi = sub.add_parser("run-stable-native-lens-hsi-codesign", help="Run stable native lens HSI co-design.")
    stable_hsi.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens"])
    stable_hsi.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    stable_hsi.add_argument("--dataset", default="synthetic")
    stable_hsi.add_argument("--max-steps", type=int, default=10)
    stable_hsi.add_argument("--optical-lr", type=float, default=1e-6)
    stable_hsi.add_argument("--recon-lr", type=float, default=1e-3)
    stable_hsi.add_argument("--optical-grad-clip", type=float, default=1.0)
    stable_hsi.add_argument("--rollback-on-loss-increase", action="store_true", default=True)
    stable_hsi.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    stable_hsi.add_argument("--remote-job-id")
    stable_abl = sub.add_parser("run-stable-native-lens-hsi-ablation", help="Run stable native lens HSI ablation.")
    stable_abl.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens"])
    stable_abl.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    stable_abl.add_argument("--dataset", default="synthetic")
    stable_abl.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    stable_abl.add_argument("--remote-job-id")
    sub.add_parser("scan-deeplens-waveoptics-paths", help="Scan DeepLens source for wave-optics differentiable paths.")
    waveoptics = sub.add_parser("run-deeplens-waveoptics-probe", help="Probe DeepLens native wave-optics PSF path.")
    waveoptics.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens", "FresnelWave", "Binary2PhaseWave", "CustomLensFile"])
    waveoptics.add_argument("--objective", required=True, choices=["minimize_psf_width", "match_target_psf", "minimize_hsi_reconstruction_loss"])
    waveoptics.add_argument("--psf-size", type=int, default=32)
    waveoptics.add_argument("--max-steps", type=int, default=3)
    waveoptics.add_argument("--learning-rate", type=float, default=1e-3)
    waveoptics.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    waveoptics.add_argument("--remote-job-id")
    wave_hsi = sub.add_parser("run-native-waveoptics-hsi-codesign", help="Run full native wave-optics HSI reconstruction co-design.")
    wave_hsi.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens"])
    wave_hsi.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    wave_hsi.add_argument("--dataset", choices=["synthetic", "local_npz"], default="synthetic")
    wave_hsi.add_argument("--max-steps", type=int, default=3)
    wave_hsi.add_argument("--optical-lr", type=float, default=1e-3)
    wave_hsi.add_argument("--recon-lr", type=float, default=1e-3)
    wave_hsi.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    wave_hsi.add_argument("--bands", type=int, default=31)
    wave_hsi.add_argument("--image-size", type=int, default=32)
    wave_hsi.add_argument("--psf-size", type=int, default=32)
    wave_hsi.add_argument("--remote-job-id")
    recon_codesign = sub.add_parser("run-native-hsi-reconstruction-codesign", help="Run full native HSI reconstruction co-design loop.")
    recon_codesign.add_argument("--optical-component", required=True, choices=["Fresnel", "Binary2Phase", "GeoLensCooke"])
    recon_codesign.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    recon_codesign.add_argument("--dataset", default="synthetic")
    recon_codesign.add_argument("--max-steps", type=int, default=5)
    recon_codesign.add_argument("--optical-lr", type=float, default=1e-3)
    recon_codesign.add_argument("--recon-lr", type=float, default=1e-3)
    recon_codesign.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    recon_codesign.add_argument("--bands", type=int, default=31)
    recon_codesign.add_argument("--image-size", type=int, default=32)
    recon_codesign.add_argument("--psf-size", type=int, default=16)
    recon_codesign.add_argument("--remote-job-id")
    ablation = sub.add_parser("run-native-hsi-reconstruction-ablation", help="Run ablation study for native HSI reconstruction co-design.")
    ablation.add_argument("--optical-component", required=True, choices=["Fresnel", "Binary2Phase"])
    ablation.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    ablation.add_argument("--max-steps", type=int, default=5)
    ablation.add_argument("--optical-lr", type=float, default=1e-3)
    ablation.add_argument("--recon-lr", type=float, default=1e-3)
    ablation.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ablation.add_argument("--bands", type=int, default=31)
    ablation.add_argument("--image-size", type=int, default=32)
    ablation.add_argument("--psf-size", type=int, default=16)
    add_worker = sub.add_parser("add-remote-worker", help="Add or update a remote worker.")
    add_worker.add_argument("--worker-id", required=True)
    add_worker.add_argument("--host", required=True)
    add_worker.add_argument("--port", type=int, default=22)
    add_worker.add_argument("--username", required=True)
    add_worker.add_argument("--ssh-key-path")
    add_worker.add_argument("--remote-project-dir", required=True)
    add_worker.add_argument("--remote-workspace-dir", required=True)
    add_worker.add_argument("--python-executable", required=True)
    add_worker.add_argument("--environment-name")
    add_worker.add_argument("--max-runtime-seconds", type=int, default=3600)
    add_worker.add_argument("--backend-tags", default="wsl,deeplens,torch,remote")
    check_worker = sub.add_parser("check-remote-worker", help="Check a registered remote worker.")
    check_worker.add_argument("--worker-id", required=True)
    remote_smoke = sub.add_parser("run-remote-deeplens-source-smoke", help="Run DeepLens source smoke on a remote worker.")
    remote_smoke.add_argument("--worker-id", required=True)
    remote_codesign = sub.add_parser("run-remote-codesign", help="Run strict DeepLens-backed co-design on a remote worker.")
    remote_codesign.add_argument("--worker-id", required=True)
    remote_codesign.add_argument("--objective", required=True)
    remote_codesign.add_argument("--psf-source", choices=["parameterized_mock", "deeplens_parameterized"], default="deeplens_parameterized")
    remote_codesign.add_argument("--backend", default="deeplens")
    remote_codesign.add_argument("--fallback-policy", choices=["fail", "fallback_to_mock", "partial_deeplens_proxy"], default="fail")
    remote_codesign.add_argument("--max-iterations", type=int, default=2)
    remote_codesign.add_argument("--strict-deeplens", action="store_true")
    remote_hsi = sub.add_parser("run-remote-hsi-reconstruction", help="Run HSI reconstruction on a remote worker.")
    remote_hsi.add_argument("--worker-id", required=True)
    remote_hsi.add_argument("--objective", default="Remote HSI reconstruction")
    remote_hsi.add_argument("--encoder", default="controlled_chromatic_edof")
    remote_hsi.add_argument("--reconstructor", default="tiny_cnn")
    remote_hsi.add_argument("--dataset", default="synthetic")
    remote_hsi.add_argument("--backend", default="deeplens")
    remote_native = sub.add_parser("run-remote-native-optimization-probe", help="Run native optimization probe on a remote worker.")
    remote_native.add_argument("--worker-id", required=True)
    remote_native.add_argument("--lens-class", required=True, choices=["ParaxialLens", "GeoLens", "DiffractiveLens", "HybridLens", "PSFNetLens"])
    remote_native.add_argument("--objective", required=True, choices=["minimize_psf_width", "maximize_center_intensity", "match_target_psf", "hsi_reconstruction_loss"])
    remote_native.add_argument("--max-steps", type=int, default=2)
    remote_native.add_argument("--learning-rate", type=float, default=1e-3)
    remote_native.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_native.add_argument("--strict-native", action="store_true")
    remote_native.add_argument("--allow-adapter-proxy", action="store_true")
    remote_surface = sub.add_parser("run-remote-deeplens-surface-optimization-probe", help="Run surface native optimization probe on a remote worker.")
    remote_surface.add_argument("--worker-id", required=True)
    remote_surface.add_argument("--surface", required=True)
    remote_surface.add_argument("--objective", required=True, choices=["minimize_phase_variance", "match_target_phase", "parameter_sanity_check"])
    remote_surface.add_argument("--max-steps", type=int, default=3)
    remote_surface.add_argument("--learning-rate", type=float, default=1e-3)
    remote_surface.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_lensfile = sub.add_parser("run-remote-deeplens-lensfile-optimization-probe", help="Run lens-file native optimization probe on a remote worker.")
    remote_lensfile.add_argument("--worker-id", required=True)
    remote_lensfile.add_argument("--lens-class", required=True, choices=["GeoLens", "HybridLens", "DiffractiveLens"])
    remote_lensfile.add_argument("--max-files", type=int, default=5)
    remote_lensfile.add_argument("--max-steps", type=int, default=2)
    remote_lensfile.add_argument("--learning-rate", type=float, default=1e-3)
    remote_lensfile.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_hsi_codesign = sub.add_parser("run-remote-native-hsi-codesign", help="Run native optical-HSI co-design on remote worker.")
    remote_hsi_codesign.add_argument("--worker-id", required=True)
    remote_hsi_codesign.add_argument("--optical-component", required=True, choices=["Fresnel", "Binary2Phase", "GeoLensCooke"])
    remote_hsi_codesign.add_argument("--objective", required=True)
    remote_hsi_codesign.add_argument("--max-steps", type=int, default=3)
    remote_hsi_codesign.add_argument("--learning-rate", type=float, default=1e-3)
    remote_hsi_codesign.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_hsi_codesign.add_argument("--bands", type=int, default=31)
    remote_hsi_codesign.add_argument("--image-size", type=int, default=32)
    remote_hsi_codesign.add_argument("--psf-size", type=int, default=16)
    remote_recon = sub.add_parser("run-remote-native-hsi-reconstruction-codesign", help="Run full native HSI reconstruction co-design on remote worker.")
    remote_recon.add_argument("--worker-id", required=True)
    remote_recon.add_argument("--optical-component", required=True, choices=["Fresnel", "Binary2Phase", "GeoLensCooke"])
    remote_recon.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    remote_recon.add_argument("--max-steps", type=int, default=5)
    remote_recon.add_argument("--optical-lr", type=float, default=1e-3)
    remote_recon.add_argument("--recon-lr", type=float, default=1e-3)
    remote_recon.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_recon.add_argument("--bands", type=int, default=31)
    remote_recon.add_argument("--image-size", type=int, default=32)
    remote_recon.add_argument("--psf-size", type=int, default=16)
    remote_waveoptics = sub.add_parser("run-remote-deeplens-waveoptics-probe", help="Run DeepLens wave-optics probe on remote worker.")
    remote_waveoptics.add_argument("--worker-id", required=True)
    remote_waveoptics.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens", "FresnelWave", "Binary2PhaseWave", "CustomLensFile"])
    remote_waveoptics.add_argument("--objective", required=True, choices=["minimize_psf_width", "match_target_psf", "minimize_hsi_reconstruction_loss"])
    remote_waveoptics.add_argument("--psf-size", type=int, default=32)
    remote_waveoptics.add_argument("--max-steps", type=int, default=3)
    remote_waveoptics.add_argument("--learning-rate", type=float, default=1e-3)
    remote_waveoptics.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_wave_hsi = sub.add_parser("run-remote-native-waveoptics-hsi-codesign", help="Run native wave-optics HSI co-design on remote worker.")
    remote_wave_hsi.add_argument("--worker-id", required=True)
    remote_wave_hsi.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens"])
    remote_wave_hsi.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    remote_wave_hsi.add_argument("--dataset", choices=["synthetic", "local_npz"], default="synthetic")
    remote_wave_hsi.add_argument("--max-steps", type=int, default=3)
    remote_wave_hsi.add_argument("--optical-lr", type=float, default=1e-3)
    remote_wave_hsi.add_argument("--recon-lr", type=float, default=1e-3)
    remote_wave_hsi.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_wave_hsi.add_argument("--bands", type=int, default=31)
    remote_wave_hsi.add_argument("--image-size", type=int, default=32)
    remote_wave_hsi.add_argument("--psf-size", type=int, default=32)
    remote_stable = sub.add_parser("run-remote-stable-native-lens-hsi-codesign", help="Run stable native lens HSI co-design on remote worker.")
    remote_stable.add_argument("--worker-id", required=True)
    remote_stable.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens"])
    remote_stable.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    remote_stable.add_argument("--dataset", default="synthetic")
    remote_stable.add_argument("--max-steps", type=int, default=10)
    remote_stable.add_argument("--optical-lr", type=float, default=1e-6)
    remote_stable.add_argument("--recon-lr", type=float, default=1e-3)
    remote_stable.add_argument("--optical-grad-clip", type=float, default=1.0)
    remote_stable.add_argument("--rollback-on-loss-increase", action="store_true", default=True)
    remote_stable.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_abl = sub.add_parser("run-remote-stable-native-lens-hsi-ablation", help="Run stable native lens HSI ablation on remote worker.")
    remote_abl.add_argument("--worker-id", required=True)
    remote_abl.add_argument("--candidate", required=True, choices=["GeoLensCooke", "DiffractiveLens", "HybridLens"])
    remote_abl.add_argument("--reconstructor", required=True, choices=["differentiable_linear", "tiny_cnn"])
    remote_abl.add_argument("--dataset", default="synthetic")
    remote_abl.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_geolens = sub.add_parser("run-remote-deeplens-native-geolens-hsi-codesign",
                                    help="Run DeepLens native GeoLens geometric HSI co-design on a remote worker.")
    remote_geolens.add_argument("--worker-id", required=True)
    remote_geolens.add_argument("--lens-file", default="auto:cooke")
    remote_geolens.add_argument("--dataset", default="synthetic")
    remote_geolens.add_argument("--reconstructor", default="differentiable_linear",
                                choices=["differentiable_linear", "tiny_cnn"])
    remote_geolens.add_argument("--max-steps", type=int, default=5)
    remote_geolens.add_argument("--optical-lr", type=float, default=1e-6)
    remote_geolens.add_argument("--rollback-on-loss-increase", action="store_true", default=True)
    remote_geolens.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_report = sub.add_parser("export-remote-execution-report", help="Export a remote execution report.")
    remote_report.add_argument("--job-id", required=True)
    geolens_hsi_report = sub.add_parser("export-native-geolens-hsi-report", help="Export a native GeoLens HSI report.")
    geolens_hsi_report.add_argument("--run-id", required=True)

    # ===================== Phase 24 CLI =====================
    sub.add_parser("list-optical-backends", help="List all registered optical backends.")
    inspect_backend = sub.add_parser("inspect-optical-backend", help="Inspect a specific optical backend.")
    inspect_backend.add_argument("--backend-id", required=True)

    run_v2 = sub.add_parser("run-experiment-v2", help="Run experiment via ExperimentControllerV2.")
    run_v2.add_argument("--backend-id", required=True)
    run_v2.add_argument("--task-type", required=True, choices=[
        "native_optimization_probe", "native_hsi_codesign",
        "native_hsi_reconstruction_codesign", "native_waveoptics_codesign",
        "stable_lens_hsi_codesign",
    ])
    run_v2.add_argument("--execution-target", choices=["local", "remote"], default="local")
    run_v2.add_argument("--worker-id")
    run_v2.add_argument("--spec-payload-json", default="{}")

    probe = sub.add_parser("run-lightweight-backend-probe", help="Run a lightweight backend availability probe.")
    probe.add_argument("--backend-id", required=True)
    probe.add_argument("--probe-depth", choices=["shallow", "deep"], default="shallow")
    probe.add_argument("--device", default="cpu")

    geo_hsi = sub.add_parser("run-deeplens-native-geolens-hsi-codesign",
                             help="Run full DeepLens native GeoLens geometric HSI co-design.")
    geo_hsi.add_argument("--lens-file", default="auto:cooke")
    geo_hsi.add_argument("--dataset", default="synthetic")
    geo_hsi.add_argument("--reconstructor", default="differentiable_linear",
                         choices=["differentiable_linear", "tiny_cnn"])
    geo_hsi.add_argument("--max-steps", type=int, default=5)
    geo_hsi.add_argument("--optical-lr", type=float, default=1e-6)
    geo_hsi.add_argument("--recon-lr", type=float, default=1e-3)
    geo_hsi.add_argument("--rollback-on-loss-increase", action="store_true", default=True)
    geo_hsi.add_argument("--device", default="cpu")

    recommend_strategy = sub.add_parser("recommend-next-strategy", help="Recommend next action via StrategyEngine.")
    recommend_strategy.add_argument("--backend-id", required=True)
    recommend_strategy.add_argument("--latest-result-json", required=True)
    recommend_strategy.add_argument("--objective", default="improve native lens simulation HSI co-design")

    sub.add_parser("compile-research-memory-v2", help="Compile ResearchMemoryV2 snapshot.")
    query_mem_v2 = sub.add_parser("query-research-memory-v2", help="Query ResearchMemoryV2.")
    query_mem_v2.add_argument("--memory-type")
    query_mem_v2.add_argument("--tag")
    query_mem_v2.add_argument("--content-contains")

    check_claim = sub.add_parser("check-claim", help="Check a claim through ClaimGateV2.")
    check_claim.add_argument("--claim-text", required=True)
    check_claim.add_argument("--backend-id", required=True)

    sub.add_parser("list-objective-profiles", help="List registered objective profiles.")
    inspect_profile = sub.add_parser("inspect-objective-profile", help="Inspect an objective profile.")
    inspect_profile.add_argument("--profile-id", required=True)

    sub.add_parser("audit-autograd-graph", help="Audit autograd graph for gradient flow breaks.")

    sub.add_parser("export-agent-system-report", help="Export agent system report markdown.")

    # ===================== Phase 25 CLI =====================
    aloop_v2_report = sub.add_parser("export-autonomous-loop-v2-report", help="Export Phase 25 autonomous loop report.")
    aloop_v2_report.add_argument("--loop-id", required=True)

    # ===================== Phase 26 CLI =====================
    plan_llm = sub.add_parser("plan-with-llm", help="Generate research proposals using LLM planner.")
    plan_llm.add_argument("--objective", required=True)
    plan_llm.add_argument("--provider", choices=["mock", "deepseek"], default="mock")
    plan_llm.add_argument("--allowed-backends", default="phase_to_fft_proxy,deeplens_geolens_geometric,local_synthetic_hsi")
    plan_llm.add_argument("--execution-mode", choices=["dry_run", "local", "remote_opt_in"], default="dry_run")
    plan_llm.add_argument("--remote-job-id")

    sub.add_parser("list-planner-traces", help="List saved planner traces.")

    inspect_trace = sub.add_parser("inspect-planner-trace", help="Inspect a planner trace.")
    inspect_trace.add_argument("--planner-run-id", required=True)

    export_plan = sub.add_parser("export-llm-planner-report", help="Export LLM planner report.")
    export_plan.add_argument("--planner-run-id", required=True)

    # ===================== Phase 27 CLI =====================
    check_prov = sub.add_parser("check-llm-provider", help="Check LLM provider availability.")
    check_prov.add_argument("--provider", choices=["mock", "deepseek"], default="deepseek")

    export_val = sub.add_parser("export-llm-provider-validation-report", help="Export LLM provider validation report.")
    export_val.add_argument("--planner-run-id", required=True)
    export_val.add_argument("--loop-id", required=True)

    # Update autonomous-loop-v2 with LLM planner options
    aloop_v2 = sub.add_parser("run-autonomous-research-loop-v2", help="Run closed-loop autonomous research loop (Phase 25+26).")
    aloop_v2.add_argument("--objective", required=True)
    aloop_v2.add_argument("--max-iterations", type=int, default=3)
    aloop_v2.add_argument("--execution-mode", choices=["dry_run", "local", "remote_opt_in"], default="dry_run")
    aloop_v2.add_argument("--allowed-backends", default="phase_to_fft_proxy,deeplens_geolens_geometric,local_synthetic_hsi")
    aloop_v2.add_argument("--allowed-task-types", default="stable_lens_hsi_codesign,native_hsi_codesign")
    aloop_v2.add_argument("--allow-remote", action="store_true")
    aloop_v2.add_argument("--remote-worker-id")
    aloop_v2.add_argument("--strict-claim-gate", action="store_true", default=True)
    aloop_v2.add_argument("--seed-result-path")
    aloop_v2.add_argument("--remote-job-id")
    # Phase 26 LLM options
    aloop_v2.add_argument("--planner-mode", choices=["rule_based", "llm_assisted", "llm_first_with_rule_fallback"], default="rule_based")
    aloop_v2.add_argument("--llm-provider", choices=["mock", "deepseek"], default="mock")
    aloop_v2.add_argument("--prefer-executable-actions", action="store_true", default=False)
    # Phase 29: multi-iteration trajectory controls
    aloop_v2.add_argument("--min-iterations-before-stop", type=int, default=2)
    aloop_v2.add_argument("--no-improvement-patience", type=int, default=2)
    aloop_v2.add_argument("--continue-on-claim-downgrade", action="store_true", default=True)
    aloop_v2.add_argument("--no-continue-on-claim-downgrade", dest="continue_on_claim_downgrade", action="store_false")
    aloop_v2.add_argument("--require-metrics-for-stop", action="store_true", default=True)
    aloop_v2.add_argument("--max-runtime-minutes-per-iter", type=int, default=10)
    # Phase 30: multi-backend switching
    aloop_v2.add_argument("--allow-backend-switching", action="store_true", default=True)
    aloop_v2.add_argument("--no-allow-backend-switching", dest="allow_backend_switching", action="store_false")
    aloop_v2.add_argument("--max-backend-switches", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "init-db":
        _init_db()
    elif args.command == "run-mvp":
        _run_mvp(args.objective, args.workspace_id, args.backend, args.use_llm, args.llm_provider)
    elif args.command == "query-memory":
        _query_memory(args.role, args.intent, args.query)
    elif args.command == "list-artifacts":
        _list_artifacts()
    elif args.command == "list-traces":
        _list_traces()
    elif args.command == "inspect-artifacts":
        _inspect_artifacts(args.run_id)
    elif args.command == "explain-claim":
        print(_compact_json(ClaimEvidenceManager().explain_claim(args.claim_id)))
    elif args.command == "list-plans":
        _list_plans()
    elif args.command == "match-plan":
        _match_plan(args.intent)
    elif args.command == "list-skills-memory":
        _list_skills_memory()
    elif args.command == "recommend-skills":
        _recommend_skills(args.intent)
    elif args.command == "run-benchmark":
        _run_benchmark(args.name, args.mode)
    elif args.command == "run-baselines":
        _run_baselines(args.objective, args.workspace_id, args.backend, args.encoder, args.output_dir, args.realization)
    elif args.command == "explain-rule":
        print(_compact_json(DesignRuleManager().explain_rule(args.rule_id)))
    elif args.command == "export-paper-summary":
        path = export_phase3_experiment_summary()
        print(f"markdown: {path}")
    elif args.command == "export-evidence-tables":
        paths = export_evidence_tables()
        print(f"claims: {paths['claims']}")
        print(f"rules: {paths['rules']}")
    elif args.command == "check-deeplens":
        _check_deeplens()
    elif args.command == "deeplens-capabilities":
        _deeplens_capabilities()
    elif args.command == "run-deeplens-smoke":
        _run_deeplens_smoke(args.objective)
    elif args.command == "compare-backends":
        _compare_backends(args.left, args.right)
    elif args.command == "export-phase6-report":
        path = export_phase6_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase7-report":
        path = export_phase7_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase8-report":
        path = export_phase8_report()
        print(f"markdown: {path}")
    elif args.command == "probe-deeplens-api":
        paths = export_deeplens_api_probe()
        print(f"json: {paths['json']}")
        print(f"markdown: {paths['markdown']}")
    elif args.command == "llm-providers":
        print(_compact_json(list_llm_providers()))
    elif args.command == "check-llm":
        _check_llm(args.provider)
    elif args.command == "test-llm":
        _test_llm(args.provider, args.prompt)
    elif args.command == "list-llm-calls":
        _list_llm_calls()
    elif args.command == "run-hsi-reconstruction":
        _run_hsi_reconstruction(args)
    elif args.command == "run-hsi-baselines":
        _run_hsi_baselines(args)
    elif args.command == "list-hsi-datasets":
        _list_hsi_datasets()
    elif args.command == "prepare-hsi-dataset":
        _prepare_hsi_dataset(args)
    elif args.command == "run-hsi-matrix":
        _run_hsi_matrix(args)
    elif args.command == "run-public-hsi-matrix":
        _run_public_hsi_matrix(args)
    elif args.command == "freeze-paper-protocol":
        path = freeze_paper_protocol()
        print(f"markdown: {path}")
    elif args.command == "export-phase9-report":
        path = export_phase9_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase10-report":
        path = export_phase10_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase11-report":
        path = export_phase11_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase12-report":
        path = export_phase12_report()
        print(f"markdown: {path}")
    elif args.command == "list-final-benchmarks":
        for b in FinalBenchmarkRegistry().list_benchmarks():
            print(f"{b['group']}\t{b['name']}\t{b['status']}")
    elif args.command == "collect-final-benchmark":
        registry = FinalBenchmarkRegistry()
        exported = registry.export_summary(Path("workspace/final_benchmark"))
        print(f"summary_json: {exported['summary_json']}")
        print(f"summary_md: {exported['summary_md']}")
        print(f"artifact_inventory: {exported['artifact_inventory']}")
    elif args.command == "export-paper-tables":
        result = export_paper_tables()
        print(f"markdown_dir: {result['markdown_dir']}")
        print(f"csv_dir: {result['csv_dir']}")
        print(f"json: {result['json_path']}")
        print(f"all_md: {result['all_md']}")
    elif args.command == "export-claim-boundary":
        import os as _os
        boundary = generate_claim_whitelist_blacklist()
        root = Path(_os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
        root.mkdir(parents=True, exist_ok=True)
        md_path = root / "claim_boundary.md"
        json_path = root / "claim_boundary.json"
        _write_boundary_files(boundary, md_path, json_path)
        print(f"markdown: {md_path}")
        print(f"json: {json_path}")
    elif args.command == "export-evidence-distribution":
        import os as _os
        dist = compute_evidence_distribution()
        root = Path(_os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
        root.mkdir(parents=True, exist_ok=True)
        md_path = root / "evidence_distribution.md"
        json_path = root / "evidence_distribution.json"
        _write_evidence_files(dist, md_path, json_path)
        print(f"markdown: {md_path}")
        print(f"json: {json_path}")
    elif args.command == "export-warnings-audit":
        audit = WarningsAudit()
        path = audit.export_report()
        print(f"markdown: {path}")
        print("If you have a pytest warnings log, run:")
        print("  python -m pytest 2>&1 | tee workspace/reports/pytest_phase13.log")
    elif args.command == "export-final-paper-package":
        result = export_final_paper_package()
        print(f"package_dir: {result['package_dir']}")
        print(f"manifest: {result['manifest_path']}")
    elif args.command == "export-phase13-report":
        path = export_phase13_report()
        print(f"markdown: {path}")
    elif args.command == "run-autonomous-loop":
        _run_autonomous_loop(args)
    elif args.command == "export-autonomous-loop-report":
        _export_autonomous_loop_report(args.loop_id)
    elif args.command == "run-codesign-loop":
        _run_codesign_loop(args)
    elif args.command == "compare-psf-sources":
        _compare_psf_sources(args)
    elif args.command == "export-phase16-report":
        path = export_phase16_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase19-report":
        from optiresearch.reports.phase19 import export_phase19_report
        path = export_phase19_report()
        print(f"markdown: {path}")
    elif args.command == "probe-deeplens-source":
        _probe_deeplens_source(args.remote_job_id)
    elif args.command == "inspect-deeplens-source":
        result = export_source_inspection()
        print(f"available: {result.get('available')}")
        print(f"modules: {list(result.get('modules', {}).keys())}")
        print(f"classes: {sum(len(v) for v in result.get('classes', {}).values())} found")
        root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
        print(f"json: {root / 'deeplens_source_inspection.json'}")
        print(f"markdown: {root / 'deeplens_source_inspection.md'}")
        if args.remote_job_id:
            export_remote_job_outputs(
                args.remote_job_id,
                "deeplens_source_inspection",
                {"status": "succeeded", "available": result.get("available"), "objective": "Inspect DeepLens source"},
                [root],
                {"job_type": "inspect_deeplens_source", "available": bool(result.get("available"))},
            )
    elif args.command == "inspect-deeplens-native-optimization":
        from optiresearch.adapters.deeplens_native_inspector import export_native_optimization_inspection
        result = export_native_optimization_inspection()
        print(f"available: {result.get('available')}")
        lens_classes = result.get("lens_classes", {})
        for cls_name, info in lens_classes.items():
            print(f"  {cls_name}: available={info.get('class_available')}, "
                  f"activate_grad={info.get('has_activate_grad')}, "
                  f"get_optimizer={info.get('has_get_optimizer')}, "
                  f"diffable={info.get('likely_differentiable')}")
        root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
        print(f"json: {root / 'deeplens_native_optimization_inspection.json'}")
        print(f"markdown: {root / 'deeplens_native_optimization_inspection.md'}")
        if getattr(args, "remote_job_id", None):
            export_remote_job_outputs(
                args.remote_job_id,
                "native_optimization_inspection",
                {"status": "succeeded" if result.get("available") else "failed", "available": result.get("available")},
                [root],
                {"job_type": "native_optimization_inspection", "available": bool(result.get("available"))},
            )
    elif args.command == "scan-deeplens-optimization-paths":
        from optiresearch.adapters.deeplens_native_inspector import export_optimization_path_scan
        result = export_optimization_path_scan()
        root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
        print(f"available: {result.get('available')}")
        print(f"entries: {result.get('summary', {}).get('entry_count', 0)}")
        print(f"json: {root / 'deeplens_optimization_path_scan.json'}")
        print(f"markdown: {root / 'deeplens_optimization_path_scan.md'}")
    elif args.command == "run-native-optimization-probe":
        _run_native_optimization_probe(args)
    elif args.command == "run-deeplens-surface-optimization-probe":
        _run_deeplens_surface_optimization_probe(args)
    elif args.command == "run-deeplens-lensfile-optimization-probe":
        _run_deeplens_lensfile_optimization_probe(args)
    elif args.command == "run-native-hsi-codesign":
        _run_native_hsi_codesign(args)
    elif args.command == "run-native-hsi-reconstruction-codesign":
        _run_native_hsi_reconstruction_codesign(args)
    elif args.command == "run-native-hsi-reconstruction-ablation":
        _run_native_hsi_reconstruction_ablation(args)
    elif args.command == "export-phase19b-report":
        from optiresearch.reports.phase19b import export_phase19b_report
        path = export_phase19b_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase20-report":
        from optiresearch.reports.phase20 import export_phase20_report
        path = export_phase20_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase21-report":
        from optiresearch.reports.phase21 import export_phase21_report
        path = export_phase21_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase22-report":
        from optiresearch.reports.phase22 import export_phase22_report
        path = export_phase22_report()
        print(f"markdown: {path}")
    elif args.command == "export-phase23-report":
        from optiresearch.reports.phase23 import export_phase23_report
        path = export_phase23_report()
        print(f"markdown: {path}")
    elif args.command == "diagnose-native-lens-hsi-codesign":
        from optiresearch.analysis.native_lens_hsi_diagnostics import diagnose_native_lens_hsi_codesign
        diagnosis = diagnose_native_lens_hsi_codesign(args.run_dir)
        import json as _json
        print(_json.dumps(diagnosis, indent=2, ensure_ascii=False, default=str))
    elif args.command == "run-stable-native-lens-hsi-codesign":
        _run_stable_native_lens_hsi_codesign(args)
    elif args.command == "run-stable-native-lens-hsi-ablation":
        _run_stable_native_lens_hsi_ablation(args)
    elif args.command == "scan-deeplens-waveoptics-paths":
        from optiresearch.adapters.deeplens_waveoptics_inspector import scan_deeplens_waveoptics_paths
        summary = scan_deeplens_waveoptics_paths()
        print(f"Scanned {summary['scanned_files']} candidates. Report: workspace/reports/")
    elif args.command == "run-deeplens-waveoptics-probe":
        _run_deeplens_waveoptics_probe(args)
    elif args.command == "run-native-waveoptics-hsi-codesign":
        _run_native_waveoptics_hsi_codesign(args)
    elif args.command == "run-deeplens-source-smoke":
        _run_deeplens_source_smoke(args.remote_job_id)
    elif args.command == "list-remote-workers":
        _list_remote_workers()
    elif args.command == "add-remote-worker":
        _add_remote_worker(args)
    elif args.command == "check-remote-worker":
        print(_compact_json(check_remote_worker(args.worker_id)))
    elif args.command == "run-remote-deeplens-source-smoke":
        payload = run_remote_deeplens_source_smoke(args.worker_id)
        _print_remote_payload(payload)
    elif args.command == "run-remote-codesign":
        payload = run_remote_codesign(
            args.worker_id,
            objective=args.objective,
            psf_source=args.psf_source,
            backend=args.backend,
            fallback_policy=args.fallback_policy,
            max_iterations=args.max_iterations,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-hsi-reconstruction":
        payload = run_remote_hsi_reconstruction(
            args.worker_id,
            objective=args.objective,
            encoder=args.encoder,
            reconstructor=args.reconstructor,
            dataset=args.dataset,
            backend=args.backend,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-native-optimization-probe":
        payload = run_remote_native_optimization_probe(
            args.worker_id,
            lens_class=args.lens_class,
            objective=args.objective,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            device=args.device,
            strict_native=args.strict_native,
            allow_adapter_proxy=args.allow_adapter_proxy,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-surface-optimization-probe":
        payload = run_remote_deeplens_surface_optimization_probe(
            args.worker_id,
            surface=args.surface,
            objective=args.objective,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-lensfile-optimization-probe":
        payload = run_remote_deeplens_lensfile_optimization_probe(
            args.worker_id,
            lens_class=args.lens_class,
            max_files=args.max_files,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-native-hsi-codesign":
        payload = run_remote_native_hsi_codesign(
            args.worker_id,
            args.optical_component,
            args.objective,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            device=args.device,
            bands=args.bands,
            image_size=args.image_size,
            psf_size=args.psf_size,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-native-hsi-reconstruction-codesign":
        payload = run_remote_native_hsi_reconstruction_codesign(
            args.worker_id,
            args.optical_component,
            args.reconstructor,
            max_steps=args.max_steps,
            optical_lr=args.optical_lr,
            recon_lr=args.recon_lr,
            device=args.device,
            bands=args.bands,
            image_size=args.image_size,
            psf_size=args.psf_size,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-waveoptics-probe":
        payload = run_remote_deeplens_waveoptics_probe(
            args.worker_id, args.candidate, args.objective,
            psf_size=args.psf_size, max_steps=args.max_steps,
            learning_rate=args.learning_rate, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-native-waveoptics-hsi-codesign":
        payload = run_remote_native_waveoptics_hsi_codesign(
            args.worker_id, args.candidate, args.reconstructor,
            dataset=args.dataset,
            max_steps=args.max_steps, optical_lr=args.optical_lr, recon_lr=args.recon_lr,
            device=args.device, bands=args.bands, image_size=args.image_size,
            psf_size=args.psf_size,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-stable-native-lens-hsi-codesign":
        payload = run_remote_stable_native_lens_hsi_codesign(
            args.worker_id, args.candidate, args.reconstructor,
            max_steps=args.max_steps, optical_lr=args.optical_lr, recon_lr=args.recon_lr,
            optical_grad_clip=args.optical_grad_clip, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-stable-native-lens-hsi-ablation":
        payload = run_remote_stable_native_lens_hsi_ablation(
            args.worker_id, args.candidate, args.reconstructor, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-native-geolens-hsi-codesign":
        payload = run_remote_deeplens_native_geolens_hsi_codesign(
            args.worker_id,
            lens_file=args.lens_file,
            dataset=args.dataset,
            reconstructor=args.reconstructor,
            max_steps=args.max_steps,
            optical_lr=args.optical_lr,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "export-remote-execution-report":
        path = export_remote_execution_report(args.job_id)
        print(f"markdown: {path}")
    elif args.command == "export-native-geolens-hsi-report":
        path = export_native_geolens_hsi_report(args.run_id)
        print(f"markdown: {path}")
    elif args.command == "list-optical-backends":
        _list_optical_backends()
    elif args.command == "inspect-optical-backend":
        _inspect_optical_backend(args.backend_id)
    elif args.command == "run-experiment-v2":
        _run_experiment_v2(args)
    elif args.command == "run-lightweight-backend-probe":
        _run_lightweight_backend_probe_cli(args)
    elif args.command == "run-deeplens-native-geolens-hsi-codesign":
        _run_deeplens_native_geolens_hsi(args)
    elif args.command == "recommend-next-strategy":
        _recommend_next_strategy(args)
    elif args.command == "compile-research-memory-v2":
        _compile_research_memory_v2()
    elif args.command == "query-research-memory-v2":
        _query_research_memory_v2(args)
    elif args.command == "check-claim":
        _check_claim_v2(args)
    elif args.command == "list-objective-profiles":
        _list_objective_profiles()
    elif args.command == "inspect-objective-profile":
        _inspect_objective_profile(args.profile_id)
    elif args.command == "audit-autograd-graph":
        _audit_autograd_graph(args)
    elif args.command == "export-agent-system-report":
        _export_agent_system_report()
    elif args.command == "run-autonomous-research-loop-v2":
        _run_autonomous_research_loop_v2(args)
    elif args.command == "export-autonomous-loop-v2-report":
        _export_autonomous_loop_v2_report(args.loop_id)
    elif args.command == "plan-with-llm":
        _plan_with_llm(args)
    elif args.command == "list-planner-traces":
        _list_planner_traces()
    elif args.command == "inspect-planner-trace":
        _inspect_planner_trace(args.planner_run_id)
    elif args.command == "export-llm-planner-report":
        _export_llm_planner_report(args.planner_run_id)
    elif args.command == "check-llm-provider":
        _check_llm_provider(args)
    elif args.command == "export-llm-provider-validation-report":
        _export_llm_provider_validation_report(args)


def _init_db() -> None:
    store = SQLiteStore()
    store.init_db()
    FileArtifactStore(store=store)
    print("OptiResearch workspace initialized.")
    print(f"database: {store.db_path}")
    print("artifact root: workspace/artifacts")


def _run_mvp(objective: str, workspace_id: str, backend: str, use_llm: bool = False, llm_provider: str | None = None) -> None:
    provider = get_llm_provider(llm_provider) if llm_provider else None
    result = run_mvp_flow(objective, workspace_id=workspace_id, backend=backend, use_llm=use_llm, llm_provider=provider)
    artifacts = FileArtifactStore().list_artifacts(run_id=result["run_id"])
    print(f"run_id: {result['run_id']}")
    print("artifacts:")
    for artifact in artifacts:
        print(f"- {artifact.uri}")
    memory = result["run_memory"]
    print("run_memory:")
    print(f"- status: {memory['current_status']}")
    print(f"- objective: {memory['objective']}")
    print(f"- metrics: {_compact_json(memory['best_metrics'])}")
    print("claims:")
    for claim in result["claims"]:
        edge_bits = [
            f"{edge.get('metric_name')}={edge.get('metric_value')}"
            for edge in claim.get("support_edges", [])
            if edge.get("metric_name")
        ]
        suffix = f" ({', '.join(edge_bits)})" if edge_bits else ""
        print(f"- {claim['status']}: {claim['text']}{suffix}")
    if result.get("errors"):
        print("errors:")
        for error in result["errors"]:
            print(f"- {_compact_json(error)}")


def _query_memory(role: str, intent: str, query: str) -> None:
    pack = MemoryRouter().query(role=role, intent=intent, query=query, scope={})
    print(_compact_json(pack))


def _list_artifacts() -> None:
    for artifact in FileArtifactStore().list_artifacts():
        print(f"{artifact.artifact_id}\t{artifact.uri}")


def _list_traces() -> None:
    for trace in MetaTraceWriter().list_traces():
        print(f"{trace.trace_id}\t{trace.actor}\t{trace.status}\t{trace.task}")


def _inspect_artifacts(run_id: str) -> None:
    store = FileArtifactStore()
    inspector = ArtifactInspector(store)
    for artifact in store.list_artifacts(run_id=run_id):
        print(_compact_json(inspector.inspect_artifact(artifact)))


def _list_plans() -> None:
    manager = PlanTemplateManager()
    manager.create_default_templates()
    for template in manager.list_templates():
        print(f"{template.template_id}\t{template.intent}\t{template.historical_success_rate}")


def _match_plan(intent: str) -> None:
    for template in PlanTemplateManager().match(intent):
        print(f"{template.template_id}\t{template.intent}\t{template.historical_success_rate}")


def _list_skills_memory() -> None:
    for memory in SkillMemoryManager().list_skill_memories():
        print(f"{memory.skill_id}\t{memory.version}\t{memory.success_rate}\tused={len(memory.used_in)}")


def _recommend_skills(intent: str) -> None:
    for memory in SkillMemoryManager().recommend_skills(intent):
        print(f"{memory.skill_id}\t{memory.version}\t{memory.success_rate}")


def _run_benchmark(name: str, mode: str) -> None:
    if name != "opti-memory":
        raise SystemExit(f"Unknown benchmark: {name}")
    report = OptiMemoryBenchRunner().run(name=name, mode=mode)
    print(_compact_json(report["summary"]))
    if "ablations" in report:
        print(_compact_json(report["ablations"].get(mode, {})))
    print(f"json: {report['paths']['json']}")
    print(f"markdown: {report['paths']['markdown']}")


def _run_baselines(objective: str, workspace_id: str, backend: str, encoder: str, output_dir: str | None, realization: str = "auto") -> None:
    report = run_baseline_batch(
        objective,
        workspace_id=workspace_id,
        backend=backend,
        encoder=encoder,
        output_root=Path(output_dir) if output_dir else None,
        realization=realization,
    )
    print(f"best_joint_tradeoff: {report['best_joint_tradeoff']['encoder_type']}")
    if report.get("design_rule_ids"):
        print(f"design_rule_ids: {', '.join(report['design_rule_ids'])}")
    for item in report["runs"]:
        metrics = item["metrics"]
        print(
            "{encoder}\t{run_id}\tdepth={depth}\tspectral={spectral}\tjoint={joint}".format(
                encoder=item["encoder_type"],
                run_id=item["run_id"],
                depth=metrics.get("psf_depth_similarity"),
                spectral=metrics.get("spectral_separability"),
                joint=item["joint_tradeoff_score"],
            )
        )
    root = Path(output_dir) if output_dir else Path("workspace/baselines") / backend
    print(f"json: {root / 'baseline_comparison.json'}")
    print(f"markdown: {root / 'baseline_comparison.md'}")


def _check_deeplens() -> None:
    environment = DeepLensAdapter().validate_environment()
    print(_compact_json({"probe": "DEEPLENS_ENVIRONMENT", **environment}))


def _deeplens_capabilities() -> None:
    environment = DeepLensAdapter().validate_environment()
    print("Capability\tAvailable\tReason\tEvidence")
    for item in environment["capabilities"]:
        print(f"{item['name']}\t{item['available']}\t{item['reason']}\t{item['evidence']}")


def _run_deeplens_smoke(objective: str) -> None:
    from optiresearch.agents.method_builder import MethodBuilder

    spec = MethodBuilder().build_mock_optical_spec(objective, backend="deeplens")
    output_dir = Path("./workspace/runs/deeplens_smoke")
    result = DeepLensAdapter().simulate_psf_cube(spec, None, output_dir)
    print(_compact_json(result.model_dump(mode="json")))


def _compare_backends(left: str, right: str) -> None:
    paths = export_backend_alignment_report(left, right)
    print(f"json: {paths['json']}")
    print(f"markdown: {paths['markdown']}")


def _check_llm(provider_name: str | None) -> None:
    provider = get_llm_provider(provider_name)
    summary = {
        "selected_provider": provider.provider_name,
        "available": provider.available(),
        "model": getattr(provider, "model", None),
        "base_url": getattr(provider, "base_url", getattr(provider, "url", None)),
        "thinking_type": getattr(provider, "thinking_type", None),
        "reasoning_effort": getattr(provider, "reasoning_effort", None),
        "error_code": None if provider.available() else f"{provider.provider_name.upper()}_UNAVAILABLE",
    }
    if hasattr(provider, "config_summary"):
        summary.update(provider.config_summary())
        summary["selected_provider"] = provider.provider_name
    print(_compact_json(summary))


def _test_llm(provider_name: str | None, prompt: str) -> None:
    provider = get_llm_provider(provider_name)
    try:
        response = provider.complete([{"role": "user", "content": prompt}])
        print(_compact_json(response.model_dump()))
    except LLMProviderError as exc:
        print(_compact_json(exc.to_dict()))


def _list_llm_calls() -> None:
    for trace in MetaTraceWriter().list_traces():
        if trace.metadata.get("llm_used") or trace.metadata.get("fallback_used"):
            print(f"{trace.trace_id}\t{trace.metadata.get('llm_provider')}\t{trace.metadata.get('llm_model')}\t{trace.metadata.get('fallback_used')}")


def _run_hsi_reconstruction(args: Any) -> None:
    result = run_hsi_reconstruction_flow(
        args.objective,
        backend=args.backend,
        encoder_type=args.encoder,
        workspace_id=args.workspace_id,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        realization=args.realization,
        forward_mode=args.forward_mode,
        reconstructor_type=args.reconstructor,
        dataset_pattern=args.dataset_pattern,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        reconstructor=args.reconstructor,
        use_optical_feature_maps=args.use_optical_feature_maps,
        tiny_cnn_epochs=args.tiny_cnn_epochs,
        tiny_cnn_hidden=args.tiny_cnn_hidden,
        device=args.device,
    )
    print(f"run_id: {result['run_id']}")
    print(f"evidence_level: {result['evidence_level']}")
    print(f"metrics: {_compact_json(result['metrics'])}")
    print("artifacts:")
    for uri in result["artifact_uris"]:
        print(f"- {uri}")
    if getattr(args, "remote_job_id", None):
        hsi_root = Path(os.getenv("OPTIRESEARCH_HSI_ROOT", "./workspace/hsi")) / "runs" / result["run_id"]
        export_dir = export_remote_job_outputs(
            args.remote_job_id,
            "hsi_reconstruction",
            {
                **result,
                "objective": args.objective,
                "backend": args.backend,
                "fallback_used": False,
            },
            [hsi_root],
            {
                "job_type": "hsi_reconstruction",
                "backend": args.backend,
                "evidence_level": result.get("evidence_level"),
                "fallback_used": False,
                **{k: v for k, v in result.get("metrics", {}).items() if isinstance(v, (int, float, bool))},
            },
        )
        print(f"remote_job_dir: {export_dir}")


def _run_hsi_baselines(args: Any) -> None:
    report = run_hsi_encoder_baselines(
        backend=args.backend,
        objective=args.objective,
        forward_mode=args.forward_mode,
        reconstructor_type=args.reconstructor,
        dataset_pattern=args.dataset_pattern,
    )
    print(f"best_reconstruction: {report['best_reconstruction']['encoder_type']}")
    for item in report["runs"]:
        print(f"{item['encoder_type']}\tPSNR={item['PSNR']}\tSAM={item['SAM']}\tscore={item['reconstruction_score']}")


def _list_hsi_datasets() -> None:
    print(_compact_json(list_hsi_dataset_adapters()))


def _prepare_hsi_dataset(args: Any) -> None:
    root = Path("workspace/hsi/datasets") / args.dataset
    adapter = get_hsi_dataset_adapter(args.dataset, path=args.path, crop_size=args.crop_size, patch_stride=args.patch_stride, normalization=args.normalization)
    result = adapter.prepare(root)
    print(_compact_json(result))
    if result.get("status") == "prepared":
        print(f"output_dir: {root}")


def _run_hsi_matrix(args: Any) -> None:
    result = run_hsi_matrix(
        datasets=_csv(args.datasets),
        backends=_csv(args.backends),
        encoders=_csv(args.encoders),
        reconstructors=_csv(args.reconstructors),
        forward_modes=_csv(args.forward_modes),
        objective=args.objective,
        workspace_id=args.workspace_id,
        dataset_path=args.dataset_path,
        use_optical_feature_maps=args.use_optical_feature_maps,
        tiny_cnn_epochs=args.tiny_cnn_epochs,
        tiny_cnn_hidden=args.tiny_cnn_hidden,
        device=args.device,
    )
    print(f"matrix_id: {result['matrix_id']}")
    print(f"summary: {_compact_json(result['summary'])}")
    root = Path("workspace/hsi/matrix") / result["matrix_id"]
    print(f"json: {root / 'hsi_matrix_results.json'}")
    print(f"markdown: {root / 'hsi_matrix_results.md'}")
    if getattr(args, "remote_job_id", None):
        export_dir = export_remote_job_outputs(
            args.remote_job_id,
            "hsi_matrix",
            {
                **result,
                "objective": args.objective,
                "backend": args.backends,
                "fallback_used": False,
            },
            [root],
            {"job_type": "hsi_matrix", "fallback_used": False, **result.get("summary", {})},
        )
        print(f"remote_job_dir: {export_dir}")


def _run_public_hsi_matrix(args: Any) -> None:
    result = run_public_hsi_matrix(
        dataset=args.dataset,
        dataset_path=args.path,
        backend=args.backend,
        encoders=_csv(args.encoders),
        reconstructors=_csv(args.reconstructors),
        forward_modes=_csv(args.forward_modes),
        realization=args.realization,
        workspace_id=args.workspace_id,
    )
    print(f"matrix_id: {result['matrix_id']}")
    print(f"status: {result['status']}")
    print(f"summary: {_compact_json(result['summary'])}")
    root = Path("workspace/hsi/public_matrix") / result["matrix_id"]
    print(f"json: {root / 'public_hsi_matrix_results.json'}")
    print(f"markdown: {root / 'public_hsi_matrix_results.md'}")


def _write_boundary_files(boundary: dict[str, Any], md_path: Path, json_path: Path) -> None:
    json_path.write_text(json.dumps(boundary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Claim Boundary", ""]
    for cat, title in [
        ("supported_claims", "Supported Claims"),
        ("qualified_claims", "Qualified Claims"),
        ("unsupported_claims", "Unsupported Claims"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        for c in boundary.get(cat, []):
            lines.append(f"- **{c['text']}** — {c['rationale']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_evidence_files(dist: dict[str, Any], md_path: Path, json_path: Path) -> None:
    json_path.write_text(json.dumps(dist, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Evidence Distribution",
        "",
        "## Count by Evidence Level",
        "",
        "| Evidence Level | Count |",
        "|---|---|",
    ]
    for level, count in dist.get("count_by_level", {}).items():
        lines.append(f"| {level} | {count} |")
    lines.extend([
        "",
        "## Status Distribution",
        "",
        "| Status | Count |",
        "|---|---|",
    ])
    for status, count in dist.get("status_counts", {}).items():
        lines.append(f"| {status} | {count} |")
    if dist.get("missing_evidence_warnings"):
        lines.extend(["", "## Missing Evidence Warnings", ""])
        for w in dist["missing_evidence_warnings"]:
            lines.append(f"- {w}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_autonomous_loop(args: Any) -> None:
    config = AutonomousLoopConfig(
        objective=args.objective,
        max_iterations=args.max_iterations,
        llm_provider=args.llm_provider,
        backend=args.backend,
        dataset=args.dataset,
        allowed_encoders=_csv(args.allowed_encoders),
        allowed_reconstructors=_csv(args.allowed_reconstructors),
        allowed_forward_modes=["depth_spectral_coded"],
        metadata={"execution_mode": args.execution_mode, "worker_id": args.worker_id or ""},
    )
    summary = run_autonomous_research_loop(config)
    print(_compact_json({
        "loop_id": summary.loop_id,
        "total_iterations": summary.total_iterations,
        "best_iteration": summary.best_iteration,
        "best_metrics": summary.best_metrics,
        "improvement_achieved": summary.improvement_achieved,
        "stopped_reason": summary.stopped_reason,
        "supported_claims": summary.supported_claims,
        "unsupported_claims": summary.unsupported_claims,
        "caveats": summary.caveats,
        "output_dir": f"workspace/autonomous_loops/{summary.loop_id}/",
    }))
    if getattr(args, "remote_job_id", None):
        output_root = Path("workspace/autonomous_loops") / summary.loop_id
        export_dir = export_remote_job_outputs(
            args.remote_job_id,
            "autonomous_loop",
            {
                **summary.model_dump(mode="json"),
                "run_id": summary.loop_id,
                "backend": args.backend,
                "fallback_used": False,
            },
            [output_root],
            {"job_type": "autonomous_loop", "fallback_used": False, **summary.best_metrics},
        )
        print(f"remote_job_dir: {export_dir}")


def _probe_deeplens_source(remote_job_id: str | None = None) -> None:
    from optiresearch.adapters.deeplens import DeepLensAdapter
    env = DeepLensAdapter().validate_environment()
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "deeplens_source_probe.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = [
        "# DeepLens Source Probe",
        "",
        f"**Available:** {env.get('available')}",
        f"**Import path:** `{env.get('import_path')}`",
        f"**Repo path:** `{env.get('repo_path')}`",
        f"**Is source checkout:** {env.get('is_source_checkout')}",
        f"**Version:** {env.get('deeplens_version')}",
        "",
        "## Available Modules",
        "| Module | Available |",
        "|---|---|",
    ]
    for mod, avail in env.get("available_modules", {}).items():
        lines.append(f"| {mod} | {avail} |")
    lines.extend([
        "",
        "## Available Classes",
        "| Class | Found |",
        "|---|---|",
    ])
    for cls, found in env.get("available_classes", {}).items():
        lines.append(f"| {cls} | {found} |")
    if env.get("missing_modules"):
        lines.extend(["", "## Missing Modules", ""])
        for m in env["missing_modules"]:
            lines.append(f"- {m}")
    (root / "deeplens_source_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"json: {root / 'deeplens_source_probe.json'}")
    print(f"markdown: {root / 'deeplens_source_probe.md'}")
    if remote_job_id:
        export_dir = export_remote_job_outputs(
            remote_job_id,
            "deeplens_source_probe",
            {
                "status": "succeeded" if env.get("available") else "failed",
                "available": env.get("available"),
                "import_path": env.get("import_path"),
                "objective": "Probe DeepLens source",
                "backend": "deeplens",
                "error_code": env.get("error_code"),
            },
            [root],
            {
                "job_type": "probe_deeplens_source",
                "available": bool(env.get("available")),
                "backend": "deeplens",
                "fallback_used": False,
                "import_path": env.get("import_path"),
            },
        )
        print(f"remote_job_dir: {export_dir}")


def _run_deeplens_source_smoke(remote_job_id: str | None = None) -> None:
    import numpy as np
    from optiresearch.adapters.deeplens import DeepLensAdapter
    output_dir = Path("workspace/runs/deeplens_source_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = DeepLensAdapter()
    env = adapter.validate_environment()
    print(f"available: {env['available']}")
    print(f"import_path: {env.get('import_path')}")
    print(f"is_source_checkout: {env.get('is_source_checkout')}")

    if not env["available"]:
        print(f"ERROR: {env.get('error_code')} - {env.get('message')}")
        if remote_job_id:
            manifest = {
                "available": False,
                "import_path": env.get("import_path"),
                "is_source_checkout": env.get("is_source_checkout"),
                "error_code": env.get("error_code"),
                "message": env.get("message"),
                "results": {},
            }
            (output_dir / "source_smoke_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            export_dir = export_remote_job_outputs(
                remote_job_id,
                "deeplens_source_smoke",
                {
                    "status": "failed",
                    "run_id": "deeplens_source_smoke",
                    "objective": "Run DeepLens source smoke",
                    "backend": "deeplens",
                    "available": False,
                    "fallback_used": False,
                    "error_code": env.get("error_code"),
                    "caveats": ["DeepLens source smoke did not run because DeepLens was unavailable."],
                },
                [output_dir],
                {
                    "job_type": "deeplens_source_smoke",
                    "available": False,
                    "backend": "deeplens",
                    "fallback_used": False,
                    "error_code": env.get("error_code"),
                },
            )
            print(f"remote_job_dir: {export_dir}")
        return

    results = {}
    for cls_name in ["ParaxialLens", "GeoLens", "DiffractiveLens", "HybridLens"]:
        if not env.get("available_classes", {}).get(cls_name):
            results[cls_name] = {"status": "unavailable"}
            continue
        try:
            mod_name = {
                "ParaxialLens": "deeplens.paraxiallens",
                "GeoLens": "deeplens.geolens",
                "DiffractiveLens": "deeplens.diffraclens",
                "HybridLens": "deeplens.hybridlens",
            }[cls_name]
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            try:
                if cls_name == "ParaxialLens":
                    lens = cls(foclen=50.0, fnum=2.8)
                    psf_fn = getattr(lens, "psf", None)
                    if callable(psf_fn):
                        psf = psf_fn(points=[0.0], ks=32)
                        arr = np.asarray(psf[0].detach().cpu().numpy() if hasattr(psf[0], 'detach') else psf[0])
                        np.savez_compressed(output_dir / f"{cls_name}_psf.npz", psf_cube=arr)
                        results[cls_name] = {"status": "success", "psf_shape": list(arr.shape)}
                    else:
                        results[cls_name] = {"status": "no_psf_method"}
                elif cls_name in ("GeoLens", "DiffractiveLens", "HybridLens"):
                    # Try basic instantiation and inspect available methods
                    sig = {}
                    for attr_name in dir(cls):
                        if not attr_name.startswith("_"):
                            obj = getattr(cls, attr_name)
                            if callable(obj):
                                sig[attr_name] = "callable"
                    results[cls_name] = {"status": "inspected", "methods_found": list(sig.keys())[:10]}
                else:
                    results[cls_name] = {"status": "unknown_class"}
            except Exception as e:
                results[cls_name] = {"status": "error", "error": str(e)[:200]}
        except Exception as e:
            results[cls_name] = {"status": "import_error", "error": str(e)[:200]}

    manifest = {
        "available": env["available"],
        "import_path": env.get("import_path"),
        "is_source_checkout": env.get("is_source_checkout"),
        "deeplens_version": env.get("deeplens_version"),
        "available_modules": env.get("available_modules", {}),
        "results": results,
    }
    (output_dir / "source_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(_compact_json({"smoke_results": results, "output_dir": str(output_dir)}))
    if remote_job_id:
        smoke_success = any(item.get("status") == "success" for item in results.values())
        export_dir = export_remote_job_outputs(
            remote_job_id,
            "deeplens_source_smoke",
            {
                "status": "succeeded" if env.get("available") else "failed",
                "run_id": "deeplens_source_smoke",
                "objective": "Run DeepLens source smoke",
                "backend": "deeplens",
                "available": env.get("available"),
                "import_path": env.get("import_path"),
                "is_source_checkout": env.get("is_source_checkout"),
                "smoke_success": smoke_success,
                "fallback_used": False,
                "error_code": None if env.get("available") else env.get("error_code"),
                "caveats": ["DeepLens source smoke validates import and minimal PSF artifact only."],
            },
            [output_dir],
            {
                "job_type": "deeplens_source_smoke",
                "available": bool(env.get("available")),
                "smoke_success": smoke_success,
                "backend": "deeplens",
                "backend_capability_level": "smoke",
                "fallback_used": False,
                "import_path": env.get("import_path"),
                "is_source_checkout": env.get("is_source_checkout"),
            },
        )
        print(f"remote_job_dir: {export_dir}")


def _run_native_optimization_probe(args: Any) -> None:
    from optiresearch.runtime.native_optimization_probe import run_native_optimization_probe
    from optiresearch.schemas.native_optimization import (
        NativeOptimizationProbeSpec,
        make_probe_id,
    )

    spec = NativeOptimizationProbeSpec(
        probe_id=make_probe_id(args.lens_class, args.objective),
        lens_class=args.lens_class,
        objective=args.objective,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        strict_native=args.strict_native,
        allow_adapter_proxy=args.allow_adapter_proxy,
        save_artifacts=True,
    )
    result = run_native_optimization_probe(spec)

    print(_compact_json({
        "probe_id": result.probe_id,
        "lens_class": result.lens_class,
        "objective": result.objective,
        "status": result.status,
        "realization_level": result.realization_level,
        "differentiable": result.differentiable,
        "native_parameter_update": result.native_parameter_update,
        "autograd_graph_exists": result.autograd_graph_exists,
        "gradient_norm": result.gradient_norm,
        "parameters_changed": result.parameters_changed,
        "loss_before": result.loss_before,
        "loss_after": result.loss_after,
        "error_code": result.error_code,
        "caveats": result.caveats,
    }))

    if getattr(args, "remote_job_id", None):
        from optiresearch.reports.remote_execution import export_remote_execution_report
        output_dir = Path("workspace/native_optimization") / result.probe_id
        export_remote_job_outputs(
            args.remote_job_id,
            "native_optimization_probe",
            {
                "status": result.status,
                "objective": args.objective,
                "lens_class": args.lens_class,
                "backend": "deeplens",
                "differentiable": result.differentiable,
                "native_parameter_update": result.native_parameter_update,
                "gradient_norm": result.gradient_norm,
                "loss_before": result.loss_before,
                "loss_after": result.loss_after,
                "fallback_used": False,
                "caveats": result.caveats,
            },
            [output_dir] if output_dir.exists() else [],
            {
                "job_type": "native_optimization_probe",
                "lens_class": args.lens_class,
                "objective": args.objective,
                "status": result.status,
                "differentiable": result.differentiable,
                "realization_level": result.realization_level,
            },
        )


def _run_deeplens_surface_optimization_probe(args: Any) -> None:
    from optiresearch.runtime.deeplens_surface_optimization_probe import run_surface_optimization_probe
    from optiresearch.schemas.surface_optimization import SurfaceOptimizationProbeSpec, make_surface_probe_id

    spec = SurfaceOptimizationProbeSpec(
        probe_id=make_surface_probe_id(args.surface, args.objective),
        surface_class=args.surface,
        objective=args.objective,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        save_artifacts=True,
    )
    result = run_surface_optimization_probe(spec)
    print(_compact_json(result.model_dump(mode="json")))

    if getattr(args, "remote_job_id", None):
        output_dir = Path("workspace/native_optimization") / f"surface_probe_{result.probe_id}"
        export_remote_job_outputs(
            args.remote_job_id,
            "deeplens_surface_optimization_probe",
            {
                **result.model_dump(mode="json"),
                "backend": "deeplens",
                "fallback_used": False,
            },
            [output_dir] if output_dir.exists() else [],
            {
                "job_type": "deeplens_surface_optimization_probe",
                "backend": "deeplens",
                "evidence_domain": "deeplens_native_optimization",
                "native_optimization_level": "component",
                "surface_class": result.surface_class,
                "status": result.status,
                "differentiable": result.differentiable,
                "requires_grad_true": result.metadata.get("requires_grad_true"),
                "gradient_norm": result.gradient_norm,
                "parameters_changed": result.parameters_changed,
                "optimizer_step_executed": result.metadata.get("optimizer_step_executed"),
            },
        )


def _run_deeplens_lensfile_optimization_probe(args: Any) -> None:
    from optiresearch.runtime.deeplens_lensfile_optimization_probe import run_lensfile_optimization_probe

    result = run_lensfile_optimization_probe(
        lens_class=args.lens_class,
        max_files=args.max_files,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        save_artifacts=True,
        remote_job_id=getattr(args, "remote_job_id", None),
    )
    print(_compact_json(result))


def _run_native_hsi_codesign(args: Any) -> None:
    from optiresearch.runtime.native_hsi_codesign_loop import run_native_optical_hsi_codesign
    from optiresearch.schemas.native_hsi_codesign import NativeOpticalHSICoDesignSpec, make_hsi_codesign_id
    from pathlib import Path

    spec = NativeOpticalHSICoDesignSpec(
        run_id=make_hsi_codesign_id(args.optical_component, args.objective),
        optical_component=args.optical_component,
        objective=args.objective,
        bands=args.bands,
        image_size=args.image_size,
        psf_size=args.psf_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    result = run_native_optical_hsi_codesign(spec)
    print(_compact_json(result.model_dump(mode="json")))

    if getattr(args, "remote_job_id", None):
        from optiresearch.runtime.remote_jobs import export_remote_job_outputs
        out_dir = Path("workspace/native_hsi_codesign") / spec.run_id
        export_remote_job_outputs(
            args.remote_job_id,
            "native_hsi_codesign",
            result.model_dump(mode="json"),
            [out_dir] if out_dir.exists() else [],
            {
                "job_type": "native_hsi_codesign",
                "evidence_domain": "deeplens_native_optimization",
                "native_optimization_level": "optical_hsi_codesign",
                "optical_component": result.optical_component,
                "differentiable": result.differentiable,
                "gradient_norm": result.gradient_norm,
                "parameters_changed": result.parameters_changed,
                "optimizer_step_executed": result.optimizer_step_executed,
                "hsi_loss_before": result.hsi_loss_before,
                "hsi_loss_after": result.hsi_loss_after,
            },
        )


def _run_native_hsi_reconstruction_codesign(args: Any) -> None:
    from optiresearch.runtime.native_hsi_reconstruction_codesign_loop import run_native_hsi_reconstruction_codesign
    from optiresearch.schemas.native_hsi_reconstruction_codesign import (
        NativeHSIReconstructionCoDesignSpec,
        make_recon_codesign_id,
    )
    from pathlib import Path

    spec = NativeHSIReconstructionCoDesignSpec(
        run_id=make_recon_codesign_id(args.optical_component, args.reconstructor),
        optical_component=args.optical_component,
        reconstructor=args.reconstructor,
        bands=args.bands, image_size=args.image_size, psf_size=args.psf_size,
        max_steps=args.max_steps, optical_lr=args.optical_lr, recon_lr=args.recon_lr,
        device=args.device,
    )
    result = run_native_hsi_reconstruction_codesign(spec)
    print(_compact_json(result.model_dump(mode="json")))

    if getattr(args, "remote_job_id", None):
        from optiresearch.runtime.remote_jobs import export_remote_job_outputs
        out_dir = Path("workspace/native_hsi_reconstruction_codesign") / spec.run_id
        export_remote_job_outputs(
            args.remote_job_id, "native_hsi_reconstruction_codesign",
            result.model_dump(mode="json"),
            [out_dir] if out_dir.exists() else [],
            {
                "job_type": "native_hsi_reconstruction_codesign",
                "evidence_domain": "deeplens_native_optimization",
                "optical_component": result.optical_component,
                "differentiable": result.differentiable,
                "optical_gradient_norm": result.optical_gradient_norm,
                "optical_parameters_changed": result.optical_parameters_changed,
                "optimizer_step_executed": result.optimizer_step_executed,
                "reconstruction_loss_before": result.reconstruction_loss_before,
                "reconstruction_loss_after": result.reconstruction_loss_after,
            },
        )


def _run_native_hsi_reconstruction_ablation(args: Any) -> None:
    from optiresearch.runtime.native_hsi_reconstruction_ablation import run_native_hsi_reconstruction_ablation
    import json

    summary = run_native_hsi_reconstruction_ablation(
        optical_component=args.optical_component,
        reconstructor_name=args.reconstructor,
        bands=args.bands, image_size=args.image_size, psf_size=args.psf_size,
        max_steps=args.max_steps, optical_lr=args.optical_lr, recon_lr=args.recon_lr,
        device=args.device,
    )
    json_summary = {k: v for k, v in summary.items() if k != "modes"}
    print(json.dumps(json_summary, indent=2, ensure_ascii=False, default=str))
    for mode, r in summary["modes"].items():
        print(f"\n{mode}: loss {r['loss_before']:.6f} -> {r['loss_after']:.6f}")


def _run_deeplens_waveoptics_probe(args: Any) -> None:
    from optiresearch.runtime.deeplens_waveoptics_probe import run_deeplens_waveoptics_probe
    from optiresearch.schemas.deeplens_waveoptics_probe import (
        DeepLensWaveOpticsProbeSpec, make_waveoptics_probe_id,
    )
    spec = DeepLensWaveOpticsProbeSpec(
        run_id=make_waveoptics_probe_id(args.candidate, args.objective),
        candidate=args.candidate, objective=args.objective,
        psf_size=args.psf_size, max_steps=args.max_steps,
        learning_rate=args.learning_rate, device=args.device,
    )
    result = run_deeplens_waveoptics_probe(spec)
    print(_compact_json(result.model_dump(mode="json")))
    if getattr(args, "remote_job_id", None):
        from optiresearch.runtime.remote_jobs import export_remote_job_outputs
        from pathlib import Path
        out_dir = Path("workspace/waveoptics_probe") / spec.run_id
        export_remote_job_outputs(args.remote_job_id, "deeplens_waveoptics_probe",
                                  result.model_dump(mode="json"),
                                  [out_dir] if out_dir.exists() else [], {})


def _run_native_waveoptics_hsi_codesign(args: Any) -> None:
    from optiresearch.runtime.native_waveoptics_hsi_codesign_loop import run_native_waveoptics_hsi_codesign
    from optiresearch.schemas.native_hsi_reconstruction_codesign import (
        NativeHSIReconstructionCoDesignSpec, make_recon_codesign_id,
    )
    spec = NativeHSIReconstructionCoDesignSpec(
        run_id=make_recon_codesign_id(args.candidate, args.reconstructor),
        optical_component=args.candidate, reconstructor=args.reconstructor,
        dataset=args.dataset,
        bands=args.bands, image_size=args.image_size, psf_size=args.psf_size,
        max_steps=args.max_steps, optical_lr=args.optical_lr, recon_lr=args.recon_lr,
        device=args.device,
    )
    result = run_native_waveoptics_hsi_codesign(spec)
    print(_compact_json(result.model_dump(mode="json")))
    if getattr(args, "remote_job_id", None):
        from optiresearch.runtime.remote_jobs import export_remote_job_outputs
        out_dir = Path("workspace/waveoptics_hsi_codesign") / spec.run_id
        export_remote_job_outputs(
            args.remote_job_id,
            "native_waveoptics_hsi_codesign",
            result.model_dump(mode="json"),
            [out_dir] if out_dir.exists() else [],
            {
                "differentiable": result.differentiable,
                "full_wave_optics": result.full_wave_optics,
                "phase_to_fft_proxy_used": result.phase_to_fft_proxy_used,
                "evidence_level": result.evidence_level,
                "reconstruction_loss_before": result.reconstruction_loss_before,
                "reconstruction_loss_after": result.reconstruction_loss_after,
                "mse_before": result.mse_before,
                "mse_after": result.mse_after,
                "psnr_before": result.psnr_before,
                "psnr_after": result.psnr_after,
                "sam_before": result.sam_before,
                "sam_after": result.sam_after,
                "optical_gradient_norm": result.optical_gradient_norm,
                "recon_gradient_norm": result.recon_gradient_norm,
                "optical_parameters_changed": result.optical_parameters_changed,
                "optimizer_step_executed": result.optimizer_step_executed,
            },
        )


def _run_stable_native_lens_hsi_codesign(args: Any) -> None:
    from optiresearch.runtime.stable_native_lens_hsi_loop import run_stable_native_lens_hsi_codesign
    from optiresearch.schemas.stable_native_lens_hsi import StableNativeLensHSISpec, make_stable_lens_id
    spec = StableNativeLensHSISpec(
        run_id=make_stable_lens_id(args.candidate, args.reconstructor),
        candidate=args.candidate, reconstructor=args.reconstructor,
        max_steps=args.max_steps, optical_lr=args.optical_lr, recon_lr=args.recon_lr,
        optical_grad_clip=args.optical_grad_clip,
        rollback_on_loss_increase=args.rollback_on_loss_increase,
        device=args.device,
    )
    result = run_stable_native_lens_hsi_codesign(spec)
    print(_compact_json(result.model_dump(mode="json")))


def _run_stable_native_lens_hsi_ablation(args: Any) -> None:
    from optiresearch.runtime.stable_native_lens_hsi_ablation import run_stable_native_lens_hsi_ablation
    import json as _json
    summary = run_stable_native_lens_hsi_ablation(
        candidate=args.candidate, reconstructor=args.reconstructor, device=args.device,
    )
    print(_json.dumps({k: v for k, v in summary.items() if k != "strategies"}, indent=2, ensure_ascii=False))
    for name, r in summary["strategies"].items():
        print(f"\n{name}: loss {r['loss_before']:.4f} -> {r['loss_after']:.4f} stable={r['stable']}")


def _run_codesign_loop(args: Any) -> None:
    spec = build_default_optimization_spec(
        target_metrics=["PSNR", "reconstruction_score"],
        backend=args.backend,
        objective=args.objective,
    )
    spec.max_iterations = args.max_iterations
    spec.encoder_type = args.encoder
    spec.reconstructor_type = args.reconstructor
    spec.forward_mode = args.forward_mode
    spec.dataset = args.dataset
    spec.llm_provider = args.llm_provider
    spec.psf_source = getattr(args, "psf_source", "parameterized_mock")
    spec.fallback_policy = getattr(args, "fallback_policy", "fallback_to_mock")
    spec.strict_deeplens = getattr(args, "strict_deeplens", False)
    result = run_codesign_loop(spec)
    print(_compact_json({
        "loop_id": result["loop_id"],
        "total_iterations": result["total_iterations"],
        "best_iteration": result.get("best_iteration", -1),
        "best_params": result.get("best_params", {}),
        "best_score": result.get("best_score"),
        "psf_source": result.get("psf_source"),
        "fallback_used": result.get("fallback_used_any"),
        "stopped_reason": result.get("stopped_reason", result.get("error")),
        "error": result.get("error"),
        "output_dir": result["output_dir"],
        "caveats": result.get("caveats", []),
    }))
    if getattr(args, "remote_job_id", None):
        output_dir = Path(result["output_dir"])
        export_dir = export_remote_job_outputs(
            args.remote_job_id,
            "codesign_loop",
            {
                **result,
                "backend": args.backend,
                "fallback_used": result.get("fallback_used_any", False),
            },
            [output_dir],
            {
                "job_type": "codesign_loop",
                "backend": args.backend,
                "psf_source": result.get("psf_source", args.psf_source),
                "fallback_used": result.get("fallback_used_any", False),
                "best_score": result.get("best_score"),
                "total_iterations": result.get("total_iterations", 0),
            },
        )
        print(f"remote_job_dir: {export_dir}")


def _compare_psf_sources(args: Any) -> None:
    output_dir = Path("workspace/codesign")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = build_default_optimization_spec(
        target_metrics=["PSNR"], backend="mock_deeplens", objective=args.objective,
    )
    spec.max_iterations = 2
    spec.llm_provider = "mock"

    # Run with left source
    spec.psf_source = args.left
    spec.fallback_policy = "fallback_to_mock"
    left_result = run_codesign_loop(spec)

    # Run with right source
    spec.psf_source = args.right
    spec.fallback_policy = "fallback_to_mock"
    right_result = run_codesign_loop(spec)

    comparison = {
        "left": {"psf_source": left_result.get("psf_source"), "best_score": left_result.get("best_score"), "trajectory": left_result.get("trajectory")},
        "right": {"psf_source": right_result.get("psf_source"), "best_score": right_result.get("best_score"), "trajectory": right_result.get("trajectory")},
    }
    (output_dir / "psf_source_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = [
        "# PSF Source Comparison",
        "",
        f"| Source | Best Score | Iterations |",
        "|---|---|---|",
        f"| {comparison['left']['psf_source']} | {comparison['left']['best_score']:.6f} | {len(comparison['left']['trajectory'])} |",
        f"| {comparison['right']['psf_source']} | {comparison['right']['best_score']:.6f} | {len(comparison['right']['trajectory'])} |",
    ]
    (output_dir / "psf_source_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"json: {output_dir / 'psf_source_comparison.json'}")
    print(f"markdown: {output_dir / 'psf_source_comparison.md'}")


def _export_autonomous_loop_report(loop_id: str) -> None:
    summary_path = Path("workspace/autonomous_loops") / loop_id / "autonomous_loop_summary.json"
    if not summary_path.exists():
        print(f"Loop summary not found: {summary_path}")
        return
    from optiresearch.schemas.autonomous import AutonomousLoopSummary
    from optiresearch.reports.autonomous_loop import export_autonomous_loop_report
    summary = AutonomousLoopSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    output_dir = summary_path.parent
    path = export_autonomous_loop_report(summary, output_dir)
    print(f"markdown: {path}")


def _list_remote_workers() -> None:
    for worker in RemoteWorkerRegistry().list_workers():
        print(
            "{worker_id}\t{host}\t{username}\t{python}".format(
                worker_id=worker.worker_id,
                host=worker.host,
                username=worker.username,
                python=worker.python_executable,
            )
        )


def _add_remote_worker(args: Any) -> None:
    worker = RemoteWorkerSpec(
        worker_id=args.worker_id,
        host=args.host,
        port=args.port,
        username=args.username,
        ssh_key_path=args.ssh_key_path,
        remote_project_dir=args.remote_project_dir,
        remote_workspace_dir=args.remote_workspace_dir,
        python_executable=args.python_executable,
        environment_name=args.environment_name,
        max_runtime_seconds=args.max_runtime_seconds,
        backend_tags=_csv(args.backend_tags),
        capabilities={},
    )
    registry = RemoteWorkerRegistry()
    registry.add_worker(worker)
    print(f"worker_id: {worker.worker_id}")
    print(f"config: {registry.config_path}")


def _print_remote_payload(payload: dict[str, Any]) -> None:
    result = payload["result"]
    print(_compact_json(result.model_dump(mode="json")))
    ingestion = payload.get("ingestion")
    if ingestion:
        print(f"ingestion: {Path(result.local_output_dir) / 'ingestion_summary.json'}")
        if ingestion.get("artifact_ids"):
            print(f"artifact_ids: {', '.join(ingestion['artifact_ids'])}")
        if ingestion.get("claims"):
            first_claim = ingestion["claims"][0]
            print(f"claim: {first_claim.get('status')} {first_claim.get('claim_id')}")




# ── Phase 26 helper functions ──────────────────────────────────────


def _plan_with_llm(args: Any) -> None:
    from optiresearch.agents.llm_planner import LLMPlanner
    planner = LLMPlanner()
    result = planner.plan(
        objective=args.objective,
        provider_name=args.provider,
        allowed_backends=_csv(args.allowed_backends),
        execution_mode=args.execution_mode,
    )
    claim_gate_decision = None
    if result.selected_proposal and result.selected_proposal.safe_wording:
        claim_gate_decision = {
            "original_claim": result.selected_proposal.proposed_claim,
            "safe_wording": result.selected_proposal.safe_wording,
            "downgraded": result.selected_proposal.proposed_claim != result.selected_proposal.safe_wording,
        }
    print(_compact_json({
        "status": result.status,
        "provider": result.provider,
        "planner_run_id": result.planner_run_id,
        "proposals_count": len(result.proposals),
        "selected_proposal": result.selected_proposal.model_dump(mode="json") if result.selected_proposal else None,
        "validation_errors": result.validation_errors,
        "claim_gate_decision": claim_gate_decision,
        "fallback_used": result.status == "fallback_used",
        "fallback_strategy": result.fallback_strategy,
        "planner_trace_path": result.planner_trace_path,
    }))


def _list_planner_traces() -> None:
    from optiresearch.agents.planner_trace import list_planner_traces
    for t in list_planner_traces():
        print(f"{t['run_id']}\tindex={t['has_index']}\t{t['path']}")


def _inspect_planner_trace(planner_run_id: str) -> None:
    from optiresearch.agents.planner_trace import inspect_planner_trace
    info = inspect_planner_trace(planner_run_id)
    if info is None:
        print(f"Trace not found: {planner_run_id}")
        return
    print(f"run_id: {info['run_id']}")
    for name, data in info.get("files", {}).items():
        if isinstance(data, (dict, list)):
            print(f"  {name}: {len(data)} entries")
        else:
            print(f"  {name}: {str(data)[:100]}")


def _export_llm_planner_report(planner_run_id: str) -> None:
    from optiresearch.reports.llm_planner_report import export_llm_planner_report
    output_dir = Path("workspace/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = export_llm_planner_report(planner_run_id, output_dir)
    print(f"markdown: {path}")


# ── Phase 27 helper functions ──────────────────────────────────────


def _check_llm_provider(args: Any) -> None:
    from optiresearch.agents.llm_provider_check import check_llm_provider
    result = check_llm_provider(args.provider)
    print(_compact_json(result))
    output_dir = Path("workspace/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_provider_check.json").write_text(
        _compact_json(result), encoding="utf-8"
    )
    lines = [
        "# LLM Provider Check",
        "",
        f"**Provider:** {result.get('provider', '-')}",
        f"**Status:** {result.get('status', '-')}",
        f"**Model:** {result.get('model', '-')}",
        f"**Base URL:** {result.get('base_url', '-')}",
        f"**Error Code:** {result.get('error_code', '-')}",
        f"**Error Message:** {result.get('error_message', '-')}",
        f"**Latency:** {result.get('latency_ms', '-')} ms",
    ]
    (output_dir / "llm_provider_check.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"json: {output_dir / 'llm_provider_check.json'}")
    print(f"markdown: {output_dir / 'llm_provider_check.md'}")


def _export_llm_provider_validation_report(args: Any) -> None:
    from optiresearch.reports.llm_provider_validation_report import (
        export_llm_provider_validation_report,
    )
    output_dir = Path("workspace/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = export_llm_provider_validation_report(
        planner_run_id=args.planner_run_id,
        loop_id=args.loop_id,
        output_dir=output_dir,
    )
    print(f"markdown: {path}")


# ── Phase 25 helper functions ──────────────────────────────────────


def _run_autonomous_research_loop_v2(args: Any) -> None:
    from optiresearch.schemas.autonomous_loop import AutonomousLoopSpec
    from optiresearch.runtime.autonomous_research_loop import run_autonomous_research_loop

    spec = AutonomousLoopSpec(
        objective=args.objective,
        max_iterations=args.max_iterations,
        execution_mode=args.execution_mode,
        allowed_backends=_csv(args.allowed_backends),
        allowed_task_types=_csv(args.allowed_task_types),
        allow_remote=getattr(args, "allow_remote", False),
        remote_worker_id=getattr(args, "remote_worker_id", None),
        strict_claim_gate=getattr(args, "strict_claim_gate", True),
        seed_result_path=getattr(args, "seed_result_path", None),
        # Phase 26 LLM options
        planner_mode=getattr(args, "planner_mode", "rule_based"),
        llm_provider=getattr(args, "llm_provider", "mock"),
        # Phase 28 executable mode
        prefer_executable_actions=getattr(args, "prefer_executable_actions", False),
        # Phase 29: multi-iteration trajectory controls
        min_iterations_before_stop=getattr(args, "min_iterations_before_stop", 2),
        no_improvement_patience=getattr(args, "no_improvement_patience", 2),
        continue_on_claim_downgrade=getattr(args, "continue_on_claim_downgrade", True),
        require_metrics_for_stop=getattr(args, "require_metrics_for_stop", True),
        max_runtime_minutes_per_iter=getattr(args, "max_runtime_minutes_per_iter", 10),
        # Phase 30: multi-backend switching
        allow_backend_switching=getattr(args, "allow_backend_switching", True),
        max_backend_switches=getattr(args, "max_backend_switches", 1),
    )
    result = run_autonomous_research_loop(spec)
    print(_compact_json({
        "loop_id": result.loop_id,
        "status": result.status,
        "objective": result.objective,
        "total_iterations": len(result.iterations),
        "iterations": [
            {
                "id": it.iteration_id,
                "action": it.strategy_recommendation.get("recommended_action"),
                "exec_status": it.execution_result.get("status"),
                "next_action": it.next_action,
                "stop_reason": it.stop_reason,
            }
            for it in result.iterations
        ],
        "final_supported_claims": result.final_supported_claims,
        "final_unsupported_claims": result.final_unsupported_claims,
        "trajectory_report_path": result.trajectory_report_path,
    }))
    if getattr(args, "remote_job_id", None):
        from optiresearch.runtime.remote_jobs import export_remote_job_outputs
        output_root = Path("workspace/autonomous_loops_v2") / result.loop_id
        export_remote_job_outputs(
            args.remote_job_id,
            "autonomous_research_loop_v2",
            {"loop_id": result.loop_id, "status": result.status},
            [output_root],
            {"job_type": "autonomous_research_loop_v2"},
        )


def _export_autonomous_loop_v2_report(loop_id: str) -> None:
    from optiresearch.schemas.autonomous_loop import AutonomousLoopResult
    from optiresearch.reports.autonomous_loop_report import export_autonomous_loop_report

    result_path = Path("workspace/autonomous_loops_v2") / loop_id / "loop_result.json"
    if not result_path.exists():
        print(f"Loop result not found: {result_path}")
        return
    result = AutonomousLoopResult.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    output_dir = result_path.parent
    path = export_autonomous_loop_report(result, output_dir)
    print(f"markdown: {path}")


# ── Phase 24 helper functions ──────────────────────────────────────

def _list_optical_backends() -> None:
    from optiresearch.backends.registry import list_backends, export_backend_registry_markdown, export_backend_registry_json
    from pathlib import Path
    import os as _os
    for b in list_backends():
        print(f"{b.backend_id}\t{b.backend_type}\t{b.differentiability_level}\t{', '.join(b.recommended_use_cases[:2])}")
    root = Path(_os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    export_backend_registry_markdown(root / "backend_registry.md")
    export_backend_registry_json(root / "backend_registry.json")
    print(f"markdown: {root / 'backend_registry.md'}")
    print(f"json: {root / 'backend_registry.json'}")


def _inspect_optical_backend(backend_id: str) -> None:
    from optiresearch.backends.registry import get_backend
    backend = get_backend(backend_id)
    if backend is None:
        print(f"Unknown backend: {backend_id}")
        return
    print(f"backend_id: {backend.backend_id}")
    print(f"label: {backend.label}")
    print(f"type: {backend.backend_type}")
    print(f"differentiability: {backend.differentiability_level}")
    print(f"claim_ceiling: {backend.claim_ceiling}")
    print(f"supports_native_optimization: {backend.supports_native_optimization}")
    print(f"supports_full_waveoptics: {backend.supports_full_waveoptics}")
    print(f"supports_remote_execution: {backend.supports_remote_execution}")
    if backend.known_failure_modes:
        print("known_failure_modes:")
        for fm in backend.known_failure_modes:
            print(f"  - {fm}")
    if backend.recommended_use_cases:
        print("recommended_use_cases:")
        for uc in backend.recommended_use_cases:
            print(f"  - {uc}")


def _run_experiment_v2(args: Any) -> None:
    import json as _json
    from optiresearch.runtime.experiment_controller_v2 import ExperimentControllerV2, ExperimentSpecV2
    from optiresearch.memory.schemas import make_deterministic_id

    payload = _json.loads(args.spec_payload_json)
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("v2", args.backend_id, args.task_type),
        task_type=args.task_type,
        backend_id=args.backend_id,
        execution_target=args.execution_target,
        worker_id=args.worker_id,
        spec_payload=payload,
    )
    ctrl = ExperimentControllerV2()
    if spec.execution_target == "remote":
        result = ctrl.run_remote(spec)
    else:
        result = ctrl.run_local(spec)
    print(_compact_json(result.model_dump(mode="json")))


def _run_lightweight_backend_probe_cli(args: Any) -> None:
    import json as _json
    from pathlib import Path as _Path
    from optiresearch.runtime.lightweight_experiments import (
        run_lightweight_backend_probe,
        run_deeplens_geolens_geometric_deep_probe,
    )

    if args.probe_depth == "deep":
        result = run_deeplens_geolens_geometric_deep_probe(
            backend_id=args.backend_id, device=args.device,
        )
    else:
        result = run_lightweight_backend_probe(
            backend_id=args.backend_id, device=args.device,
        )

    output_dir = _Path("workspace/backend_probes") / (result.run_id or "unknown")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_spec = {
        "backend_id": args.backend_id,
        "probe_depth": args.probe_depth,
        "device": args.device,
    }
    (output_dir / "probe_spec.json").write_text(
        _json.dumps(probe_spec, indent=2, default=str), encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        _json.dumps(result.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    print(_compact_json(result.model_dump(mode="json")))


def _run_deeplens_native_geolens_hsi(args: Any) -> None:
    import json as _json
    from pathlib import Path as _Path
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stable_native_lens_hsi_codesign,
    )
    from optiresearch.schemas.stable_native_lens_hsi import (
        StableNativeLensHSISpec, make_stable_lens_id,
    )

    run_id = make_stable_lens_id(
        "GeoLensCooke", args.reconstructor,
    )
    spec = StableNativeLensHSISpec(
        run_id=run_id,
        candidate="GeoLensCooke",
        reconstructor=args.reconstructor,
        dataset=args.dataset,
        max_steps=args.max_steps,
        optical_lr=args.optical_lr,
        recon_lr=args.recon_lr,
        rollback_on_loss_increase=args.rollback_on_loss_increase,
        device=args.device,
        full_wave_optics=False,
        phase_to_fft_proxy_used=False,
    )
    result = run_stable_native_lens_hsi_codesign(spec)

    output_dir = _Path("workspace/native_geolens_hsi") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (_Path(output_dir) / "spec.json").write_text(
        _json.dumps(spec.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    (_Path(output_dir) / "result.json").write_text(
        _json.dumps(result.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    print(_compact_json({
        "run_id": run_id,
        "status": result.status,
        "execution_fidelity": "deeplens_native_geometric",
        "full_wave_optics": result.full_wave_optics,
        "phase_to_fft_proxy_used": result.phase_to_fft_proxy_used,
        "deeplens_native_psf_path": result.deeplens_native_psf_path,
        "evidence_level": result.evidence_level,
        "error_code": result.error_code,
        "output_dir": str(output_dir),
    }))


def _recommend_next_strategy(args: Any) -> None:
    import json as _json
    from optiresearch.agents.strategy_engine import StrategyEngine

    latest = _json.loads(args.latest_result_json)
    engine = StrategyEngine()
    rec = engine.recommend(latest, args.backend_id)
    print(_json.dumps({
        "recommended_action": rec.recommended_action,
        "rationale": rec.rationale,
        "expected_claim_gain": rec.expected_claim_gain,
        "risk_level": rec.risk_level,
        "required_evidence": rec.required_evidence,
        "proposed_cli_commands": rec.proposed_cli_commands,
    }, indent=2, ensure_ascii=False))


def _compile_research_memory_v2() -> None:
    from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
    from pathlib import Path
    import os as _os
    mem = ResearchMemoryV2()
    snapshot = mem.compile_snapshot()
    total = sum(len(v) for v in snapshot.values())
    root = Path(_os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    path = mem.export_markdown(root / "research_memory_v2.md")
    print(f"Total entries: {total}")
    for mtype, entries in sorted(snapshot.items()):
        print(f"  {mtype}: {len(entries)} entries")
    print(f"markdown: {path}")


def _query_research_memory_v2(args: Any) -> None:
    from optiresearch.memory.research_memory_v2 import ResearchMemoryV2
    mem = ResearchMemoryV2()
    results = mem.query(
        memory_type=args.memory_type if args.memory_type else None,
        tags=[args.tag] if args.tag else None,
        content_contains=args.content_contains,
    )
    for entry in results:
        print(f"[{entry.memory_type}] {entry.content[:150]} (confidence={entry.confidence:.2f})")
    if not results:
        print("No matching entries found.")


def _check_claim_v2(args: Any) -> None:
    import json as _json
    from optiresearch.memory.claim_gate_v2 import ClaimGateV2
    gate = ClaimGateV2()
    decision = gate.check_claim(args.claim_text, args.backend_id)
    print(_json.dumps({
        "decision": decision.decision,
        "max_allowed_claim": decision.max_allowed_claim,
        "violation_reason": decision.violation_reason,
        "violation_type": decision.violation_type,
        "required_additional_evidence": decision.required_additional_evidence,
        "safe_wording": decision.safe_wording,
        "applicable_caveats": decision.applicable_caveats,
    }, indent=2, ensure_ascii=False))


def _list_objective_profiles() -> None:
    from optiresearch.objectives.optical_objectives import list_objective_profiles
    for p in list_objective_profiles():
        print(f"{p.profile_id}\t{', '.join(p.losses)}\tcompatible: {', '.join(p.compatible_backends)}")


def _inspect_objective_profile(profile_id: str) -> None:
    import json as _json
    from optiresearch.objectives.optical_objectives import get_objective_profile
    p = get_objective_profile(profile_id)
    if p:
        print(_json.dumps(p.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(f"Unknown profile: {profile_id}")


def _audit_autograd_graph(args: Any) -> None:
    import json as _json
    from optiresearch.diagnostics.autograd_auditor import audit_autograd_graph
    import torch
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 2
    loss = (y - 3.0) ** 2
    loss.backward()
    report = audit_autograd_graph(loss, {"x": x})
    print(_json.dumps({
        "loss_requires_grad": report.loss_requires_grad,
        "parameter_count": report.parameter_count,
        "parameters_with_grad": report.parameters_with_grad,
        "gradient_norms": report.gradient_norms,
        "zero_grad_parameters": report.zero_grad_parameters,
        "missing_grad_parameters": report.missing_grad_parameters,
        "suspected_breaks": report.suspected_breaks,
        "verdict": report.verdict,
        "recommendations": report.recommendations,
    }, indent=2, ensure_ascii=False, default=str))


def _export_agent_system_report() -> None:
    from optiresearch.reports.agent_system_report import export_agent_system_report
    path = export_agent_system_report()
    print(f"markdown: {path}")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
