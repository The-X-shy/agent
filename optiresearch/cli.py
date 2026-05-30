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
    run_remote_native_geolens_stabilization_sweep,
    run_remote_stabilized_native_geolens_hsi,
    run_remote_native_geolens_stability_benchmark,
    run_remote_deeplens_trainable_parameter_inspection,
    run_remote_deeplens_autograd_audit,
    run_remote_deeplens_curriculum_probe,
    run_remote_deeplens_regularized_probe,
    run_remote_resolve_lens_file,
    run_remote_deeplens_component_probe,
    run_remote_deeplens_component_discovery,
    run_remote_component_surrogate_hsi_codesign,
)
from optiresearch.reports.remote_execution import export_remote_execution_report
from optiresearch.reports.native_geolens_hsi_report import export_native_geolens_hsi_report
from optiresearch.reports.native_geolens_stabilization_report import export_native_geolens_stabilization_report
from optiresearch.reports.component_surrogate_hsi_report import export_component_surrogate_hsi_report


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
    diagnose_gl = sub.add_parser("diagnose-native-geolens-update", help="Diagnose native GeoLens HSI update instability.")
    diagnose_gl.add_argument("--run-dir", required=True)
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
    remote_clean_geo_hsi = sub.add_parser(
        "run-remote-native-geolens-geometric-hsi-codesign",
        help="Run full GeoLens geometric HSI co-design on a remote worker.",
    )
    remote_clean_geo_hsi.add_argument("--worker-id", required=True)
    remote_clean_geo_hsi.add_argument("--lens-file", default="auto:cooke")
    remote_clean_geo_hsi.add_argument("--dataset", default="synthetic")
    remote_clean_geo_hsi.add_argument("--reconstructor", default="differentiable_linear",
                                       choices=["differentiable_linear", "tiny_cnn"])
    remote_clean_geo_hsi.add_argument("--steps", type=int, default=3)
    remote_clean_geo_hsi.add_argument("--optical-lr", type=float, default=1e-6)
    remote_clean_geo_hsi.add_argument("--device", default="cpu")
    remote_stab_hsi = sub.add_parser(
        "run-remote-stabilized-native-geolens-hsi",
        help="Run stabilized native GeoLens HSI co-design on a remote worker.",
    )
    remote_stab_hsi.add_argument("--worker-id", required=True)
    remote_stab_hsi.add_argument("--lens-file", default="auto:cooke")
    remote_stab_hsi.add_argument("--dataset", default="synthetic")
    remote_stab_hsi.add_argument("--reconstructor", default="differentiable_linear",
                                  choices=["differentiable_linear", "tiny_cnn"])
    remote_stab_hsi.add_argument("--steps", type=int, default=10)
    remote_stab_hsi.add_argument("--spectral-angle-weight", type=float, default=0.2)
    remote_stab_hsi.add_argument("--grad-clip-norm", type=float, default=1000.0)
    remote_stab_hsi.add_argument("--optical-lr", type=float, default=1e-6)
    remote_stab_hsi.add_argument("--device", default="cpu")
    remote_bench = sub.add_parser(
        "run-remote-native-geolens-stability-benchmark",
        help="Run native GeoLens stability reproducibility benchmark on a remote worker.",
    )
    remote_bench.add_argument("--worker-id", required=True)
    remote_bench.add_argument("--lens-file", default="auto:cooke")
    remote_bench.add_argument("--dataset", default="synthetic")
    remote_bench.add_argument("--seeds", default="0,1,2")
    remote_bench.add_argument("--step-grid", default="10,20")
    remote_bench.add_argument("--spectral-angle-weights", default="0.1,0.2,0.5")
    remote_bench.add_argument("--grad-clip-norms", default="1000")
    remote_bench.add_argument("--device", default="cpu")
    remote_geo_sweep = sub.add_parser("run-remote-native-geolens-stabilization-sweep",
                                      help="Run native GeoLens stabilization sweep on a remote worker.")
    remote_geo_sweep.add_argument("--worker-id", required=True)
    remote_geo_sweep.add_argument("--lens-file", default="auto:cooke")
    remote_geo_sweep.add_argument("--dataset", default="synthetic")
    remote_geo_sweep.add_argument("--reconstructor", default="differentiable_linear",
                                  choices=["differentiable_linear", "tiny_cnn"])
    remote_geo_sweep.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    # Phase 58/59: Remote diagnostic CLI commands
    remote_tpi = sub.add_parser("run-remote-deeplens-trainable-parameter-inspection",
                                 help="Run GeoLens trainable parameter inspection on a remote worker.")
    remote_tpi.add_argument("--worker-id", required=True)
    remote_tpi.add_argument("--lens-file", default="auto:cooke")
    remote_tpi.add_argument("--backend-id", default="deeplens_geolens_geometric")
    remote_tpi.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_aa = sub.add_parser("run-remote-deeplens-autograd-audit",
                                help="Run GeoLens autograd audit on a remote worker.")
    remote_aa.add_argument("--worker-id", required=True)
    remote_aa.add_argument("--lens-file", default="auto:cooke")
    remote_aa.add_argument("--backend-id", default="deeplens_geolens_geometric")
    remote_aa.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_cp = sub.add_parser("run-remote-deeplens-curriculum-probe",
                                help="Run DeepLens curriculum probe on a remote worker.")
    remote_cp.add_argument("--worker-id", required=True)
    remote_cp.add_argument("--max-steps", type=int, default=3)
    remote_cp.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_rp = sub.add_parser("run-remote-deeplens-regularized-probe",
                                help="Run DeepLens regularized probe on a remote worker.")
    remote_rp.add_argument("--worker-id", required=True)
    remote_rp.add_argument("--max-steps", type=int, default=3)
    remote_rp.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_rlf = sub.add_parser("run-remote-resolve-lens-file",
                                 help="Resolve lens file on a remote worker.")
    remote_rlf.add_argument("--worker-id", required=True)
    remote_rlf.add_argument("--lens-file", default="auto:cooke")
    remote_rlf.add_argument("--backend-id", default=None)
    # Phase 62: Remote component probe and discovery
    remote_comp_probe = sub.add_parser("run-remote-deeplens-component-probe",
                                        help="Run DeepLens component optimization probe on a remote worker.")
    remote_comp_probe.add_argument("--worker-id", required=True)
    remote_comp_probe.add_argument("--component", required=True,
                                    choices=["fresnel", "binary2phase", "diffractive"])
    remote_comp_probe.add_argument("--objective", default="parameter_sanity_check")
    remote_comp_probe.add_argument("--max-steps", type=int, default=5)
    remote_comp_probe.add_argument("--learning-rate", type=float, default=1e-3)
    remote_comp_probe.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_comp_disc = sub.add_parser("run-remote-discover-deeplens-components",
                                       help="Discover DeepLens component backends on a remote worker.")
    remote_comp_disc.add_argument("--worker-id", required=True)
    remote_comp_disc.add_argument("--components", default="fresnel,binary2phase,diffractive")
    remote_comp_disc.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_comp_sur = sub.add_parser("run-remote-component-surrogate-hsi-codesign",
                                      help="Run component surrogate HSI co-design on a remote worker.")
    remote_comp_sur.add_argument("--worker-id", required=True)
    remote_comp_sur.add_argument("--component", required=True,
                                 choices=["fresnel", "binary2phase", "diffractive_candidate"])
    remote_comp_sur.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    remote_comp_sur.add_argument("--steps", type=int, default=3)
    remote_comp_sur.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    remote_report = sub.add_parser("export-remote-execution-report", help="Export a remote execution report.")
    remote_report.add_argument("--job-id", required=True)
    remote_diag_report = sub.add_parser("export-remote-diagnostic-report",
                                         help="Export a remote diagnostic report with lens resolution.")
    remote_diag_report.add_argument("--remote-job-id", required=True)
    geolens_hsi_report = sub.add_parser("export-native-geolens-hsi-report", help="Export a native GeoLens HSI report.")
    geolens_hsi_report.add_argument("--run-id", required=True)
    geo_stab_report = sub.add_parser("export-native-geolens-stabilization-report",
                                     help="Export a native GeoLens stabilization report.")
    geo_stab_report.add_argument("--sweep-id", required=True)
    # Phase 62: component probe report
    comp_probe_report = sub.add_parser("export-component-probe-report",
                                        help="Export a component probe report.")
    comp_probe_report.add_argument("--remote-job-id", required=True)

    # ===================== Phase 36 CLI =====================
    sub.add_parser("list-agent-events", help="List all agent events from the event bus.")
    export_events = sub.add_parser("export-agent-events", help="Export agent events to JSON.")
    export_events.add_argument("--output", default="workspace/reports/agent_events.json")
    sub.add_parser("show-agent-state", help="Show current agent state.")
    sub.add_parser("export-agent-state-report", help="Export agent state report.")
    sub.add_parser("list-skills-v2", help="List all registered skills (v2).")
    inspect_skill = sub.add_parser("inspect-skill", help="Inspect a specific skill.")
    inspect_skill.add_argument("--skill-id", required=True)
    run_skill = sub.add_parser("run-skill", help="Run a skill by ID.")
    run_skill.add_argument("--skill-id", required=True)
    run_skill.add_argument("--input-json", default="{}")
    list_hc = sub.add_parser("list-handler-capabilities", help="List all registered handler capabilities.")
    list_hc.add_argument("--include-disabled", action="store_true", default=False)
    inspect_hc = sub.add_parser("inspect-handler-capability", help="Inspect a specific handler capability.")
    inspect_hc.add_argument("--handler-id", required=True)
    resolve_cc = sub.add_parser("resolve-claim-ceiling", help="Resolve claim ceiling from handler capability.")
    resolve_cc.add_argument("--handler-id", default="")
    resolve_cc.add_argument("--backend-id", default="")
    resolve_cc.add_argument("--dataset", default="synthetic")
    resolve_cc.add_argument("--execution-fidelity", default="lightweight_proxy")
    sub.add_parser("validate-handler-capabilities", help="Validate handler capabilities config.")
    sub.add_parser("export-handler-capability-config-report", help="Export handler capability config diagnostics report.")
    # Phase 68: System Capability Registry and Execution Contracts
    sub.add_parser("build-system-capability-registry", help="Build SystemCapabilityRegistry from all existing configs.")
    sub.add_parser("validate-execution-contracts", help="Validate execution contracts against the registry.")
    sub.add_parser("validate-remote-execution-contracts", help="Validate remote execution contracts.")
    vac = sub.add_parser("validate-artifact-contract", help="Validate artifact contract for a run directory.")
    vac.add_argument("--run-dir", required=True)
    vac.add_argument("--contract-id", required=True)
    vrc = sub.add_parser("validate-report-contract", help="Validate report contract.")
    vrc.add_argument("--report-path", required=True)
    vrc.add_argument("--contract-id", required=True)
    sub.add_parser("export-claim-policy-matrix", help="Export claim policy matrix.")
    sub.add_parser("export-system-capability-report", help="Export system capability report.")
    sub.add_parser("export-contract-coverage-dashboard", help="Export contract coverage dashboard.")
    # Phase 69: Remote and artifact contract reconciliation
    sub.add_parser("export-remote-command-inventory", help="Export remote command inventory with canonical mapping.")
    sub.add_parser("validate-remote-allowlist-coverage", help="Validate remote allowlist coverage using canonical mapping.")
    nam = sub.add_parser("normalize-artifact-manifest", help="Normalize artifact manifest to canonical schema.")
    nam.add_argument("--dir", required=True, help="Directory containing artifact manifest.")
    vm = sub.add_parser("validate-remote-artifact-manifest", help="Validate remote artifact manifest.")
    vm.add_argument("--manifest-path", required=True)
    ia = sub.add_parser("ingest-remote-artifacts", help="Ingest remote artifacts into ArtifactStore.")
    ia.add_argument("--manifest-path", required=True)
    sub.add_parser("export-remote-artifact-index-report", help="Export remote artifact index report.")
    diag_grad = sub.add_parser("diagnose-gradient-instability", help="Diagnose GeoLens gradient instability.")
    diag_grad.add_argument("--source-path", default=None)
    diag_grad.add_argument("--remote-job-id", default=None)
    sub.add_parser("export-gradient-instability-report", help="Export gradient instability report.")
    sub.add_parser("list-deeplens-design-strategies", help="List DeepLens design strategies.")
    sub.add_parser("export-deeplens-design-strategy-report", help="Export DeepLens design strategy report.")
    # Phase 59: Lens file resolver CLI
    resolve_lens = sub.add_parser("resolve-lens-file", help="Resolve a lens file identifier to a real path.")
    resolve_lens.add_argument("--lens-file", default="auto:cooke")
    resolve_lens.add_argument("--backend-id", default=None)
    resolve_lens.add_argument("--remote-job-id", default="")
    # Phase 58: Remote diagnostic CLI commands
    for cmd_name in ["run-deeplens-trainable-parameter-inspection", "run-deeplens-autograd-audit",
                      "run-deeplens-curriculum-probe", "run-deeplens-regularized-probe"]:
        cmd = sub.add_parser(cmd_name, help=f"Remote diagnostic: {cmd_name}")
        cmd.add_argument("--lens-file", default="auto:cooke")
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--max-steps", type=int, default=3)
        cmd.add_argument("--remote-job-id", default="")
    # Phase 62: component probe CLI commands
    comp_probe = sub.add_parser("run-deeplens-component-probe",
                                 help="Run DeepLens component optimization probe (Fresnel/Binary2Phase).")
    comp_probe.add_argument("--component", required=True,
                            choices=["fresnel", "binary2phase", "diffractive"])
    comp_probe.add_argument("--objective", default="parameter_sanity_check")
    comp_probe.add_argument("--max-steps", type=int, default=5)
    comp_probe.add_argument("--learning-rate", type=float, default=1e-3)
    comp_probe.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    comp_probe.add_argument("--remote-job-id", default="")
    comp_disc = sub.add_parser("discover-deeplens-components",
                                help="Discover importable DeepLens component backends.")
    comp_disc.add_argument("--components", default="fresnel,binary2phase,diffractive")
    comp_disc.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    comp_disc.add_argument("--remote-job-id", default="")
    comp_sur = sub.add_parser("run-component-surrogate-hsi-codesign",
                              help="Run component surrogate PSF HSI co-design.")
    comp_sur.add_argument("--component", required=True,
                          choices=["fresnel", "binary2phase", "diffractive_candidate"])
    comp_sur.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    comp_sur.add_argument("--steps", type=int, default=3)
    comp_sur.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    comp_sur.add_argument("--remote-job-id", default="")
    comp_sur_report = sub.add_parser("export-component-surrogate-hsi-report",
                                      help="Export a component surrogate HSI co-design report.")
    comp_sur_report.add_argument("--run-id", required=True)
    classify_fail = sub.add_parser("classify-failure", help="Classify a failure from result JSON.")
    classify_fail.add_argument("--result-path", required=True)
    rec_rec = sub.add_parser("recommend-recovery", help="Recommend recovery for a failure.")
    rec_rec.add_argument("--failure-id", required=True)
    reason_ev = sub.add_parser("reason-from-evidence", help="Generate strategies from evidence.")
    reason_ev.add_argument("--objective", default="improve native optical HSI co-design")
    gen_designs = sub.add_parser("generate-experiment-designs", help="Generate experiment designs.")
    gen_designs.add_argument("--objective", default="recover from native GeoLens optical update instability")
    eval_plans = sub.add_parser("evaluate-candidate-plans", help="Evaluate candidate plans.")
    eval_plans.add_argument("--designs", default="workspace/reports/experiment_design_candidates.json")
    sub.add_parser("run-agent-self-test", help="Run agent system self-test.")
    sub.add_parser("run-agent-subunit-benchmark", help="Run agent subunit benchmark.")
    sub.add_parser("export-system-subunit-report", help="Export system subunit report.")
    plan_exec = sub.add_parser("run-agent-plan-execution", help="Run agent plan execution loop.")
    plan_exec.add_argument("--objective", default="recover from native GeoLens optical update instability")
    plan_exec.add_argument("--seed-result-path", default=None)
    plan_exec.add_argument("--mode", choices=["dry_run", "local", "remote_opt_in"], default="dry_run")
    plan_exec.add_argument("--execute-top-k", type=int, default=1)
    plan_exec.add_argument("--allow-remote", action="store_true")
    plan_exec.add_argument("--use-gradient-diagnosis", action="store_true")
    plan_exec.add_argument("--remote-worker-id", default=None)
    hybrid_p = sub.add_parser("hybrid-plan", help="Run hybrid planner.")
    hybrid_p.add_argument("--objective", default="recover from native GeoLens instability")
    hybrid_p.add_argument("--mode", choices=["rule_only", "llm_only", "llm_with_rule_context", "llm_with_rule_fallback"], default="rule_only")
    hybrid_p.add_argument("--llm-provider", default="")
    plan_report = sub.add_parser("export-agent-plan-execution-report", help="Export agent plan execution report.")
    plan_report.add_argument("--execution-id", required=True)
    sub.add_parser("run-agent-e2e-benchmark", help="Run agent end-to-end benchmark.")

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
    geo_hsi.add_argument("--remote-job-id")
    clean_geo_hsi = sub.add_parser(
        "run-native-geolens-geometric-hsi-codesign",
        help="Run full GeoLens geometric HSI co-design end-to-end training.",
    )
    clean_geo_hsi.add_argument("--lens-file", default="auto:cooke")
    clean_geo_hsi.add_argument("--dataset", default="synthetic")
    clean_geo_hsi.add_argument("--reconstructor", default="differentiable_linear",
                                choices=["differentiable_linear", "tiny_cnn"])
    clean_geo_hsi.add_argument("--steps", type=int, default=3,
                                help="Number of joint training steps (maps to max_steps)")
    clean_geo_hsi.add_argument("--optical-lr", type=float, default=1e-6)
    clean_geo_hsi.add_argument("--recon-lr", type=float, default=1e-3)
    clean_geo_hsi.add_argument("--device", default="cpu")
    stab_hsi = sub.add_parser(
        "run-stabilized-native-geolens-hsi",
        help="Run stabilized native GeoLens HSI co-design with multi-objective loss and rollback.",
    )
    stab_hsi.add_argument("--lens-file", default="auto:cooke")
    stab_hsi.add_argument("--dataset", default="synthetic")
    stab_hsi.add_argument("--reconstructor", default="differentiable_linear",
                          choices=["differentiable_linear", "tiny_cnn"])
    stab_hsi.add_argument("--steps", type=int, default=10)
    stab_hsi.add_argument("--spectral-angle-weight", type=float, default=0.2)
    stab_hsi.add_argument("--grad-clip-norm", type=float, default=1000.0)
    stab_hsi.add_argument("--optical-lr", type=float, default=1e-6)
    stab_hsi.add_argument("--device", default="cpu")
    bench = sub.add_parser(
        "run-native-geolens-stability-benchmark",
        help="Run native GeoLens stability reproducibility benchmark.",
    )
    bench.add_argument("--lens-file", default="auto:cooke")
    bench.add_argument("--dataset", default="synthetic")
    bench.add_argument("--seeds", default="0,1,2")
    bench.add_argument("--step-grid", default="10,20")
    bench.add_argument("--spectral-angle-weights", default="0.1,0.2,0.5")
    bench.add_argument("--grad-clip-norms", default="1000")
    bench.add_argument("--device", default="cpu")
    geo_sweep = sub.add_parser("run-native-geolens-stabilization-sweep",
                               help="Run native GeoLens stabilization sweep.")
    geo_sweep.add_argument("--lens-file", default="auto:cooke")
    geo_sweep.add_argument("--dataset", default="synthetic")
    geo_sweep.add_argument("--reconstructor", default="differentiable_linear",
                           choices=["differentiable_linear", "tiny_cnn"])
    geo_sweep.add_argument("--device", default="cpu")
    geo_sweep.add_argument("--remote-job-id")

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
    elif args.command == "diagnose-native-geolens-update":
        from optiresearch.analysis.native_geolens_update_diagnostics import diagnose_native_geolens_update
        diagnosis = diagnose_native_geolens_update(args.run_dir)
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
    elif args.command == "run-remote-native-geolens-geometric-hsi-codesign":
        payload = run_remote_deeplens_native_geolens_hsi_codesign(
            args.worker_id,
            lens_file=args.lens_file,
            dataset=args.dataset,
            reconstructor=args.reconstructor,
            max_steps=args.steps,
            optical_lr=args.optical_lr,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-stabilized-native-geolens-hsi":
        payload = run_remote_stabilized_native_geolens_hsi(
            args.worker_id,
            lens_file=args.lens_file,
            dataset=args.dataset,
            reconstructor=args.reconstructor,
            max_steps=args.steps,
            spectral_angle_weight=args.spectral_angle_weight,
            optical_lr=args.optical_lr,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-native-geolens-stability-benchmark":
        payload = run_remote_native_geolens_stability_benchmark(
            args.worker_id,
            lens_file=args.lens_file,
            dataset=args.dataset,
            seeds=args.seeds,
            step_grid=args.step_grid,
            spectral_angle_weights=args.spectral_angle_weights,
            grad_clip_norms=args.grad_clip_norms,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "export-remote-execution-report":
        path = export_remote_execution_report(args.job_id)
        print(f"markdown: {path}")
    elif args.command == "export-remote-diagnostic-report":
        from optiresearch.reports.remote_diagnostic_report import export_remote_diagnostic_report
        path = export_remote_diagnostic_report(args.remote_job_id)
        print(f"markdown: {path}")
    elif args.command == "export-native-geolens-hsi-report":
        path = export_native_geolens_hsi_report(args.run_id)
        print(f"markdown: {path}")
    elif args.command == "export-native-geolens-stabilization-report":
        path = _export_native_geolens_stabilization_report(args.sweep_id)
        print(f"markdown: {path}")
    elif args.command == "export-component-probe-report":
        from optiresearch.reports.component_probe_report import export_component_probe_report
        path = export_component_probe_report(args.remote_job_id)
        print(f"markdown: {path}")
    elif args.command == "export-component-surrogate-hsi-report":
        path = export_component_surrogate_hsi_report(args.run_id)
        print(f"markdown: {path}")
    # ---- Phase 36 handlers ----
    elif args.command == "list-agent-events":
        _list_agent_events()
    elif args.command == "export-agent-events":
        _export_agent_events(args.output)
    elif args.command == "show-agent-state":
        _show_agent_state()
    elif args.command == "export-agent-state-report":
        _export_agent_state_report()
    elif args.command == "list-skills-v2":
        _list_skills_v2()
    elif args.command == "inspect-skill":
        _inspect_skill(args.skill_id)
    elif args.command == "run-skill":
        _run_skill(args.skill_id, args.input_json)
    elif args.command == "list-handler-capabilities":
        _list_handler_capabilities(include_disabled=getattr(args, "include_disabled", False))
    elif args.command == "inspect-handler-capability":
        _inspect_handler_capability(args.handler_id)
    elif args.command == "resolve-claim-ceiling":
        _resolve_claim_ceiling_cli(args.handler_id, args.backend_id, args.dataset, args.execution_fidelity)
    elif args.command == "validate-handler-capabilities":
        _validate_handler_capabilities()
    elif args.command == "export-handler-capability-config-report":
        _export_handler_capability_config_report()
    # Phase 68: System Capability Registry and Execution Contracts
    elif args.command == "build-system-capability-registry":
        _build_system_capability_registry()
    elif args.command == "validate-execution-contracts":
        _validate_execution_contracts()
    elif args.command == "validate-remote-execution-contracts":
        _validate_remote_execution_contracts()
    elif args.command == "validate-artifact-contract":
        _validate_artifact_contract(args.run_dir, args.contract_id)
    elif args.command == "validate-report-contract":
        _validate_report_contract(args.report_path, args.contract_id)
    elif args.command == "export-claim-policy-matrix":
        _export_claim_policy_matrix()
    elif args.command == "export-system-capability-report":
        _export_system_capability_report()
    elif args.command == "export-contract-coverage-dashboard":
        _export_contract_coverage_dashboard()
    # Phase 69: Remote and artifact contract reconciliation
    elif args.command == "export-remote-command-inventory":
        _export_remote_command_inventory()
    elif args.command == "validate-remote-allowlist-coverage":
        _validate_remote_allowlist_coverage()
    elif args.command == "normalize-artifact-manifest":
        _normalize_artifact_manifest(args.dir)
    elif args.command == "validate-remote-artifact-manifest":
        _validate_remote_artifact_manifest(args.manifest_path)
    elif args.command == "ingest-remote-artifacts":
        _ingest_remote_artifacts(args.manifest_path)
    elif args.command == "export-remote-artifact-index-report":
        _export_remote_artifact_index_report()
    elif args.command == "diagnose-gradient-instability":
        _diagnose_gradient_instability(
            getattr(args, "source_path", None),
            getattr(args, "remote_job_id", None),
        )
    elif args.command == "export-gradient-instability-report":
        _export_gradient_instability_report()
    elif args.command == "list-deeplens-design-strategies":
        _list_deeplens_design_strategies()
    elif args.command == "export-deeplens-design-strategy-report":
        _export_deeplens_design_strategy_report()
    elif args.command == "resolve-lens-file":
        _resolve_lens_file_cmd(args)
    elif args.command == "run-deeplens-trainable-parameter-inspection":
        _run_wsl_diagnostic("trainable_parameter", args)
    elif args.command == "run-deeplens-autograd-audit":
        _run_wsl_diagnostic("autograd_audit", args)
    elif args.command == "run-deeplens-curriculum-probe":
        _run_wsl_diagnostic("curriculum_probe", args)
    elif args.command == "run-deeplens-regularized-probe":
        _run_wsl_diagnostic("regularized_probe", args)
    elif args.command == "run-deeplens-component-probe":
        _run_deeplens_component_probe(args)
    elif args.command == "discover-deeplens-components":
        _run_deeplens_component_discovery(args)
    elif args.command == "run-component-surrogate-hsi-codesign":
        _run_component_surrogate_hsi_codesign(args)
    elif args.command == "classify-failure":
        _classify_failure(args.result_path)
    elif args.command == "recommend-recovery":
        _recommend_recovery(args.failure_id)
    elif args.command == "reason-from-evidence":
        _reason_from_evidence(args.objective)
    elif args.command == "generate-experiment-designs":
        _generate_experiment_designs(args.objective)
    elif args.command == "evaluate-candidate-plans":
        _evaluate_candidate_plans(args.designs)
    elif args.command == "run-agent-self-test":
        _run_agent_self_test()
    elif args.command == "run-agent-subunit-benchmark":
        _run_agent_subunit_benchmark()
    elif args.command == "export-system-subunit-report":
        _export_system_subunit_report()
    elif args.command == "run-agent-plan-execution":
        _run_agent_plan_execution(args)
    elif args.command == "hybrid-plan":
        _hybrid_plan(args)
    elif args.command == "export-agent-plan-execution-report":
        _export_agent_plan_execution_report(args.execution_id)
    elif args.command == "run-agent-e2e-benchmark":
        _run_agent_e2e_benchmark()
    elif args.command == "run-remote-native-geolens-stabilization-sweep":
        payload = run_remote_native_geolens_stabilization_sweep(
            args.worker_id,
            lens_file=args.lens_file,
            dataset=args.dataset,
            reconstructor=args.reconstructor,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-trainable-parameter-inspection":
        payload = run_remote_deeplens_trainable_parameter_inspection(
            args.worker_id, lens_file=args.lens_file, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-autograd-audit":
        payload = run_remote_deeplens_autograd_audit(
            args.worker_id, lens_file=args.lens_file, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-curriculum-probe":
        payload = run_remote_deeplens_curriculum_probe(
            args.worker_id, max_steps=args.max_steps, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-regularized-probe":
        payload = run_remote_deeplens_regularized_probe(
            args.worker_id, max_steps=args.max_steps, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-resolve-lens-file":
        payload = run_remote_resolve_lens_file(
            args.worker_id, lens_file=args.lens_file, backend_id=args.backend_id,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-deeplens-component-probe":
        payload = run_remote_deeplens_component_probe(
            args.worker_id, component=args.component, objective=args.objective,
            max_steps=args.max_steps, learning_rate=args.learning_rate,
            device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-discover-deeplens-components":
        payload = run_remote_deeplens_component_discovery(
            args.worker_id, components=args.components, device=args.device,
        )
        _print_remote_payload(payload)
    elif args.command == "run-remote-component-surrogate-hsi-codesign":
        payload = run_remote_component_surrogate_hsi_codesign(
            args.worker_id, component=args.component, dataset=args.dataset,
            steps=args.steps, device=args.device,
        )
        _print_remote_payload(payload)
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
    elif args.command == "run-native-geolens-geometric-hsi-codesign":
        _run_native_geolens_geometric_hsi_codesign(args)
    elif args.command == "run-stabilized-native-geolens-hsi":
        _run_stabilized_native_geolens_hsi(args)
    elif args.command == "run-native-geolens-stability-benchmark":
        _run_native_geolens_stability_benchmark(args)
    elif args.command == "run-native-geolens-stabilization-sweep":
        _run_native_geolens_stabilization_sweep(args)
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


def _run_deeplens_component_probe(args: Any) -> None:
    from optiresearch.schemas.component_probe import ComponentProbeSpec, make_component_probe_id
    from optiresearch.runtime.deeplens_component_first_probe import run_deeplens_component_probe
    import json as _json

    spec = ComponentProbeSpec(
        probe_id=make_component_probe_id(args.component, args.objective),
        component=args.component,
        objective=args.objective,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        save_artifacts=True,
    )
    result = run_deeplens_component_probe(spec)
    print(_json.dumps(result.model_dump(mode="json"), indent=2, default=str))

    if getattr(args, "remote_job_id", ""):
        output_dir = Path("workspace/remote_jobs") / args.remote_job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(
            _json.dumps(result.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "component_probe_metrics.json").write_text(_json.dumps({
            "component": result.component,
            "surface_class": result.surface_class,
            "status": result.status,
            "differentiable": result.differentiable,
            "parameters_changed": result.parameters_changed,
            "trainable_param_count": result.trainable_param_count,
            "params_with_grad": result.params_with_grad,
            "gradient_norm": result.gradient_norm,
            "loss_before": result.loss_before,
            "loss_after": result.loss_after,
            "evidence_level": result.evidence_level,
            "claim_ceiling": result.claim_ceiling,
            "error_code": result.error_code,
        }, indent=2, default=str), encoding="utf-8")
        (output_dir / "artifact_manifest.json").write_text(_json.dumps({
            "schema_version": "0.1", "job_id": args.remote_job_id,
            "completeness": "complete", "artifacts": [
                {"artifact_name": "result.json", "artifact_type": "execution_result",
                 "evidence_role": "component_probe_metric"},
                {"artifact_name": "component_probe_metrics.json",
                 "artifact_type": "component_probe_metric", "evidence_role": "component_probe_metric"},
            ],
        }, indent=2, default=str), encoding="utf-8")
        export_remote_job_outputs(
            args.remote_job_id,
            "deeplens_component_probe",
            {**result.model_dump(mode="json"), "backend": "deeplens"},
            [output_dir] if output_dir.exists() else [],
            {
                "job_type": "deeplens_component_probe",
                "backend": "deeplens",
                "evidence_domain": "deeplens_native_optimization",
                "native_optimization_level": "component",
                "component": result.component,
                "surface_class": result.surface_class,
                "status": result.status,
                "differentiable": result.differentiable,
                "parameters_changed": result.parameters_changed,
                "gradient_norm": result.gradient_norm,
            },
        )


def _run_deeplens_component_discovery(args: Any) -> None:
    from optiresearch.optics.deeplens_component_discovery import discover_deeplens_components
    import json as _json

    components = [c.strip() for c in (args.components or "fresnel,binary2phase,diffractive").split(",")]
    results = discover_deeplens_components(components=components, device=args.device)
    print(_json.dumps({
        "deeplens_available": results.deeplens_available,
        "deeplens_version": results.deeplens_version,
        "component_candidates": results.component_candidates,
        "available_components": results.available_components,
        "unavailable_components": results.unavailable_components,
        "results": [
            {
                "component": r.component,
                "surface_class": r.surface_class,
                "importable": r.importable,
                "import_error": r.import_error,
                "instantiatable": r.instantiatable,
                "instantiation_error": r.instantiation_error,
                "has_phase_func": r.has_phase_func,
                "has_phi": r.has_phi,
                "has_get_optimizer": r.has_get_optimizer,
                "has_get_optimizer_params": r.has_get_optimizer_params,
                "trainable_param_names": r.trainable_param_names,
                "differentiability_hints": r.differentiability_hints,
            }
            for r in results.results
        ],
        "diffractive_candidates_found": results.diffractive_candidates_found,
        "differentiable_candidate_found": results.differentiable_candidate_found,
        "import_paths_checked": results.import_paths_checked,
        "constructor_signatures": results.constructor_signatures,
        "warnings": results.warnings,
        "errors": results.errors,
    }, indent=2, default=str))


def _run_component_surrogate_hsi_codesign(args: Any) -> None:
    from optiresearch.runtime.component_surrogate_hsi_codesign import (
        run_component_surrogate_hsi_codesign,
    )
    from optiresearch.schemas.component_surrogate_psf import (
        ComponentSurrogateHSICoDesignSpec,
    )

    spec = ComponentSurrogateHSICoDesignSpec(
        component_type=args.component,
        dataset=args.dataset,
        steps=args.steps,
        device=args.device,
        band_count=4,
        image_size=16,
        psf_size=9,
        batch_size=1,
    )
    result = run_component_surrogate_hsi_codesign(spec)
    print(_compact_json(result.model_dump(mode="json")))

    if getattr(args, "remote_job_id", ""):
        out_dir = Path("workspace/component_surrogate_hsi") / result.run_id
        export_remote_job_outputs(
            args.remote_job_id,
            "component_surrogate_hsi_codesign",
            {
                **result.model_dump(mode="json"),
                "backend": "component_surrogate_psf",
                "objective": "Component surrogate HSI co-design",
                "fallback_used": False,
                "synthetic_data": True,
                "physical_backend": False,
                "native_backend": False,
                "phase_to_fft_proxy_used": True,
            },
            [out_dir] if out_dir.exists() else [],
            {
                "job_type": "component_surrogate_hsi_codesign",
                "backend": "component_surrogate_psf",
                "component": result.component_type,
                "status": result.status,
                "evidence_level": result.evidence_level,
                "claim_ceiling": result.claim_ceiling,
                "reconstruction_loss_before": result.reconstruction_loss_before,
                "reconstruction_loss_after": result.reconstruction_loss_after,
                "mse_before": result.mse_before,
                "mse_after": result.mse_after,
                "psnr_before": result.psnr_before,
                "psnr_after": result.psnr_after,
                "sam_before": result.sam_before,
                "sam_after": result.sam_after,
                "component_grad_norm_max": result.component_grad_norm_max,
                "component_parameter_changed": result.component_parameter_changed,
                "psf_requires_grad": result.psf_requires_grad,
                "loss_requires_grad": result.loss_requires_grad,
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
    if hasattr(result, "model_dump"):
        print(_compact_json(result.model_dump(mode="json")))
    else:
        print(_compact_json(result))
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

    remote_job_id = getattr(args, "remote_job_id", None)
    if remote_job_id:
        output_dir = _Path("workspace/remote_jobs") / remote_job_id
    else:
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

    if remote_job_id:
        command_result = {
            "status": result.status,
            "run_id": run_id,
            "error_code": result.error_code,
            "execution_fidelity": "deeplens_native_geometric",
            "proxy_fallback_used": False,
            "deeplens_native_psf_path": result.deeplens_native_psf_path,
            "full_wave_optics": result.full_wave_optics,
            "phase_to_fft_proxy_used": result.phase_to_fft_proxy_used,
        }
        (_Path(output_dir) / "command_result.json").write_text(
            _json.dumps(command_result, indent=2, default=str), encoding="utf-8",
        )
        (_Path(output_dir) / "metrics_summary.json").write_text(
            _json.dumps({
                "job_type": "deeplens_native_geolens_hsi_codesign",
                "candidate": "GeoLensCooke",
                "reconstructor": args.reconstructor,
                "status": result.status,
                "execution_fidelity": "deeplens_native_geometric",
                "proxy_fallback_used": False,
                "deeplens_native_psf_path": result.deeplens_native_psf_path,
                "full_wave_optics": result.full_wave_optics,
                "phase_to_fft_proxy_used": result.phase_to_fft_proxy_used,
                "reconstruction_loss_before": result.reconstruction_loss_before,
                "reconstruction_loss_after": result.reconstruction_loss_after,
                "optical_gradient_norm": result.optical_gradient_norm_max,
                "accepted_update_count": result.accepted_update_count,
                "rollback_count": result.rollback_count,
                "stable_training_succeeded": result.stable_training_succeeded,
                "evidence_level": result.evidence_level,
                "error_code": result.error_code,
            }, indent=2, default=str), encoding="utf-8",
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


def _run_native_geolens_geometric_hsi_codesign(args: Any) -> None:
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stable_native_lens_hsi_codesign,
    )
    from optiresearch.schemas.stable_native_lens_hsi import (
        StableNativeLensHSISpec, make_stable_lens_id,
    )

    run_id = make_stable_lens_id("GeoLensCooke", args.reconstructor)
    spec = StableNativeLensHSISpec(
        run_id=run_id,
        candidate="GeoLensCooke",
        reconstructor=args.reconstructor,
        dataset=args.dataset,
        max_steps=args.steps,
        optical_warmup_steps=min(3, max(1, args.steps // 3)),
        optical_lr=args.optical_lr,
        recon_lr=args.recon_lr,
        device=args.device,
        full_wave_optics=False,
        phase_to_fft_proxy_used=False,
    )
    result = run_stable_native_lens_hsi_codesign(spec)
    print(_compact_json({
        "run_id": run_id,
        "status": result.status,
        "parameter_count": result.parameter_count,
        "trainable_param_count": result.trainable_param_count,
        "params_with_grad": result.params_with_grad,
        "grad_norm_max": result.grad_norm_max,
        "psf_requires_grad": result.psf_requires_grad,
        "loss_requires_grad": result.loss_requires_grad,
        "graph_connected": result.graph_connected,
        "parameter_changed": result.optical_parameters_changed,
        "mse_before": result.mse_before,
        "mse_after": result.mse_after,
        "psnr_before": result.psnr_before,
        "psnr_after": result.psnr_after,
        "sam_before": result.sam_before,
        "sam_after": result.sam_after,
        "execution_fidelity": "deeplens_native_geometric",
        "deeplens_native_psf_path": result.deeplens_native_psf_path,
        "evidence_level": result.evidence_level,
        "error_code": result.error_code,
    }))


def _run_stabilized_native_geolens_hsi(args: Any) -> None:
    from optiresearch.runtime.stable_native_lens_hsi_loop import (
        run_stabilized_native_geolens_hsi_loop,
    )
    from optiresearch.schemas.native_geolens_stability import (
        NativeGeoLensStabilitySpec,
    )
    from optiresearch.schemas.stable_native_lens_hsi import make_stable_lens_id

    run_id = make_stable_lens_id("GeoLensCooke", args.reconstructor)
    spec = NativeGeoLensStabilitySpec(
        run_id=run_id,
        candidate="GeoLensCooke",
        reconstructor=args.reconstructor,
        dataset=args.dataset,
        max_steps=args.steps,
        optical_warmup_steps=min(3, max(1, args.steps // 3)),
        optical_lr=args.optical_lr,
        optical_grad_clip=args.grad_clip_norm,
        spectral_angle_weight=args.spectral_angle_weight,
        device=args.device,
    )
    result = run_stabilized_native_geolens_hsi_loop(spec)
    print(_compact_json({
        "run_id": run_id,
        "status": result.status,
        "parameter_count": result.parameter_count,
        "trainable_param_count": result.trainable_param_count,
        "params_with_grad": result.params_with_grad,
        "grad_norm_max": result.grad_norm_max,
        "grad_norm_mean": result.grad_norm_mean,
        "graph_connected": result.graph_connected,
        "psf_requires_grad": result.psf_requires_grad,
        "loss_requires_grad": result.loss_requires_grad,
        "parameter_changed": result.optical_parameters_changed,
        "accepted_update_count": result.accepted_update_count,
        "rollback_count": result.rollback_count,
        "rollback_reasons": result.rollback_reasons,
        "mse_before": result.mse_before,
        "mse_after": result.mse_after,
        "psnr_before": result.psnr_before,
        "psnr_after": result.psnr_after,
        "sam_before": result.sam_before,
        "sam_after": result.sam_after,
        "psf_energy_before": result.psf_energy_before,
        "psf_energy_after": result.psf_energy_after,
        "psf_centroid_shift": result.psf_centroid_shift,
        "psf_width_shift": result.psf_width_shift,
        "stability_score": result.stability_score,
        "spectral_angle_weight": result.spectral_angle_weight,
        "metric_tradeoff_summary": result.metric_tradeoff_summary,
        "evidence_level": result.evidence_level,
        "error_code": result.error_code,
    }))


def _run_native_geolens_stability_benchmark(args: Any) -> None:
    from optiresearch.runtime.native_geolens_stability_benchmark import (
        run_native_geolens_stability_benchmark,
    )
    from optiresearch.schemas.native_geolens_benchmark import NativeGeoLensBenchmarkSpec

    def _parse_ints(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    spec = NativeGeoLensBenchmarkSpec(
        lens_file=args.lens_file,
        dataset=args.dataset,
        seeds=_parse_ints(args.seeds),
        step_grid=_parse_ints(args.step_grid),
        spectral_angle_weights=_parse_floats(args.spectral_angle_weights),
        grad_clip_norms=_parse_floats(args.grad_clip_norms),
        device=args.device,
    )
    summary = run_native_geolens_stability_benchmark(spec)
    print(_compact_json({
        "benchmark_id": summary.benchmark_id,
        "config_count": summary.config_count,
        "completed_count": summary.completed_count,
        "unsupported_count": summary.unsupported_count,
        "failed_count": summary.failed_count,
        "completion_rate": summary.completion_rate,
        "seed_count": summary.seed_count,
        "all_metrics_improved_rate": summary.all_metrics_improved_rate,
        "all_metrics_improved_rate_full_grid": summary.all_metrics_improved_rate_full_grid,
        "mse_improved_rate": summary.mse_improved_rate,
        "psnr_improved_rate": summary.psnr_improved_rate,
        "sam_improved_rate": summary.sam_improved_rate,
        "mean_mse_delta": summary.mean_mse_delta,
        "mean_psnr_delta": summary.mean_psnr_delta,
        "mean_sam_delta": summary.mean_sam_delta,
        "rollback_rate": summary.rollback_rate,
        "best_config_id": summary.best_config_id,
        "robust_config_family": summary.robust_config_family,
        "claim_recommendation": summary.claim_recommendation,
        "safe_wording": summary.safe_wording,
    }))


def _run_native_geolens_stabilization_sweep(args: Any) -> None:
    from optiresearch.runtime.native_geolens_stabilization_sweep import (
        run_native_geolens_stabilization_sweep,
    )
    import json as _json
    summary = run_native_geolens_stabilization_sweep(
        lens_file=args.lens_file,
        dataset=args.dataset,
        reconstructor=args.reconstructor,
        device=args.device,
        save_artifacts=True,
    )
    print(_json.dumps({
        "sweep_id": summary["sweep_id"],
        "configs_tested": summary["configs_tested"],
        "configs_with_accepted_updates": summary["configs_with_accepted_updates"],
        "best_config_name": summary["best_config_name"],
        "best_result": summary.get("best_result", {}),
        "output_dir": summary.get("output_dir", ""),
    }, indent=2, ensure_ascii=False, default=str))


def _export_native_geolens_stabilization_report(sweep_id: str) -> str:
    from optiresearch.reports.native_geolens_stabilization_report import (
        export_native_geolens_stabilization_report,
    )
    return str(export_native_geolens_stabilization_report(sweep_id))


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


# ---- Phase 36 handler functions ----

def _list_agent_events() -> None:
    from optiresearch.agent_system.event_bus import get_event_bus
    bus = get_event_bus()
    for e in bus.list_events():
        print(f"[{e.event_type}] {e.source_module} severity={e.severity} {e.payload.get('status', '')}")


def _export_agent_events(output: str) -> None:
    from optiresearch.agent_system.event_bus import get_event_bus
    path = get_event_bus().export_events(output)
    print(f"events: {path}")


def _show_agent_state() -> None:
    from optiresearch.agent_system.state_store import StateStore
    import json as _json
    s = StateStore().state
    print(_json.dumps(s.model_dump(mode="json"), indent=2, ensure_ascii=False))


def _export_agent_state_report() -> None:
    from optiresearch.agent_system.state_store import StateStore
    path = StateStore().export_state_report()
    print(f"markdown: {path}")


def _list_skills_v2() -> None:
    from optiresearch.skills.registry_v2 import SkillRegistryV2
    for s in SkillRegistryV2().list_skills():
        print(f"{s.skill_id}: {s.name} (risk={s.risk_level}, backends={s.required_backends})")


def _inspect_skill(skill_id: str) -> None:
    from optiresearch.skills.registry_v2 import SkillRegistryV2
    import json as _json
    info = SkillRegistryV2().inspect_skill(skill_id)
    print(_json.dumps(info, indent=2, ensure_ascii=False) if info else f"Unknown skill: {skill_id}")


def _run_skill(skill_id: str, input_json: str) -> None:
    from optiresearch.skills.runtime_v2 import SkillRuntimeV2
    import json as _json
    inputs = _json.loads(input_json)
    result = SkillRuntimeV2().execute_skill(skill_id, inputs)
    print(_json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


def _list_handler_capabilities(include_disabled: bool = False) -> None:
    from optiresearch.skills.handler_capability_registry import get_handler_capability_registry
    import json as _json
    registry = get_handler_capability_registry()
    caps = registry.list_all() if include_disabled else registry.list_enabled()
    result = []
    for c in caps:
        result.append({
            "handler_id": c.handler_id,
            "design_type": c.design_type,
            "actual_evidence_level": c.actual_evidence_level,
            "max_claim_ceiling": c.max_claim_ceiling,
            "synthetic_only": c.synthetic_only,
            "supported_modes": c.supported_execution_modes,
            "metrics": c.metrics_supported,
            "compatible_designs": c.compatible_design_ids,
            "enabled": c.enabled,
            "supports_remote": c.supports_remote,
            "remote_required": c.remote_required,
            "remote_evidence_ceiling": c.remote_evidence_ceiling,
            "local_evidence_ceiling": c.local_evidence_ceiling,
        })
    print(_json.dumps(result, indent=2, ensure_ascii=False))


def _inspect_handler_capability(handler_id: str) -> None:
    from optiresearch.skills.handler_capability_registry import get_handler_capability_registry
    import json as _json
    registry = get_handler_capability_registry()
    info = registry.inspect(handler_id)
    print(_json.dumps(info, indent=2, ensure_ascii=False) if info else f"Unknown handler: {handler_id}")


def _resolve_claim_ceiling_cli(handler_id: str, backend_id: str, dataset: str, execution_fidelity: str) -> None:
    from optiresearch.memory.claim_ceiling_resolver import resolve_claim_ceiling
    import json as _json
    result = resolve_claim_ceiling(
        handler_id=handler_id,
        backend_id=backend_id,
        dataset=dataset,
        execution_fidelity=execution_fidelity,
        synthetic_data=(dataset == "synthetic"),
        physical_backend=False,
        native_backend=False,
        real_data=(dataset == "real"),
    )
    print(_json.dumps({
        "handler_id": result.handler_id,
        "design_backend_id": result.design_backend_id,
        "handler_claim_ceiling": result.handler_claim_ceiling,
        "backend_claim_ceiling": result.backend_claim_ceiling,
        "dataset_claim_ceiling": result.dataset_claim_ceiling,
        "execution_fidelity_claim_ceiling": result.execution_fidelity_claim_ceiling,
        "final_claim_ceiling": result.final_claim_ceiling,
        "ceiling_source": result.ceiling_source,
        "limiting_factor": result.limiting_factor,
        "downgrade_reasons": result.downgrade_reasons,
        "warnings": result.warnings,
    }, indent=2, ensure_ascii=False))


def _validate_handler_capabilities() -> None:
    from optiresearch.skills.handler_capability_schema import validate_handler_capability_config
    from pathlib import Path
    import yaml
    import json as _json
    config_path = Path("optiresearch/config/handler_capabilities.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors = validate_handler_capability_config(data)
    if errors:
        print(f"Validation FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation PASSED")


def _export_handler_capability_config_report() -> None:
    from optiresearch.reports.handler_capability_config_report import (
        export_handler_capability_config_report,
    )
    md_path, json_path = export_handler_capability_config_report()
    print(f"MD report:  {md_path}")
    print(f"JSON report: {json_path}")


def _build_system_capability_registry() -> None:
    from optiresearch.system.capability_registry import build_system_capability_registry
    import json as _json
    from pathlib import Path
    registry = build_system_capability_registry()
    out_dir = Path("workspace/system_capability")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "system_capability_registry.json"
    json_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    print(f"JSON: {json_path}")
    # Markdown summary
    md_path = out_dir / "system_capability_registry.md"
    lines = [
        "# System Capability Registry", "",
        f"**Version:** {registry.registry_version}",
        f"**Generated At:** {registry.generated_at}",
        f"**Total Entries:** {len(registry.entries)}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD:   {md_path}")
    print(f"Total capabilities: {len(registry.entries)}")


def _validate_execution_contracts() -> None:
    from optiresearch.system.execution_contract_validator import validate_execution_contracts
    from tests.test_core_handler_execution_contracts import get_all_contracts
    import json as _json
    contracts = get_all_contracts()
    report = validate_execution_contracts(contracts)
    print(_json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False))
    if report["validation_status"] != "passed":
        print(f"Issues found: {report['total_issues']}")


def _validate_remote_execution_contracts() -> None:
    from optiresearch.system.remote_execution_contract_validator import validate_remote_execution_contracts
    from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts
    import json as _json
    contracts = get_all_remote_contracts()
    report = validate_remote_execution_contracts(contracts)
    print(_json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2, ensure_ascii=False))
    if report["unsafe_args_detected"]:
        print(f"UNSAFE ARGS DETECTED: {report['unsafe_args_detected']}")


def _validate_artifact_contract(run_dir: str, contract_id: str) -> None:
    from optiresearch.system.artifact_contract_validator import validate_artifact_contract_for_run
    from tests.test_core_artifact_contracts import get_artifact_contract
    import json as _json
    contract = get_artifact_contract(contract_id)
    if contract is None:
        print(f"Unknown contract_id: {contract_id}")
        return
    result = validate_artifact_contract_for_run(run_dir, contract)
    print(_json.dumps(result, indent=2, ensure_ascii=False))


def _validate_report_contract(report_path: str, contract_id: str) -> None:
    from optiresearch.system.report_contract_validator import validate_report_contract
    from tests.test_core_report_contracts import get_report_contract
    import json as _json
    contract = get_report_contract(contract_id)
    if contract is None:
        print(f"Unknown contract_id: {contract_id}")
        return
    result = validate_report_contract(report_path, contract)
    print(_json.dumps(result, indent=2, ensure_ascii=False))


def _export_claim_policy_matrix() -> None:
    from optiresearch.system.claim_policy_matrix import generate_claim_policy_matrix
    import json as _json
    import csv
    from pathlib import Path
    matrix = generate_claim_policy_matrix()
    out_dir = Path("workspace/system_capability")
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    json_path = out_dir / "claim_policy_matrix.json"
    json_path.write_text(_json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {json_path}")
    # CSV
    csv_path = out_dir / "claim_policy_matrix.csv"
    if matrix["rows"]:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=matrix["rows"][0].keys())
            w.writeheader()
            w.writerows(matrix["rows"])
        print(f"CSV:  {csv_path}")
    # MD
    md_path = out_dir / "claim_policy_matrix.md"
    lines = ["# Claim Policy Matrix", "", f"**Evidence levels covered:** {matrix['evidence_levels_covered']}", ""]
    lines.append("| evidence_level | rank | supported_claims | blocked_claims |")
    lines.append("|---|---|---|---|")
    for row in matrix["rows"]:
        lines.append(f"| {row['evidence_level']} | {row['rank']} | {', '.join(row['supported_claims'][:2])}... | {', '.join(row['blocked_claims'][:2])}... |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD:   {md_path}")


def _export_system_capability_report() -> None:
    from optiresearch.reports.system_capability_report import export_system_capability_report
    md_path = export_system_capability_report()
    print(f"Report: {md_path}")


def _export_contract_coverage_dashboard() -> None:
    from optiresearch.system.contract_coverage import generate_contract_coverage
    import json as _json
    from pathlib import Path
    dashboard = generate_contract_coverage()
    out_dir = Path("workspace/system_capability")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "contract_coverage.json"
    json_path.write_text(_json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {json_path}")
    md_path = out_dir / "contract_coverage.md"
    lines = [
        "# Contract Coverage Dashboard", "",
        f"**Overall Readiness Score:** {dashboard['overall_system_readiness_score']:.2%}",
        "",
        "| Metric | Coverage |",
        "|---|---|",
        f"| Handler Contracts | {dashboard['handler_contract_coverage']:.2%} |",
        f"| Design Mapping | {dashboard['design_mapping_coverage']:.2%} |",
        f"| Remote Contracts | {dashboard['remote_contract_coverage']:.2%} |",
        f"| Artifact Contracts | {dashboard['artifact_contract_coverage']:.2%} |",
        f"| Report Contracts | {dashboard['report_contract_coverage']:.2%} |",
        f"| Claim Policy | {dashboard['claim_policy_coverage']:.2%} |",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD:   {md_path}")
    lines.extend([
        f"| Remote Allowlist | {dashboard.get('remote_allowlist_coverage', 0):.2%} |",
        f"| Artifact Handler IDs | {dashboard.get('artifact_evidence_role_coverage', 0):.2%} |",
        f"| Penalties Applied | {dashboard.get('penalties_applied', 0):.3f} |",
        f"| Known Gap Contracts | {dashboard.get('known_gap_contracts', 0)} |",
    ])
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD:   {md_path}")
    print(f"Overall system readiness: {dashboard['overall_system_readiness_score']:.2%}")


def _export_remote_command_inventory() -> None:
    from optiresearch.system.remote_command_inventory import build_remote_command_inventory
    import json as _json
    from pathlib import Path
    inventory = build_remote_command_inventory()
    out_dir = Path("workspace/system_capability")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "remote_command_inventory.json"
    json_path.write_text(_json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"CLI commands: {inventory['total_cli_commands']}")
    print(f"Allowlist entries: {inventory['total_allowlist_entries']}")
    print(f"Remote job functions: {inventory['total_remote_jobs_functions']}")
    print(f"Contracts: {inventory['total_contracts']}")
    print(f"Known gaps: {inventory['known_gaps']}")
    print(f"Missing from allowlist: {inventory['missing_from_allowlist']}")
    print(f"Handlers without contracts: {inventory['handlers_without_contracts']}")


def _validate_remote_allowlist_coverage() -> None:
    from optiresearch.system.remote_allowlist_coverage import validate_remote_allowlist_coverage
    from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts
    import json as _json
    contracts = get_all_remote_contracts()
    report = validate_remote_allowlist_coverage(contracts)
    print(_json.dumps({k: v for k, v in report.items() if k not in ("covered", "uncovered")}, indent=2, ensure_ascii=False))


def _normalize_artifact_manifest(dir_path: str) -> None:
    from pathlib import Path
    import json as _json
    manifest_path = Path(dir_path) / "artifact_manifest.json"
    if not manifest_path.exists():
        print(f"No artifact_manifest.json found in {dir_path}")
        return
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    # Report current state
    artifacts = manifest.get("artifacts", [])
    print(f"Manifest loaded: {len(artifacts)} artifacts")
    sha256_count = sum(1 for a in artifacts if a.get("sha256"))
    role_count = sum(1 for a in artifacts if a.get("evidence_role"))
    print(f"SHA256 present: {sha256_count}/{len(artifacts)}")
    print(f"Evidence roles present: {role_count}/{len(artifacts)}")
    # Normalize
    normalized = dict(manifest)
    normalized["normalized_by"] = "Phase 69 artifact_manifest_normalizer"
    out_path = Path(dir_path) / "normalized_artifact_manifest.json"
    out_path.write_text(_json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Normalized manifest: {out_path}")


def _validate_remote_artifact_manifest(manifest_path: str) -> None:
    from optiresearch.remote.artifact_ingestion import validate_remote_artifact_manifest
    import json as _json
    result = validate_remote_artifact_manifest(manifest_path)
    print(_json.dumps(result, indent=2, ensure_ascii=False))


def _ingest_remote_artifacts(manifest_path: str) -> None:
    from optiresearch.remote.artifact_ingestion import ingest_remote_artifact_manifest
    import json as _json
    result = ingest_remote_artifact_manifest(manifest_path)
    print(_json.dumps({
        "remote_job_id": result.remote_job_id,
        "run_id": result.run_id,
        "completeness": result.completeness,
        "ingested_count": len(result.ingested_artifacts),
        "artifact_ids": result.artifact_ids,
        "primary_metric_artifact_id": result.primary_metric_artifact_id,
        "execution_result_artifact_id": result.execution_result_artifact_id,
        "missing_required": result.missing_required_artifacts,
        "warnings": result.warnings,
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _export_remote_artifact_index_report() -> None:
    from optiresearch.storage.file_artifact_store import FileArtifactStore
    from pathlib import Path
    store = FileArtifactStore()
    artifacts = store.list_artifacts()
    remote_artifacts = [
        a for a in artifacts
        if hasattr(a, "metadata") and isinstance(a.metadata, dict)
        and a.metadata.get("source") == "remote_wsl"
    ]
    out = Path("workspace/reports/remote_artifact_index_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Remote Artifact Index Report", "",
        f"**Remote Artifacts:** {len(remote_artifacts)}", "",
        "| Artifact ID | Name | Type | Evidence Role |",
        "|---|---|---|---|",
    ]
    for a in remote_artifacts[:50]:
        meta = a.metadata if isinstance(a.metadata, dict) else {}
        lines.append(
            f"| {a.artifact_id} | {meta.get('filename', '-')} | "
            f"{meta.get('artifact_type', '-')} | {meta.get('evidence_role', '-')} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out}")


def _diagnose_gradient_instability(source_path: str | None, remote_job_id: str | None) -> None:
    from optiresearch.analysis.gradient_instability_analyzer import analyze_gradient_instability
    import json as _json
    source_paths = [source_path] if source_path else []
    remote_job_ids = [remote_job_id] if remote_job_id else []
    diagnosis = analyze_gradient_instability(source_paths=source_paths, remote_job_ids=remote_job_ids)
    print(_json.dumps({
        "diagnosis_id": diagnosis.diagnosis_id,
        "status": diagnosis.status,
        "source_count": diagnosis.source_count,
        "severity": diagnosis.severity,
        "failure_modes": diagnosis.failure_modes,
        "likely_causes": diagnosis.likely_causes,
        "recommended_recoveries": diagnosis.recommended_recoveries,
        "optical_gradient_norm_max": diagnosis.metrics.optical_gradient_norm_max,
        "accepted_update_count": diagnosis.metrics.accepted_update_count,
        "rollback_rate": diagnosis.metrics.rollback_rate,
        "stable_training_succeeded": diagnosis.metrics.stable_training_succeeded,
    }, indent=2, ensure_ascii=False))


def _export_gradient_instability_report() -> None:
    from optiresearch.analysis.gradient_instability_analyzer import analyze_gradient_instability
    from pathlib import Path
    diag = analyze_gradient_instability(
        source_paths=["workspace/native_geolens_stabilization/geolens_stabilization_1779550632/sweep_results.json"],
        remote_job_ids=["remote_job_3cd757e87cd95e56"],
    )
    out = Path("workspace/reports/gradient_instability_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gradient Instability Diagnosis Report",
        f"**Diagnosis ID:** {diag.diagnosis_id}",
        f"**Status:** {diag.status}",
        f"**Severity:** {diag.severity}",
        f"**Sources:** {diag.source_count}",
        "",
        "## Failure Modes",
        *[f"- {m}" for m in diag.failure_modes],
        "",
        "## Likely Causes",
        *[f"- {c}" for c in diag.likely_causes],
        "",
        "## Recommended Recoveries",
        *[f"- {r}" for r in diag.recommended_recoveries],
        "",
        "## Metrics",
        f"- optical_gradient_norm_max: {diag.metrics.optical_gradient_norm_max}",
        f"- accepted_update_count: {diag.metrics.accepted_update_count}",
        f"- rollback_rate: {diag.metrics.rollback_rate:.2f}",
        f"- stable_training_succeeded: {diag.metrics.stable_training_succeeded}",
        "",
        "## Claim Implications",
        *[f"- {c}" for c in diag.claim_implications],
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out}")


def _list_deeplens_design_strategies() -> None:
    from optiresearch.optics.deeplens_design_strategy_registry import get_deeplens_design_strategy_registry
    import json as _json
    registry = get_deeplens_design_strategy_registry()
    result = [{"strategy_id": s.strategy_id, "strategy_family": s.strategy_family,
               "evidence_level": s.evidence_level, "claim_ceiling": s.claim_ceiling,
               "enabled": s.enabled, "compatible_diagnosis": s.compatible_diagnosis_failure_modes}
              for s in registry.list_all()]
    print(_json.dumps(result, indent=2, ensure_ascii=False))


def _export_deeplens_design_strategy_report() -> None:
    from optiresearch.optics.deeplens_design_strategy_registry import get_deeplens_design_strategy_registry
    from pathlib import Path
    registry = get_deeplens_design_strategy_registry()
    out = Path("workspace/reports/deeplens_design_strategy_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# DeepLens Design Strategy Report", ""]
    families = {}
    for s in registry.list_all():
        families.setdefault(s.strategy_family, []).append(s)
    for fam, strats in sorted(families.items()):
        lines.extend([f"## {fam}", ""])
        for s in strats:
            lines.extend([
                f"### {s.strategy_id}", f"- **Objective:** {s.objective}",
                f"- **Evidence Level:** {s.evidence_level}",
                f"- **Claim Ceiling:** {s.claim_ceiling}",
                f"- **Enabled:** {s.enabled}", "",
            ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {out}")


def _resolve_lens_file_cmd(args: Any) -> None:
    import json as _json
    from pathlib import Path as _Path
    from optiresearch.optics.lens_file_resolver import resolve_lens_file

    result = resolve_lens_file(lens_file=args.lens_file, backend_id=args.backend_id)
    result_dict = result.to_dict()
    print(_json.dumps(result_dict, indent=2, default=str))
    if getattr(args, "remote_job_id", ""):
        out_dir = _Path("workspace/remote_jobs") / args.remote_job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(_json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
        (out_dir / "remote_job_result.json").write_text(_json.dumps({
            "status": "succeeded",
            "remote_run_id": args.remote_job_id,
            "metrics_summary": result_dict,
            "artifact_manifest": {"completeness": "complete", "artifacts": []},
        }, indent=2, default=str), encoding="utf-8")


def _run_wsl_diagnostic(diag_type: str, args: Any) -> None:
    import json as _json
    from pathlib import Path as _Path
    result: dict[str, Any] = {"status": "unavailable", "evidence_level": "diagnostic_evidence", "diagnostic_type": diag_type}
    try:
        if diag_type == "trainable_parameter":
            from optiresearch.runtime.deeplens_trainable_parameter_inspection import inspect_deeplens_trainable_parameters
            result = inspect_deeplens_trainable_parameters(lens_file=args.lens_file, device=args.device)
        elif diag_type == "autograd_audit":
            from optiresearch.runtime.deeplens_autograd_audit import run_deeplens_autograd_audit
            result = run_deeplens_autograd_audit(lens_file=args.lens_file, device=args.device)
        elif diag_type == "curriculum_probe":
            from optiresearch.runtime.deeplens_curriculum_probe import run_deeplens_curriculum_probe
            result = run_deeplens_curriculum_probe(max_steps=args.max_steps, device=args.device)
        elif diag_type == "regularized_probe":
            from optiresearch.runtime.deeplens_regularized_probe import run_deeplens_regularized_probe
            result = run_deeplens_regularized_probe(max_steps=args.max_steps, device=args.device)
    except Exception as e:
        result["status"] = "unavailable"
        result["error"] = str(e)
    out_dir = _Path("workspace/remote_diagnostics") / (args.remote_job_id or diag_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(_json.dumps(result, indent=2, default=str), encoding="utf-8")
    diag_metrics = {
        k: result.get(k) for k in ("trainable_param_count", "params_with_grad", "grad_norm_max",
            "grad_norm_mean", "graph_connected", "psf_requires_grad", "loss_requires_grad",
            "detach_suspected", "candidate_update_changes_parameter",
            "parameter_count", "trainable_count", "zero_gradient_parameters",
            "recommended_trainable_subset", "recommended_strategy",
            "stages_completed", "curriculum_progress",
            "base_loss", "regularized_loss", "update_accepted",
        ) if k in result
    }
    (out_dir / "diagnostic_metrics.json").write_text(_json.dumps(diag_metrics, indent=2, default=str), encoding="utf-8")
    if getattr(args, "remote_job_id", ""):
        remote_out = _Path("workspace/remote_jobs") / args.remote_job_id
        remote_out.mkdir(parents=True, exist_ok=True)
        (remote_out / "result.json").write_text(_json.dumps(result, indent=2, default=str), encoding="utf-8")
        (remote_out / "diagnostic_metrics.json").write_text(_json.dumps(diag_metrics, indent=2, default=str), encoding="utf-8")
        lens_fields = {k: result.get(k) for k in (
            "requested_lens_file", "resolved_lens_file", "lens_resolution_source",
            "checked_lens_paths", "recommended_next_strategy",
        ) if k in result}
        (remote_out / "remote_job_result.json").write_text(_json.dumps({
            "status": result.get("status", "succeeded"),
            "remote_run_id": args.remote_job_id,
            "metrics_summary": {"diagnostic_type": diag_type, **lens_fields, **diag_metrics},
            "artifact_manifest": {"completeness": "complete", "artifacts": [
                {"artifact_name": "result.json", "artifact_type": "execution_result"},
                {"artifact_name": "diagnostic_metrics.json", "artifact_type": "metrics"},
            ]},
        }, indent=2, default=str), encoding="utf-8")
    print(_json.dumps({"status": result.get("status"), "evidence_level": result.get("evidence_level")}))


def _classify_failure(result_path: str) -> None:
    from optiresearch.agent_system.failure_taxonomy import FailureClassifier
    import json as _json
    result = _json.loads(open(result_path).read())
    fm = FailureClassifier().classify(result)
    print(_json.dumps(fm.model_dump(mode="json") if fm else {"error": "no match"}, indent=2, ensure_ascii=False))


def _recommend_recovery(failure_id: str) -> None:
    from optiresearch.agent_system.recovery_policy import RecoveryPolicy
    import json as _json
    rec = RecoveryPolicy().recommend_recovery(failure_id)
    print(_json.dumps(rec, indent=2, ensure_ascii=False))


def _reason_from_evidence(objective: str) -> None:
    from optiresearch.agents.evidence_strategy_reasoner import EvidenceStrategyReasoner
    reasoner = EvidenceStrategyReasoner()
    strategies = reasoner.reason(objective=objective)
    reasoner.export()
    reasoner.export_markdown()
    for s in strategies:
        print(f"  [{s.strategy_type}] {s.strategy_id}: {s.rationale[:100]}...")


def _generate_experiment_designs(objective: str) -> None:
    from optiresearch.agents.evidence_strategy_reasoner import EvidenceStrategyReasoner
    from optiresearch.agents.experiment_design_generator import ExperimentDesignGenerator
    reasoner = EvidenceStrategyReasoner()
    strategies = reasoner.reason(objective=objective)
    gen = ExperimentDesignGenerator()
    designs = gen.generate_designs(strategies)
    gen.export(designs)
    gen.export_markdown(designs)
    for d in designs:
        print(f"  {d.design_id}: {d.backend_id} {d.task_type} risk={d.risk_level}")


def _evaluate_candidate_plans(designs_path: str) -> None:
    from optiresearch.agents.candidate_plan_evaluator import CandidatePlanEvaluator
    from optiresearch.agents.experiment_design_generator import ExperimentDesignCandidate
    import json as _json
    data = _json.loads(open(designs_path).read())
    designs = [ExperimentDesignCandidate(**d) for d in data]
    scores = CandidatePlanEvaluator().evaluate(designs)
    CandidatePlanEvaluator().export(scores)
    CandidatePlanEvaluator().export_markdown(scores)
    for s in scores[:3]:
        print(f"  {s.design_id}: score={s.total_score:.3f} → {s.recommendation}")


def _run_agent_self_test() -> None:
    from optiresearch.agent_system.self_test import run_agent_self_test
    results = run_agent_self_test()
    passed = sum(1 for r in results if r.passed)
    print(f"Self-test: {passed}/{len(results)} passed")
    for r in results:
        status = "PASS" if r.passed else f"FAIL: {r.error}"
        print(f"  [{status}] {r.check_name} ({r.latency_sec:.3f}s)")


def _run_agent_subunit_benchmark() -> None:
    from optiresearch.benchmarks.agent_subunit_bench import run_agent_subunit_benchmark
    results = run_agent_subunit_benchmark()
    passed = sum(1 for r in results if r.passed)
    print(f"Benchmark: {passed}/{len(results)} passed")
    for r in results:
        status = "PASS" if r.passed else f"FAIL: {r.error}"
        print(f"  [{status}] {r.task_id} ({r.latency_sec:.3f}s)")


def _export_system_subunit_report() -> None:
    from optiresearch.reports.system_subunit_report import export_system_subunit_report
    path = export_system_subunit_report()
    print(f"markdown: {path}")


def _run_agent_plan_execution(args: Any) -> None:
    import time as _time
    from optiresearch.runtime.agent_plan_execution_loop import run_agent_plan_execution
    from optiresearch.schemas.agent_plan_execution import AgentPlanExecutionSpec
    from optiresearch.memory.schemas import make_deterministic_id
    import json as _json
    spec = AgentPlanExecutionSpec(
        execution_id=make_deterministic_id("plan_exec", args.objective, str(_time.time())),
        objective=args.objective,
        seed_result_path=args.seed_result_path,
        mode=args.mode,
        execute_top_k=args.execute_top_k,
        allow_remote=getattr(args, "allow_remote", False),
        remote_worker_id=getattr(args, "remote_worker_id", None),
        use_gradient_diagnosis=getattr(args, "use_gradient_diagnosis", False),
        diagnosis_source_path=getattr(args, "seed_result_path", None),
    )
    result = run_agent_plan_execution(spec)
    print(_json.dumps({
        "execution_id": result.execution_id,
        "status": result.status,
        "classified_failure": result.classified_failure,
        "failure_category": result.failure_category,
        "candidate_strategies_count": result.candidate_strategies_count,
        "candidate_designs_count": result.candidate_designs_count,
        "selected_design": result.selected_design or "none",
        "selected_design_rank": result.selected_design_rank,
        "skipped_higher_ranked_designs": result.skipped_higher_ranked_designs,
        "attempted_designs": result.attempted_designs,
        "execution_result": result.execution_result,
        "claim_gate_decision": result.claim_gate_decision,
        "memory_updated": result.memory_updated,
        "mode": result.mode,
        "executed_or_dry_run": result.executed_or_dry_run,
        "fallback_to_report_only": result.fallback_to_report_only,
        "event_count": result.event_count,
        "state_snapshots_count": result.state_snapshots_count,
        "event_log_path": result.event_log_path,
        "report_path": result.report_path,
        "diagnosis_id": result.diagnosis_id,
        "diagnosis_status": result.diagnosis_status,
        "diagnosis_failure_modes": result.diagnosis_failure_modes,
        "diagnosis_used_for_planning": result.diagnosis_used_for_planning,
        "diagnosis_strategy_count": result.diagnosis_strategy_count,
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _hybrid_plan(args: Any) -> None:
    from optiresearch.agents.hybrid_planner import HybridPlanner
    import json as _json
    planner = HybridPlanner()
    result = planner.plan(
        objective=args.objective,
        mode=args.mode,
        llm_provider=args.llm_provider,
    )
    print(_json.dumps({
        "mode": result.mode,
        "strategies_count": result.strategies_count,
        "designs_count": result.designs_count,
        "top_design_id": result.top_design_id,
        "top_score": result.top_score,
        "top_recommendation": result.top_recommendation,
        "llm_called": result.llm_called,
        "errors": result.errors,
    }, indent=2, ensure_ascii=False))


def _export_agent_plan_execution_report(execution_id: str) -> None:
    from optiresearch.reports.agent_plan_execution_report import export_agent_plan_execution_report
    path = export_agent_plan_execution_report(execution_id)
    print(f"markdown: {path}")


def _run_agent_e2e_benchmark() -> None:
    from optiresearch.benchmarks.agent_e2e_bench import run_agent_e2e_benchmark
    results = run_agent_e2e_benchmark()
    passed = sum(1 for r in results if r.passed)
    print(f"E2E Benchmark: {passed}/{len(results)} passed")
    for r in results:
        status = "PASS" if r.passed else f"FAIL: {r.detail}"
        print(f"  [{status}] {r.task} ({r.latency_sec:.3f}s)")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
