# Architecture

OptiResearch Agent MVP has five layers:

- Agent Orchestration: rule-based `LeadInvestigator`, `MethodBuilder`, `SimulationExperimentalist`, and `CriticalReviewer`.
- Skill Runtime: manifest scanning, skill routing, progressive L0/L1 loading, allowlisted execution, and validation.
- Research Memory OS: immutable Meta-Trace plus RunMemory and ClaimEvidence projections.
- Experiment Spec: normalized `ExperimentSpec` with optical, sweep, and metric sections.
- Storage: SQLite JSON tables and local filesystem artifacts.
- API/CLI: FastAPI endpoints and `python -m optiresearch.cli`.

The source of truth is trace plus artifact. Summaries, claims, and future design rules are derived views.

Phase 2 adds derived PlanTemplate and SkillMemory projections plus OptiMemoryBench. These remain projections from traces, artifacts, and run memory rather than independent truth sources.

Phase 3 adds baseline batch execution across five mock encoder families and DesignRule memory for comparing evidence-backed optical tradeoffs.

Phase 4 freezes `ExperimentSpec v0.1` and adds a backend compatibility layer. `MockDeepLensAdapter` and `DeepLensAdapter` share `AdapterRunResult`, `AdapterArtifact`, and `AdapterMetricBundle`, so downstream artifact registration, memory compilation, claim review, and paper exports do not depend on a specific optical backend.

The real DeepLens contract lives in `optiresearch/adapters/deeplens.py`:

- `validate_environment()`
- `translate_experiment_spec()`
- `simulate_psf_cube()`
- `compute_mtf()`
- `run_optimization()`
- `collect_artifacts()`

If real DeepLens is not installed, the adapter returns structured `DEEPLENS_NOT_INSTALLED` errors instead of raising during import.

Phase 9 adds synthetic HSI reconstruction:

- `SyntheticHSIDataset` generates deterministic HSI cubes.
- `HSIForwardModel` renders measurements from PSF cube artifacts.
- `LinearSpectralReconstructor` reconstructs HSI bands without GPU dependencies.
- Reconstruction metrics are registered as artifacts and used as separate reconstruction-level evidence.
