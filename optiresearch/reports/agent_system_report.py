"""Agent system report generator — exports a comprehensive markdown report.

Aggregates information from all Phase 24 components into a single
structured report at workspace/reports/agent_system_report.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def export_agent_system_report(output_dir: Optional[Path] = None) -> Path:
    """Generate the agent system report and return the output path."""
    out = output_dir or Path("workspace/reports")
    out.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []

    # ── 1. System overview ──────────────────────────────────────────
    sections.append(_section_overview())

    # ── 2. Backend registry ─────────────────────────────────────────
    sections.append(_section_backend_registry())

    # ── 3. Experiment controller v2 ─────────────────────────────────
    sections.append(_section_experiment_controller())

    # ── 4. Strategy engine ──────────────────────────────────────────
    sections.append(_section_strategy_engine())

    # ── 5. Research memory v2 ───────────────────────────────────────
    sections.append(_section_research_memory())

    # ── 6. Claim gate v2 ────────────────────────────────────────────
    sections.append(_section_claim_gate())

    # ── 7. Objective library ────────────────────────────────────────
    sections.append(_section_objective_library())

    # ── 8. Autograd auditor ─────────────────────────────────────────
    sections.append(_section_autograd_auditor())

    # ── 9. Example workflow ─────────────────────────────────────────
    sections.append(_section_example_workflow())

    # ── 10. Current capability limits ───────────────────────────────
    sections.append(_section_capability_limits())

    # ── 11. Next development roadmap ────────────────────────────────
    sections.append(_section_roadmap())

    report_text = "\n\n".join(sections)
    path = out / "agent_system_report.md"
    path.write_text(report_text, encoding="utf-8")
    return path


def _section_overview() -> str:
    return """# Agentic Differentiable Optics Framework — System Report

## 1. System Overview

The OptiResearch Agent is an agentic differentiable optics framework for
computational imaging research. It combines:

- **Differentiable optical backends** (mock, proxy, DeepLens geometric, DeepLens component)
- **HSI reconstruction pipeline** (synthetic data, differentiable forward model, learned reconstructors)
- **Experiment automation** (local CLI, remote WSL workers, ablation studies)
- **Agent intelligence** (strategy engine, research memory, claim gates, autograd auditing)

**Version:** Phase 24
**Key capability:** Native differentiable optimization of DeepLens optical parameters
through HSI reconstruction loss backpropagation, with stability guarantees via
rollback protection and gradient clipping.

### Architecture Layers

| Layer | Components |
|-------|-----------|
| Backends | 8 registered optical backends with capability declarations |
| Objectives | Composable optical + HSI loss functions + regularizers |
| Runtime | ExperimentControllerV2 wrapping Phase 18-23 loops |
| Diagnostics | Autograd auditor for gradient flow verification |
| Memory | ResearchMemoryV2 with 9 seeded rules |
| Safety | ClaimGateV2 pre-checking 8 violation types |
| Strategy | StrategyEngine with 8 built-in recommendation rules |
| Reports | Agent system report, backend registry export |
"""


def _section_backend_registry() -> str:
    from optiresearch.backends.registry import list_backends

    backends = list_backends()
    lines = [
        "## 2. Backend Registry",
        "",
        f"**Total backends:** {len(backends)}",
        "",
        "| backend_id | type | diff_level | claim_ceiling |",
        "|---|---|---|---|",
    ]
    for b in sorted(backends, key=lambda x: x.backend_id):
        lines.append(
            f"| {b.backend_id} | {b.backend_type} | "
            f"{b.differentiability_level} | {b.claim_ceiling} |"
        )

    lines.append("")
    lines.append("### Claim Ceiling Hierarchy")
    lines.append("")
    lines.append(
        "1. `unsupported` → 2. `mock_simulation` → 3. `deeplens_integration_smoke` → "
        "4. `native_component_optimization` → 5. `native_hsi_proxy` → "
        "6. `native_full_reconstruction_proxy` → 7. `native_lens_simulation` → "
        "8. `native_waveoptics` → 9. `stable_native_lens_hsi_codesign` → "
        "10. `rollback_protected_native_lens_hsi` → 11. `real_hsi_performance`"
    )

    lines.append("")
    lines.append("### Key Constraints")
    lines.append("")
    lines.append(f"- `phase_to_fft_proxy` claim ceiling: `native_full_reconstruction_proxy` — cannot exceed proxy")
    lines.append(f"- `deeplens_geolens_geometric` claim ceiling: `native_lens_simulation` — cannot claim wave-optics")
    lines.append(f"- `deeplens_coherent_asm` is **not differentiable** — requires_grad=False")
    lines.append(f"- `local_synthetic_hsi` cannot support real HSI performance claims")

    return "\n".join(lines)


def _section_experiment_controller() -> str:
    return """## 3. Experiment Controller v2

`ExperimentControllerV2` provides a unified entry point for all experiment types.

### Supported Task Types

| task_type | Runtime Loop | Minimum Claim Ceiling |
|---|---|---|
| `native_optimization_probe` | Phase 19 probe | `native_component_optimization` |
| `native_hsi_codesign` | Phase 20 loop | `native_hsi_proxy` |
| `native_hsi_reconstruction_codesign` | Phase 21 loop | `native_full_reconstruction_proxy` |
| `native_waveoptics_codesign` | Phase 22 loop | `native_waveoptics` |
| `stable_lens_hsi_codesign` | Phase 23 loop | `native_lens_simulation` |
| `psf_probe` | PSF generation | `deeplens_integration_smoke` |
| `component_optimization` | Component-level | `native_component_optimization` |

### Key Features

- **Precondition validation:** Checks backend capabilities against task requirements
- **Claim ceiling enforcement:** Automatically downgrades claims when backend is insufficient
- **Lazy delegation:** Wraps existing runtime loops without modifying them
- **Remote execution:** Delegates to SSH/remote worker system

### CLI

```
python -m optiresearch.cli run-experiment-v2 \\
  --backend-id deeplens_geolens_geometric \\
  --task-type stable_lens_hsi_codesign \\
  --execution-target local
```
"""


def _section_strategy_engine() -> str:
    return """## 4. Strategy Engine

`StrategyEngine` automatically recommends next experimental actions based on
result metrics, backend capabilities, and known failure patterns.

### Built-in Rules (Priority-Ordered)

| # | Rule | Trigger | Action |
|---|------|---------|--------|
| 1 | Large gradient | `optical_gradient_norm > 100` | Reduce LR 100x + rollback |
| 2 | High rollback ratio | `rollback_count / steps > 0.5` | Freeze optics / ablation |
| 3 | Zero gradient | `grad_norm == 0` after optimizer step | Audit autograd |
| 4 | Loss increase | `loss_after > loss_before` without rollback | Enable rollback |
| 5 | Claim downgraded | `claim_downgraded == True` | Reword claim |
| 6 | Recon loss increase | `recon_loss_after > recon_loss_before` | More warmup steps |
| 7 | Max gradient spike | `grad_max > 10` | Clip + reduce LR |
| 8 | PSF energy drift | `psf_energy_delta > 0.5` | Increase PSF reg |

### CLI

```
python -m optiresearch.cli recommend-next-strategy \\
  --backend-id deeplens_geolens_geometric \\
  --latest-result-json '{"optical_gradient_norm": 500}'
```
"""


def _section_research_memory() -> str:
    from optiresearch.memory.research_memory_v2 import ResearchMemoryV2

    mem = ResearchMemoryV2()
    snapshot = mem.compile_snapshot()
    total = sum(len(v) for v in snapshot.values())

    lines = [
        "## 5. Research Memory v2",
        "",
        f"**Total entries:** {total}",
        f"**Memory types:** {len(snapshot)}",
        "",
    ]
    for mtype in sorted(snapshot.keys()):
        entries = snapshot[mtype]
        lines.append(f"### {mtype} ({len(entries)} entries)")
        for entry in entries:
            lines.append(f"- **{entry.memory_id}:** {entry.content[:120]}...")
        lines.append("")

    return "\n".join(lines)


def _section_claim_gate() -> str:
    return """## 6. Claim Gate v2

`ClaimGateV2` pre-checks proposed claims before they enter the evidence system.

### Detected Violation Types

| Violation | Description | Decision |
|-----------|-------------|----------|
| `proxy_as_waveoptics` | Proxy FFT claiming wave-optics | `unsupported` |
| `geometric_as_coherent` | Geometric PSF claiming coherent | `unsupported` |
| `synthetic_as_real` | Synthetic data claiming real | `unsupported` |
| `differentiable_as_improves` | Differentiability claiming better | `qualified` |
| `local_only_as_robust` | Local-only claiming robust | `needs_followup` |
| `rollback_protection_as_improvement` | Rollback claiming improvement | `qualified` |
| `unsupported_path_as_supported` | Broken path claiming supported | `unsupported` |
| `black_box_as_native` | Black-box claiming native grad | `unsupported` |

### CLI

```
python -m optiresearch.cli check-claim \\
  --claim-text "Full DeepLens wave-optics native HSI co-design is supported" \\
  --backend-id deeplens_geolens_geometric
```
"""


def _section_objective_library() -> str:
    from optiresearch.objectives.optical_objectives import list_objective_profiles

    profiles = list_objective_profiles()
    lines = [
        "## 7. Objective Library",
        "",
        f"**Registered profiles:** {len(profiles)}",
        "",
    ]
    for p in profiles:
        lines.append(f"### {p.profile_id}")
        lines.append(f"- **Losses:** {', '.join(p.losses)}")
        lines.append(f"- **Weights:** {p.weights}")
        lines.append(f"- **Compatible backends:** {', '.join(p.compatible_backends)}")
        if p.claim_implications:
            lines.append(f"- **Claim implications:** {p.claim_implications}")
        if p.description:
            lines.append(f"- **Description:** {p.description}")
        lines.append("")

    lines.append("### Available Loss Functions")
    lines.append("")
    lines.append("**Optical:** psf_width_loss, psf_centroid_loss, psf_energy_loss, "
                 "psf_smoothness_loss, spot_size_loss, field_consistency_loss")
    lines.append("")
    lines.append("**HSI:** reconstruction_mse, spectral_angle_loss, measurement_consistency_loss, "
                 "spectral_smoothness_loss, band_weighted_mse, task_aligned_hsi_loss")
    lines.append("")
    lines.append("**Regularizers:** optical_param_l2, optical_param_delta_limit, "
                 "psf_energy_preservation, psf_centroid_preservation, psf_width_preservation, "
                 "rollback_penalty")

    return "\n".join(lines)


def _section_autograd_auditor() -> str:
    return """## 8. Autograd Auditor

`audit_autograd_graph()` analyses gradient flow through optical parameters.

### Capabilities

- `inspect_tensor_requires_grad` — check which tensors track gradients
- `trace_loss_to_parameters` — verify loss can reach parameters
- `detect_detach` — find detached tensors in the graph
- `detect_numpy_conversion_risk` — scan for numpy conversions
- `detect_no_grad_region` — detect no_grad contexts
- `summarize_gradient_flow` — gradient norm statistics
- `compare_gradient_strength` — relative gradient magnitudes

### Rollback Awareness

The auditor distinguishes between:
- **Normal zero-grad:** `rollback_parameters_changed=False` → `verdict: clean`
  (parameters were restored from snapshot — zero change is expected)
- **True autograd break:** `loss.requires_grad=False` → `verdict: broken`
  (graph is detached — no gradient can flow)

### CLI

```
python -m optiresearch.cli audit-autograd-graph
```
"""


def _section_example_workflow() -> str:
    return """## 9. Example Workflow

### Typical Research Iteration

```bash
# 1. List available backends
python -m optiresearch.cli list-optical-backends

# 2. Inspect a specific backend
python -m optiresearch.cli inspect-optical-backend \\
  --backend-id deeplens_geolens_geometric

# 3. Check a proposed claim before running
python -m optiresearch.cli check-claim \\
  --claim-text "GeoLens geometric PSF improves HSI reconstruction" \\
  --backend-id deeplens_geolens_geometric

# 4. Run stable training experiment
python -m optiresearch.cli run-experiment-v2 \\
  --backend-id deeplens_geolens_geometric \\
  --task-type stable_lens_hsi_codesign \\
  --execution-target local

# 5. Audit autograd if training failed
python -m optiresearch.cli audit-autograd-graph

# 6. Get strategy recommendation
python -m optiresearch.cli recommend-next-strategy \\
  --backend-id deeplens_geolens_geometric \\
  --latest-result-json '{"optical_gradient_norm": 500, "rollback_count": 7}'

# 7. Query research memory for relevant rules
python -m optiresearch.cli query-research-memory-v2 \\
  --tag gradient

# 8. Export full system report
python -m optiresearch.cli export-agent-system-report
```
"""


def _section_capability_limits() -> str:
    return """## 10. Current Capability Limits

### Supported

| Capability | Evidence Level | Backend |
|-----------|---------------|---------|
| Component-level native diff optimization | `native_component_optimization` | Fresnel, Binary2Phase |
| Native HSI proxy co-design | `native_hsi_proxy` | phase_to_fft_proxy |
| Native HSI reconstruction co-design | `native_full_reconstruction_proxy` | phase_to_fft_proxy |
| Native lens simulation HSI co-design | `native_lens_simulation` | GeoLens geometric |
| Stable native lens HSI co-design | `stable_native_lens_hsi_codesign` | GeoLens geometric (local small_lr + remote rollback) |

### Needs Follow-up

| Capability | Blocker |
|-----------|---------|
| Full wave-optics native HSI co-design | Coherent ASM `requires_grad=False` |
| Remote-validated stable training | Remote job shows 0 accepted updates |

### Unsupported

| Capability | Reason |
|-----------|--------|
| Real HSI performance | No real HSI dataset |
| Coherent wave-optics gradient flow | ASM ray sampling breaks autograd |
| Black-box → native gradient claim | Black-box has no gradient interface |
"""


def _section_roadmap() -> str:
    return """## 11. Next Development Roadmap

### Phase 25 (Suggested)
1. **Coherent wave-optics gradient path:** Investigate `DiffractiveLens` or pure wave propagation
   as alternatives to the broken GeoLens coherent ASM path
2. **Real HSI dataset integration:** Acquire or generate a real HSI dataset to move beyond
   synthetic-only claims
3. **Remote stability validation:** Debug why remote GeoLens geometric produces 0 accepted updates
   despite local success — environment diff, stochasticity, or DeepLens version issue
4. **Multi-surface optimization:** Extend beyond single-surface (Fresnel, Binary2Phase) to
   compound lens co-design
5. **Task-aligned HSI optimization:** Replace generic reconstruction loss with downstream-task
   metrics (classification, detection, material identification)

### Phase 26+ (Long-term)
1. **Meta-learning for optical design:** Train a hypernetwork that proposes optical parameters
   given task requirements
2. **Bayesian optimization for non-differentiable backends:** Use the black-box PSF backend
   with BO for cases where gradients are unavailable
3. **Physical prototype validation:** If hardware access becomes available, close the
   sim-to-real gap
"""
