# Agentic Differentiable Optics Framework

## Overview

The OptiResearch Agent is an agentic differentiable optics framework that combines
differentiable optical simulation, HSI reconstruction, and automated experiment
management into a unified research system.

## Architecture

```
CLI (cli.py)
  ├── Backend Registry (backends/)
  │   ├── 8 registered optical backends
  │   ├── Claim ceiling enforcement
  │   └── Capability introspection
  ├── Experiment Controller v2 (runtime/experiment_controller_v2.py)
  │   ├── Unified experiment entry point
  │   ├── Precondition validation
  │   └── Claim ceiling downgrade
  ├── Strategy Engine (agents/strategy_engine.py)
  │   ├── 8 built-in recommendation rules
  │   ├── Automatic next-step planning
  │   └── Risk-aware decision making
  ├── Research Memory v2 (memory/research_memory_v2.py)
  │   ├── 7 memory types
  │   ├── 9 seeded rules from Phase 18-23
  │   └── Queryable by type, tag, content
  ├── Claim Gate v2 (memory/claim_gate_v2.py)
  │   ├── 8 violation type detectors
  │   ├── Pre-claim boundary enforcement
  │   └── Safe wording generation
  ├── Objective Library (objectives/)
  │   ├── 6 optical losses
  │   ├── 6 HSI losses
  │   ├── 6 regularizers
  │   └── 3 preset profiles
  ├── Autograd Auditor (diagnostics/autograd_auditor.py)
  │   ├── Gradient flow analysis
  │   ├── Detach detection
  │   └── Rollback-aware verdicts
  └── Agent System Report (reports/agent_system_report.py)
      └── Comprehensive framework documentation
```

## Key Concepts

### Backend Registry

All optical simulation backends are registered with declared capabilities and
claim ceilings. The registry enables automatic backend selection, capability
checking, and claim enforcement.

### Claim Ceiling

Every backend declares a maximum claim level it can support. The experiment
controller enforces this ceiling — if a task requires a higher claim level
than the backend supports, the claim is automatically downgraded.

### Strategy Engine

After each experiment, the strategy engine analyzes results and recommends
the next action. Rules are based on Phase 18-23 experience:

- Large gradients → reduce learning rate
- High rollback ratio → freeze optics
- Zero gradient → audit autograd
- Loss increase → enable rollback

### Research Memory

Long-term knowledge from experiments is stored as structured memory entries.
The memory system supports querying by type, tag, and content search.

## CLI Commands

```bash
# Backend registry
python -m optiresearch.cli list-optical-backends
python -m optiresearch.cli inspect-optical-backend --backend-id X

# Experiment controller
python -m optiresearch.cli run-experiment-v2 --backend-id X --task-type Y

# Strategy
python -m optiresearch.cli recommend-next-strategy --backend-id X --latest-result-json '...'

# Memory
python -m optiresearch.cli compile-research-memory-v2
python -m optiresearch.cli query-research-memory-v2 --tag gradient

# Claims
python -m optiresearch.cli check-claim --claim-text "..." --backend-id X

# Objectives
python -m optiresearch.cli list-objective-profiles
python -m optiresearch.cli inspect-objective-profile --profile-id X

# Diagnostics
python -m optiresearch.cli audit-autograd-graph

# Reports
python -m optiresearch.cli export-agent-system-report
```

## Adding New Backends

```python
from optiresearch.backends.base import OpticalBackend
from optiresearch.backends.registry import register_backend

backend = OpticalBackend(
    backend_id="my_new_backend",
    label="My New Optical Backend",
    backend_type="deeplens",
    differentiability_level="native_component",
    supports_psf_generation=True,
    claim_ceiling="my_claim_level",
    known_failure_modes=["example failure mode"],
    recommended_use_cases=["example use case"],
)
register_backend(backend)
```

## Adding New Objective Profiles

```python
from optiresearch.objectives.optical_objectives import ObjectiveProfile, register_objective_profile

profile = ObjectiveProfile(
    profile_id="my_profile",
    losses=["reconstruction_mse", "psf_energy_preservation"],
    weights={"reconstruction_mse": 1.0, "psf_energy_preservation": 0.1},
    compatible_backends=["deeplens_geolens_geometric"],
    claim_implications="my_claim_implication",
    description="My custom objective profile",
)
register_objective_profile(profile)
```
