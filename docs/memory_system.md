# Memory System

The MVP follows three rules:

- Trace-first: every meaningful research action writes an immutable `MetaTrace`.
- Projection-based: `RunMemory` and `ClaimEvidence` are derived from traces and artifacts.
- Provenance-enforced: claims must link to artifact evidence or remain unsupported.

## Objects

- `MetaTrace`: append-only event record.
- `RunMemory`: run-level summary, metrics, decisions, blockers, and next actions.
- `ClaimEvidence`: claim status, support edges, contradiction edges, and caveats.
- `EvidenceEdge`: artifact, trace, metric, score, relation, and rationale for one evidence link.
- `DesignRule`: reserved for reusable optical design rules.
- `PlanTemplate`: reusable research workflow with success count, artifact types, and metric names.
- `SkillMemory`: skill success/failure patterns, emitted artifact types, commands, and best practices.

## Router Rules

- evidence / claim / 证据: claims, artifacts, traces.
- plan / 计划: plan templates and run memories.
- skill / 技能: skill memories and skill registry.
- default: run memories and traces.

Selective forgetting is reserved for later versions. The MVP avoids deleting source traces.

## Phase 2 Managers

- `PlanTemplateManager` creates default templates, matches by intent, and compiles successful runs into reusable plan memory.
- `SkillMemoryManager` updates skill usage statistics from Meta-Trace entries and recommends skills by intent.
- `ClaimEvidenceManager.explain_claim()` returns an evidence table with artifact IDs, metric names, metric values, trace IDs, and caveats.
- `OptiMemoryBench` evaluates recipe reuse, claim QA, and skill-load efficiency with deterministic toy tasks.

## DesignRule Memory

Phase 3 adds `DesignRuleManager`, which compiles design rules from claims and artifact metrics. It can detect contradictions when a claim is weaker than later metric evidence, and it can supersede old rules without deleting the original record.

See `docs/design_rule_memory.md`.
