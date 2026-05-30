# System Capability Registry

The System Capability Registry provides a unified view of all registered system capabilities across handlers, skills, design strategies, optical backends, and claim policies.

## Architecture

The registry is a **read-only aggregation layer** built over existing domain registries:

- Handler capabilities from `optiresearch/skills/handler_capability_registry.py`
- Skills from `optiresearch/skills/registry_v2.py`
- Design strategies from `optiresearch/optics/deeplens_design_strategy_registry.py`
- Optical backends from `optiresearch/backends/registry.py`
- Claim policies from `optiresearch/memory/claim_gate_v2.py`

## Usage

```bash
# Build and export the registry
python -m optiresearch.cli build-system-capability-registry

# Export system capability report
python -m optiresearch.cli export-system-capability-report

# Export claim policy matrix
python -m optiresearch.cli export-claim-policy-matrix
```

## Output

- `workspace/system_capability/system_capability_registry.json`
- `workspace/system_capability/system_capability_registry.md`
- `workspace/system_capability/system_capability_report.md`
- `workspace/system_capability/claim_policy_matrix.{json,csv,md}`

## Schema

### SystemCapabilityEntry

Each entry captures the capability profile of a single system component:

| Field | Type | Description |
|---|---|---|
| capability_id | str | Unique identifier |
| capability_type | Literal | handler, skill, design, backend, dataset, remote_worker, artifact, report, benchmark, claim_policy |
| evidence_level | str | Actual evidence level produced |
| max_claim_ceiling | str | Maximum claim level supported |
| maturity_level | Literal | experimental, validated_local, validated_remote, benchmarked, production_ready |
| supports_remote | bool | Whether remote execution is supported |
### SystemCapabilityRegistry

Wrapper aggregating all entries with metadata:

- `registry_version`: Schema version
- `entries`: List of SystemCapabilityEntry
- `generated_at`: ISO timestamp
- `source_files`: List of source configs
- `validation_summary`: Diagnostics (missing fields, orphans, inconsistencies)
