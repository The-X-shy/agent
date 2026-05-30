# Report Contracts

Report contracts specify required structure and content for each report type.

## Contract Fields

| Field | Description |
|---|---|
| report_contract_id | Unique identifier |
| report_type | Report category |
| exporter_cli | CLI command that generates the report |
| required_sections | Must-include markdown sections |
| optional_sections | May-include sections |
| required_tables | Required data tables |
| required_fields | Required data fields |
| linked_artifacts | Artifacts that must be referenced |
| linked_claims | Claims that must be referenced |
| safe_wording_required | Whether safe wording section is mandatory |
| blocked_claims_section_required | Whether blocked claims section is mandatory |
| evidence_level_section_required | Whether evidence level must be stated |

## Core Contracts

8 core report contracts:

1. `rc_agent_plan` - Agent plan execution report
2. `rc_remote_diagnostic` - Remote diagnostic report
3. `rc_component_probe` - Component probe report
4. `rc_native_geolens_stability` - GeoLens stability report
5. `rc_native_geolens_benchmark` - GeoLens benchmark report
6. `rc_benchmark_failure` - Benchmark failure report
7. `rc_design_strategy` - Design strategy report
8. `rc_evidence_tables` - Evidence tables

## Benchmark Report Requirements

Benchmark reports must include **both**:
- Completed-Only improvement rates
- Full-Grid improvement rates

This ensures that selective reporting bias is prevented and the reader
understands the difference between best-case and overall performance.

## Validation

```bash
python -m optiresearch.cli validate-report-contract --report-path <path> --contract-id <id>
```
