# Config-Driven Handler Capabilities

Phase 42 moves handler capabilities from hardcoded Python to a YAML config file.

## Config File

`optiresearch/config/handler_capabilities.yaml`

## Schema Version

`capability_schema_version: "0.1"`

## Environment Variable Override

```bash
OPTIRESEARCH_HANDLER_CAPABILITY_CONFIG=/custom/path/to/config.yaml
```

## Enabled vs Disabled Handlers

- **Enabled** (5): `objective_redesign_simpler_metric`, `param_reduction_sweep`, `backend_switch_waveoptics_coherent`, `report_negative_result_doc`, `real_data_request`
- **Disabled** (4): `deeplens_native_geolens_hsi_codesign`, `native_geolens_stabilization_sweep`, `coherent_asm_waveoptics_probe`, `remote_native_geolens_validation`

Disabled handlers can be inspected but not selected by the AgentPlanExecutionLoop.

## CLI

```bash
python -m optiresearch.cli validate-handler-capabilities
python -m optiresearch.cli list-handler-capabilities --include-disabled
python -m optiresearch.cli export-handler-capability-config-report
```
