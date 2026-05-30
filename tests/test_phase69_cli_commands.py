"""Tests for Phase 69 CLI command functions."""
from __future__ import annotations

import json
from pathlib import Path


def test_remote_command_inventory_cli():
    from optiresearch.system.remote_command_inventory import build_remote_command_inventory
    inv = build_remote_command_inventory()
    out_dir = Path("workspace/system_capability")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_remote_command_inventory.json"
    json_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    assert json_path.exists()


def test_remote_allowlist_coverage_cli():
    from optiresearch.system.remote_allowlist_coverage import validate_remote_allowlist_coverage
    from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts
    contracts = get_all_remote_contracts()
    report = validate_remote_allowlist_coverage(contracts)
    assert report["allowlist_coverage"] == 1.0


def test_contract_coverage_dashboard_cli():
    from optiresearch.system.contract_coverage import generate_contract_coverage
    dashboard = generate_contract_coverage()
    out_dir = Path("workspace/system_capability")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_contract_coverage.json"
    json_path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    assert json_path.exists()
    assert dashboard["overall_system_readiness_score"] > 0.7
