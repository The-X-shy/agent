"""Remote command inventory with canonical mapping for Phase 69."""
from __future__ import annotations

from typing import Any

# Canonical mapping: orchestrator-side contract command_name → worker-side allowlist command_name
# Rule: strip "run-remote-" prefix, keeping "run-"
CONTRACT_TO_ALLOWLIST_COMMAND: dict[str, str | None] = {
    "run-remote-deeplens-trainable-parameter-inspection": "run-deeplens-trainable-parameter-inspection",
    "run-remote-deeplens-autograd-audit": "run-deeplens-autograd-audit",
    "run-remote-deeplens-component-probe": "run-deeplens-component-probe",
    "run-remote-stabilized-native-geolens-hsi": "run-stabilized-native-geolens-hsi",
    "run-remote-native-geolens-stability-benchmark": "run-native-geolens-stability-benchmark",
    "run-remote-native-geolens-benchmark-failure-analysis": None,
    "run-remote-resume-native-geolens-benchmark": None,
    "run-remote-component-surrogate-hsi-codesign": "run-component-surrogate-hsi-codesign",
    "run-remote-deeplens-curriculum-probe": "run-deeplens-curriculum-probe",
    "run-remote-deeplens-regularized-probe": "run-deeplens-regularized-probe",
}

# Known gap contracts: planned but no CLI/remote_jobs implementation exists yet
KNOWN_GAP_CONTRACT_IDS = {
    "rec_benchmark_failure_analysis",
    "rec_resume_benchmark",
}


def get_canonical_command_name(contract_command_name: str) -> str | None:
    """Return the allowlist-side command name for a contract's command_name."""
    return CONTRACT_TO_ALLOWLIST_COMMAND.get(contract_command_name)


def get_allowlist_command_name(contract_command_name: str) -> str | None:
    """Return the allowlist command name. Returns None for known gaps."""
    canonical = CONTRACT_TO_ALLOWLIST_COMMAND.get(contract_command_name)
    if canonical is None and contract_command_name in CONTRACT_TO_ALLOWLIST_COMMAND:
        return None
    return canonical


def build_remote_command_inventory() -> dict[str, Any]:
    """Scan CLI, allowlist, remote_jobs.py, and contracts, build inventory."""
    from optiresearch.remote.command_allowlist import ALLOWED_CLI_COMMANDS

    # Collect CLI commands that start with "run-remote-"
    cli_commands: list[str] = _collect_cli_remote_commands()

    # Collect allowlist commands
    allowlist_commands = list(ALLOWED_CLI_COMMANDS.keys())

    # Collect remote_jobs functions
    remote_jobs_functions: list[str] = _collect_remote_jobs_functions()

    # Collect contracts
    contract_commands: dict[str, str | None] = dict(CONTRACT_TO_ALLOWLIST_COMMAND)

    # Analyze
    allowlist_entries = set(allowlist_commands)
    orchestrator_entries = set(cli_commands)
    remote_jobs_entries = set(remote_jobs_functions)

    mapped_contracts: list[dict] = []
    missing_from_cli: list[str] = []
    missing_from_allowlist: list[str] = []
    missing_from_remote_jobs: list[str] = []

    for contract_name, allowlist_name in contract_commands.items():
        entry = {
            "contract_command_name": contract_name,
            "mapped_allowlist_name": allowlist_name,
            "is_known_gap": allowlist_name is None,
        }
        if allowlist_name is None:
            entry["cli_exists"] = False
            entry["allowlist_exists"] = False
            entry["remote_jobs_exists"] = False
            missing_from_allowlist.append(contract_name)
        else:
            # Check CLI: orchestrator name is contract_name
            cli_exists = contract_name in orchestrator_entries
            # Check allowlist
            allowlist_exists = allowlist_name in allowlist_entries
            # Check remote_jobs: function name is run_remote_X
            func_name = _contract_to_function_name(contract_name)
            remote_exists = func_name in remote_jobs_entries

            entry["cli_exists"] = cli_exists
            entry["allowlist_exists"] = allowlist_exists
            entry["remote_jobs_exists"] = remote_exists

            if not cli_exists:
                missing_from_cli.append(contract_name)
            if not allowlist_exists:
                missing_from_allowlist.append(contract_name)
            if not remote_exists:
                missing_from_remote_jobs.append(contract_name)

        mapped_contracts.append(entry)

    # Find handlers with supports_remote but no contract
    try:
        from optiresearch.skills.handler_capability_registry import get_handler_capability_registry
        reg = get_handler_capability_registry()
        remote_handlers = [h for h in reg.list_enabled() if h.supports_remote]
        contracted_handler_ids = _collect_contracted_handler_ids()
        handlers_without_contracts = [
            h.handler_id for h in remote_handlers
            if h.handler_id not in contracted_handler_ids
        ]
    except Exception:
        remote_handlers = []
        handlers_without_contracts = []

    return {
        "inventory_version": "0.1",
        "total_cli_commands": len(cli_commands),
        "total_allowlist_entries": len(allowlist_commands),
        "total_remote_jobs_functions": len(remote_jobs_functions),
        "total_contracts": len(contract_commands),
        "known_gaps": list(KNOWN_GAP_CONTRACT_IDS),
        "mapped_contracts": mapped_contracts,
        "missing_from_cli": missing_from_cli,
        "missing_from_allowlist": missing_from_allowlist,
        "missing_from_remote_jobs": missing_from_remote_jobs,
        "handlers_without_contracts": handlers_without_contracts,
        "cli_commands": cli_commands,
        "allowlist_commands": allowlist_commands,
        "remote_jobs_functions": remote_jobs_functions,
    }


def _collect_cli_remote_commands() -> list[str]:
    """Parse cli.py for run-remote-* subcommands."""
    import ast
    from pathlib import Path

    cli_path = Path(__file__).resolve().parent.parent / "cli.py"
    if not cli_path.exists():
        return []

    commands: list[str] = []
    try:
        tree = ast.parse(cli_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and
                        node.func.attr == "add_parser"):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value.startswith("run-remote-"):
                                commands.append(arg.value)
    except Exception:
        pass
    return sorted(commands)


def _collect_remote_jobs_functions() -> list[str]:
    """Parse remote_jobs.py for run_remote_* function definitions."""
    import ast
    from pathlib import Path

    rj_path = Path(__file__).resolve().parent.parent / "runtime" / "remote_jobs.py"
    if not rj_path.exists():
        return []

    functions: list[str] = []
    try:
        tree = ast.parse(rj_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("run_remote_"):
                    functions.append(node.name)
    except Exception:
        pass
    return sorted(functions)


def _contract_to_function_name(contract_command_name: str) -> str:
    """Convert contract command name to expected remote_jobs function name."""
    # run-remote-deeplens-trainable-parameter-inspection -> run_remote_deeplens_trainable_parameter_inspection
    return contract_command_name.replace("-", "_")


def _collect_contracted_handler_ids() -> set[str]:
    """Get handler_ids from all remote execution contracts."""
    try:
        from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts
        contracts = get_all_remote_contracts()
        return {c.handler_id for c in contracts.values() if c.handler_id}
    except Exception:
        return set()
