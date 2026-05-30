"""Tests for result parser contracts."""
from __future__ import annotations

from tests.test_remote_execution_contracts_core_commands import get_all_remote_contracts


def test_all_remote_contracts_have_result_parsers():
    contracts = get_all_remote_contracts()
    for cid, c in contracts.items():
        assert c.result_parser, f"Contract {cid} missing result_parser"


def test_all_remote_contracts_have_failure_parsers():
    contracts = get_all_remote_contracts()
    for cid, c in contracts.items():
        assert c.failure_parser, f"Contract {cid} missing failure_parser"


def test_result_parsers_are_expected_files():
    contracts = get_all_remote_contracts()
    for cid, c in contracts.items():
        assert c.result_parser.endswith(".json") or c.result_parser.endswith(".txt"), \
            f"Contract {cid} has unexpected result_parser: {c.result_parser}"
