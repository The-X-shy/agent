"""Tests for core result parsers."""
from __future__ import annotations

from tests.test_remote_execution_contracts_core_commands import get_remote_contract


def test_diagnostic_parser():
    c = get_remote_contract("rec_trainable_param_inspection")
    assert c.result_parser == "result.json"


def test_benchmark_parser():
    c = get_remote_contract("rec_native_geolens_benchmark")
    assert c.result_parser == "result.json"


def test_surrogate_hsi_parser():
    c = get_remote_contract("rec_component_surrogate_hsi_codesign")
    assert c.result_parser == "result.json"
