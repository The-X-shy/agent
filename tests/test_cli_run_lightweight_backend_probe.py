"""Phase 32: CLI run-lightweight-backend-probe tests."""

import json
import subprocess
import sys


def test_cli_shallow_probe():
    result = subprocess.run(
        [sys.executable, "-m", "optiresearch.cli", "run-lightweight-backend-probe",
         "--backend-id", "phase_to_fft_proxy", "--probe-depth", "shallow"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["status"] == "succeeded"
    assert data["backend_id"] == "phase_to_fft_proxy"


def test_cli_deep_probe_deeplens():
    result = subprocess.run(
        [sys.executable, "-m", "optiresearch.cli", "run-lightweight-backend-probe",
         "--backend-id", "deeplens_geolens_geometric", "--probe-depth", "deep"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["status"] in ("succeeded", "failed")
    assert data["backend_id"] == "deeplens_geolens_geometric"
