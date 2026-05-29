"""CLI contract tests for component surrogate HSI co-design."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "optiresearch.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_run_component_surrogate_hsi_help():
    result = _run_cli(["run-component-surrogate-hsi-codesign", "--help"])
    assert result.returncode == 0
    assert "--component" in result.stdout
    assert "--dataset" in result.stdout


def test_run_component_surrogate_hsi_fresnel_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_cli([
        "run-component-surrogate-hsi-codesign",
        "--component", "fresnel",
        "--dataset", "synthetic",
        "--steps", "3",
        "--device", "cpu",
    ])
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["component_type"] == "fresnel"
    assert output["status"] == "succeeded"
    assert output["psf_requires_grad"] is True
    assert output["loss_requires_grad"] is True
    assert output["claim_ceiling"] == "component_surrogate_hsi_codesign"


def test_run_remote_component_surrogate_hsi_command_uses_remote_wrapper(monkeypatch):
    captured = {}

    def _fake_remote(worker_id, component, dataset, steps, device, **_kwargs):
        captured.update({
            "worker_id": worker_id,
            "component": component,
            "dataset": dataset,
            "steps": steps,
            "device": device,
        })
        return {"result": {"job_id": "remote_job_0000000000000001", "status": "succeeded"}, "ingestion": None}

    monkeypatch.setattr(
        "optiresearch.cli.run_remote_component_surrogate_hsi_codesign",
        _fake_remote,
    )
    from optiresearch.cli import main

    main([
        "run-remote-component-surrogate-hsi-codesign",
        "--worker-id", "windows_wsl",
        "--component", "binary2phase",
        "--dataset", "synthetic",
        "--steps", "3",
        "--device", "cpu",
    ])

    assert captured == {
        "worker_id": "windows_wsl",
        "component": "binary2phase",
        "dataset": "synthetic",
        "steps": 3,
        "device": "cpu",
    }
