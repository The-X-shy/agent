"""Allowlisted skill execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.mock_deeplens import MockDeepLensAdapter


class SkillExecutor:
    """Execute curated skill commands only."""

    def execute(self, skill_id: str, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        if skill_id != "deeplens-adapter" or command not in {"run_mock_psf", "run_deeplens_psf"}:
            return {
                "status": "failed",
                "artifacts": [],
                "metrics": {},
                "logs": [],
                "errors": [f"Command not allowlisted: {skill_id}/{command}"],
            }
        try:
            output_dir = Path(args.get("output_dir", "./workspace/runs/mock_psf"))
            if command == "run_deeplens_psf":
                result = DeepLensAdapter().simulate_psf_cube(
                    spec=args.get("spec", {}),
                    sweep=args.get("sweep", {}),
                    output_dir=output_dir,
                    realization=args.get("realization", "auto"),
                )
            else:
                result = MockDeepLensAdapter(seed=int(args.get("seed", 42))).simulate_psf_cube(
                    spec=args.get("spec", {}),
                    sweep=args.get("sweep", {}),
                    output_dir=output_dir,
                )
            return {
                "status": result["status"],
                "artifacts": [str(path) for path in result["artifacts"]],
                "metrics": result["metrics"],
                "logs": result["logs"],
                "errors": result["errors"],
                "metadata": result["metadata"],
            }
        except Exception as exc:  # pragma: no cover - defensive return path
            return {"status": "failed", "artifacts": [], "metrics": {}, "logs": [], "errors": [str(exc)]}
