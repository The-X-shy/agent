"""CLI wrapper for the mock PSF generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optiresearch.adapters.mock_deeplens import MockDeepLensAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic mock PSF simulation.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec-json", default="{}")
    parser.add_argument("--sweep-json", default="{}")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = MockDeepLensAdapter(seed=args.seed).simulate_psf_cube(
        spec=json.loads(args.spec_json),
        sweep=json.loads(args.sweep_json),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"artifacts": [str(path) for path in result["artifacts"]], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
