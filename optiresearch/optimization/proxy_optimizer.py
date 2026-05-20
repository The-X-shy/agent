"""Black-box optical parameter search for co-design optimization.

Supports coordinate search and random perturbation strategies
for any PSF generator (parameterized_mock or deeplens_parameterized).
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np


class BlackBoxOpticalSearch:
    """Black-box search over optical variables.

    Supports:
    - coordinate_search: tweak one variable at a time
    - random_perturbation: randomly perturb all variables
    - deterministic seed for reproducibility
    - records rejected candidates for audit
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._rejected: list[dict[str, Any]] = []

    @property
    def rejected_candidates(self) -> list[dict[str, Any]]:
        return list(self._rejected)

    def optimize(
        self,
        psf_generator: Any,
        initial_vars: dict[str, float],
        objective_fn: Callable[[np.ndarray], dict[str, Any]],
        max_steps: int = 10,
        strategy: str = "coordinate_search",
        variable_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Run black-box optimization over optical variables.

        Args:
            psf_generator: Callable that takes optical_vars dict and returns PSF result dict.
            initial_vars: Starting optical variable values.
            objective_fn: Function that takes PSF cube and returns metrics dict with 'score'.
            max_steps: Maximum search steps.
            strategy: 'coordinate_search' or 'random_perturbation'.
            variable_bounds: Dict of variable_name → (min, max). Defaults to [0, 1].

        Returns:
            Dict with best_vars, best_score, trajectory, rejected, strategy.
        """
        bounds = variable_bounds or {}
        current_vars = dict(initial_vars)
        psf_result = psf_generator(current_vars)
        psf_cube = psf_result.get("psf_cube")

        if psf_cube is None:
            return {
                "best_vars": current_vars,
                "best_score": 0.0,
                "trajectory": [],
                "rejected": [],
                "strategy": strategy,
                "status": "psf_generation_failed",
            }

        obj_result = objective_fn(psf_cube)
        best_score = float(obj_result.get("score", 0.0))
        best_vars = dict(current_vars)
        trajectory = [{
            "step": 0,
            "vars": dict(current_vars),
            "score": best_score,
            "action": "initial",
        }]

        for step in range(1, max_steps + 1):
            if strategy == "coordinate_search":
                candidate_vars, action = self._coordinate_step(
                    current_vars, bounds, step
                )
            elif strategy == "random_perturbation":
                candidate_vars, action = self._random_step(
                    current_vars, bounds
                )
            else:
                break

            psf_result = psf_generator(candidate_vars)
            psf_cube = psf_result.get("psf_cube")
            if psf_cube is None:
                self._rejected.append({"step": step, "vars": candidate_vars, "reason": "psf_failed"})
                continue

            obj_result = objective_fn(psf_cube)
            score = float(obj_result.get("score", 0.0))

            if score > best_score:
                best_score = score
                best_vars = dict(candidate_vars)
                current_vars = dict(candidate_vars)
                trajectory.append({
                    "step": step,
                    "vars": dict(current_vars),
                    "score": score,
                    "action": action,
                    "improvement": score - trajectory[-1]["score"],
                })
            else:
                self._rejected.append({
                    "step": step,
                    "vars": dict(candidate_vars),
                    "score": score,
                    "reason": "no_improvement",
                })

        return {
            "best_vars": best_vars,
            "best_score": best_score,
            "trajectory": trajectory,
            "rejected": list(self._rejected),
            "strategy": strategy,
            "status": "completed",
        }

    def _coordinate_step(
        self,
        current_vars: dict[str, float],
        bounds: dict[str, tuple[float, float]],
        step: int,
    ) -> tuple[dict[str, float], str]:
        """Tweak one variable at a time, cycling through them."""
        var_names = list(current_vars.keys())
        if not var_names:
            return current_vars, "no_variables"

        target = var_names[(step - 1) % len(var_names)]
        lo, hi = bounds.get(target, (0.0, 1.0))
        current = current_vars[target]
        step_size = (hi - lo) * 0.1

        # Alternate direction
        direction = 1.0 if (step // len(var_names)) % 2 == 0 else -1.0
        new_val = current + direction * step_size

        # If hitting boundary, try other direction
        if new_val > hi:
            new_val = current - step_size
        elif new_val < lo:
            new_val = current + step_size

        new_val = max(lo, min(hi, new_val))
        candidate = dict(current_vars)
        candidate[target] = round(new_val, 4)

        return candidate, f"coordinate_step_{target}_{direction:+.1f}"

    def _random_step(
        self,
        current_vars: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> tuple[dict[str, float], str]:
        """Perturb all variables randomly within bounds."""
        candidate = {}
        for name, value in current_vars.items():
            lo, hi = bounds.get(name, (0.0, 1.0))
            noise = self._rng.normal(0, (hi - lo) * 0.1)
            new_val = value + noise
            candidate[name] = round(max(lo, min(hi, new_val)), 4)
        return candidate, "random_perturbation"
