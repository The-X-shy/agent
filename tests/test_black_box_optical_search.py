"""Test BlackBoxOpticalSearch with mock PSF generator."""
import numpy as np
from optiresearch.optimization.proxy_optimizer import BlackBoxOpticalSearch
from optiresearch.adapters.parameterized_psf import generate_parameterized_psf, compute_psf_metrics


def _mock_psf_generator(optical_vars):
    psf = generate_parameterized_psf(optical_vars, seed=42)
    return {"psf_cube": psf, "status": "succeeded"}


def _mock_objective_fn(psf_cube):
    metrics = compute_psf_metrics(psf_cube)
    score = metrics.get("coding_strength", 0)
    return {"score": float(score), "metrics": metrics}


def test_coordinate_search_improves_score():
    search = BlackBoxOpticalSearch(seed=42)
    initial = {"surface_curvature": 0.5, "depth_variation": 0.5, "chromatic_shift": 0.3}

    result = search.optimize(
        psf_generator=_mock_psf_generator,
        initial_vars=initial,
        objective_fn=_mock_objective_fn,
        max_steps=5,
        strategy="coordinate_search",
    )

    assert result["status"] == "completed"
    assert result["best_score"] > 0
    assert len(result["trajectory"]) >= 1


def test_random_perturbation_runs():
    search = BlackBoxOpticalSearch(seed=42)
    initial = {"surface_curvature": 0.5, "depth_variation": 0.5}

    result = search.optimize(
        psf_generator=_mock_psf_generator,
        initial_vars=initial,
        objective_fn=_mock_objective_fn,
        max_steps=3,
        strategy="random_perturbation",
    )

    assert result["status"] == "completed"
    assert result["best_score"] >= 0


def test_rejected_candidates_recorded():
    search = BlackBoxOpticalSearch(seed=42)
    initial = {"surface_curvature": 0.1}

    result = search.optimize(
        psf_generator=_mock_psf_generator,
        initial_vars=initial,
        objective_fn=_mock_objective_fn,
        max_steps=5,
        strategy="coordinate_search",
        variable_bounds={"surface_curvature": (0.0, 1.0)},
    )

    rejected = search.rejected_candidates
    assert isinstance(rejected, list)


def test_bounds_respected():
    search = BlackBoxOpticalSearch(seed=42)
    initial = {"surface_curvature": 0.5}
    bounds = {"surface_curvature": (0.0, 1.0)}

    result = search.optimize(
        psf_generator=_mock_psf_generator,
        initial_vars=initial,
        objective_fn=_mock_objective_fn,
        max_steps=10,
        strategy="coordinate_search",
        variable_bounds=bounds,
    )

    for t in result["trajectory"]:
        for var_name, val in t["vars"].items():
            lo, hi = bounds.get(var_name, (0.0, 1.0))
            assert lo <= val <= hi, f"{var_name}={val} outside [{lo}, {hi}]"


def test_deterministic_seed():
    initial = {"surface_curvature": 0.5, "depth_variation": 0.5}

    s1 = BlackBoxOpticalSearch(seed=42)
    r1 = s1.optimize(_mock_psf_generator, initial, _mock_objective_fn, max_steps=3, strategy="coordinate_search")

    s2 = BlackBoxOpticalSearch(seed=42)
    r2 = s2.optimize(_mock_psf_generator, initial, _mock_objective_fn, max_steps=3, strategy="coordinate_search")

    assert r1["best_score"] == r2["best_score"]
