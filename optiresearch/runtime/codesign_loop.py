"""Optical–HSI co-design optimization loop.

Implements the closed-loop co-design flow:
  Agent proposes optical parameters
  → Parameterized PSF generation
  → HSI forward model + reconstruction
  → Loss/metric evaluation
  → Agent reviews and proposes new parameters
  → Repeat

The "backward update" is agent-driven: the LLM analyzes
reconstruction metrics and proposes new optical variable values.
This is black-box optimization with LLM guidance + rule fallback.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from optiresearch.adapters.parameterized_psf import (
    compute_psf_metrics,
    generate_parameterized_psf,
    optical_vars_to_dict,
)
from optiresearch.adapters.deeplens_parameterized_psf import (
    DeepLensParameterizedPSFGenerator,
)
from optiresearch.hsi.dataset import SyntheticHSIDataset
from optiresearch.hsi.forward_model import HSIForwardModel
from optiresearch.hsi.optical_features import OpticalFeatureExtractor
from optiresearch.schemas.hsi import (
    build_default_hsi_forward_model_spec,
    build_default_hsi_reconstruction_spec,
    build_default_synthetic_hsi_dataset_spec,
)
from optiresearch.schemas.optimization import (
    CoDesignState,
    OptimizationSpec,
    build_default_optical_variables,
)
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.memory.meta_trace import MetaTrace, MetaTraceWriter
from optiresearch.memory.schemas import make_deterministic_id
from optiresearch.llm.registry import get_llm_provider
from optiresearch.llm.base import LLMProviderError
from optiresearch.schemas.autonomous import ResearchIterationPlan


def run_codesign_loop(spec: OptimizationSpec) -> dict[str, Any]:
    """Run the full optical–HSI co-design optimization loop.

    Args:
        spec: OptimizationSpec with optical variables, target metrics, etc.

    Returns:
        Dict with loop_id, states, best_params, best_metrics, trajectory, claims.
    """
    loop_id = make_deterministic_id("codesign", spec.objective or "codesign", str(time.time()))
    output_dir = Path("workspace/codesign_loops") / loop_id
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "optimization_spec.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )

    claim_manager = ClaimEvidenceManager(workspace_id=loop_id)
    trace_writer = MetaTraceWriter()
    provider = get_llm_provider(spec.llm_provider)
    llm_available = provider.available()

    # Initialize optical variables
    if spec.optical_variables:
        optical_vars = {v.name: v.current_value for v in spec.optical_variables}
    else:
        optical_vars = optical_vars_to_dict(build_default_optical_variables())

    states: list[CoDesignState] = []
    best_score = -1.0
    best_params: dict[str, float] = dict(optical_vars)
    best_metrics: dict[str, Any] = {}
    best_iteration = -1
    trajectory: list[dict[str, Any]] = []
    stopped_reason = ""

    # Setup PSF generator based on psf_source
    deeplens_gen = None
    fallback_used = False
    unsupported_vars: list[dict[str, Any]] = []

    if spec.psf_source == "deeplens_parameterized":
        deeplens_gen = DeepLensParameterizedPSFGenerator(strict_deeplens=spec.strict_deeplens)
        if spec.fallback_policy == "fail" and not deeplens_gen.deeplens_available:
            return {
                "loop_id": loop_id,
                "objective": spec.objective,
                "total_iterations": 0,
                "best_params": {},
                "best_metrics": {},
                "error": "DEEPLENS_UNAVAILABLE",
                "fallback_policy": "fail",
                "output_dir": str(output_dir),
                "caveats": ["DeepLens SDK not available and fallback_policy=fail."],
            }

    for iteration in range(1, spec.max_iterations + 1):
        # Phase 1: Generate PSF from current optical parameters
        unsupported_vars = []
        fallback_used = False

        if spec.psf_source == "deeplens_parameterized" and deeplens_gen is not None:
            psf_result = deeplens_gen.generate_psf_cube(
                optical_vars,
                output_dir / f"iteration_{iteration:03d}",
            )
            unsupported_vars = psf_result.get("unsupported_variables", [])
            fallback_used = psf_result.get("fallback_used", False)
            psf_cube = psf_result.get("psf_cube")

            if psf_cube is None:
                if spec.fallback_policy == "fallback_to_mock":
                    psf_cube = generate_parameterized_psf(
                        optical_vars, encoder_type=spec.encoder_type,
                    )
                    fallback_used = True
                else:
                    continue

            psf_metrics = compute_psf_metrics(psf_cube) if psf_cube is not None else {}
            psf_path_obj = output_dir / f"iteration_{iteration:03d}_psf_cube.npz"
            if psf_cube is not None:
                np.savez_compressed(psf_path_obj, psf_cube=psf_cube)
            psf_path = str(psf_path_obj)
        else:
            psf_cube = generate_parameterized_psf(
                optical_vars,
                depth_planes=5,
                wavelength_bands=31,
                psf_size=32,
                encoder_type=spec.encoder_type,
            )
            psf_metrics = compute_psf_metrics(psf_cube)
            psf_path = str(output_dir / f"iteration_{iteration:03d}_psf_cube.npz")
            np.savez_compressed(psf_path, psf_cube=psf_cube)

        # Phase 2: Run HSI pipeline with this PSF
        hsi_metrics = _run_hsi_with_psf(
            psf_cube=psf_cube,
            psf_path=str(psf_path),
            encoder_type=spec.encoder_type,
            reconstructor_type=spec.reconstructor_type,
            forward_mode=spec.forward_mode,
            dataset=spec.dataset,
            output_dir=output_dir,
            iteration=iteration,
        )

        # Phase 3: Compute composite score
        score = _compute_score(hsi_metrics, spec.target_metrics)
        loss = 1.0 - score if score > 0 else 1.0
        improvement = (score - best_score) if best_score >= 0 else None

        # Phase 4: Agent review (or rule fallback)
        agent_decision, agent_rationale, next_vars = _agent_review(
            spec=spec,
            provider=provider,
            llm_available=llm_available,
            iteration=iteration,
            optical_vars=optical_vars,
            psf_metrics=psf_metrics,
            hsi_metrics=hsi_metrics,
            score=score,
            improvement=improvement,
            best_score=best_score,
            previous_states=states,
        )

        state = CoDesignState(
            iteration=iteration,
            optical_variables=dict(optical_vars),
            psf_metrics=psf_metrics,
            hsi_metrics=hsi_metrics,
            reconstruction_score=score,
            loss_value=loss,
            improvement_from_previous=improvement,
            agent_decision=agent_decision,
            agent_rationale=agent_rationale,
        )
        states.append(state)
        trajectory.append({
            "iteration": iteration,
            "optical_vars": dict(optical_vars),
            "psf_depth_stability": psf_metrics.get("depth_stability_score", 0),
            "psf_spectral_sep": psf_metrics.get("spectral_separability_score", 0),
            "psnr": hsi_metrics.get("PSNR", 0),
            "sam": hsi_metrics.get("SAM", 0),
            "score": score,
            "loss": loss,
            "improvement": improvement,
            "psf_source": spec.psf_source,
            "backend": spec.backend,
            "unsupported_variables": [u.get("variable", "") for u in unsupported_vars],
            "fallback_used": fallback_used,
            "differentiable": False,
            "native_parameter_update": False,
        })

        # Save state
        (output_dir / f"iteration_{iteration:03d}_state.json").write_text(
            json.dumps(state.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Write trace
        trace_writer.write_trace(MetaTrace(
            trace_id=make_deterministic_id("trace", loop_id, f"codesign_{iteration}"),
            workspace_id=loop_id,
            run_id=loop_id,
            branch_id=None,
            step_id=None,
            phase="Execute",
            actor="System",
            task=f"Co-design iteration {iteration}: score={score:.4f}",
            skill_id=None,
            skill_version=None,
            tool=None,
            next_action=None,
            status="succeeded",
            timestamp_start=None,
            timestamp_end=None,
            content_hash=None,
            metadata={
                "iteration": iteration,
                "score": score,
                "optical_vars": optical_vars,
                "psf_metrics": psf_metrics,
                "hsi_metrics": {k: v for k, v in hsi_metrics.items() if isinstance(v, (int, float, str, bool))},
                "agent_decision": agent_decision,
            },
        ))

        # Track best
        if score > best_score:
            best_score = score
            best_params = dict(optical_vars)
            best_metrics = dict(hsi_metrics)
            best_iteration = iteration

        # Create claim
        claim_text = f"Co-design iteration {iteration}: {spec.encoder_type} with optical_vars={json.dumps(optical_vars)} achieves score={score:.4f}"
        claim = claim_manager.create_claim(claim_text, {
            "backend": spec.backend,
            "evidence_domain": "codesign",
            "iteration": iteration,
            "optical_vars": optical_vars,
        })
        claim_manager.review_claim(claim.claim_id)

        # Stopping criteria
        if agent_decision == "stop":
            stopped_reason = agent_rationale or "Agent requested stop"
            break
        if iteration >= spec.max_iterations:
            stopped_reason = f"Max iterations ({spec.max_iterations}) reached"
            break
        if improvement is not None and improvement <= spec.convergence_threshold and iteration > 2:
            stopped_reason = f"Converged (improvement {improvement:.6f} <= threshold {spec.convergence_threshold})"
            break

        # Update optical variables for next iteration
        if next_vars:
            optical_vars.update(next_vars)

    if not stopped_reason:
        stopped_reason = f"Max iterations ({spec.max_iterations}) reached"

    # Build result
    result = {
        "loop_id": loop_id,
        "objective": spec.objective,
        "total_iterations": len(states),
        "best_iteration": best_iteration,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "trajectory": trajectory,
        "stopped_reason": stopped_reason,
        "output_dir": str(output_dir),
        "psf_source": spec.psf_source,
        "backend": spec.backend,
        "fallback_policy": spec.fallback_policy,
        "fallback_used_any": any(t.get("fallback_used", False) for t in trajectory),
        "differentiable": False,
        "native_parameter_update": False,
        "caveats": _build_caveats(spec, fallback_used),
    }

    # Export summary and report
    (output_dir / "codesign_loop_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _export_codesign_report(result, output_dir)

    return result


def _run_hsi_with_psf(
    psf_cube: np.ndarray,
    psf_path: str,
    encoder_type: str,
    reconstructor_type: str,
    forward_mode: str,
    dataset: str,
    output_dir: Path,
    iteration: int,
) -> dict[str, Any]:
    """Run HSI forward + reconstruction using a pre-computed PSF cube."""
    try:
        # Build dataset
        dataset_spec = build_default_synthetic_hsi_dataset_spec()
        hsi_dataset = SyntheticHSIDataset(dataset_spec)
        train_data = hsi_dataset.generate_split("train")
        if train_data is None:
            return {"PSNR": 0.0, "SAM": 1.0, "SSIM": 0.0, "error": "empty_dataset"}

        # Dataset returns dict with 'hsi' [N, B, H, W] and 'depth_indices' [N]
        if isinstance(train_data, dict):
            hsi_cubes = train_data.get("hsi")
            depth_indices = train_data.get("depth_indices")
        elif hasattr(train_data, "hsi"):
            hsi_cubes = train_data.hsi
            depth_indices = getattr(train_data, "depth_indices", None)
        else:
            hsi_cubes = train_data

        if hsi_cubes is None:
            return {"PSNR": 0.0, "SAM": 1.0, "SSIM": 0.0, "error": "empty_dataset"}

        hsi_cubes = np.asarray(hsi_cubes, dtype=np.float64)
        if hsi_cubes.ndim == 4:
            num_samples = hsi_cubes.shape[0]
        elif hsi_cubes.ndim == 3:
            hsi_cubes = hsi_cubes[np.newaxis, ...]
            num_samples = 1
        else:
            return {"PSNR": 0.0, "SAM": 1.0, "SSIM": 0.0, "error": f"bad_shape_{hsi_cubes.shape}"}

        # Build forward model with the parameterized PSF
        forward_spec = build_default_hsi_forward_model_spec(forward_mode=forward_mode)
        forward_spec.optical_artifact_id = f"codesign_psf_{iteration}"
        forward_model = HSIForwardModel(forward_spec)

        # Load PSF from the saved file
        loaded_psf = forward_model.load_psf_cube(psf_path)

        # Extract optical features
        extractor = OpticalFeatureExtractor()
        features = extractor.extract(psf_cube)

        # Render measurements for a small batch
        measurements = []
        targets = []
        batch_size = min(4, num_samples)
        for i in range(batch_size):
            cube = hsi_cubes[i]  # [B, H, W]
            cube = np.asarray(cube, dtype=np.float64)
            if cube.ndim == 3:
                depth_idx = int(depth_indices[i]) if depth_indices is not None and i < len(depth_indices) else i % 5
                meas = forward_model.render_measurement(cube, loaded_psf, depth_index=depth_idx)
                measurements.append(meas)
                targets.append(cube)

        if not measurements:
            return {"PSNR": 0.0, "SAM": 1.0, "SSIM": 0.0, "error": "no_measurements"}

        # Split into train/test
        n = len(measurements)
        split = max(1, n * 3 // 4)
        train_meas = np.stack(measurements[:split])
        train_tgt = np.stack(targets[:split])
        test_meas = np.stack(measurements[split:]) if split < n else train_meas
        test_tgt = np.stack(targets[split:]) if split < n else train_tgt
        test_depths = np.arange(len(test_tgt)) % 5

        # Run reconstruction via the standard path
        from optiresearch.hsi.reconstruction import run_reconstruction as run_recon
        recon_dir = output_dir / f"recon_{iteration:03d}"
        recon_result = run_recon(
            reconstructor_type=reconstructor_type,
            train_measurements=train_meas,
            train_targets=train_tgt,
            test_measurements=test_meas,
            test_targets=test_tgt,
            test_depth_indices=test_depths,
            output_dir=recon_dir,
            optical_features=features,
            use_optical_feature_maps=False,
        )

        # Extract metrics
        metrics = dict(recon_result.get("metrics", {}))
        metrics.update({
            "optical_coding_strength": features.get("coding_strength", 0),
            "optical_depth_stability_score": features.get("depth_stability_score", 0),
            "optical_spectral_separability_score": features.get("spectral_separability_score", 0),
            "optical_band_condition_score": features.get("band_condition_score", 0),
        })

        return metrics

    except Exception as exc:
        return {"PSNR": 0.0, "SAM": 1.0, "SSIM": 0.0, "error": str(exc)}


def _compute_score(metrics: dict[str, Any], target_metrics: list[str]) -> float:
    """Compute composite score from metrics."""
    if not target_metrics:
        target_metrics = ["PSNR", "reconstruction_score"]

    score = 0.0
    weight = 1.0 / max(len(target_metrics), 1)

    for metric in target_metrics:
        value = metrics.get(metric, 0.0)
        if isinstance(value, (int, float)):
            if metric in ("SAM", "ERGAS", "loss"):
                # Lower is better
                score += weight * max(0.0, 1.0 - float(value))
            else:
                # Higher is better
                norm = float(value) / 100.0 if metric == "PSNR" else float(value)
                score += weight * min(1.0, max(0.0, norm))

    return round(score, 6)


def _agent_review(
    spec: OptimizationSpec,
    provider: Any,
    llm_available: bool,
    iteration: int,
    optical_vars: dict[str, float],
    psf_metrics: dict[str, Any],
    hsi_metrics: dict[str, Any],
    score: float,
    improvement: float | None,
    best_score: float,
    previous_states: list[CoDesignState],
) -> tuple[str, str, dict[str, float] | None]:
    """Ask agent (or rule fallback) to review and propose next parameters."""
    if llm_available and spec.llm_provider != "mock":
        try:
            prompt = _build_codesign_review_prompt(
                spec, iteration, optical_vars, psf_metrics, hsi_metrics,
                score, improvement, best_score, previous_states,
            )
            from optiresearch.schemas.autonomous import ReviewerOutput
            response = provider.structured_complete(
                [{"role": "user", "content": prompt}],
                ReviewerOutput,
            )
            if isinstance(response, ReviewerOutput):
                next_vars = _parse_next_vars_from_decision(
                    response.next_action, response.recommendation_for_human, optical_vars
                )
                return response.next_action, response.iteration_assessment, next_vars
        except (LLMProviderError, Exception):
            pass

    return _rule_based_codesign_review(iteration, optical_vars, score, improvement, best_score, spec)


def _rule_based_codesign_review(
    iteration: int,
    optical_vars: dict[str, float],
    score: float,
    improvement: float | None,
    best_score: float,
    spec: OptimizationSpec,
) -> tuple[str, str, dict[str, float] | None]:
    """Deterministic fallback for co-design review."""
    max_iter = spec.max_iterations
    if iteration >= max_iter:
        return "stop", f"Max iterations ({max_iter}) reached.", None

    # Simple hill-climbing: tweak the variable with the most range remaining
    next_vars: dict[str, float] = {}
    var_names = list(optical_vars.keys())
    if var_names:
        target_var = var_names[iteration % len(var_names)]
        current = optical_vars[target_var]
        step = 0.1
        if improvement is not None and improvement <= 0:
            step = -step  # reverse direction if no improvement
        new_val = max(0.0, min(1.0, current + step))
        if target_var == "doe_grating_period":
            new_val = max(0.1, min(2.0, current + step * 2.0))
        next_vars[target_var] = round(new_val, 3)

    rationale = f"Rule-based: adjusting {target_var} from {current:.3f} to {new_val:.3f}"
    return "continue", rationale, next_vars


def _build_codesign_review_prompt(
    spec: OptimizationSpec,
    iteration: int,
    optical_vars: dict[str, float],
    psf_metrics: dict[str, Any],
    hsi_metrics: dict[str, Any],
    score: float,
    improvement: float | None,
    best_score: float,
    previous_states: list[CoDesignState],
) -> str:
    """Build LLM prompt for co-design review."""
    prev_text = "No previous iterations."
    if previous_states:
        lines = []
        for s in previous_states[-3:]:
            lines.append(f"- Iter {s.iteration}: score={s.reconstruction_score:.4f}, vars={json.dumps(s.optical_variables)}")
        prev_text = "\n".join(lines)

    return f"""You are an optical-HSI co-design optimization agent.

## Objective
{spec.objective or 'Optimize optical parameters for best HSI reconstruction'}

## Current Iteration: {iteration}
- Optical variables: {json.dumps(optical_vars, indent=2)}
- PSF metrics: {json.dumps(psf_metrics, indent=2)}
- HSI metrics: {json.dumps({k: v for k, v in hsi_metrics.items() if isinstance(v, (int, float, str))}, indent=2)}
- Composite score: {score:.6f}
- Improvement from previous: {improvement}
- Best score so far: {best_score:.6f}

## Previous Iterations
{prev_text}

## Allowed Optical Variables
{json.dumps([{'name': v.name, 'min': v.min_value, 'max': v.max_value, 'current': v.current_value} for v in spec.optical_variables], indent=2)}

## Evidence Limitations
- This is mock/synthetic co-design optimization.
- Results do NOT represent native DeepLens physical optimization.
- Do NOT claim real optical performance.

## Instructions
Evaluate the current iteration and decide next action. Output valid JSON with:
- iteration_assessment: brief assessment
- improvement_detected: boolean
- improvement_detail: what changed
- evidence_level: "codesign_mock"
- caveats: list of honest limitations
- next_action: "continue" or "stop"
- recommendation_for_human: what to do next, including suggested optical variable changes
"""


def _parse_next_vars_from_decision(
    next_action: str,
    recommendation: str,
    current_vars: dict[str, float],
) -> dict[str, float] | None:
    """Attempt to parse optical variable changes from agent recommendation."""
    if next_action == "stop":
        return None

    next_vars: dict[str, float] = {}
    for var_name in current_vars:
        # Look for patterns like "increase phase_mask_strength to 0.7" or "set doe_grating_period=1.5"
        import re
        for pattern in [
            rf'{var_name}[=\s]+([0-9.]+)',
            rf'{var_name.replace("_", " ")}[=\s]+([0-9.]+)',
            rf'increase {var_name}[=\s]+([0-9.]+)',
            rf'decrease {var_name}[=\s]+([0-9.]+)',
        ]:
            match = re.search(pattern, recommendation.lower())
            if match:
                try:
                    next_vars[var_name] = float(match.group(1))
                except ValueError:
                    pass
                break

    return next_vars if next_vars else None


def _build_caveats(spec: OptimizationSpec, fallback_used: bool) -> list[str]:
    caveats = [
        "Optical parameter updates are agent-driven (black-box), not gradient-based.",
        "Results do NOT represent native DeepLens differentiable optimization.",
        "Agent decisions are recommendations, not validated conclusions.",
    ]
    if spec.psf_source == "parameterized_mock" or fallback_used:
        caveats.append("PSF generated by parameterized mock model, NOT real DeepLens.")
    if spec.psf_source == "deeplens_parameterized" and not fallback_used:
        caveats.append("PSF generated by DeepLens-backed black-box search, NOT differentiable optimization.")
    if spec.backend == "mock_deeplens":
        caveats.append("Mock backend — not real optical hardware.")
    return caveats


def _export_codesign_report(result: dict[str, Any], output_dir: Path) -> Path:
    """Export co-design loop report as markdown."""
    lines = [
        "# Co-Design Optimization Loop Report",
        "",
        f"**Loop ID:** `{result['loop_id']}`",
        f"**Objective:** {result.get('objective', 'N/A')}",
        f"**Total iterations:** {result['total_iterations']}",
        f"**Stopped reason:** {result['stopped_reason']}",
        f"**Best iteration:** {result['best_iteration']}",
        f"**Best score:** {result.get('best_score', 'N/A')}",
        "",
        "## 1. Optical Variable Trajectory",
        "",
        "| Iter | " + " | ".join(result.get('best_params', {}).keys()) + " | Score | Improvement |",
        "|---|" + "|".join(["---" for _ in result.get('best_params', {})]) + "|---:|---:|",
    ]
    for t in result.get("trajectory", []):
        vars_str = " | ".join(
            f"{t['optical_vars'].get(k, 0):.3f}"
            for k in result.get("best_params", {})
        )
        imp = t.get("improvement")
        imp_str = f"{imp:+.4f}" if imp is not None else "—"
        lines.append(f"| {t['iteration']} | {vars_str} | {t.get('score', 0):.4f} | {imp_str} |")

    lines.extend([
        "",
        "## 2. Metric Trajectory",
        "",
        "| Iter | PSNR | SAM | Depth Stab | Spectral Sep | Coding Str |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for t in result.get("trajectory", []):
        lines.append(
            f"| {t['iteration']} | {t.get('psnr', 0):.4f} | {t.get('sam', 0):.4f} | "
            f"{t.get('psf_depth_stability', 0):.4f} | {t.get('psf_spectral_sep', 0):.4f} | "
            f"{float(t.get('psf_depth_stability', 0)) * float(t.get('psf_spectral_sep', 0)):.4f} |"
        )

    lines.extend([
        "",
        "## 3. Best Parameters",
        "",
        "| Variable | Value |",
        "|---|---|",
    ])
    for k, v in result.get("best_params", {}).items():
        lines.append(f"| {k} | {v:.4f} |")

    lines.extend([
        "",
        "## 4. Best Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ])
    for k, v in result.get("best_metrics", {}).items():
        if isinstance(v, (int, float)):
            lines.append(f"| {k} | {v:.4f} |")

    lines.extend([
        "",
        "## 5. Caveats",
        "",
    ])
    for c in result.get("caveats", []):
        lines.append(f"- **{c}**")

    lines.extend([
        "",
        "## 6. Limitations",
        "",
        "- Co-design optimization is agent-driven (black-box), not gradient-based.",
        "- Optical parameters affect PSF through a parameterized mock model.",
        "- Native DeepLens physical optimization requires DeepLens SDK integration.",
        "- All HSI data is synthetic; real camera validation requires lab measurements.",
        "",
        "## 7. Next Steps",
        "",
        "1. Review best optical parameters and their PSF characteristics.",
        "2. Run additional iterations with different encoder types.",
        "3. Integrate native DeepLens optimization when SDK supports it.",
        "4. Validate best parameters with real HSI data.",
    ])

    path = output_dir / "codesign_loop_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
