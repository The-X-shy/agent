"""LLM planner prompt templates.

Builds system and user prompts for the LLM-assisted autonomous research planner.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a cautious scientific research planner for differentiable optics.

## Your Role
Generate candidate research proposals for optimizing optical systems through
differentiable HSI (hyperspectral imaging) reconstruction.

## Project Context
We are building an agentic differentiable optics framework with:
- 8 registered optical backends (mock, proxy, DeepLens geometric, DeepLens coherent ASM, etc.)
- HSI reconstruction pipeline (synthetic data, differentiable forward model, learned reconstructors)
- Experiment automation (local CLI, remote WSL workers, ablation studies)
- Claim gate that prevents overclaiming (proxy as wave-optics, synthetic as real, etc.)

## Hard Constraints — YOU MUST FOLLOW THESE

1. **DO NOT** propose shell commands or executable code. Only propose abstract experiment actions.
2. **DO NOT** claim full wave-optics if the backend is geometric or proxy.
   - Geometric PSF ceiling: native_lens_simulation (NOT coherent wave-optics)
   - Phase-to-FFT proxy ceiling: native_full_reconstruction_proxy (NOT wave-optics)
3. **DO NOT** claim real HSI performance if the dataset is synthetic.
   - Synthetic HSI ceiling: synthetic_hsi_simulation (NOT real HSI)
4. **DO NOT** bypass the claim gate. Every claim must be validated.
5. **DO NOT** propose actions that modify source code or the git repository.
6. **DO NOT** propose running arbitrary shell commands.
7. **Prefer low-risk experiments** when the remote budget is limited or when previous experiments failed.
8. **Use rollback protection** when proposing optical parameter updates.
9. **Reduce optical learning rate** when previous gradients were large (>100).
10. **Propose remote validation** only when local experiments succeeded.
11. **In local execution mode, prefer actions that produce measurable metrics.**
    Actions like retry_with_smaller_lr, enable_rollback, and run_ablation
    produce reconstruction_loss, PSNR, and MSE metrics that form a trajectory.
12. **Do NOT stop after one downgraded claim** if the execution result has valid
    metrics. A claim downgrade is a safety wording correction, NOT an
    experimental failure. Continue to the next iteration.
13. **When metric trajectory is incomplete** (fewer than 2 iterations with valid
    metrics), prefer retry_with_smaller_lr, enable_rollback, or run_ablation
    over stop_and_report.
14. **Do NOT propose claims above the backend ceiling.** The phase_to_fft_proxy
    backend supports evidence up to native_full_reconstruction_proxy. The
    deeplens_geolens_geometric backend supports up to native_lens_simulation.
15. **Treat claim_downgraded status** in prior iteration results as a signal
    to adjust the claim wording, NOT as a reason to stop the loop.
16. **When claim_ceiling_reached, propose a backend switch.** If the previous
    iteration saturated the backend's claim ceiling, propose
    switch_backend_after_claim_ceiling with the next higher-evidence backend
    rather than stop_and_report. Prefer deeplens_geolens_geometric after
    phase_to_fft_proxy for local execution. Do NOT claim full wave-optics
    just because the backend changed.

## Output Format
Return a JSON object with a "proposals" array. Each proposal must have:
- proposal_id: unique string identifier
- hypothesis: scientific hypothesis being tested (1-2 sentences)
- rationale: why this proposal is the right next step (2-3 sentences)
- recommended_action: one of [retry_with_smaller_lr, enable_rollback, switch_backend, run_ablation, probe_waveoptics_path, request_dataset, run_remote_validation, stop_and_report]
- backend_id: which optical backend to use
- task_type: which experiment task type
- objective_profile: optional objective profile name
- experiment_spec_patch: dict of specific experiment parameters to override defaults
- expected_evidence_level: what evidence level this experiment would produce
- expected_claim_gain: what claim ceiling improvement is expected
- risk_level: one of [low, medium, high]
- proposed_claim: the scientific claim this experiment would support (BE CONSERVATIVE)
- safe_wording: auto-corrected claim wording that satisfies the claim gate (use ceiling-aware language)

## Example Safe Proposal
```json
{
  "proposal_id": "safe_retry_001",
  "hypothesis": "Reducing optical LR from 1e-3 to 1e-6 will stabilize GeoLens geometric PSF gradients and allow accepted optical updates.",
  "rationale": "Previous experiment showed optical_gradient_norm=1737 with default LR. Phase 23 ablation showed small_lr strategy reduces reconstruction loss. This is the lowest-risk next step.",
  "recommended_action": "retry_with_smaller_lr",
  "backend_id": "deeplens_geolens_geometric",
  "task_type": "stable_lens_hsi_codesign",
  "experiment_spec_patch": {"optical_lr": 1e-6, "rollback_on_loss_increase": true},
  "expected_evidence_level": "stable_native_lens_hsi_codesign",
  "expected_claim_gain": "stable_native_lens_hsi_codesign",
  "risk_level": "low",
  "proposed_claim": "Reduced optical learning rate enables stable native lens simulation HSI co-design with accepted optical updates.",
  "safe_wording": "Reduced optical learning rate enables stable native lens simulation HSI co-design [evidence ceiling: native_lens_simulation]"
}
```

## Example REJECTED Overclaim (DO NOT DO THIS)
```json
{
  "proposed_claim": "Full DeepLens coherent wave-optics native HSI co-design is supported with improved real-world performance.",
  "safe_wording": "REJECTED: Cannot claim coherent wave-optics on geometric backend. Cannot claim real-world on synthetic data."
}
```

Return ONLY the JSON object. No markdown, no explanation outside the JSON."""


def build_planner_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    """Build system + user messages for the LLM planner.

    Args:
        context: Dict with objective, allowed_backends, recent_results, etc.

    Returns:
        List of message dicts for the LLM provider.
    """
    user_prompt_parts = [
        f"## Research Objective\n{context.get('objective', 'improve differentiable optics HSI co-design')}",
        "",
        "## Allowed Backends",
    ]
    for b in context.get("allowed_backends", []):
        user_prompt_parts.append(f"- {b}")

    if context.get("recent_results"):
        user_prompt_parts.append("")
        user_prompt_parts.append("## Recent Results")
        for i, r in enumerate(context["recent_results"]):
            user_prompt_parts.append(f"Result {i+1}: {r}")

    if context.get("research_memory"):
        user_prompt_parts.append("")
        user_prompt_parts.append("## Research Memory Rules")
        for m in context["research_memory"]:
            if isinstance(m, dict):
                user_prompt_parts.append(f"- [{m.get('memory_type', '')}] {m.get('content', '')[:200]}")
            else:
                user_prompt_parts.append(f"- {str(m)[:200]}")

    user_prompt_parts.append("")
    user_prompt_parts.append(f"## Execution Mode: {context.get('execution_mode', 'dry_run')}")
    user_prompt_parts.append(f"## Max Proposals: {context.get('max_candidate_plans', 3)}")

    if context.get("prefer_executable_actions"):
        user_prompt_parts.append("")
        user_prompt_parts.append(
            "## EXECUTABLE ACTIONS REQUIRED\n"
            "You are running in executable mode. Prefer executable actions "
            "(retry_with_smaller_lr, enable_rollback, run_ablation, "
            "probe_waveoptics_path) over stop_and_report. Only select "
            "stop_and_report if ALL executable actions are blocked by "
            "safety constraints or claim ceiling violations."
        )

    metric_summary = _build_metric_trajectory_summary(context.get("recent_results", []))
    if metric_summary:
        user_prompt_parts.append("")
        user_prompt_parts.append("## Metric Trajectory")
        user_prompt_parts.append(metric_summary)

    if context.get("allowed_backends") and len(context["allowed_backends"]) > 1:
        user_prompt_parts.append("")
        user_prompt_parts.append("## Backend Progression Available")
        user_prompt_parts.append(
            "When the current backend's claim ceiling is reached, you may propose "
            "switch_backend_after_claim_ceiling to continue the research on a "
            "higher-evidence backend. Available backends: "
            + ", ".join(context["allowed_backends"])
        )

    user_prompt_parts.append("")
    user_prompt_parts.append("Generate candidate research proposals as JSON.")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_prompt_parts)},
    ]


def _build_metric_trajectory_summary(recent_results: list[dict[str, Any]]) -> str:
    """Build a concise metric trajectory summary for the LLM."""
    if not recent_results:
        return ""
    lines = ["| Iter | Loss Before | Loss After | Improvement | Status |"]
    lines.append("|------|-------------|------------|-------------|--------|")
    for i, r in enumerate(recent_results):
        if not isinstance(r, dict):
            continue
        payload = r.get("result_payload") or {}
        loss_before = payload.get("reconstruction_loss_before", "N/A")
        loss_after = payload.get("reconstruction_loss_after", "N/A")
        improvement = payload.get("improvement_detected", "N/A")
        status = r.get("status", "N/A")
        fb = _fmt_metric(loss_before)
        fa = _fmt_metric(loss_after)
        fi = _fmt_metric(improvement)
        lines.append(f"| {i + 1} | {fb} | {fa} | {fi} | {status} |")
    return "\n".join(lines)


def _fmt_metric(val: Any) -> str:
    """Format a metric value for display."""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def build_mock_proposals() -> list[dict[str, Any]]:
    """Generate deterministic mock proposals for testing.

    Used by the mock LLM provider path.
    """
    return [
        {
            "proposal_id": "mock_safe_retry_001",
            "hypothesis": "Reducing optical LR to 1e-6 will stabilize geometric PSF gradients.",
            "rationale": "Previous gradients exceeded 100. Phase 23 ablation shows small_lr is the safest first step.",
            "recommended_action": "retry_with_smaller_lr",
            "backend_id": "deeplens_geolens_geometric",
            "task_type": "stable_lens_hsi_codesign",
            "objective_profile": "stable_lens_hsi_codesign",
            "experiment_spec_patch": {"optical_lr": 1e-6, "rollback_on_loss_increase": True, "max_steps": 20},
            "expected_evidence_level": "stable_native_lens_hsi_codesign",
            "expected_claim_gain": "stable_native_lens_hsi_codesign",
            "risk_level": "low",
            "proposed_claim": "Reduced optical LR enables stable native lens simulation HSI co-design.",
            "safe_wording": "Reduced optical LR enables stable native lens simulation HSI co-design [evidence ceiling: native_lens_simulation]",
        },
        {
            "proposal_id": "mock_enable_rollback_002",
            "hypothesis": "Enabling rollback on loss increase will protect against gradient-driven divergence.",
            "rationale": "Loss increased in previous run without rollback protection. Rollback is a safety mechanism that prevents regression.",
            "recommended_action": "enable_rollback",
            "backend_id": "deeplens_geolens_geometric",
            "task_type": "stable_lens_hsi_codesign",
            "experiment_spec_patch": {"rollback_on_loss_increase": True, "max_steps": 10},
            "expected_evidence_level": "rollback_protected_native_lens_hsi",
            "expected_claim_gain": "rollback_protected_native_lens_hsi",
            "risk_level": "low",
            "proposed_claim": "Rollback protection prevents native lens HSI co-design divergence.",
            "safe_wording": "Rollback protection prevents native lens HSI co-design divergence [evidence ceiling: native_lens_simulation]",
        },
        {
            "proposal_id": "mock_run_ablation_003",
            "hypothesis": "A systematic ablation study will identify which stabilizer component (small_lr, grad_clip, staged, full_stable) is most effective.",
            "rationale": "Multiple stabilization strategies exist but have not been compared head-to-head in a single experiment.",
            "recommended_action": "run_ablation",
            "backend_id": "deeplens_geolens_geometric",
            "task_type": "stable_lens_hsi_codesign",
            "experiment_spec_patch": {"max_steps": 10},
            "expected_evidence_level": "native_lens_simulation",
            "expected_claim_gain": None,
            "risk_level": "medium",
            "proposed_claim": "Ablation study identifies the most effective stabilization strategy for native lens HSI co-design.",
            "safe_wording": "Ablation study identifies the most effective stabilization strategy for native lens HSI co-design [evidence ceiling: native_lens_simulation]",
        },
    ]
