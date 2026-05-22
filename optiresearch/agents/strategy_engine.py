"""Strategy Engine — automatic next-step recommendation from experiment results.

Takes the latest experiment result, backend registry, claim boundary,
memory rules, and failure diagnostics, then outputs a structured
StrategyRecommendation with concrete CLI commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

RiskLevel = Literal["low", "medium", "high"]


@dataclass
class StrategyRecommendation:
    """Output of the strategy engine."""

    recommended_action: str
    rationale: str
    expected_claim_gain: Optional[str] = None
    risk_level: RiskLevel = "low"
    required_evidence: list[str] = field(default_factory=list)
    proposed_cli_commands: list[str] = field(default_factory=list)
    proposed_experiment_spec: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StrategyEngine:
    """Rule-based strategy engine for differentiable optics experiments.

    Inputs are pure dicts — no coupling to specific schema versions.
    """

    def __init__(self):
        self._rules: list[dict[str, Any]] = self._build_builtin_rules()

    def recommend(
        self,
        latest_result: dict[str, Any],
        backend_id: str,
        claim_boundary: Optional[dict[str, Any]] = None,
        memory_rules: Optional[list[dict[str, Any]]] = None,
        failure_diagnostics: Optional[dict[str, Any]] = None,
    ) -> StrategyRecommendation:
        """Analyse result and recommend next action.

        Args:
            latest_result: Dict with metrics from the latest experiment run.
                Expected keys (all optional): optical_gradient_norm,
                rollback_count, total_steps, loss_before, loss_after,
                reconstruction_loss_before, reconstruction_loss_after,
                accepted_update_count, rejected_update_count,
                optical_parameters_changed, optimizer_step_executed,
                claim_downgraded, downgraded_to, stable_training_succeeded,
                psf_energy_delta, optical_gradient_norm_max.
            backend_id: Which optical backend was used.
            claim_boundary: Optional claim whitelist/blacklist.
            memory_rules: Optional research memory rules.
            failure_diagnostics: Optional failure diagnostic data.

        Returns:
            StrategyRecommendation with concrete next steps.
        """
        recommendations: list[StrategyRecommendation] = []

        for rule in self._rules:
            condition: Callable[[dict[str, Any]], bool] = rule["condition"]
            try:
                if condition(latest_result):
                    rec = self._build_recommendation(rule, latest_result, backend_id)
                    recommendations.append(rec)
            except Exception:
                continue

        if not recommendations:
            return self._default_recommendation(latest_result, backend_id)

        # Return the highest-priority recommendation (first match)
        return recommendations[0]

    def _build_builtin_rules(self) -> list[dict[str, Any]]:
        """8 built-in rules ordered by priority (highest first)."""
        return [
            {
                "id": "large_grad_small_lr",
                "condition": lambda r: r.get("optical_gradient_norm", 0) > 100,
                "action": "retry_with_smaller_lr",
                "rationale": (
                    "Optical gradient norm ({grad_norm:.1f}) is large. "
                    "Reduce optical_lr by 100x and re-run with rollback enabled."
                ),
                "expected_claim_gain": "stable_native_lens_hsi_codesign",
                "risk_level": "low",
                "proposed_commands": [
                    "python -m optiresearch.cli run-stable-native-lens-hsi-codesign "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn "
                    "--optical-lr {new_lr} --rollback-on-loss-increase"
                ],
                "required_evidence": [
                    "Show that reduced LR leads to accepted optical updates",
                    "Show reconstruction loss decreases after accepted update",
                ],
            },
            {
                "id": "rollback_count_freeze",
                "condition": lambda r: (
                    r.get("rollback_count", 0)
                    / max(r.get("total_steps", r.get("max_steps", 1)), 1)
                    > 0.5
                ),
                "action": "switch_backend",
                "rationale": (
                    "More than 50% of optical updates were rejected by rollback "
                    "({rollback_count}/{total_steps}). Consider freezing optics "
                    "or switching to a simpler optical objective."
                ),
                "expected_claim_gain": None,
                "risk_level": "medium",
                "proposed_commands": [
                    "python -m optiresearch.cli run-stable-native-lens-hsi-ablation "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn"
                ],
                "required_evidence": [
                    "Ablation study to identify which stabilizer component helps most",
                ],
            },
            {
                "id": "zero_grad_all_params",
                "condition": lambda r: (
                    r.get("optical_gradient_norm", 1) == 0
                    and r.get("optimizer_step_executed", False)
                ),
                "action": "run_ablation",
                "rationale": (
                    "Zero optical gradient but optimizer step executed — "
                    "possible autograd break. Run autograd auditor."
                ),
                "expected_claim_gain": None,
                "risk_level": "high",
                "proposed_commands": [
                    "python -m optiresearch.cli audit-autograd-graph"
                ],
                "required_evidence": [
                    "Autograd audit report showing clean gradient flow",
                ],
            },
            {
                "id": "loss_increase_no_rollback",
                "condition": lambda r: (
                    r.get("loss_after", 0) > r.get("loss_before", 0)
                    and not r.get("rollback_protected", False)
                    and r.get("rollback_count", 0) == 0
                ),
                "action": "enable_rollback",
                "rationale": (
                    "Loss increased from {loss_before:.4f} to {loss_after:.4f} "
                    "without rollback protection. Enable rollback."
                ),
                "expected_claim_gain": "rollback_protected_native_lens_hsi",
                "risk_level": "low",
                "proposed_commands": [
                    "python -m optiresearch.cli run-stable-native-lens-hsi-codesign "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn "
                    "--rollback-on-loss-increase"
                ],
                "required_evidence": [
                    "Show loss does not increase after enabling rollback",
                ],
            },
            {
                "id": "claim_downgrade_required",
                "condition": lambda r: r.get("claim_downgraded", False),
                "action": "downgrade_claim",
                "rationale": (
                    "Claim was downgraded from {downgraded_from} to {downgraded_to}. "
                    "Update claim wording to match evidence ceiling."
                ),
                "expected_claim_gain": None,
                "risk_level": "low",
                "proposed_commands": [
                    "python -m optiresearch.cli check-claim "
                    '--claim-text "<updated claim>" '
                    "--backend-id {backend_id}"
                ],
                "required_evidence": [],
            },
            {
                "id": "recon_loss_increase",
                "condition": lambda r: (
                    r.get("reconstruction_loss_after", 0)
                    > r.get("reconstruction_loss_before", 0)
                ),
                "action": "retry_with_smaller_lr",
                "rationale": (
                    "Reconstruction loss increased. Increase reconstructor warmup "
                    "steps before joint finetune."
                ),
                "expected_claim_gain": None,
                "risk_level": "medium",
                "proposed_commands": [
                    "python -m optiresearch.cli run-stable-native-lens-hsi-codesign "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn "
                    "--max-steps 20"
                ],
                "required_evidence": [
                    "Show reconstruction loss decreases with more warmup steps",
                ],
            },
            {
                "id": "gradient_clip_required",
                "condition": lambda r: r.get("optical_gradient_norm_max", 0) > 10,
                "action": "retry_with_smaller_lr",
                "rationale": (
                    "Max optical gradient norm ({grad_max:.1f}) exceeds 10. "
                    "Apply gradient clipping at 1.0 and reduce LR."
                ),
                "expected_claim_gain": "stable_native_lens_hsi_codesign",
                "risk_level": "medium",
                "proposed_commands": [
                    "python -m optiresearch.cli run-stable-native-lens-hsi-codesign "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn "
                    "--optical-lr 1e-6 --optical-grad-clip 1.0"
                ],
                "required_evidence": [
                    "Show max gradient norm stays below 10 after clipping",
                ],
            },
            {
                "id": "psf_reg_needed",
                "condition": lambda r: abs(r.get("psf_energy_delta", 0)) > 0.5,
                "action": "run_ablation",
                "rationale": (
                    "PSF energy changed significantly (delta={psf_energy_delta:.2f}). "
                    "Consider increasing PSF energy preservation weight."
                ),
                "expected_claim_gain": None,
                "risk_level": "low",
                "proposed_commands": [
                    "python -m optiresearch.cli run-stable-native-lens-hsi-ablation "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn"
                ],
                "required_evidence": [
                    "Show PSF energy is preserved within 10% of initial value",
                ],
            },
        ]

    def _build_recommendation(
        self,
        rule: dict[str, Any],
        result: dict[str, Any],
        backend_id: str,
    ) -> StrategyRecommendation:
        """Format a rule into a concrete recommendation with filled-in values."""
        rationale = rule["rationale"].format(
            grad_norm=result.get("optical_gradient_norm", 0),
            rollback_count=result.get("rollback_count", 0),
            total_steps=result.get("total_steps", result.get("max_steps", 1)),
            loss_before=result.get("loss_before", result.get("reconstruction_loss_before", 0)),
            loss_after=result.get("loss_after", result.get("reconstruction_loss_after", 0)),
            downgraded_from=result.get("downgraded_from", "unknown"),
            downgraded_to=result.get("downgraded_to", "unknown"),
            grad_max=result.get("optical_gradient_norm_max", 0),
            psf_energy_delta=result.get("psf_energy_delta", 0),
            backend_id=backend_id,
        )

        new_lr = max(result.get("optical_lr", 1e-3) * 0.01, 1e-8)
        commands = [
            cmd.format(new_lr=new_lr, backend_id=backend_id)
            for cmd in rule["proposed_commands"]
        ]

        return StrategyRecommendation(
            recommended_action=rule["action"],
            rationale=rationale,
            expected_claim_gain=rule.get("expected_claim_gain"),
            risk_level=rule.get("risk_level", "low"),
            required_evidence=rule.get("required_evidence", []),
            proposed_cli_commands=commands,
            metadata={"rule_id": rule["id"]},
        )

    def _default_recommendation(
        self,
        result: dict[str, Any],
        backend_id: str,
    ) -> StrategyRecommendation:
        """Fallback recommendation when no specific rule fires."""
        stable = result.get("stable_training_succeeded", None)
        loss_before = result.get("reconstruction_loss_before", result.get("loss_before"))
        loss_after = result.get("reconstruction_loss_after", result.get("loss_after"))

        if stable is True:
            return StrategyRecommendation(
                recommended_action="run_remote_validation",
                rationale=(
                    "Local stable training succeeded. "
                    "Run remote WSL validation to confirm robustness."
                ),
                expected_claim_gain="stable_native_lens_hsi_codesign (remote-validated)",
                risk_level="low",
                required_evidence=["Remote validation on WSL DeepLens instance"],
                proposed_cli_commands=[
                    "python -m optiresearch.cli run-remote-stable-native-lens-hsi-codesign "
                    "--candidate GeoLensCooke --reconstructor tiny_cnn "
                    "--worker-id wslbox"
                ],
            )

        if loss_before is not None and loss_after is not None and loss_after <= loss_before:
            return StrategyRecommendation(
                recommended_action="run_remote_validation",
                rationale=(
                    f"Loss decreased from {loss_before:.4f} to {loss_after:.4f}. "
                    "Validate remotely to confirm environment-independent result."
                ),
                expected_claim_gain="native_lens_simulation",
                risk_level="low",
                required_evidence=["Remote validation on WSL DeepLens instance"],
                proposed_cli_commands=[
                    "python -m optiresearch.cli run-experiment-v2 "
                    f"--backend-id {backend_id} "
                    "--task-type stable_lens_hsi_codesign "
                    "--execution-target remote"
                ],
            )

        return StrategyRecommendation(
            recommended_action="stop_and_report",
            rationale=(
                "No specific improvement path identified from current metrics. "
                "Review experiment configuration and consider running an ablation study."
            ),
            risk_level="medium",
            required_evidence=["Review experiment configuration", "Consider ablation study"],
            proposed_cli_commands=[
                "python -m optiresearch.cli export-agent-system-report"
            ],
        )
