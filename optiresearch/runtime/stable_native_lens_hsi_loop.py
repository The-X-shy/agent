"""Stable native lens-simulation HSI co-design loop for Phase 23.

Addresses Phase 22 instability (optical gradient 1737 blowing up reconstruction
loss) with: small optical LR, gradient clipping, staged training
(reconstructor warmup -> joint finetune), PSF regularization, and
rollback on loss increase.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from optiresearch.adapters.geolens_waveoptics_bridge import GeoLensWaveOpticsBridge
from optiresearch.hsi.differentiable_proxy import (
    generate_torch_synthetic_hsi,
    make_measurement_from_psf_torch,
)
from optiresearch.hsi.differentiable_reconstructor import (
    DifferentiableLinearHSIReconstructor,
    TinyDifferentiableHSIReconstructor,
    hsi_reconstruction_losses,
)
from optiresearch.hsi.torch_metrics import torch_psnr, torch_sam
from optiresearch.schemas.stable_native_lens_hsi import (
    StableNativeLensHSIResult,
    StableNativeLensHSISpec,
)

RECON_CLASSES = {
    "differentiable_linear": DifferentiableLinearHSIReconstructor,
    "tiny_cnn": TinyDifferentiableHSIReconstructor,
}


def _normalize_psf_cube_for_hsi(psf_cube: torch.Tensor) -> torch.Tensor:
    if psf_cube.dim() == 4:
        psf_cube = psf_cube[:, 0, :, :]
    if psf_cube.dim() == 3 and psf_cube.shape[1] == 1:
        psf_cube = psf_cube[:, 0, :, :]
    return psf_cube


def run_stable_native_lens_hsi_codesign(
    spec: StableNativeLensHSISpec,
) -> StableNativeLensHSIResult:
    import torch

    metadata: dict[str, Any] = {}
    caveats: list[str] = []

    recon_cls = RECON_CLASSES.get(spec.reconstructor)
    if recon_cls is None:
        return _unsupported(spec, "UNSUPPORTED_RECONSTRUCTOR", caveats, metadata)

    try:
        bridge = GeoLensWaveOpticsBridge(device=spec.device)
        bridge.build_component()
    except Exception as exc:
        return _unsupported(spec, "BUILD_FAILED", caveats, metadata, str(exc))

    # Phase 33: Smoke test PSF generation to catch macOS GeoLens IndexError early
    try:
        _test_psf = bridge.psf_cube_torch(num_bands=2, ks=16)
        if _test_psf is None or _test_psf.numel() == 0:
            return _unsupported(spec, "GEOLENS_PSF_GEOMETRIC_FAILED", caveats, metadata,
                              "PSF generation returned empty tensor")
    except IndexError as exc:
        return _unsupported(spec, "GEOLENS_PSF_GEOMETRIC_FAILED_INDEXERROR",
                           caveats, metadata,
                           f"GeoLens geometric PSF path failed on this platform: {exc}")
    except Exception as exc:
        return _unsupported(spec, "GEOLENS_PSF_GEOMETRIC_FAILED", caveats, metadata, str(exc))

    try:
        opt_optimizer = bridge.get_optimizer(learning_rate=spec.optical_lr)
    except Exception as exc:
        return _unsupported(spec, "OPTIMIZER_UNAVAILABLE", caveats, metadata, str(exc))

    trainable_params = bridge.get_trainable_parameters()
    trainable_param_count = len(trainable_params)
    parameter_count = sum(p.numel() for p in trainable_params)
    if trainable_param_count == 0:
        return _unsupported(spec, "NO_NATIVE_TRAINABLE_PARAMETERS", caveats, metadata)

    try:
        reconstructor = recon_cls(bands=spec.bands).to(spec.device)
        recon_optimizer = torch.optim.Adam(reconstructor.parameters(), lr=spec.recon_lr)
    except Exception as exc:
        return _unsupported(spec, "RECONSTRUCTOR_BUILD_FAILED", caveats, metadata, str(exc))

    hsi_target = generate_torch_synthetic_hsi(
        batch=1, bands=spec.bands, height=spec.image_size, width=spec.image_size,
        device=spec.device,
    )

    opt_before = bridge.parameter_snapshot()
    metadata["optical_parameter_before"] = opt_before

    accepted = 0
    rejected = 0
    rollbacks = 0
    best_loss = float("inf")
    opt_grad_norms: list[float] = []
    recon_grad_norms: list[float] = []
    rollback_trace: list[dict[str, Any]] = []
    trust_region_activated = False
    params_with_grad = 0
    psf_requires_grad = False
    loss_requires_grad = False
    graph_connected = False

    # --- Phase 1: Reconstructor Warmup ---
    initial_psf_raw = _normalize_psf_cube_for_hsi(
        bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
    )
    # Detach PSF during warmup — optics are frozen, and GeoLens diff_float
    # doesn't support multiple backward calls without retain_graph
    initial_psf = initial_psf_raw.detach().clone()
    psf_energy_initial = float(initial_psf.sum(dim=(-2, -1)).mean().cpu().item())
    psf_width_initial = _psf_width_metric(initial_psf)

    for step in range(spec.optical_warmup_steps):
        recon_optimizer.zero_grad()
        measurement = make_measurement_from_psf_torch(hsi_target, initial_psf)
        recon = reconstructor(measurement, initial_psf)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, initial_psf, spec.loss_weights)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), spec.recon_grad_clip)
        recon_optimizer.step()

    # --- Phase 2: Joint Finetune ---
    for step in range(spec.optical_warmup_steps, spec.max_steps):
        recon_optimizer.zero_grad()
        opt_optimizer.zero_grad()

        psf = _normalize_psf_cube_for_hsi(bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size))
        measurement = make_measurement_from_psf_torch(hsi_target, psf)
        recon = reconstructor(measurement, psf)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf, spec.loss_weights)

        # PSF regularization
        psf_reg = torch.tensor(0.0, device=spec.device)
        if spec.psf_energy_reg_weight > 0:
            per_band_energy = psf.sum(dim=(-2, -1))
            psf_reg = psf_reg + spec.psf_energy_reg_weight * (per_band_energy - 1.0).abs().mean()
        if spec.psf_width_reg_weight > 0:
            psf_width = _psf_width_metric(psf)
            psf_reg = psf_reg + spec.psf_width_reg_weight * abs(psf_width - psf_width_initial)

        total_loss = losses["total_loss"] + psf_reg
        psf_requires_grad = psf_requires_grad or bool(getattr(psf, "requires_grad", False))
        loss_requires_grad = loss_requires_grad or bool(getattr(total_loss, "requires_grad", False))

        optical_params = bridge.get_trainable_parameters()
        optical_params_before_step = [p.detach().clone() for p in optical_params]
        loss_before_step = float(total_loss.detach().cpu().item())

        total_loss.backward()
        step_params_with_grad = sum(1 for p in optical_params if p.grad is not None)
        step_nonzero_grad_params = sum(
            1
            for p in optical_params
            if p.grad is not None and float(p.grad.detach().abs().max().cpu().item()) > 0.0
        )
        params_with_grad = max(params_with_grad, step_params_with_grad)
        graph_connected = graph_connected or (
            bool(psf_requires_grad) and bool(loss_requires_grad) and step_nonzero_grad_params > 0
        )

        opt_gn = torch.nn.utils.clip_grad_norm_(optical_params, spec.optical_grad_clip)
        recon_gn = torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), spec.recon_grad_clip)
        opt_gn = float(opt_gn.detach().cpu().item()) if isinstance(opt_gn, torch.Tensor) else float(opt_gn)
        recon_gn = float(recon_gn.detach().cpu().item()) if isinstance(recon_gn, torch.Tensor) else float(recon_gn)
        opt_grad_norms.append(opt_gn)
        recon_grad_norms.append(recon_gn)

        if step % spec.optical_update_interval == 0:
            opt_optimizer.step()

            # Trust region: scale down optical update if it exceeds max_optical_param_delta
            if spec.trust_region_enabled:
                max_delta = 0.0
                deltas: list[torch.Tensor] = []
                for p_before, p_after in zip(optical_params_before_step, optical_params):
                    d = (p_after - p_before).abs().max().item()
                    deltas.append(p_after.data - p_before.data)
                    if d > max_delta:
                        max_delta = d
                if max_delta > spec.max_optical_param_delta:
                    scale = spec.max_optical_param_delta / max_delta
                    for p_before, p_after, delta in zip(optical_params_before_step, optical_params, deltas):
                        p_after.data.copy_(p_before.data + delta * scale)
                    trust_region_activated = True

            recon_optimizer.step()
            metadata["optimizer_step_executed"] = True

            if spec.rollback_on_loss_increase:
                psf_after = _normalize_psf_cube_for_hsi(
                    bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
                )
                measurement_after = make_measurement_from_psf_torch(hsi_target, psf_after)
                recon_after = reconstructor(measurement_after, psf_after)
                losses_after = hsi_reconstruction_losses(recon_after, hsi_target, measurement_after, psf_after, spec.loss_weights)
                loss_after_step = float(losses_after["total_loss"].detach().cpu().item())

                tolerance = spec.accept_tolerance if spec.accept_tolerance else spec.accept_if_loss_delta_below
                loss_increased = loss_after_step > loss_before_step + tolerance

                psf_unstable = False
                psf_instability_reason = ""
                if spec.rollback_on_psf_instability:
                    psf_energy_step = float(psf_after.sum(dim=(-2, -1)).mean().cpu().item())
                    psf_width_step = _psf_width_metric(psf_after)
                    energy_delta = abs(psf_energy_step - psf_energy_initial)
                    width_delta = abs(psf_width_step - psf_width_initial)
                    if energy_delta > spec.max_psf_energy_delta:
                        psf_unstable = True
                        psf_instability_reason = f"psf_energy_delta={energy_delta:.4f}>{spec.max_psf_energy_delta}"
                    elif width_delta > spec.max_psf_width_delta:
                        psf_unstable = True
                        psf_instability_reason = f"psf_width_delta={width_delta:.4f}>{spec.max_psf_width_delta}"

                should_rollback = loss_increased or psf_unstable
                if should_rollback:
                    trace_entry: dict[str, Any] = {
                        "step": step,
                        "loss_before": loss_before_step,
                        "loss_after": loss_after_step,
                    }
                    if loss_increased:
                        trace_entry["reason"] = "loss_increase"
                    if psf_unstable:
                        trace_entry["reason"] = trace_entry.get("reason", "") + (";psf_instability" if trace_entry.get("reason") else "psf_instability")
                        trace_entry["psf_instability_detail"] = psf_instability_reason
                    for p, saved in zip(optical_params, optical_params_before_step):
                        p.data.copy_(saved.data)
                    rejected += 1
                    rollbacks += 1
                    rollback_trace.append(trace_entry)
                else:
                    accepted += 1
                    if loss_after_step < best_loss:
                        best_loss = loss_after_step
            else:
                accepted += 1
                if loss_before_step < best_loss:
                    best_loss = loss_before_step
        else:
            recon_optimizer.step()
            metadata["optimizer_step_executed"] = True

    # --- Phase 3: Final Reconstructor Adaptation (PSF detached, optics frozen) ---
    psf_final_raw = _normalize_psf_cube_for_hsi(bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size))
    psf_final_detached = psf_final_raw.detach().clone()
    for _ in range(2):
        recon_optimizer.zero_grad()
        measurement = make_measurement_from_psf_torch(hsi_target, psf_final_detached)
        recon = reconstructor(measurement, psf_final_detached)
        losses = hsi_reconstruction_losses(recon, hsi_target, measurement, psf_final_detached, spec.loss_weights)
        losses["total_loss"].backward()
        recon_optimizer.step()

    # --- Final Evaluation (PSF detached, no backward needed) ---
    psf_eval = _normalize_psf_cube_for_hsi(bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size))
    psf_eval_detached = psf_eval.detach().clone()
    measurement_final = make_measurement_from_psf_torch(hsi_target, psf_eval_detached)
    recon_final = reconstructor(measurement_final, psf_eval_detached)
    losses_final = hsi_reconstruction_losses(recon_final, hsi_target, measurement_final, psf_eval_detached, spec.loss_weights)

    # Before metrics
    measurement_before = make_measurement_from_psf_torch(hsi_target, initial_psf)
    recon_before = reconstructor(measurement_before, initial_psf)
    losses_before = hsi_reconstruction_losses(recon_before, hsi_target, measurement_before, initial_psf, spec.loss_weights)

    loss_before = float(losses_before["total_loss"].detach().cpu().item())
    loss_after = float(losses_final["total_loss"].detach().cpu().item())
    if best_loss == float("inf"):
        best_loss = loss_after

    opt_after = bridge.parameter_snapshot()
    opt_changed = _params_changed(opt_before, opt_after)

    psf_energy_after = float(initial_psf.sum(dim=(-2, -1)).mean().cpu().item())
    psf_width_after = _psf_width_metric(initial_psf)

    loss_stable_or_decreased = loss_after <= loss_before
    optics_improved = accepted > 0
    stable = loss_stable_or_decreased and optics_improved
    rollback_protected = loss_stable_or_decreased and not optics_improved and rejected > 0
    evidence = "stable_native_lens_hsi_codesign" if stable else (
        "rollback_protected_native_lens_hsi" if rollback_protected else None
    )

    if rollback_protected and not stable:
        caveats.append("All optical updates rejected; rollback protected from loss increase")

    result = StableNativeLensHSIResult(
        run_id=spec.run_id,
        status="succeeded" if (stable or rollback_protected) else "unsupported",
        candidate=spec.candidate,
        reconstructor=spec.reconstructor,
        reconstruction_loss_before=loss_before,
        reconstruction_loss_after=loss_after,
        best_reconstruction_loss=best_loss,
        accepted_update_count=accepted,
        rejected_update_count=rejected,
        rollback_count=rollbacks,
        optical_gradient_norm_max=max(opt_grad_norms) if opt_grad_norms else None,
        optical_gradient_norm_mean=sum(opt_grad_norms) / len(opt_grad_norms) if opt_grad_norms else None,
        recon_gradient_norm_max=max(recon_grad_norms) if recon_grad_norms else None,
        recon_gradient_norm_mean=sum(recon_grad_norms) / len(recon_grad_norms) if recon_grad_norms else None,
        trainable_param_count=trainable_param_count,
        parameter_count=parameter_count,
        params_with_grad=params_with_grad,
        grad_norm_max=max(opt_grad_norms) if opt_grad_norms else 0.0,
        graph_connected=graph_connected,
        psf_requires_grad=psf_requires_grad,
        loss_requires_grad=loss_requires_grad,
        optical_parameters_changed=opt_changed,
        component_parameter_changed=opt_changed,
        psf_energy_delta=psf_energy_after - psf_energy_initial,
        psf_width_delta=psf_width_after - psf_width_initial,
        mse_before=float(losses_before["mse_loss"].detach().cpu().item()),
        mse_after=float(losses_final["mse_loss"].detach().cpu().item()),
        psnr_before=float(torch_psnr(recon_before, hsi_target).detach().cpu().item()),
        psnr_after=float(torch_psnr(recon_final, hsi_target).detach().cpu().item()),
        sam_before=float(torch_sam(recon_before, hsi_target).detach().cpu().item()),
        sam_after=float(torch_sam(recon_final, hsi_target).detach().cpu().item()),
        stable_training_succeeded=stable,
        full_wave_optics=False,
        phase_to_fft_proxy_used=False,
        deeplens_native_psf_path="geolens.psf_geometric",
        evidence_level=evidence,
        optimizer_step_executed=metadata.get("optimizer_step_executed", False),
        rollback_trace=rollback_trace,
        trust_region_activated=trust_region_activated,
        caveats=caveats,
        metadata=metadata,
    )

    if spec.save_artifacts:
        out_dir = Path("workspace/stable_native_lens_hsi") / spec.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.json").write_text(json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out_dir / "report.md").write_text(
            f"# Stable Native Lens HSI CoDesign\n"
            f"- status: {result.status}\n"
            f"- stable_training_succeeded: {stable}\n"
            f"- reconstruction_loss: {loss_before:.6f} -> {loss_after:.6f}\n"
            f"- best_loss: {best_loss:.6f}\n"
            f"- accepted/rejected/rollback: {accepted}/{rejected}/{rollbacks}\n"
            f"- optical_grad max/mean: {result.optical_gradient_norm_max}/{result.optical_gradient_norm_mean}\n"
            f"- evidence: {evidence}\n",
            encoding="utf-8",
        )
        result.artifact_paths = [str((out_dir / n).relative_to(Path("workspace"))) for n in ["result.json", "report.md"]]
    return result


# ── Stabilized GeoLens HSI Loop (Phase 65) ───────────────────────────


def run_stabilized_native_geolens_hsi_loop(
    spec: "NativeGeoLensStabilitySpec",
) -> "NativeGeoLensStabilityResult":
    """Stabilized native GeoLens HSI co-design with multi-objective loss and
    multi-metric rollback acceptance.

    Builds on the existing 3-phase structure (warmup -> joint -> adaptation)
    but replaces the simple MSE+proxy loss with a full multi-objective loss
    and gates each optical update through the rollback policy.
    """
    from optiresearch.schemas.native_geolens_stability import (
        NativeGeoLensStabilityResult,
    )
    from optiresearch.runtime.native_geolens_multiobjective_loss import (
        compute_native_geolens_multiobjective_loss,
    )
    from optiresearch.runtime.native_geolens_rollback_policy import (
        AcceptanceDecision,
        RollbackPolicy,
        evaluate_native_geolens_update_acceptance,
    )

    metadata: dict[str, Any] = {}
    caveats: list[str] = []
    warnings: list[str] = []

    recon_cls = RECON_CLASSES.get(spec.reconstructor)
    if recon_cls is None:
        return _stability_unsupported(spec, "UNSUPPORTED_RECONSTRUCTOR", caveats, metadata)

    try:
        bridge = GeoLensWaveOpticsBridge(device=spec.device)
        bridge.build_component()
    except Exception as exc:
        return _stability_unsupported(spec, "BUILD_FAILED", caveats, metadata, str(exc))

    try:
        _test_psf = bridge.psf_cube_torch(num_bands=2, ks=16)
        if _test_psf is None or _test_psf.numel() == 0:
            return _stability_unsupported(spec, "GEOLENS_PSF_GEOMETRIC_FAILED",
                                          caveats, metadata, "empty PSF")
    except IndexError as exc:
        return _stability_unsupported(spec, "GEOLENS_PSF_GEOMETRIC_FAILED_INDEXERROR",
                                      caveats, metadata, str(exc))
    except Exception as exc:
        return _stability_unsupported(spec, "GEOLENS_PSF_GEOMETRIC_FAILED",
                                      caveats, metadata, str(exc))

    try:
        opt_optimizer = bridge.get_optimizer(learning_rate=spec.optical_lr)
    except Exception as exc:
        return _stability_unsupported(spec, "OPTIMIZER_UNAVAILABLE", caveats, metadata, str(exc))

    trainable_params = bridge.get_trainable_parameters()
    trainable_param_count = len(trainable_params)
    parameter_count = sum(p.numel() for p in trainable_params)
    if trainable_param_count == 0:
        return _stability_unsupported(spec, "NO_NATIVE_TRAINABLE_PARAMETERS", caveats, metadata)

    try:
        reconstructor = recon_cls(bands=spec.bands).to(spec.device)
        recon_optimizer = torch.optim.Adam(reconstructor.parameters(), lr=spec.recon_lr)
    except Exception as exc:
        return _stability_unsupported(spec, "RECONSTRUCTOR_BUILD_FAILED", caveats, metadata, str(exc))

    torch.manual_seed(spec.seed)

    hsi_target = generate_torch_synthetic_hsi(
        batch=1, bands=spec.bands, height=spec.image_size, width=spec.image_size,
        device=spec.device,
    )

    # Snapshot initial optical state for delta regularization and rollback ref
    params_initial = [p.detach().clone() for p in trainable_params]
    opt_before = bridge.parameter_snapshot()
    metadata["optical_parameter_before"] = opt_before

    # Rollback policy from spec
    policy = RollbackPolicy(
        enabled=spec.enable_rollback_policy,
        max_grad_norm=spec.rollback_max_grad_norm,
        sam_tolerance=spec.rollback_sam_tolerance,
        allow_tradeoff=spec.rollback_allow_tradeoff,
    )

    # Multi-objective loss weights
    loss_weights = {
        "reconstruction_mse": 1.0,
        "reconstruction_mae": 0.0,
        "spectral_angle": spec.spectral_angle_weight,
        "measurement_consistency": 0.1,
        "psf_energy": 0.1,
        "psf_centroid": 0.1,
        "psf_width": 0.05,
        "optical_param_delta": 1e-4,
        "smoothness": 0.0,
    }

    accepted = 0
    rejected = 0
    rollbacks = 0
    best_stability_score = float("-inf")
    opt_grad_norms: list[float] = []
    params_with_grad = 0
    psf_requires_grad = False
    loss_requires_grad = False
    graph_connected = False
    rollback_reasons: list[str] = []

    # --- Phase 1: Reconstructor Warmup ---
    initial_psf_raw = _normalize_psf_cube_for_hsi(
        bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
    )
    initial_psf = initial_psf_raw.detach().clone()
    psf_energy_initial = float(initial_psf.sum(dim=(-2, -1)).mean().cpu().item())
    psf_width_initial = _psf_width_metric(initial_psf)

    for _step in range(spec.optical_warmup_steps):
        recon_optimizer.zero_grad()
        measurement = make_measurement_from_psf_torch(hsi_target, initial_psf)
        recon = reconstructor(measurement, initial_psf)
        losses = compute_native_geolens_multiobjective_loss(
            recon, hsi_target, measurement, initial_psf, initial_psf,
            trainable_params, params_initial,
            weights=loss_weights,
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), spec.recon_grad_clip)
        recon_optimizer.step()

    # --- Phase 2: Joint Finetune ---
    for step in range(spec.optical_warmup_steps, spec.max_steps):
        recon_optimizer.zero_grad()
        opt_optimizer.zero_grad()

        psf = _normalize_psf_cube_for_hsi(
            bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
        )
        measurement = make_measurement_from_psf_torch(hsi_target, psf)
        recon = reconstructor(measurement, psf)

        losses = compute_native_geolens_multiobjective_loss(
            recon, hsi_target, measurement, psf, initial_psf,
            trainable_params, params_initial,
            weights=loss_weights,
        )

        psf_requires_grad = psf_requires_grad or bool(getattr(psf, "requires_grad", False))
        loss_requires_grad = loss_requires_grad or bool(getattr(losses["total_loss"], "requires_grad", False))

        optical_params = trainable_params
        optical_params_before_step = [p.detach().clone() for p in optical_params]

        # Metric snapshots before update
        mse_before_step = float(losses["reconstruction_mse"].detach().cpu().item())
        psnr_before_step = float(torch_psnr(recon, hsi_target).detach().cpu().item())
        sam_before_step = float(torch_sam(recon, hsi_target).detach().cpu().item())
        psf_centroid_before = _psf_centroid_stats(psf.detach())
        psf_width_before = _psf_width_metric(psf)
        psf_energy_before = float(psf.sum(dim=(-2, -1)).mean().cpu().item())

        losses["total_loss"].backward()

        step_params_with_grad = sum(1 for p in optical_params if p.grad is not None)
        step_nonzero_grad_params = sum(
            1 for p in optical_params
            if p.grad is not None and float(p.grad.detach().abs().max().cpu().item()) > 0.0
        )
        params_with_grad = max(params_with_grad, step_params_with_grad)
        graph_connected = graph_connected or (
            bool(psf_requires_grad) and bool(loss_requires_grad) and step_nonzero_grad_params > 0
        )

        opt_gn = torch.nn.utils.clip_grad_norm_(optical_params, spec.optical_grad_clip)
        opt_gn = float(opt_gn.detach().cpu().item()) if isinstance(opt_gn, torch.Tensor) else float(opt_gn)
        opt_grad_norms.append(opt_gn)

        if step % spec.optical_update_interval == 0:
            opt_optimizer.step()

            # Post-update metrics for rollback evaluation
            psf_after = _normalize_psf_cube_for_hsi(
                bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
            )
            measurement_after = make_measurement_from_psf_torch(hsi_target, psf_after)
            recon_after = reconstructor(measurement_after, psf_after)
            losses_after = compute_native_geolens_multiobjective_loss(
                recon_after, hsi_target, measurement_after, psf_after, initial_psf,
                trainable_params, params_initial,
                weights=loss_weights,
            )

            mse_after_step = float(losses_after["reconstruction_mse"].detach().cpu().item())
            psnr_after_step = float(torch_psnr(recon_after, hsi_target).detach().cpu().item())
            sam_after_step = float(torch_sam(recon_after, hsi_target).detach().cpu().item())
            psf_centroid_after = _psf_centroid_stats(psf_after.detach())
            psf_width_after = _psf_width_metric(psf_after)
            psf_energy_after = float(psf_after.sum(dim=(-2, -1)).mean().cpu().item())

            decision = evaluate_native_geolens_update_acceptance(
                metrics_before={"mse": mse_before_step, "psnr": psnr_before_step,
                                "sam": sam_before_step},
                metrics_after={"mse": mse_after_step, "psnr": psnr_after_step,
                               "sam": sam_after_step},
                grad_norm_max=opt_gn,
                psf_stats_before={"centroid_y": psf_centroid_before.get("centroid_y", 0.0),
                                  "centroid_x": psf_centroid_before.get("centroid_x", 0.0),
                                  "width": psf_width_before},
                psf_stats_after={"centroid_y": psf_centroid_after.get("centroid_y", 0.0),
                                 "centroid_x": psf_centroid_after.get("centroid_x", 0.0),
                                 "width": psf_width_after},
                policy=policy,
            )

            if not decision.accepted:
                for p, saved in zip(optical_params, optical_params_before_step):
                    p.data.copy_(saved.data)
                rejected += 1
                rollbacks += 1
                rollback_reasons.extend(decision.reasons)
            else:
                accepted += 1
                if decision.stability_score_after > best_stability_score:
                    best_stability_score = decision.stability_score_after

            recon_optimizer.step()
            metadata["optimizer_step_executed"] = True
        else:
            recon_optimizer.step()
            metadata["optimizer_step_executed"] = True

    # --- Phase 3: Final Reconstructor Adaptation ---
    psf_final_raw = _normalize_psf_cube_for_hsi(
        bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
    )
    psf_final_detached = psf_final_raw.detach().clone()
    for _ in range(2):
        recon_optimizer.zero_grad()
        measurement = make_measurement_from_psf_torch(hsi_target, psf_final_detached)
        recon = reconstructor(measurement, psf_final_detached)
        losses = compute_native_geolens_multiobjective_loss(
            recon, hsi_target, measurement, psf_final_detached, initial_psf,
            trainable_params, params_initial,
            weights=loss_weights,
        )
        losses["total_loss"].backward()
        recon_optimizer.step()

    # --- Final Evaluation ---
    psf_eval = _normalize_psf_cube_for_hsi(
        bridge.psf_cube_torch(num_bands=spec.bands, ks=spec.psf_size)
    )
    psf_eval_detached = psf_eval.detach().clone()
    measurement_final = make_measurement_from_psf_torch(hsi_target, psf_eval_detached)
    recon_final = reconstructor(measurement_final, psf_eval_detached)
    losses_final = compute_native_geolens_multiobjective_loss(
        recon_final, hsi_target, measurement_final, psf_eval_detached, initial_psf,
        trainable_params, params_initial,
        weights=loss_weights,
    )

    measurement_before = make_measurement_from_psf_torch(hsi_target, initial_psf)
    recon_before = reconstructor(measurement_before, initial_psf)
    losses_before = compute_native_geolens_multiobjective_loss(
        recon_before, hsi_target, measurement_before, initial_psf, initial_psf,
        trainable_params, params_initial,
        weights=loss_weights,
    )

    mse_before = float(losses_before["reconstruction_mse"].detach().cpu().item())
    mse_after = float(losses_final["reconstruction_mse"].detach().cpu().item())
    psnr_before_val = float(torch_psnr(recon_before, hsi_target).detach().cpu().item())
    psnr_after_val = float(torch_psnr(recon_final, hsi_target).detach().cpu().item())
    sam_before_val = float(torch_sam(recon_before, hsi_target).detach().cpu().item())
    sam_after_val = float(torch_sam(recon_final, hsi_target).detach().cpu().item())

    psf_energy_after = float(psf_eval.sum(dim=(-2, -1)).mean().cpu().item())
    psf_centroid_after = _psf_centroid_stats(psf_eval.detach())
    psf_centroid_before = _psf_centroid_stats(initial_psf.detach())
    centroid_shift = abs(psf_centroid_after.get("centroid_y", 0.0) - psf_centroid_before.get("centroid_y", 0.0))
    centroid_shift += abs(psf_centroid_after.get("centroid_x", 0.0) - psf_centroid_before.get("centroid_x", 0.0))
    psf_width_after = _psf_width_metric(psf_eval)
    width_shift = abs(psf_width_after - psf_width_initial)

    opt_after = bridge.parameter_snapshot()
    opt_changed = _params_changed(opt_before, opt_after)

    grad_norm_max = max(opt_grad_norms) if opt_grad_norms else 0.0
    grad_norm_mean = sum(opt_grad_norms) / len(opt_grad_norms) if opt_grad_norms else 0.0

    # Stability score using final metrics
    stability_score = -mse_after + 0.1 * psnr_after_val - sam_after_val

    # Tradeoff summary
    metric_tradeoff_summary = ""
    sam_worsened = sam_after_val > sam_before_val
    mse_improved = mse_after < mse_before
    if sam_worsened and mse_improved:
        metric_tradeoff_summary = (
            f"MSE improved ({mse_before:.4f} -> {mse_after:.4f}) "
            f"but SAM degraded ({sam_before_val:.4f} -> {sam_after_val:.4f}) — "
            f"spectral_angle_weight={spec.spectral_angle_weight} may need increase"
        )
        warnings.append(metric_tradeoff_summary)
        caveats.append("Spectral angle worsened despite MSE improvement; multi-metric stability not yet demonstrated")
    elif not sam_worsened and mse_improved:
        metric_tradeoff_summary = "All metrics improved — stable multi-objective optimization achieved"

    loss_stable_or_decreased = mse_after <= mse_before
    optics_improved = accepted > 0
    stable = loss_stable_or_decreased and optics_improved
    rollback_protected = loss_stable_or_decreased and not optics_improved and rejected > 0
    evidence = "stable_native_lens_hsi_codesign" if stable else (
        "rollback_protected_native_lens_hsi" if rollback_protected else None
    )

    if sam_worsened and stable:
        evidence = "stable_native_lens_hsi_codesign"
        caveats.append("Stable MSE optimization achieved but spectral angle tradeoff observed")

    result = NativeGeoLensStabilityResult(
        run_id=spec.run_id,
        status="succeeded" if (stable or rollback_protected) else "unsupported",
        candidate=spec.candidate,
        reconstructor=spec.reconstructor,
        reconstruction_loss_before=mse_before,
        reconstruction_loss_after=mse_after,
        best_reconstruction_loss=min(mse_before, mse_after),
        accepted_update_count=accepted,
        rejected_update_count=rejected,
        rollback_count=rollbacks,
        optical_gradient_norm_max=grad_norm_max if grad_norm_max > 0 else None,
        optical_gradient_norm_mean=grad_norm_mean if grad_norm_mean > 0 else None,
        trainable_param_count=trainable_param_count,
        parameter_count=parameter_count,
        params_with_grad=params_with_grad,
        grad_norm_max=grad_norm_max if grad_norm_max > 0 else None,
        graph_connected=graph_connected,
        psf_requires_grad=psf_requires_grad,
        loss_requires_grad=loss_requires_grad,
        optical_parameters_changed=opt_changed,
        component_parameter_changed=opt_changed,
        psf_energy_delta=psf_energy_after - psf_energy_initial,
        psf_width_delta=width_shift,
        mse_before=mse_before,
        mse_after=mse_after,
        psnr_before=psnr_before_val,
        psnr_after=psnr_after_val,
        sam_before=sam_before_val,
        sam_after=sam_after_val,
        stable_training_succeeded=stable,
        full_wave_optics=False,
        phase_to_fft_proxy_used=False,
        deeplens_native_psf_path="geolens.psf_geometric",
        evidence_level=evidence,
        optimizer_step_executed=metadata.get("optimizer_step_executed", False),
        rollback_trace=[],
        trust_region_activated=False,
        caveats=caveats,
        metadata=metadata,
        spectral_angle_weight=spec.spectral_angle_weight,
        seed=spec.seed,
        optimizer_name=spec.optimizer_name,
        rollback_policy_enabled=spec.enable_rollback_policy,
        rollback_reasons=rollback_reasons,
        grad_norm_mean=grad_norm_mean if grad_norm_mean > 0 else None,
        psf_energy_before=psf_energy_initial,
        psf_energy_after=psf_energy_after,
        psf_centroid_shift=centroid_shift,
        psf_width_shift=width_shift,
        stability_score=stability_score,
        loss_terms_final={k: float(v.detach().cpu().item()) for k, v in losses_final.items() if k != "total_loss" and v.numel() > 0},
        metric_tradeoff_summary=metric_tradeoff_summary,
        warnings=warnings,
    )

    if spec.save_artifacts:
        out_dir = Path("workspace/stabilized_native_geolens_hsi") / spec.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.json").write_text(json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (out_dir / "report.md").write_text(
            f"# Stabilized Native GeoLens HSI CoDesign\n"
            f"- status: {result.status}\n"
            f"- stable_training_succeeded: {stable}\n"
            f"- mse: {mse_before:.6f} -> {mse_after:.6f}\n"
            f"- psnr: {psnr_before_val:.4f} -> {psnr_after_val:.4f}\n"
            f"- sam: {sam_before_val:.4f} -> {sam_after_val:.4f}\n"
            f"- spectral_angle_weight: {spec.spectral_angle_weight}\n"
            f"- accepted/rejected/rollbacks: {accepted}/{rejected}/{rollbacks}\n"
            f"- optical_grad max/mean: {grad_norm_max}/{grad_norm_mean}\n"
            f"- stability_score: {stability_score:.3f}\n"
            f"- rollback_reasons: {rollback_reasons}\n"
            f"- evidence: {evidence}\n"
            f"- tradeoff: {metric_tradeoff_summary}\n",
            encoding="utf-8",
        )
        result.artifact_paths = [str((out_dir / n).relative_to(Path("workspace"))) for n in ["result.json", "report.md"]]
    return result


def _psf_centroid_stats(psf: torch.Tensor) -> dict[str, float]:
    """Compute PSF centroid y/x for each band, return mean values."""
    B, H, W = psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=psf.dtype, device=psf.device),
        torch.arange(W, dtype=psf.dtype, device=psf.device),
        indexing="ij",
    )
    mass = psf.sum(dim=(-2, -1)) + 1e-8
    cy = (psf * yy).sum(dim=(-2, -1)) / mass
    cx = (psf * xx).sum(dim=(-2, -1)) / mass
    return {"centroid_y": float(cy.mean().cpu().item()),
            "centroid_x": float(cx.mean().cpu().item())}


def _stability_unsupported(spec, code, cav, meta, msg=""):
    from optiresearch.schemas.native_geolens_stability import (
        NativeGeoLensStabilityResult,
    )
    return NativeGeoLensStabilityResult(
        run_id=spec.run_id, status="unsupported",
        candidate=spec.candidate, reconstructor=spec.reconstructor,
        error_code=code, error_message=msg or code,
        caveats=[*cav, msg or code], metadata=meta,
    )


def _psf_width_metric(psf: torch.Tensor) -> float:
    _, k, _ = psf.shape
    half = k // 2
    yy, xx = torch.meshgrid(
        torch.arange(k, dtype=psf.dtype, device=psf.device) - half,
        torch.arange(k, dtype=psf.dtype, device=psf.device) - half,
        indexing="ij",
    )
    r2 = yy * yy + xx * xx
    energy = psf.sum(dim=(-2, -1)) + 1e-8
    moment2 = (psf * r2).sum(dim=(-2, -1)) / energy
    return float(moment2.mean().detach().cpu().item())


def _params_changed(before: dict, after: dict) -> bool:
    for key, bv in before.items():
        av = after.get(key)
        if isinstance(bv, dict):
            if abs(float((av or {}).get("norm", 0)) - float((bv or {}).get("norm", 0))) > 1e-12:
                return True
        elif av is not None and abs(float(av) - float(bv)) > 1e-12:
            return True
    return False


def _unsupported(spec, code, cav, meta, msg=""):
    return StableNativeLensHSIResult(
        run_id=spec.run_id, status="unsupported",
        candidate=spec.candidate, reconstructor=spec.reconstructor,
        error_code=code, error_message=msg or code,
        caveats=[*cav, msg or code], metadata=meta,
    )
