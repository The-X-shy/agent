"""Lens-file native optimization probe for DeepLens Phase 19B."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from optiresearch.memory.schemas import make_deterministic_id


LENS_MODULES = {
    "GeoLens": "deeplens.geolens",
    "HybridLens": "deeplens.hybridlens",
    "DiffractiveLens": "deeplens.diffraclens",
}


def run_lensfile_optimization_probe(
    lens_class: str,
    repo_path: str | Path | None = None,
    max_files: int = 5,
    max_steps: int = 2,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    save_artifacts: bool = True,
    remote_job_id: str | None = None,
) -> dict[str, Any]:
    probe_id = make_deterministic_id("lensfile_probe", lens_class, max_files, max_steps, time.time())
    output_dir = Path("workspace/native_optimization") / f"lensfile_probe_{probe_id}"
    result: dict[str, Any] = {
        "probe_id": probe_id,
        "status": "unsupported",
        "lens_class": lens_class,
        "max_files": max_files,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "device": device,
        "repo_path": str(repo_path) if repo_path else None,
        "successful_file": None,
        "attempts": [],
        "differentiable": False,
        "gradient_norm": None,
        "parameters_changed": False,
        "loss_before": None,
        "loss_after": None,
        "output_dir": str(output_dir),
        "error_code": None,
        "error_message": None,
    }

    lens_cls, module_path, import_error = _import_lens_class(lens_class)
    result["module_path"] = module_path
    if lens_cls is None:
        result.update({"error_code": "IMPORT_FAILED", "error_message": import_error or "lens class not importable"})
        if save_artifacts:
            _save_result(output_dir, result)
        return result

    repo = _resolve_repo_path(repo_path)
    files = _find_lens_files(repo, lens_class, max_files=max_files) if repo is not None else []
    result["repo_path"] = str(repo) if repo else None
    if not files:
        result.update({"error_code": "NO_LENS_FILES_FOUND", "error_message": "No candidate lens files found"})
        if save_artifacts:
            _save_result(output_dir, result)
        return result

    for lens_file in files:
        attempt = _probe_file(
            lens_cls=lens_cls,
            lens_class=lens_class,
            lens_file=lens_file,
            max_steps=max_steps,
            learning_rate=learning_rate,
            device=device,
        )
        result["attempts"].append(attempt)
        if attempt["status"] == "succeeded":
            result.update(
                {
                    "status": "succeeded",
                    "successful_file": str(lens_file),
                    "differentiable": attempt["differentiable"],
                    "gradient_norm": attempt["gradient_norm"],
                    "parameters_changed": attempt["parameters_changed"],
                    "loss_before": attempt["loss_before"],
                    "loss_after": attempt["loss_after"],
                    "parameter_before": attempt["parameter_before"],
                    "parameter_after": attempt["parameter_after"],
                    "optimizer_step_executed": attempt["optimizer_step_executed"],
                    "surface_count": attempt.get("surface_count"),
                    "error_code": None,
                    "error_message": None,
                }
            )
            break

    if result["status"] != "succeeded":
        result["error_code"] = "NO_LENSFILE_NATIVE_OPTIMIZATION_PATH_SUCCEEDED"
        result["error_message"] = "Candidate lens files did not complete backward + optimizer.step with parameter change"

    if save_artifacts:
        _save_result(output_dir, result)
    if remote_job_id and save_artifacts:
        from optiresearch.runtime.remote_jobs import export_remote_job_outputs

        export_remote_job_outputs(
            remote_job_id,
            "deeplens_lensfile_optimization_probe",
            result,
            [output_dir],
            _metrics_summary(result),
        )
    return result


def _probe_file(
    lens_cls: type,
    lens_class: str,
    lens_file: Path,
    max_steps: int,
    learning_rate: float,
    device: str,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "lens_file": str(lens_file),
        "status": "failed",
        "error_code": None,
        "error_message": None,
        "differentiable": False,
        "gradient_norm": None,
        "parameters_changed": False,
        "loss_before": None,
        "loss_after": None,
        "optimizer_step_executed": False,
    }
    try:
        lens = _instantiate_lens(lens_cls, lens_file, device)
        attempt["surface_count"] = len(getattr(lens, "surfaces", []) or getattr(getattr(lens, "geolens", None), "surfaces", []) or [])
        optimizer = _get_optimizer(lens, lens_class, learning_rate)
        parameter_before = _parameter_snapshot(optimizer)
        target = None
        max_gradient_norm = 0.0
        for step in range(max_steps):
            optimizer.zero_grad()
            output = _generate_lens_output(lens, lens_class, device)
            loss, target = _loss(output, target)
            if step == 0:
                attempt["loss_before"] = float(loss.detach().cpu().item())
            loss.backward()
            step_gradient_norm = _gradient_norm(optimizer)
            max_gradient_norm = max(max_gradient_norm, step_gradient_norm)
            optimizer.step()
            attempt["optimizer_step_executed"] = True
        output_after = _generate_lens_output(lens, lens_class, device)
        loss_after, _ = _loss(output_after, target)
        parameter_after = _parameter_snapshot(optimizer)
        attempt["loss_after"] = float(loss_after.detach().cpu().item())
        attempt["gradient_norm"] = max_gradient_norm
        attempt["parameter_before"] = parameter_before
        attempt["parameter_after"] = parameter_after
        attempt["parameters_changed"] = _parameters_changed(parameter_before, parameter_after)
        attempt["differentiable"] = bool(
            attempt["gradient_norm"] is not None
            and attempt["gradient_norm"] > 0
            and attempt["parameters_changed"]
            and attempt["optimizer_step_executed"]
        )
        attempt["status"] = "succeeded" if attempt["differentiable"] else "unsupported"
        if not attempt["differentiable"]:
            attempt["error_code"] = "NO_PARAMETER_CHANGE"
            attempt["error_message"] = "backward/optimizer.step did not change parameters"
    except Exception as exc:
        attempt["error_code"] = "LENSFILE_PROBE_FAILED"
        attempt["error_message"] = str(exc)
    return attempt


def _import_lens_class(lens_class: str) -> tuple[Any | None, str | None, str | None]:
    repo_path = os.getenv("DEEPLENS_REPO_PATH")
    if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    module_path = LENS_MODULES.get(lens_class)
    if module_path is None:
        return None, None, f"Unsupported lens class: {lens_class}"
    try:
        module = importlib.import_module(module_path)
        return getattr(module, lens_class), module_path, None
    except Exception as exc:
        return None, module_path, str(exc)


def _instantiate_lens(lens_cls: type, lens_file: Path, device: str) -> Any:
    attempts = [
        ((str(lens_file),), {"device": device}),
        ((str(lens_file),), {}),
    ]
    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            return lens_cls(*args, **kwargs)
        except TypeError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError("; ".join(errors) or f"failed to instantiate {lens_cls}")


def _get_optimizer(lens: Any, lens_class: str, learning_rate: float) -> Any:
    import torch

    if callable(getattr(lens, "get_optimizer", None)):
        attempts = [
            ((), {"lr": learning_rate}),
            ((learning_rate,), {}),
            ((), {"lrs": [learning_rate, learning_rate, learning_rate, learning_rate]}),
            ((), {}),
        ]
        for args, kwargs in attempts:
            try:
                optimizer = lens.get_optimizer(*args, **kwargs)
                if optimizer is not None:
                    return optimizer
            except TypeError:
                continue
            except Exception:
                continue
    params = []
    for surface in _surfaces(lens):
        if callable(getattr(surface, "get_optimizer_params", None)):
            try:
                params.extend(surface.get_optimizer_params(lr=learning_rate))
            except TypeError:
                try:
                    params.extend(surface.get_optimizer_params(lrs=[learning_rate, learning_rate]))
                except TypeError:
                    continue
    if params:
        return torch.optim.Adam(params)
    raise RuntimeError(f"{lens_class} did not expose optimizer parameters")


def _surfaces(lens: Any) -> list[Any]:
    surfaces = list(getattr(lens, "surfaces", []) or [])
    geolens = getattr(lens, "geolens", None)
    if geolens is not None:
        surfaces.extend(list(getattr(geolens, "surfaces", []) or []))
    doe = getattr(lens, "doe", None)
    if doe is not None:
        surfaces.append(doe)
    return surfaces


def _generate_lens_output(lens: Any, lens_class: str, device: str) -> Any:
    import torch

    errors: list[str] = []
    if callable(getattr(lens, "psf", None)):
        patterns = [
            ((), {}),
            ((), {"points": torch.zeros(1, 3, device=device), "ks": 16}),
            ((), {"point": [0.0, 0.0, getattr(lens, "obj_depth", -1000.0)], "ks": 16}),
            ((torch.zeros(1, 3, device=device),), {"ks": 16}),
        ]
        for args, kwargs in patterns:
            try:
                output = lens.psf(*args, **kwargs)
                return output if hasattr(output, "detach") else torch.as_tensor(output, device=device)
            except Exception as exc:
                errors.append(str(exc))
    if callable(getattr(lens, "loss_rms", None)):
        return lens.loss_rms(num_grid=(1, 1), num_rays=64)
    raise RuntimeError(f"{lens_class} did not produce differentiable PSF/loss output: {'; '.join(errors)}")


def _loss(output: Any, target: Any = None) -> tuple[Any, Any]:
    import torch

    tensor = output if hasattr(output, "pow") else torch.as_tensor(output)
    if tensor.ndim == 0:
        return tensor, target
    if target is None:
        target = torch.zeros_like(tensor)
    return torch.nn.functional.mse_loss(tensor, target), target


def _resolve_repo_path(repo_path: str | Path | None) -> Path | None:
    if repo_path is not None:
        candidate = Path(repo_path)
        return candidate if candidate.exists() else None
    env_path = os.getenv("DEEPLENS_REPO_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    try:
        spec = importlib.util.find_spec("deeplens")
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        repo = Path(spec.origin).parent.parent
        if (repo / "deeplens").is_dir():
            return repo
    return None


def _find_lens_files(repo: Path, lens_class: str, max_files: int) -> list[Path]:
    roots = [
        repo / "datasets" / "lenses",
        repo / "deeplens" / "samples",
        repo / "samples",
        repo / "examples",
        repo / "test",
        repo / "tests",
        repo / "lenses",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.json", "*.zmx", "*.pth"):
            candidates.extend(sorted(root.rglob(suffix)))
    excluded = {"materials_data.json"}
    filtered = [path for path in candidates if path.name not in excluded and "sensor" not in path.as_posix().lower()]
    keywords = {
        "GeoLens": ["geo", "lens", "cooke", "zemax"],
        "HybridLens": ["hybrid", "doe", "diffractive", "lens"],
        "DiffractiveLens": ["diffractive", "fresnel", "doe", "lens"],
    }.get(lens_class, ["lens"])
    ranked = sorted(
        filtered,
        key=lambda path: (
            0 if any(keyword in path.name.lower() for keyword in keywords) else 1,
            path.as_posix(),
        ),
    )
    return ranked[:max_files]


def _parameter_snapshot(optimizer: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for index, param in enumerate(_optimizer_params(optimizer)):
        tensor = param.detach().cpu()
        if tensor.numel() == 1:
            snapshot[f"param_{index}"] = float(tensor.item())
        else:
            snapshot[f"param_{index}"] = {"shape": list(tensor.shape), "norm": float(tensor.norm().item())}
    return snapshot


def _optimizer_params(optimizer: Any) -> list[Any]:
    params: list[Any] = []
    for group in optimizer.param_groups:
        group_params = group.get("params", [])
        if hasattr(group_params, "detach"):
            params.append(group_params)
        else:
            params.extend(group_params)
    return params


def _gradient_norm(optimizer: Any) -> float:
    total = 0.0
    for param in _optimizer_params(optimizer):
        grad = getattr(param, "grad", None)
        if grad is not None:
            total += float(grad.detach().norm().cpu().item() ** 2)
    return float(total ** 0.5)


def _parameters_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for key, before_value in before.items():
        after_value = after.get(key)
        if isinstance(before_value, dict) or isinstance(after_value, dict):
            before_norm = float((before_value or {}).get("norm", 0.0))
            after_norm = float((after_value or {}).get("norm", 0.0))
            if abs(after_norm - before_norm) > 1e-12:
                return True
        elif after_value is not None and abs(float(after_value) - float(before_value)) > 1e-12:
            return True
    return False


def _save_result(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    if "parameter_before" in result:
        (output_dir / "parameter_before.json").write_text(
            json.dumps(result["parameter_before"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if "parameter_after" in result:
        (output_dir / "parameter_after.json").write_text(
            json.dumps(result["parameter_after"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (output_dir / "loss_trace.json").write_text(
        json.dumps(
            {"loss_before": result.get("loss_before"), "loss_after": result.get("loss_after")},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text("\n".join(_report_lines(result)), encoding="utf-8")


def _report_lines(result: dict[str, Any]) -> list[str]:
    return [
        f"# Lens-file Optimization Probe: {result.get('lens_class')}",
        "",
        f"- status: {result.get('status')}",
        f"- successful_file: {result.get('successful_file')}",
        f"- differentiable: {result.get('differentiable')}",
        f"- gradient_norm: {result.get('gradient_norm')}",
        f"- parameters_changed: {result.get('parameters_changed')}",
        f"- loss_before: {result.get('loss_before')}",
        f"- loss_after: {result.get('loss_after')}",
    ]


def _metrics_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_type": "deeplens_lensfile_optimization_probe",
        "backend": "deeplens",
        "evidence_domain": "deeplens_native_optimization",
        "native_optimization_level": "lens",
        "lens_file_loaded": bool(result.get("successful_file")),
        "differentiable": bool(result.get("differentiable")),
        "gradient_norm": result.get("gradient_norm"),
        "parameters_changed": bool(result.get("parameters_changed")),
        "optimizer_step_executed": bool(result.get("optimizer_step_executed")),
        "status": result.get("status"),
    }
