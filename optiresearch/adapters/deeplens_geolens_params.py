"""Utilities for DeepLens GeoLens native trainable parameters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


DEFAULT_GEOLENS_LRS = [1e-6, 1e-6, 0.0, 0.0]


def _as_param_iter(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (list, tuple)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def flatten_optimizer_param_groups(param_groups: Any) -> list[torch.Tensor]:
    """Return unique trainable tensors from torch optimizer-style param groups."""
    if param_groups is None:
        return []
    groups = param_groups if isinstance(param_groups, (list, tuple)) else [param_groups]
    params: list[torch.Tensor] = []
    seen: set[int] = set()
    for group in groups:
        raw_params = group.get("params") if isinstance(group, dict) else group
        for param in _as_param_iter(raw_params):
            if not isinstance(param, torch.Tensor) or not bool(param.requires_grad):
                continue
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(param)
    return params


def activate_geolens_trainable_parameters(
    geolens: Any,
    lrs: list[float] | tuple[float, ...] | None = None,
    optim_mat: bool = False,
) -> tuple[list[Any], list[torch.Tensor]]:
    """Activate GeoLens native optimizer parameters and return groups + tensors.

    DeepLens GeoLens is not an nn.Module and may return a scalar tensor directly
    in each optimizer group. This helper uses the native optimizer API and then
    normalizes the result into a plain tensor list for audits and clipping.
    """
    if not callable(getattr(geolens, "get_optimizer_params", None)):
        return [], []

    lr_values = list(lrs or DEFAULT_GEOLENS_LRS)
    attempts = [
        ((), {"lrs": lr_values, "optim_mat": optim_mat}),
        ((), {"lrs": lr_values}),
        ((), {"lr": lr_values[0]}),
        ((), {}),
    ]
    errors: list[str] = []
    for args, kwargs in attempts:
        try:
            groups = geolens.get_optimizer_params(*args, **kwargs)
            groups_list = groups if isinstance(groups, list) else [groups]
            params = flatten_optimizer_param_groups(groups_list)
            return groups_list, params
        except Exception as exc:  # pragma: no cover - surfaced by callers
            errors.append(str(exc))
            continue
    raise RuntimeError("; ".join(errors) or "GeoLens get_optimizer_params failed")
