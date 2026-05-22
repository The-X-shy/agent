"""Optical backend registry for differentiable optics framework."""

from optiresearch.backends.base import OpticalBackend
from optiresearch.backends.registry import (
    get_backend,
    get_backend_by_claim_ceiling,
    get_backend_registry,
    list_backends,
    register_backend,
    export_backend_registry_markdown,
    export_backend_registry_json,
)

__all__ = [
    "OpticalBackend",
    "get_backend",
    "get_backend_by_claim_ceiling",
    "get_backend_registry",
    "list_backends",
    "register_backend",
    "export_backend_registry_markdown",
    "export_backend_registry_json",
]
