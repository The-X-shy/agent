"""Runtime handler for component surrogate HSI co-design."""

from __future__ import annotations

from optiresearch.hsi.component_surrogate_forward import run_component_surrogate_hsi_forward
from optiresearch.schemas.component_surrogate_psf import (
    ComponentSurrogateHSICoDesignResult,
    ComponentSurrogateHSICoDesignSpec,
)


def run_component_surrogate_hsi_codesign(
    spec: ComponentSurrogateHSICoDesignSpec,
) -> ComponentSurrogateHSICoDesignResult:
    return run_component_surrogate_hsi_forward(spec)
