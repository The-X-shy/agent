"""Composable optical and HSI objective library."""

from optiresearch.objectives.optical_objectives import (
    psf_width_loss,
    psf_centroid_loss,
    psf_energy_loss,
    psf_smoothness_loss,
    spot_size_loss,
    field_consistency_loss,
    ObjectiveProfile,
    list_objective_profiles,
    get_objective_profile,
    register_objective_profile,
    PRESET_PROFILES,
)
from optiresearch.objectives.hsi_objectives import (
    reconstruction_mse,
    spectral_angle_loss,
    measurement_consistency_loss,
    spectral_smoothness_loss,
    band_weighted_mse,
    task_aligned_hsi_loss,
)
from optiresearch.objectives.regularizers import (
    optical_param_l2,
    optical_param_delta_limit,
    psf_energy_preservation,
    psf_centroid_preservation,
    psf_width_preservation,
    rollback_penalty,
)

__all__ = [
    # Optical
    "psf_width_loss",
    "psf_centroid_loss",
    "psf_energy_loss",
    "psf_smoothness_loss",
    "spot_size_loss",
    "field_consistency_loss",
    # HSI
    "reconstruction_mse",
    "spectral_angle_loss",
    "measurement_consistency_loss",
    "spectral_smoothness_loss",
    "band_weighted_mse",
    "task_aligned_hsi_loss",
    # Regularizers
    "optical_param_l2",
    "optical_param_delta_limit",
    "psf_energy_preservation",
    "psf_centroid_preservation",
    "psf_width_preservation",
    "rollback_penalty",
    # Profiles
    "ObjectiveProfile",
    "list_objective_profiles",
    "get_objective_profile",
    "register_objective_profile",
    "PRESET_PROFILES",
]
