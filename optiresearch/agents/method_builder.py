"""Rule-based Method Builder."""

from __future__ import annotations

from optiresearch.schemas.experiment import ExperimentSpec, build_default_mock_edof_hsi_experiment
from optiresearch.llm.registry import get_llm_provider
from optiresearch.llm.structured_output import ExperimentSpecDraft


class MethodBuilder:
    """Build a deterministic mock optical simulation spec."""

    def build_mock_optical_spec(
        self,
        objective: str,
        encoder_type: str = "controlled_chromatic_edof",
        backend: str = "mock_deeplens",
    ) -> ExperimentSpec:
        if backend not in {"mock_deeplens", "deeplens"}:
            raise ValueError(f"Unsupported backend: {backend}")
        if encoder_type == "controlled_chromatic_edof":
            detected = _detect_encoder_from_objective(objective)
            if detected:
                encoder_type = detected
        spec = build_default_mock_edof_hsi_experiment(objective, encoder_type=encoder_type)
        if backend == "mock_deeplens":
            return spec
        optical = spec.optical_spec.model_copy(
            update={"metadata": {**spec.optical_spec.metadata, "backend": backend}},
            deep=True,
        )
        return spec.model_copy(
            update={
                "backend": backend,
                "optical_spec": optical,
                "metadata": {**spec.metadata, "backend": backend},
            },
            deep=True,
        )

    def build_experiment_spec_with_llm(self, objective: str, memory_context: dict | None = None, provider=None) -> ExperimentSpec:
        provider = provider or get_llm_provider()
        if not getattr(provider, "available", lambda: False)():
            return self.build_mock_optical_spec(objective).model_copy(update={"metadata": {"llm_used": False, "fallback_used": True}}, deep=True)
        try:
            draft = provider.structured_complete(
                [
                    {"role": "system", "content": "Draft an ExperimentSpec-compatible optical setup. Return JSON only."},
                    {"role": "user", "content": objective},
                ],
                ExperimentSpecDraft,
            )
            encoder = draft.encoder_type if draft.encoder_type in {"conventional", "achromatic", "edof", "chromatic_coded", "controlled_chromatic_edof", "mock"} else "controlled_chromatic_edof"
            backend = draft.backend if draft.backend in {"mock_deeplens", "deeplens"} else "mock_deeplens"
            return self.build_mock_optical_spec(objective, encoder_type=encoder, backend=backend).model_copy(
                update={"metadata": {"llm_used": True, "fallback_used": False, "llm_caveats": draft.caveats}},
                deep=True,
            )
        except Exception:
            return self.build_mock_optical_spec(objective).model_copy(update={"metadata": {"llm_used": False, "fallback_used": True}}, deep=True)


def _detect_encoder_from_objective(objective: str) -> str | None:
    """Detect encoder type from brackets in objective, e.g. '... [chromatic_coded]'.
    Only bracket-delimited tokens are checked to avoid false matches on words like 'EDOF-HSI'."""
    import re
    known = {"conventional", "achromatic", "edof", "chromatic_coded", "controlled_chromatic_edof"}
    bracket_match = re.findall(r'\[([^\]]+)\]', objective)
    for match in bracket_match:
        clean = match.strip().lower()
        if clean in known:
            return clean
    return None
