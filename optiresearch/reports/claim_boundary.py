"""Claim whitelist / blacklist generator.

Generates three categories of claims:
  1. Supported Claims — what the system CAN claim with current evidence
  2. Qualified Claims — what requires caveats about mock/synthetic scope
  3. Unsupported Claims — what MUST NOT be claimed without further evidence

IMPORTANT: This module enforces evidence boundaries.
- Synthetic/mock results MUST NOT be written as real HSI performance.
- DeepLens adapter_proxy MUST NOT be written as native validation.
- Public dataset + mock optical MUST NOT be written as real camera experiment.
"""

from __future__ import annotations

from typing import Any


def generate_claim_whitelist_blacklist() -> dict[str, Any]:
    return {
        "supported_claims": _supported_claims(),
        "qualified_claims": _qualified_claims(),
        "unsupported_claims": _unsupported_claims(),
    }


def _supported_claims() -> list[dict[str, str]]:
    return [
        {
            "text": "The agent system can run an end-to-end optical-HSI evaluation loop.",
            "rationale": "Demonstrated across Phases 9-12 with mock and conditional DeepLens backends.",
        },
        {
            "text": "Local/public HSI datasets can be prepared through the adapter interface.",
            "rationale": "Local NPZ, CAVE, and ICVL adapters pass preparation contracts.",
        },
        {
            "text": "Optical encoder choice affects synthetic HSI reconstruction ranking under the current optical-sensitive forward model.",
            "rationale": "Phase 10-11 baseline comparisons show distinct encoder-dependent metrics.",
        },
        {
            "text": "The system maintains a structured ClaimEvidence pipeline with support/contradict/qualify edges.",
            "rationale": "ClaimEvidenceManager enforces downgrade rules and evidence boundaries.",
        },
        {
            "text": "Wavelength-aware PSF contract records metadata about PSF wavelength axis.",
            "rationale": "DeepLens adapter exposes wavelength_aware_psf fields in metadata.",
        },
        {
            "text": "Structured skips prevent false results when data or backends are unavailable.",
            "rationale": "Public HSI matrix returns structured skip entries with reasons.",
        },
    ]


def _qualified_claims() -> list[dict[str, str]]:
    return [
        {
            "text": "controlled_chromatic_edof improves over conventional under synthetic/mock setting.",
            "rationale": "Supported in mock baseline but requires native DeepLens for physical validation.",
        },
        {
            "text": "public/local HSI data can be used with mock optical measurement.",
            "rationale": "Pipeline is functional but this is NOT real camera validation.",
        },
        {
            "text": "DeepLens wavelength-aware PSF contract is interface-level.",
            "rationale": "Contract exists but native wavelength physics requires real DeepLens SDK.",
        },
        {
            "text": "DeepLens adapter_proxy can produce encoder-specific PSF artifacts.",
            "rationale": "Proxy transform generates differentiated outputs but does not prove native behavior.",
        },
    ]


def _unsupported_claims() -> list[dict[str, str]]:
    return [
        {
            "text": "controlled_chromatic_edof is best for real HSI reconstruction.",
            "rationale": "Requires native DeepLens optimization and real lab HSI data. Current evidence is mock/synthetic only.",
        },
        {
            "text": "current DeepLens proxy results validate native physical performance.",
            "rationale": "adapter_proxy uses Python-level transforms; native physics requires full DeepLens SDK integration.",
        },
        {
            "text": "public dataset + mock optical measurement proves real optical design.",
            "rationale": "Mock optical is a numerical proxy; real optical design requires physical measurements.",
        },
        {
            "text": "the system achieves state-of-the-art real HSI reconstruction.",
            "rationale": "No real camera HSI data; no comparison with published real-HSI methods.",
        },
        {
            "text": "DeepLens semi_native results are equivalent to native results.",
            "rationale": "semi_native is a partial transform; native results require full DeepLens optimization.",
        },
    ]
