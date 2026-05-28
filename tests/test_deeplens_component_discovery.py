"""Tests for DeepLens component backend discovery."""

from unittest.mock import patch

import pytest

from optiresearch.optics.deeplens_component_discovery import (
    COMPONENT_TO_SURFACE,
    ComponentDiscoveryResult,
    discover_deeplens_components,
)


class TestComponentMapping:
    def test_fresnel_maps_to_fresnel(self):
        assert COMPONENT_TO_SURFACE["fresnel"] == "Fresnel"

    def test_binary2phase_maps_to_binary2phase(self):
        assert COMPONENT_TO_SURFACE["binary2phase"] == "Binary2Phase"

    def test_diffractive_maps_to_fresnel(self):
        assert COMPONENT_TO_SURFACE["diffractive"] == "Fresnel"


class TestComponentDiscoveryResult:
    def test_default_construction(self):
        r = ComponentDiscoveryResult(component="fresnel", surface_class="Fresnel")
        assert r.importable is False
        assert r.instantiatable is False
        assert r.available is False

    def test_available_when_both_importable_and_instantiatable(self):
        r = ComponentDiscoveryResult(
            component="fresnel", surface_class="Fresnel",
            importable=True, instantiatable=True,
        )
        assert r.available is True

    def test_not_available_when_importable_only(self):
        r = ComponentDiscoveryResult(
            component="fresnel", surface_class="Fresnel",
            importable=True, instantiatable=False,
        )
        assert r.available is False


class TestDiscoverDeeplensComponents:
    def test_returns_manifest_when_deeplens_not_installed(self):
        manifest = discover_deeplens_components()
        # DeepLens may or may not be installed — both are valid.
        assert len(manifest.results) == 3
        if manifest.deeplens_available:
            assert manifest.deeplens_version != ""
        else:
            assert manifest.deeplens_version == ""

    def test_maps_components_correctly(self):
        manifest = discover_deeplens_components(
            components=["fresnel", "binary2phase"],
        )
        surface_classes = [r.surface_class for r in manifest.results]
        assert "Fresnel" in surface_classes
        assert "Binary2Phase" in surface_classes

    def test_import_paths_checked(self):
        manifest = discover_deeplens_components()
        assert len(manifest.import_paths_checked) == 3

    def test_warnings_accumulated(self):
        manifest = discover_deeplens_components(components=["fresnel"])
        if not manifest.results[0].importable:
            assert len(manifest.warnings) > 0

    def test_empty_components_list(self):
        manifest = discover_deeplens_components(components=[])
        assert len(manifest.results) == 0

    def test_timestamp_present(self):
        manifest = discover_deeplens_components()
        assert len(manifest.timestamp) > 0

    def test_manifest_has_component_candidates_field(self):
        manifest = discover_deeplens_components(components=["fresnel", "binary2phase"])
        assert "fresnel" in manifest.component_candidates
        assert "binary2phase" in manifest.component_candidates
