from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.cli import main


def test_deeplens_capability_schema_is_stable(capsys):
    environment = DeepLensAdapter().validate_environment()

    assert "capabilities" in environment
    assert "capability_names" in environment
    assert isinstance(environment["capabilities"], list)
    assert all(
        set(item).issuperset({"name", "available", "reason", "evidence"})
        for item in environment["capabilities"]
    )
    assert "import_deeplens" in environment["capability_names"]
    assert "paraxial_lens_available" in environment["capability_names"]
    assert "encoder_specific_proxy_available" in environment["capability_names"]
    assert "encoder_specific_native_available" in environment["capability_names"]
    assert "proxy_transform_available" in environment["capability_names"]
    assert "raw_base_psf_export_available" in environment["capability_names"]
    assert "proxy_manifest_export_available" in environment["capability_names"]

    main(["deeplens-capabilities"])
    output = capsys.readouterr().out
    assert "Capability" in output
    assert "import_deeplens" in output
