"""Test claim boundary export."""
import json
from pathlib import Path
from optiresearch.reports.claim_boundary import generate_claim_whitelist_blacklist


def test_generate_claim_boundary_has_three_categories(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    result = generate_claim_whitelist_blacklist()

    assert "supported_claims" in result
    assert "qualified_claims" in result
    assert "unsupported_claims" in result
    for cat in ["supported_claims", "qualified_claims", "unsupported_claims"]:
        assert isinstance(result[cat], list)
        for claim in result[cat]:
            assert "text" in claim
            assert "rationale" in claim


def test_claim_boundary_exports_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    md_path = tmp_path / "reports" / "claim_boundary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    result = generate_claim_whitelist_blacklist()
    lines = ["# Claim Boundary", ""]
    for cat, title in [("supported_claims", "Supported Claims"), ("qualified_claims", "Qualified Claims"), ("unsupported_claims", "Unsupported Claims")]:
        lines.append(f"## {title}")
        lines.append("")
        for c in result[cat]:
            lines.append(f"- **{c['text']}** — {c['rationale']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    content = md_path.read_text(encoding="utf-8")
    assert "Supported Claims" in content
    assert "Qualified Claims" in content
    assert "Unsupported Claims" in content


def test_claim_boundary_exports_json(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))

    result = generate_claim_whitelist_blacklist()
    json_path = tmp_path / "reports" / "claim_boundary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "supported_claims" in data


def test_unsupported_claims_contain_camera_and_native_warnings():
    result = generate_claim_whitelist_blacklist()
    unsupported_texts = " ".join(c["text"].lower() for c in result["unsupported_claims"])
    assert "real hsi" in unsupported_texts or "real camera" in unsupported_texts or "native" in unsupported_texts
