import hashlib

from src import license_validation


def test_validate_license_accepts_matching_hash(tmp_path, monkeypatch):
    mac_address = "a1:b2:c3:d4:e5:f6"
    expected_hash = hashlib.sha256(f"{mac_address}dongyuan".encode()).hexdigest()
    (tmp_path / "salt").write_text(expected_hash, encoding="utf-8")
    monkeypatch.setattr(license_validation, "get_mac_address", lambda: mac_address)

    assert license_validation.validate_license(tmp_path)


def test_validate_license_rejects_missing_or_mismatched_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        license_validation,
        "get_mac_address",
        lambda: "a1:b2:c3:d4:e5:f6",
    )

    assert not license_validation.validate_license(tmp_path)

    (tmp_path / "salt").write_text("not-a-valid-hash", encoding="utf-8")
    assert not license_validation.validate_license(tmp_path)
