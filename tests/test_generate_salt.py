import hashlib
import sys

from tools import generate_salt


def test_generate_salt_writes_expected_hash(tmp_path, monkeypatch):
    mac_address = "a1:b2:c3:d4:e5:f6"
    monkeypatch.setattr(generate_salt, "get_mac_address", lambda: mac_address)

    salt_path = generate_salt.generate_salt(tmp_path)

    assert salt_path == tmp_path / "salt"
    assert salt_path.read_text(encoding="utf-8") == hashlib.sha256(
        f"{mac_address}dongyuan".encode("utf-8")
    ).hexdigest()


def test_frozen_generator_defaults_to_executable_directory(tmp_path, monkeypatch):
    executable = tmp_path / "SaltGenerator.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert generate_salt.get_default_output_dir() == tmp_path
