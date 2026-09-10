"""Tests for the machine-readable CLI capability catalog."""

from src.cli.capabilities import get_capabilities


def test_capabilities_are_sorted_and_describe_contracts():
    payload = get_capabilities()

    node_types = [item["type"] for item in payload["nodes"]]
    layer_types = [item["type"] for item in payload["layers"]]

    assert payload["schema_version"] == "1.0"
    assert payload["scope"] == "audio_acoustic"
    assert node_types == sorted(node_types)
    assert layer_types == sorted(layer_types)
    assert "classification_trainer" in node_types
    assert "trainer" not in node_types
    assert "input" in layer_types

    audio_folder = next(item for item in payload["nodes"] if item["type"] == "audio_folder")
    assert any(item["name"] == "folder_path" for item in audio_folder["parameters"])
    assert any(item["name"] == "audio" for item in audio_folder["outputs"])


def test_capabilities_can_include_hidden_legacy_nodes():
    payload = get_capabilities(include_hidden=True)

    assert "trainer" in [item["type"] for item in payload["nodes"]]
