"""Tests for generated classification model and workflow definitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.generation import (
    ModelGenerationRequest,
    WorkflowGenerationRequest,
    create_model_definition,
    create_workflow_definition,
    mel_input_shape,
)
from src.cli.validation import validate_model_file, validate_workflow_file


def test_create_model_definition_is_valid_and_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    result = create_model_definition(
        ModelGenerationRequest(output=first, classes=3, name="example")
    )
    create_model_definition(
        ModelGenerationRequest(output=second, classes=3, name="example")
    )

    report, _ = validate_model_file(first)
    definition = json.loads(first.read_text(encoding="utf-8"))
    classifier = next(layer for layer in definition["layers"] if layer["id"] == "classifier")
    assert report.valid
    assert result["classes"] == 3
    assert classifier["parameters"]["units"] == 3
    assert first.read_bytes() == second.read_bytes()


def test_create_model_definition_preserves_existing_file(tmp_path):
    output = tmp_path / "model.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_model_definition(ModelGenerationRequest(output=output, classes=2))

    assert output.read_text(encoding="utf-8") == "keep"


def test_mel_input_shape_matches_current_centered_stft_contract():
    assert mel_input_shape(sample_rate=16000, duration=3.0, n_mels=64, hop_length=512) == "(64, 94, 1)"


def test_create_workflow_definition_is_valid(tmp_path):
    dataset = tmp_path / "audio"
    dataset.mkdir()
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\na.wav,yes\n", encoding="utf-8")
    model = tmp_path / "initial.keras"
    model.write_bytes(b"placeholder")
    output = tmp_path / "workflow.json"

    result = create_workflow_definition(
        WorkflowGenerationRequest(
            output=output,
            dataset=dataset,
            labels=labels,
            model=model,
            output_model="trained",
        )
    )

    report, _ = validate_workflow_file(output)
    definition = json.loads(output.read_text(encoding="utf-8"))
    assert report.valid
    assert result["saved_model"] == "models/trained.keras"
    assert len(definition["nodes"]) == 18
    assert len(definition["connections"]) == 27
    assert next(node for node in definition["nodes"] if node["id"] == "split")["parameters"]["stratify"] is True


def test_create_workflow_rejects_invalid_ratios_before_writing(tmp_path):
    dataset = tmp_path / "audio"
    dataset.mkdir()
    labels = tmp_path / "labels.json"
    labels.write_text("{}", encoding="utf-8")
    model = tmp_path / "initial.keras"
    model.write_bytes(b"placeholder")
    output = tmp_path / "workflow.json"

    with pytest.raises(ValueError, match="sum to 1"):
        create_workflow_definition(
            WorkflowGenerationRequest(
                output=output,
                dataset=dataset,
                labels=labels,
                model=model,
                output_model="trained.keras",
                train_ratio=0.8,
            )
        )

    assert not output.exists()
