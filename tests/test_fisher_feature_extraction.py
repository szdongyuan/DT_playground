import json
import sys

import numpy as np
import pytest

from src.cli.validation import validate_workflow_file
from src.workflow.nodes.feature import fisher_embedding
from src.workflow.nodes.feature.ai_embedding import AIFeatureExtractionNode
from src.workflow.nodes.feature.base import AudioData, FeatureData
from src.workflow.path_resolver import resolve_workflow_input_paths, restore_workflow_input_paths


def test_legacy_keras_activation():
    from tensorflow import keras

    node = AIFeatureExtractionNode.from_dict({
        "id": "features", "type": "ai_feature_extraction",
        "parameters": {"layer_name": "embedding", "output_type": "feature_1d"},
    })
    model_input = keras.Input(shape=(2,))
    embeddings = keras.layers.Dense(3, kernel_initializer="ones", name="embedding")(model_input)
    node.inputs["input_data"].set_connected(True)
    node.inputs["input_data"].data = [np.ones(2, dtype=np.float32)] * 2
    assert not node.validate()[0]
    node.inputs["model"].set_connected(True)
    node.inputs["model"].data = keras.Model(model_input, embeddings)

    assert node.validate()[0]
    assert node.execute(), node.error_message
    assert len(node.outputs["feature"].data) == 2
    np.testing.assert_array_equal(node.outputs["feature"].data[0].data, [[2, 2, 2]])


def test_local_mode_conflict_and_failed_execution_clears_output(monkeypatch):
    node = AIFeatureExtractionNode()
    node.set_parameter("model_source", "local")
    node.set_parameter("model_dir", "weights")
    node.inputs["input_data"].set_connected(True)
    node.inputs["input_data"].data = AudioData(np.zeros(3200, dtype=np.float32), 16000, "sample.wav")
    feature = FeatureData(np.ones((1, 3)), "fisher_embedding", 16000, 160, "sample.wav")
    monkeypatch.setattr(fisher_embedding, "extract", lambda *args: feature)

    assert node.validate()[0]
    assert node.execute()
    assert node.outputs["feature"].data is feature

    node.inputs["model"].set_connected(True)
    assert not node.validate()[0]
    assert not node.execute()
    assert node.outputs["feature"].data is None


def test_cli_validates_local_relative_directory_without_importing_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    (tmp_path / "audio.wav").touch()
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(fisher_embedding.MODEL_CONFIG), encoding="utf-8")
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({
        "nodes": [
            {"id": "audio", "type": "audio_file", "parameters": {"file_path": "audio.wav"}},
            {"id": "features", "type": "ai_feature_extraction",
             "parameters": {"model_source": "local", "model_dir": "weights"}},
        ],
        "connections": [{"source": {"node": "audio", "port": "audio"},
                         "target": {"node": "features", "port": "input_data"}}],
    }), encoding="utf-8")
    report, _ = validate_workflow_file(path)
    assert {error["code"] for error in report.errors} == {"model.invalid_fisher"}

    (model / "model.safetensors").touch()
    report, workflow = validate_workflow_file(path)
    assert report.valid, report.errors
    originals = resolve_workflow_input_paths(workflow)
    assert workflow.get_node("features").get_parameter("model_dir") == str(model)
    restore_workflow_input_paths(workflow, originals)
    assert workflow.get_node("features").get_parameter("model_dir") == "weights"


@pytest.mark.parametrize(
    ("shape", "rate", "message"),
    [((2, 3200), 16000, "mono"), ((1, 3200), 48000, "Resample"),
     ((1, 500), 16000, "STFT frames"), ((1, 200000), 16000, "STFT frames")],
)
def test_invalid_waveforms_fail_before_loading_model(shape, rate, message):
    audio = AudioData(np.zeros(shape, dtype=np.float32), rate, "sample.wav")
    with pytest.raises(ValueError, match=message):
        fisher_embedding.extract(audio, "unused", 16000, 8)
