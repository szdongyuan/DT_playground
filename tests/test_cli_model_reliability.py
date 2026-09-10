"""Real Keras regressions for metric names, optimizer state and build seeds."""

import json
import warnings

import numpy as np
import pytest

from src.cli.main import main
from src.model_builder import ModelGraph
from src.workflow.node_base import create_node


def definition(optimizer="Adam"):
    return {
        "name": "reliability", "version": "1.0",
        "layers": [
            {"id": "input", "type": "input", "parameters": {"shape": "(4,)"}},
            {"id": "dense", "type": "dense", "parameters": {"units": 2}},
            {"id": "output", "type": "output", "parameters": {
                "activation": "softmax", "optimizer": optimizer,
                "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"],
            }},
        ],
        "connections": [{"source": "input", "target": "dense"}, {"source": "dense", "target": "output"}],
    }


@pytest.mark.parametrize("optimizer", ["Adam", "SGD", "RMSprop", "AdamW", "Nadam"])
def test_initial_optimizer_roundtrip_without_skipped_state(tmp_path, optimizer):
    from tensorflow import keras

    model = ModelGraph.from_dict(definition(optimizer)).build_keras_model()
    path = tmp_path / "initial.keras"
    model.save(path)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loaded = keras.models.load_model(path)
    assert not any("Skipping variable loading for optimizer" in str(w.message) for w in captured)
    assert loaded.compiled
    assert int(loaded.optimizer.iterations) == 0
    assert len(model.optimizer.variables) == len(loaded.optimizer.variables)
    for expected, actual in zip(model.optimizer.variables, loaded.optimizer.variables):
        np.testing.assert_array_equal(expected.numpy(), actual.numpy())


def test_trained_optimizer_roundtrip_and_continuation(tmp_path):
    from tensorflow import keras

    model = ModelGraph.from_dict(definition()).build_keras_model()
    x = np.ones((4, 4), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    model.train_on_batch(x, y)
    path = tmp_path / "checkpoint.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    assert int(loaded.optimizer.iterations) == 1
    for expected, actual in zip(model.optimizer.variables, loaded.optimizer.variables):
        np.testing.assert_array_equal(expected.numpy(), actual.numpy())
    model.train_on_batch(x, y)
    loaded.train_on_batch(x, y)
    assert int(loaded.optimizer.iterations) == 2
    for expected, actual in zip(model.get_weights(), loaded.get_weights()):
        np.testing.assert_allclose(expected, actual, atol=1e-7)


@pytest.mark.parametrize("no_compile", [False, True])
def test_cli_seed_initializes_weights_reproducibly(tmp_path, capsys, no_compile):
    from tensorflow import keras

    source = tmp_path / "model.json"
    source.write_text(json.dumps(definition()), encoding="utf-8")
    weights = []
    for index, seed in enumerate([42, 42, 7]):
        output = tmp_path / f"model-{index}.keras"
        args = ["build-model", str(source), "--output", str(output), "--json"]
        if index != 0:
            args.extend(["--seed", str(seed)])
        if no_compile:
            args.append("--no-compile")
        assert main(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["seed"] == seed
        assert payload["compiled"] == (not no_compile)
        weights.append(keras.models.load_model(output, compile=False).get_weights())
    for a, b in zip(weights[0], weights[1]):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(weights[0], weights[2]))


@pytest.mark.parametrize("seed", ["-1", "4294967296", "oops"])
def test_invalid_seed_is_usage_error(seed):
    with pytest.raises(SystemExit) as exc:
        main(["build-model", "unused.json", "--output", "unused.keras", "--seed", seed])
    assert exc.value.code == 2


def test_model_build_failure_keeps_validation_exit_code(tmp_path, capsys):
    data = definition()
    data["layers"][1] = {"id": "dense", "type": "conv2d", "parameters": {"filters": 2}}
    source = tmp_path / "invalid-shape.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "new/model.keras"
    assert main(["build-model", str(source), "--output", str(output), "--json"]) == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["validation"]["errors"][0]["code"] == "model.build"
    assert not output.parent.exists()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_evaluator_returns_all_named_metrics(task):
    from tensorflow import keras

    classification = task == "classification"
    model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(2 if classification else 1, activation="softmax" if classification else "linear")])
    metric_names = ["accuracy", "sparse_categorical_crossentropy"] if classification else ["mae", "mse"]
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy" if classification else "mse", metrics=metric_names)
    x = np.ones((4, 4), dtype=np.float32)
    y = np.array([0, 1, 0, 1]) if classification else np.array([0.2, 1.2, 0.3, 1.3])
    node = create_node(f"{task}_evaluator")
    node.inputs["model"].data = model
    node.inputs["x_test"].data = x
    node.inputs["y_test"].data = y
    assert node.execute(), node.error_message
    actual = node.outputs["metrics"].data
    expected = model.evaluate(x, y, verbose=0, return_dict=True)
    assert set(actual) == {"loss", *metric_names}
    assert actual == pytest.approx(expected)
