"""Shared relative input-path behavior for GUI and CLI workflow execution."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from src.workflow.engine import WorkflowEngine
from src.workflow.workflow import Workflow


def test_engine_resolves_loaded_relative_path_and_restores_serialized_value(tmp_path):
    audio = tmp_path / "sample.wav"
    sf.write(audio, np.zeros(800, dtype=np.float32), 8000)
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "source",
                        "type": "audio_file",
                        "parameters": {"file_path": "sample.wav", "target_sr": 8000},
                    }
                ],
                "connections": [],
            }
        ),
        encoding="utf-8",
    )
    workflow = Workflow.load(str(workflow_path))
    assert workflow is not None
    engine = WorkflowEngine()
    engine.set_workflow(workflow)

    result = engine.execute_sync()

    assert result.success, result.message
    assert workflow.get_node("source").get_parameter("file_path") == "sample.wav"
    assert workflow.to_dict()["nodes"][0]["parameters"]["file_path"] == "sample.wav"
