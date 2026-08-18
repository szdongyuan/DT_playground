import logging

import numpy as np
import pytest

from src.ui.widgets.preview.label_preview import LabelPreviewWidget
from src.workflow.node_base import NodeCategory, create_node
from src.workflow.port import DataType


def _node():
    node = create_node("filter_dataset_by_label")
    assert node is not None
    return node


def _set_inputs(node, *, data, labels, file_paths=None, label_map=None):
    node.inputs["data"].data = data
    node.inputs["labels"].data = labels
    if file_paths is not None:
        node.inputs["file_paths"].data = file_paths
    if label_map is not None:
        node.inputs["label_map"].data = label_map


def test_filter_node_exposes_expected_catalog_metadata_and_contract():
    node = _node()

    assert node.category == NodeCategory.CONTROL
    assert node.subcategory == "Dataset processing"
    assert list(node.inputs) == ["data", "labels", "file_paths", "label_map"]
    assert list(node.outputs) == [
        "filtered_data",
        "filtered_labels",
        "filtered_file_paths",
        "filtered_label_map",
    ]
    assert node.inputs["data"].data_type == DataType.ANY
    assert node.inputs["labels"].data_type == DataType.LABEL
    assert not node.inputs["file_paths"].required
    assert not node.inputs["label_map"].required
    assert node.get_parameter("keep_labels") == ""


def test_filter_by_names_preserves_sample_order_and_reencodes_by_requested_order():
    node = _node()
    node.set_parameter("keep_labels", " NG, OK, NG ")
    _set_inputs(
        node,
        data=["audio-a", "audio-b", "audio-c", "audio-d"],
        labels=[0, 2, 1, 2],
        file_paths=["a.wav", "b.wav", "c.wav", "d.wav"],
        label_map={
            "label_names": {"OK": 0, "not_label": 1, "NG": 2},
            "filename_map": {
                "a.wav": 0,
                "b.wav": 2,
                "c.wav": 1,
                "d.wav": 2,
            },
        },
    )
    statuses = []
    node.status_callback = statuses.append

    assert node.execute(), node.error_message

    assert node.outputs["filtered_data"].data == ["audio-a", "audio-b", "audio-d"]
    assert node.outputs["filtered_labels"].data == [1, 0, 0]
    assert node.outputs["filtered_file_paths"].data == ["a.wav", "b.wav", "d.wav"]
    assert node.outputs["filtered_label_map"].data == {
        "label_names": {"NG": 0, "OK": 1},
        "filename_map": {"a.wav": 1, "b.wav": 0, "d.wav": 0},
    }
    assert statuses
    assert "input 4" in statuses[-1]
    assert "kept 3" in statuses[-1]
    assert "removed 1" in statuses[-1]
    assert "NG=2" in statuses[-1]
    assert "OK=1" in statuses[-1]


def test_filter_logs_one_summary_without_full_label_maps(caplog):
    node = _node()
    node.set_parameter("keep_labels", "OK")
    _set_inputs(
        node,
        data=["audio-a", "audio-b"],
        labels=[0, 1],
        file_paths=["sensitive-a.wav", "sensitive-b.wav"],
        label_map={
            "label_names": {"OK": 0, "NG": 1},
            "filename_map": {"sensitive-a.wav": 0, "sensitive-b.wav": 1},
        },
    )

    with caplog.at_level(logging.INFO, logger="src.workflow.nodes.control"):
        assert node.execute(), node.error_message

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "input 2" in messages[0]
    assert "kept 1" in messages[0]
    assert "removed 1" in messages[0]
    assert "OK=1" in messages[0]
    assert "filename_map" not in messages[0]
    assert "sensitive-a.wav" not in messages[0]


def test_filter_without_mapping_matches_string_label_values():
    node = _node()
    node.set_parameter("keep_labels", "OK,NG")
    _set_inputs(
        node,
        data=["audio-a", "audio-b", "audio-c"],
        labels=["OK", "not_label", "NG"],
    )

    assert node.execute(), node.error_message

    assert node.outputs["filtered_data"].data == ["audio-a", "audio-c"]
    assert node.outputs["filtered_labels"].data == [0, 1]
    assert node.outputs["filtered_file_paths"].data == []
    assert node.outputs["filtered_label_map"].data == {
        "label_names": {"OK": 0, "NG": 1}
    }


def test_filter_without_mapping_matches_numeric_ids_in_requested_order():
    node = _node()
    node.set_parameter("keep_labels", "2,0")
    _set_inputs(
        node,
        data=np.array([[10], [20], [30]]),
        labels=np.array([0, 1, 2]),
    )

    assert node.execute(), node.error_message

    assert [item.tolist() for item in node.outputs["filtered_data"].data] == [[10], [30]]
    assert node.outputs["filtered_labels"].data == [1, 0]
    assert node.outputs["filtered_label_map"].data == {
        "label_names": {"2": 0, "0": 1}
    }


@pytest.mark.parametrize(
    ("data", "labels", "file_paths", "message"),
    [
        (["a", "b"], [0], None, "Label count"),
        (["a", "b"], [0, 0], ["a.wav"], "File path count"),
    ],
)
def test_filter_rejects_misaligned_inputs(data, labels, file_paths, message):
    node = _node()
    node.set_parameter("keep_labels", "0")
    _set_inputs(node, data=data, labels=labels, file_paths=file_paths)

    assert not node.execute()
    assert message in node.error_message


def test_filter_rejects_missing_label_with_case_sensitive_available_names():
    node = _node()
    node.set_parameter("keep_labels", "OK,ng")
    _set_inputs(
        node,
        data=["a", "b"],
        labels=[0, 1],
        label_map={"label_names": {"OK": 0, "NG": 1}},
    )

    assert not node.execute()
    assert "ng" in node.error_message
    assert "OK" in node.error_message
    assert "NG" in node.error_message


def test_filter_rejects_mapped_class_without_input_samples():
    node = _node()
    node.set_parameter("keep_labels", "NG")
    _set_inputs(
        node,
        data=["a", "b"],
        labels=[0, 0],
        label_map={"label_names": {"OK": 0, "NG": 1}},
    )

    assert not node.execute()
    assert "NG" in node.error_message
    assert "no input samples" in node.error_message


def test_filter_rejects_label_values_missing_from_mapping():
    node = _node()
    node.set_parameter("keep_labels", "OK")
    _set_inputs(
        node,
        data=["a", "b"],
        labels=[0, 2],
        label_map={"label_names": {"OK": 0, "NG": 1}},
    )

    assert not node.execute()
    assert "2" in node.error_message
    assert "label mapping" in node.error_message


def test_filter_rejects_non_dictionary_label_mapping():
    node = _node()
    node.set_parameter("keep_labels", "OK")
    _set_inputs(node, data=["a"], labels=["OK"], label_map=["OK"])

    assert not node.execute()
    assert "Label map must be a dictionary" in node.error_message


def test_filter_rejects_empty_data():
    node = _node()
    node.set_parameter("keep_labels", "OK")
    _set_inputs(node, data=[], labels=[])

    assert not node.execute()
    assert "Data list is empty" in node.error_message


@pytest.mark.parametrize("keep_labels", ["", " , , "])
def test_filter_rejects_empty_keep_label_parameter(keep_labels):
    node = _node()
    node.set_parameter("keep_labels", keep_labels)
    _set_inputs(node, data=["a"], labels=[0])

    assert not node.execute()
    assert "Keep labels" in node.error_message


def test_filter_parameter_survives_node_serialization():
    node = _node()
    node.set_parameter("keep_labels", "NG,OK")

    restored = type(node).from_dict(node.to_dict())

    assert restored.get_parameter("keep_labels") == "NG,OK"


def test_filtered_label_mapping_is_compatible_with_existing_label_preview(qapp):
    node = _node()
    node.set_parameter("keep_labels", "OK,NG")
    _set_inputs(
        node,
        data=["a", "b"],
        labels=[0, 1],
        file_paths=["a.wav", "b.wav"],
        label_map={"label_names": {"OK": 0, "NG": 1}},
    )

    assert node.execute(), node.error_message

    output = node.outputs["filtered_label_map"].data
    assert LabelPreviewWidget.can_display(output)
    preview = LabelPreviewWidget()
    assert preview.set_data(output)
