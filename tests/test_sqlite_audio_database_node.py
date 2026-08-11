import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.workflow.node_base import create_node
from src.workflow.port import DataType


AUDIO_COLUMNS = (
    "audio_data_id",
    "file_path",
    "product_model",
    "sample_rate",
    "record_date",
    "labels",
    "barcode",
    "stimulus_id",
)


def _create_database(path: Path, rows=(), columns_sql: str | None = None) -> Path:
    schema = columns_sql or """
        audio_data_id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL UNIQUE,
        product_model TEXT NOT NULL,
        sample_rate INTEGER NOT NULL,
        record_date DATETIME NOT NULL,
        labels TEXT,
        barcode TEXT,
        stimulus_id TEXT
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE audio_data_table ({schema})")
        if rows:
            placeholders = ", ".join("?" for _ in AUDIO_COLUMNS)
            connection.executemany(
                f"INSERT INTO audio_data_table ({', '.join(AUDIO_COLUMNS)}) VALUES ({placeholders})",
                rows,
            )
    return path


def _row(
    row_id: str,
    file_path: str,
    label: str | None,
    sample_rate: int = 44100,
):
    return (
        row_id,
        file_path,
        "product-a",
        sample_rate,
        "2026-08-11",
        label,
        None,
        None,
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"audio-placeholder")
    return path


def _fake_audio_loader(sample_rates=None, failures=()):
    rates = sample_rates or {}
    failed_names = set(failures)

    def load(file_path: str, sr, mono: bool):
        assert sr is None
        assert mono is False
        name = Path(file_path).name
        if name in failed_names:
            raise RuntimeError("decoder failure")
        value = float(sum(ord(char) for char in name))
        data = np.array([[value, value + 1.0]], dtype=np.float32)
        return data, rates.get(name, 44100)

    return load


def _configured_node(database_path: Path):
    node = create_node("sqlite_audio_database")
    assert node is not None
    assert node.set_parameter("database_path", str(database_path))[0]
    assert node.set_parameter("sample_rate", 44100)[0]
    return node


def test_sqlite_audio_database_node_contract():
    node = create_node("sqlite_audio_database")

    assert node is not None
    assert list(node.parameters) == [
        "database_path",
        "audio_root",
        "sample_rate",
        "labels",
        "file_path_regex",
        "max_files",
    ]
    assert node.get_parameter("sample_rate") == 44100
    assert node.get_parameter("max_files") == 0
    assert list(node.outputs) == ["audio", "labels", "file_paths", "label_map", "metadata"]
    assert node.outputs["audio"].data_type == DataType.AUDIO
    assert node.outputs["labels"].data_type == DataType.LABEL


def test_explicit_labels_sort_limit_and_keep_outputs_aligned(tmp_path):
    root = tmp_path
    database_path = root / "database" / "audio_data.db"
    for relative_path in ("audio/z_ok.wav", "audio/a_ng.wav", "audio/m_ok.wav"):
        _touch(root / relative_path)
    _create_database(
        database_path,
        [
            _row("1", "audio/z_ok.wav", "OK"),
            _row("2", "audio/a_ng.wav", "NG"),
            _row("3", "audio/m_ok.wav", "OK"),
            _row("4", "audio/wrong_rate.wav", "OK", sample_rate=48000),
        ],
    )
    node = _configured_node(database_path)
    node.set_parameter("labels", " OK, NG, OK ")
    node.set_parameter("max_files", 2)

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(),
    ):
        assert node.execute(), node.error_message

    expected_paths = [
        str((root / "audio/a_ng.wav").resolve()),
        str((root / "audio/m_ok.wav").resolve()),
    ]
    assert node.outputs["file_paths"].data == expected_paths
    assert node.outputs["labels"].data == [1, 0]
    assert [item.file_path for item in node.outputs["audio"].data] == expected_paths
    assert [item["audio_data_id"] for item in node.outputs["metadata"].data] == ["2", "3"]
    assert node.outputs["label_map"].data == {
        "filename_map": {expected_paths[0]: 1, expected_paths[1]: 0},
        "label_names": {"OK": 0, "NG": 1},
    }


def test_empty_labels_build_lexical_map_and_exclude_null_or_empty(tmp_path):
    root = tmp_path
    database_path = root / "database" / "audio_data.db"
    valid_rows = [
        _row("1", "audio/ng.wav", "NG"),
        _row("2", "audio/ok.wav", "OK"),
        _row("3", "audio/unlabeled.wav", "not_labeled"),
    ]
    for row in valid_rows:
        _touch(root / row[1])
    _create_database(
        database_path,
        valid_rows
        + [
            _row("4", "audio/empty.wav", ""),
            _row("5", "audio/null.wav", None),
        ],
    )
    node = _configured_node(database_path)

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(),
    ):
        assert node.execute(), node.error_message

    assert node.outputs["label_map"].data["label_names"] == {
        "NG": 0,
        "OK": 1,
        "not_labeled": 2,
    }
    assert node.outputs["labels"].data == [0, 1, 2]
    assert len(node.outputs["metadata"].data) == 3


def test_file_path_regex_is_case_sensitive_and_uses_search(tmp_path):
    root = tmp_path
    database_path = root / "database" / "audio_data.db"
    _touch(root / "audio/lower.wav")
    _touch(root / "audio/upper.WAV")
    _create_database(
        database_path,
        [
            _row("lower", "audio/lower.wav", "OK"),
            _row("upper", "audio/upper.WAV", "OK"),
        ],
    )
    node = _configured_node(database_path)
    node.set_parameter("file_path_regex", r"lower\.wav$")

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(),
    ):
        assert node.execute(), node.error_message

    assert [item["audio_data_id"] for item in node.outputs["metadata"].data] == ["lower"]


def test_manual_audio_root_overrides_default_for_relative_paths(tmp_path):
    database_path = tmp_path / "database" / "audio_data.db"
    audio_root = tmp_path / "mounted-dataset"
    expected_path = _touch(audio_root / "clips/a.wav").resolve()
    _create_database(database_path, [_row("1", "clips/a.wav", "OK")])
    node = _configured_node(database_path)
    node.set_parameter("audio_root", str(audio_root))

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(),
    ):
        assert node.execute(), node.error_message

    assert node.outputs["file_paths"].data == [str(expected_path)]


def test_absolute_audio_path_is_not_rebased(tmp_path):
    database_path = tmp_path / "database" / "audio_data.db"
    absolute_audio = _touch(tmp_path / "external" / "absolute.wav").resolve()
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    _create_database(database_path, [_row("1", str(absolute_audio), "OK")])
    node = _configured_node(database_path)
    node.set_parameter("audio_root", str(unrelated_root))

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(),
    ):
        assert node.execute(), node.error_message

    assert node.outputs["file_paths"].data == [str(absolute_audio)]


def test_skips_missing_and_broken_files_and_reports_rate_mismatch(tmp_path):
    root = tmp_path
    database_path = root / "database" / "audio_data.db"
    _touch(root / "audio/good.wav")
    _touch(root / "audio/broken.wav")
    _create_database(
        database_path,
        [
            _row("broken", "audio/broken.wav", "OK"),
            _row("good", "audio/good.wav", "OK"),
            _row("missing", "audio/missing.wav", "OK"),
        ],
    )
    node = _configured_node(database_path)
    messages = []
    node.status_callback = messages.append

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(
            sample_rates={"good.wav": 48000},
            failures={"broken.wav"},
        ),
    ):
        assert node.execute(), node.error_message

    assert [item["audio_data_id"] for item in node.outputs["metadata"].data] == ["good"]
    assert node.outputs["audio"].data[0].sample_rate == 48000
    assert "loaded 1" in messages[-1]
    assert "missing 1" in messages[-1]
    assert "failed 1" in messages[-1]
    assert "sample-rate mismatches 1" in messages[-1]


def test_all_candidate_audio_failures_make_node_fail(tmp_path):
    root = tmp_path
    database_path = root / "database" / "audio_data.db"
    _touch(root / "audio/broken.wav")
    _create_database(
        database_path,
        [
            _row("broken", "audio/broken.wav", "OK"),
            _row("missing", "audio/missing.wav", "OK"),
        ],
    )
    node = _configured_node(database_path)

    with patch(
        "src.workflow.nodes.data_source.librosa.load",
        side_effect=_fake_audio_loader(failures={"broken.wav"}),
    ):
        assert not node.execute()

    assert "No valid database audio files" in node.error_message
    assert all(port.data is None for port in node.outputs.values())


@pytest.mark.parametrize(
    "schema,missing_item",
    [
        ("audio_data_id TEXT PRIMARY KEY", "file_path"),
    ],
)
def test_missing_schema_columns_are_reported(tmp_path, schema, missing_item):
    database_path = _create_database(
        tmp_path / "database" / "audio_data.db",
        columns_sql=schema,
    )
    node = _configured_node(database_path)

    assert not node.execute()
    assert missing_item in node.error_message


def test_missing_audio_table_is_reported(tmp_path):
    database_path = tmp_path / "database" / "audio_data.db"
    database_path.parent.mkdir(parents=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated_table (id INTEGER PRIMARY KEY)")
    node = _configured_node(database_path)

    assert not node.execute()
    assert "audio_data_table" in node.error_message


@pytest.mark.parametrize(
    "parameter,value,expected_message",
    [
        ("sample_rate", 0, "positive integer"),
        ("max_files", -1, "non-negative integer"),
    ],
)
def test_invalid_numeric_parameters_are_rejected_at_execution(
    tmp_path,
    parameter,
    value,
    expected_message,
):
    database_path = _create_database(tmp_path / "database" / "audio_data.db")
    node = _configured_node(database_path)
    node.parameter_values[parameter] = value

    assert not node.execute()
    assert expected_message in node.error_message


def test_invalid_regular_expression_fails_before_database_query(tmp_path):
    database_path = tmp_path / "invalid.db"
    database_path.write_bytes(b"not a sqlite database")
    node = _configured_node(database_path)
    node.set_parameter("file_path_regex", "[")

    assert not node.execute()
    assert "Invalid file path regular expression" in node.error_message
