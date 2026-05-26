import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.workflow.node_base import create_node


class TargetFileNodeTests(unittest.TestCase):
    def _node_for_file(self, file_path: Path):
        node = create_node("target_file")
        self.assertIsNotNone(node)
        node.set_parameter("file_path", str(file_path))
        return node

    def test_csv_continuous_target_preserves_float_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.csv"
            path.write_text(
                "filename,angle\n"
                "a.wav,0\n"
                "b.wav,15.5\n"
                "c.wav,359\n",
                encoding="utf-8",
            )

            node = self._node_for_file(path)
            node.set_parameter("target_columns", "angle")
            node.set_parameter("target_kind", "continuous")

            self.assertTrue(node.execute(), msg=node.error_message)

            self.assertEqual(node.outputs["targets"].data, [0.0, 15.5, 359.0])
            self.assertEqual(
                node.outputs["target_map"].data,
                {"a.wav": 0.0, "b.wav": 15.5, "c.wav": 359.0},
            )
            metadata = node.outputs["target_metadata"].data
            self.assertEqual(metadata["kind"], "continuous")
            self.assertEqual(metadata["shape"], [3])
            self.assertEqual(metadata["columns"], ["angle"])
            self.assertEqual(metadata["dtype"], "float32")
            self.assertEqual(metadata["count"], 3)

    def test_csv_categorical_target_outputs_ids_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.csv"
            path.write_text(
                "filename,label\n"
                "cat_001.wav,cat\n"
                "dog_001.wav,dog\n"
                "cat_002.wav,cat\n",
                encoding="utf-8",
            )

            node = self._node_for_file(path)
            node.set_parameter("target_columns", "label")
            node.set_parameter("target_kind", "categorical")

            self.assertTrue(node.execute(), msg=node.error_message)

            self.assertEqual(node.outputs["targets"].data, [0, 1, 0])
            self.assertEqual(
                node.outputs["target_map"].data,
                {"cat_001.wav": 0, "dog_001.wav": 1, "cat_002.wav": 0},
            )
            metadata = node.outputs["target_metadata"].data
            self.assertEqual(metadata["kind"], "categorical")
            self.assertEqual(metadata["category_mapping"], {"cat": 0, "dog": 1})
            self.assertEqual(metadata["dtype"], "int64")

    def test_csv_multi_column_target_outputs_row_vectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.csv"
            path.write_text(
                "filename,x,y\n"
                "a.wav,1.5,2.5\n"
                "b.wav,3.5,4.5\n",
                encoding="utf-8",
            )

            node = self._node_for_file(path)
            node.set_parameter("target_columns", "x,y")
            node.set_parameter("target_kind", "continuous")

            self.assertTrue(node.execute(), msg=node.error_message)

            self.assertEqual(node.outputs["targets"].data, [[1.5, 2.5], [3.5, 4.5]])
            self.assertEqual(
                node.outputs["target_map"].data,
                {"a.wav": [1.5, 2.5], "b.wav": [3.5, 4.5]},
            )
            metadata = node.outputs["target_metadata"].data
            self.assertEqual(metadata["shape"], [2, 2])
            self.assertEqual(metadata["columns"], ["x", "y"])

    def test_json_list_outputs_ordered_targets_without_filename_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.json"
            path.write_text(json.dumps([0, 15.5, 359]), encoding="utf-8")

            node = self._node_for_file(path)
            node.set_parameter("target_kind", "continuous")

            self.assertTrue(node.execute(), msg=node.error_message)

            self.assertEqual(node.outputs["targets"].data, [0.0, 15.5, 359.0])
            self.assertEqual(node.outputs["target_map"].data, {})
            self.assertEqual(node.outputs["target_metadata"].data["source"], str(path))

    def test_json_filename_map_outputs_targets_and_target_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.json"
            path.write_text(
                json.dumps({"a.wav": 1.25, "b.wav": 2.5}),
                encoding="utf-8",
            )

            node = self._node_for_file(path)
            node.set_parameter("target_kind", "continuous")

            self.assertTrue(node.execute(), msg=node.error_message)

            self.assertEqual(node.outputs["targets"].data, [1.25, 2.5])
            self.assertEqual(node.outputs["target_map"].data, {"a.wav": 1.25, "b.wav": 2.5})

    def test_json_structured_targets_and_target_map_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": [1.25, 2.5],
                        "target_map": {"a.wav": 1.25, "b.wav": 2.5},
                    }
                ),
                encoding="utf-8",
            )

            node = self._node_for_file(path)
            node.set_parameter("target_kind", "continuous")

            self.assertTrue(node.execute(), msg=node.error_message)

            self.assertEqual(node.outputs["targets"].data, [1.25, 2.5])
            self.assertEqual(node.outputs["target_map"].data, {"a.wav": 1.25, "b.wav": 2.5})

    def test_output_format_numpy_returns_numpy_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.txt"
            path.write_text("1\n2\n3\n", encoding="utf-8")

            node = self._node_for_file(path)
            node.set_parameter("target_kind", "continuous")
            node.set_parameter("output_format", "numpy")

            self.assertTrue(node.execute(), msg=node.error_message)

            targets = node.outputs["targets"].data
            self.assertIsInstance(targets, np.ndarray)
            np.testing.assert_array_equal(targets, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_continuous_target_rejects_string_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.txt"
            path.write_text("1\n2\n3\n", encoding="utf-8")

            node = self._node_for_file(path)
            node.set_parameter("target_kind", "continuous")
            node.set_parameter("dtype", "str")

            self.assertFalse(node.execute())
            self.assertIn("Continuous targets require numeric dtype", node.error_message)


if __name__ == "__main__":
    unittest.main()
