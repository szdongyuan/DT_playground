import tempfile
import unittest
from pathlib import Path

from src.workflow.node_base import create_node


class AlignTargetsNodeTests(unittest.TestCase):
    def _node(self):
        node = create_node("align_targets")
        self.assertIsNotNone(node)
        return node

    def test_basename_matching_aligns_data_targets_and_paths(self):
        node = self._node()
        node.inputs["data"].data = ["audio-a", "audio-b"]
        node.inputs["file_paths"].data = [r"C:\data\a.wav", r"C:\data\b.wav"]
        node.inputs["target_map"].data = {"b.wav": 2.5, "a.wav": 1.5, "unused.wav": 9.0}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_data"].data, ["audio-a", "audio-b"])
        self.assertEqual(node.outputs["aligned_targets"].data, [1.5, 2.5])
        self.assertEqual(node.outputs["aligned_file_paths"].data, [r"C:\data\a.wav", r"C:\data\b.wav"])
        report = node.outputs["alignment_report"].data
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual(report["unused_count"], 1)
        self.assertEqual(report["unused_target_keys"], ["unused.wav"])

    def test_csv_label_file_output_aligns_by_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "labels.csv"
            csv_path.write_text(
                "filename,label\n"
                "a.wav,cat\n"
                "b.wav,dog\n",
                encoding="utf-8",
            )

            label_node = create_node("label_file")
            self.assertIsNotNone(label_node)
            label_node.set_parameter("file_path", str(csv_path))
            self.assertTrue(label_node.execute(), msg=label_node.error_message)

            align_node = self._node()
            align_node.inputs["file_paths"].data = [
                str(Path(tmpdir) / "a.wav"),
                str(Path(tmpdir) / "b.wav"),
            ]
            align_node.inputs["target_map"].data = label_node.outputs["label_map"].data

            self.assertTrue(align_node.execute(), msg=align_node.error_message)
            self.assertEqual(align_node.outputs["aligned_targets"].data, [0, 1])
            self.assertEqual(align_node.outputs["alignment_report"].data["matched_count"], 2)

    def test_csv_label_metadata_enables_downstream_stratified_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "labels.csv"
            rows = ["filename,label"] + [
                f"sample_{index}.wav,{'cat' if index < 10 else 'dog'}"
                for index in range(20)
            ]
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            label_node = create_node("label_file")
            self.assertIsNotNone(label_node)
            label_node.set_parameter("file_path", str(csv_path))
            self.assertTrue(label_node.execute(), msg=label_node.error_message)
            label_metadata = label_node.outputs["label_map"].data
            self.assertEqual(label_metadata["kind"], "categorical")

            align_node = self._node()
            align_node.inputs["file_paths"].data = [
                str(Path(tmpdir) / f"sample_{index}.wav") for index in range(20)
            ]
            align_node.inputs["target_map"].data = label_metadata
            self.assertTrue(align_node.execute(), msg=align_node.error_message)

            split_node = create_node("split")
            self.assertIsNotNone(split_node)
            split_node.set_parameter("train_ratio", 0.8)
            split_node.set_parameter("val_ratio", 0.0)
            split_node.set_parameter("test_ratio", 0.2)
            split_node.set_parameter("shuffle", True)
            split_node.set_parameter("stratify", True)
            split_node.inputs["data"].data = list(range(20))
            split_node.inputs["targets"].data = align_node.outputs["aligned_targets"].data
            split_node.inputs["target_metadata"].data = label_metadata
            statuses = []
            split_node.status_callback = statuses.append

            self.assertTrue(split_node.execute(), msg=split_node.error_message)
            self.assertIn("target_kind=categorical", statuses[-1])
            self.assertIn("test_stratified=true", statuses[-1])
            self.assertEqual(split_node.outputs["test_targets"].data.count(0), 2)
            self.assertEqual(split_node.outputs["test_targets"].data.count(1), 2)

    def test_json_label_map_is_wrapped_with_categorical_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "labels.json"
            json_path.write_text(
                '{"a.wav": 0, "b.wav": 1}',
                encoding="utf-8",
            )

            label_node = create_node("label_file")
            self.assertIsNotNone(label_node)
            label_node.set_parameter("file_path", str(json_path))
            self.assertTrue(label_node.execute(), msg=label_node.error_message)

            label_metadata = label_node.outputs["label_map"].data
            self.assertEqual(label_metadata["kind"], "categorical")
            self.assertEqual(
                label_metadata["filename_map"],
                {"a.wav": 0, "b.wav": 1},
            )

            align_node = self._node()
            align_node.inputs["file_paths"].data = ["a.wav", "b.wav"]
            align_node.inputs["target_map"].data = label_metadata
            self.assertTrue(align_node.execute(), msg=align_node.error_message)
            self.assertEqual(align_node.outputs["aligned_targets"].data, [0, 1])

    def test_stem_matching_aligns_extensionless_target_keys(self):
        node = self._node()
        node.set_parameter("match_mode", "stem")
        node.inputs["file_paths"].data = ["/data/a.wav", "/data/b.flac"]
        node.inputs["target_map"].data = {"a": 1, "b": 2}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_targets"].data, [1, 2])
        self.assertEqual(node.outputs["aligned_data"].data, [])

    def test_relative_path_matching_uses_explicit_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            first = root / "Actor_01" / "sample.wav"
            second = root / "Actor_02" / "sample.wav"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)

            node = self._node()
            node.set_parameter("match_mode", "relative_path")
            node.set_parameter("dataset_root", str(root))
            node.inputs["file_paths"].data = [str(first), str(second)]
            node.inputs["target_map"].data = {
                "Actor_01/sample.wav": 0,
                "Actor_02\\sample.wav": 1,
            }

            self.assertTrue(node.execute(), msg=node.error_message)
            self.assertEqual(node.outputs["aligned_targets"].data, [0, 1])
            report = node.outputs["alignment_report"].data
            self.assertEqual(report["dataset_root"], str(root.resolve()))

    def test_relative_path_matching_requires_dataset_root(self):
        node = self._node()
        node.set_parameter("match_mode", "relative_path")
        node.inputs["file_paths"].data = ["dataset/a.wav"]
        node.inputs["target_map"].data = {"a.wav": 1}

        self.assertFalse(node.execute())
        self.assertIn("Dataset root is required", node.error_message)

    def test_relative_path_matching_rejects_file_outside_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            root.mkdir()
            outside = Path(tmpdir) / "outside.wav"

            node = self._node()
            node.set_parameter("match_mode", "relative_path")
            node.set_parameter("dataset_root", str(root))
            node.inputs["file_paths"].data = [str(outside)]
            node.inputs["target_map"].data = {"outside.wav": 1}

            self.assertFalse(node.execute())
            self.assertIn("outside dataset root", node.error_message)

    def test_relative_path_matching_normalizes_absolute_target_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            sample = root / "nested" / "sample.wav"
            sample.parent.mkdir(parents=True)

            node = self._node()
            node.set_parameter("match_mode", "relative_path")
            node.set_parameter("dataset_root", str(root))
            node.inputs["file_paths"].data = [str(sample)]
            node.inputs["target_map"].data = {str(sample): 7}

            self.assertTrue(node.execute(), msg=node.error_message)
            self.assertEqual(node.outputs["aligned_targets"].data, [7])

    def test_relative_path_matching_rejects_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            root.mkdir()

            node = self._node()
            node.set_parameter("match_mode", "relative_path")
            node.set_parameter("dataset_root", str(root))
            node.inputs["file_paths"].data = [str(root / "sample.wav")]
            node.inputs["target_map"].data = {"../sample.wav": 1}

            self.assertFalse(node.execute())
            self.assertIn("outside dataset root", node.error_message)

    def test_missing_policy_error_fails_with_clear_message(self):
        node = self._node()
        node.inputs["file_paths"].data = ["a.wav", "b.wav"]
        node.inputs["target_map"].data = {"a.wav": 1}

        self.assertFalse(node.execute())
        self.assertIn("Missing targets", node.error_message)
        self.assertIn("b.wav", node.error_message)

    def test_missing_error_limits_examples_and_shows_normalized_keys(self):
        node = self._node()
        node.inputs["file_paths"].data = [f"folder/{index}.wav" for index in range(10)]
        node.inputs["target_map"].data = {"target.wav": 1}

        self.assertFalse(node.execute())
        self.assertIn("Missing targets for 10 file path(s)", node.error_message)
        self.assertIn("0.wav", node.error_message)
        self.assertIn("4.wav", node.error_message)
        self.assertNotIn("5.wav", node.error_message)
        self.assertIn("file=0.wav, target=target.wav", node.error_message)

    def test_missing_policy_drop_outputs_aligned_subset_and_report(self):
        node = self._node()
        node.set_parameter("missing_policy", "drop")
        node.inputs["data"].data = ["audio-a", "audio-b", "audio-c"]
        node.inputs["file_paths"].data = ["a.wav", "b.wav", "c.wav"]
        node.inputs["target_map"].data = {"a.wav": 1, "c.wav": 3}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_data"].data, ["audio-a", "audio-c"])
        self.assertEqual(node.outputs["aligned_targets"].data, [1, 3])
        self.assertEqual(node.outputs["aligned_file_paths"].data, ["a.wav", "c.wav"])
        report = node.outputs["alignment_report"].data
        self.assertEqual(report["missing_count"], 1)
        self.assertEqual(report["dropped_count"], 1)
        self.assertEqual(report["missing_file_paths"], ["b.wav"])

    def test_missing_policy_fill_uses_fill_value(self):
        node = self._node()
        node.set_parameter("missing_policy", "fill")
        node.set_parameter("fill_value", "0")
        node.inputs["file_paths"].data = ["a.wav", "b.wav"]
        node.inputs["target_map"].data = {"a.wav": 1}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_targets"].data, [1, "0"])
        self.assertEqual(node.outputs["alignment_report"].data["filled_count"], 1)

    def test_duplicate_policy_error_reports_normalized_duplicate_keys(self):
        node = self._node()
        node.inputs["file_paths"].data = ["a.wav"]
        node.inputs["target_map"].data = {"cat/a.wav": 1, "dog/a.wav": 2}

        self.assertFalse(node.execute())
        self.assertIn("Duplicate target keys", node.error_message)
        self.assertIn("a.wav", node.error_message)

    def test_duplicate_policy_last_uses_last_target_value(self):
        node = self._node()
        node.set_parameter("duplicate_policy", "last")
        node.inputs["file_paths"].data = ["a.wav"]
        node.inputs["target_map"].data = {"cat/a.wav": 1, "dog/a.wav": 2}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_targets"].data, [2])
        self.assertEqual(node.outputs["alignment_report"].data["duplicate_count"], 1)

    def test_duplicate_file_path_keys_fail_before_alignment(self):
        node = self._node()
        node.inputs["file_paths"].data = ["cat/a.wav", "dog/a.wav"]
        node.inputs["target_map"].data = {"a.wav": 1}

        self.assertFalse(node.execute())
        self.assertIn("Duplicate file path keys", node.error_message)
        self.assertIn("a.wav", node.error_message)

    def test_empty_target_map_uses_missing_policy_instead_of_ordered_fallback(self):
        node = self._node()
        node.inputs["file_paths"].data = ["a.wav"]
        node.inputs["target_map"].data = {}
        node.inputs["targets"].data = [1]

        self.assertFalse(node.execute())
        self.assertIn("Missing targets", node.error_message)

    def test_ordered_targets_fallback_validates_sample_count(self):
        node = self._node()
        node.inputs["file_paths"].data = ["a.wav", "b.wav"]
        node.inputs["targets"].data = [1, 2]

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_targets"].data, [1, 2])
        self.assertEqual(node.outputs["aligned_file_paths"].data, ["a.wav", "b.wav"])

    def test_ordered_targets_fallback_rejects_count_mismatch(self):
        node = self._node()
        node.inputs["file_paths"].data = ["a.wav", "b.wav"]
        node.inputs["targets"].data = [1]

        self.assertFalse(node.execute())
        self.assertIn("Target count", node.error_message)


if __name__ == "__main__":
    unittest.main()
