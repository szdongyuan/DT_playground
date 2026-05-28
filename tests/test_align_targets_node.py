import unittest

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

    def test_stem_matching_aligns_extensionless_target_keys(self):
        node = self._node()
        node.set_parameter("match_mode", "stem")
        node.inputs["file_paths"].data = ["/data/a.wav", "/data/b.flac"]
        node.inputs["target_map"].data = {"a": 1, "b": 2}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["aligned_targets"].data, [1, 2])
        self.assertEqual(node.outputs["aligned_data"].data, [])

    def test_missing_policy_error_fails_with_clear_message(self):
        node = self._node()
        node.inputs["file_paths"].data = ["a.wav", "b.wav"]
        node.inputs["target_map"].data = {"a.wav": 1}

        self.assertFalse(node.execute())
        self.assertIn("Missing targets", node.error_message)
        self.assertIn("b.wav", node.error_message)

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
