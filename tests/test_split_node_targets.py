import unittest

from src.workflow.node_base import create_node


class SplitNodeTargetsTests(unittest.TestCase):
    def _node(self):
        node = create_node("split")
        self.assertIsNotNone(node)
        node.set_parameter("train_ratio", 0.6)
        node.set_parameter("val_ratio", 0.0)
        node.set_parameter("test_ratio", 0.4)
        node.set_parameter("shuffle", False)
        node.set_parameter("stratify", False)
        return node

    def test_data_only_split_outputs_empty_target_ports(self):
        node = self._node()
        node.inputs["data"].data = ["a", "b", "c", "d", "e"]

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["train_data"].data, ["a", "b", "c"])
        self.assertEqual(node.outputs["val_data"].data, [])
        self.assertEqual(node.outputs["test_data"].data, ["d", "e"])
        self.assertEqual(node.outputs["train_targets"].data, [])
        self.assertEqual(node.outputs["val_targets"].data, [])
        self.assertEqual(node.outputs["test_targets"].data, [])

    def test_continuous_targets_split_into_target_outputs(self):
        node = self._node()
        node.inputs["data"].data = ["a", "b", "c", "d", "e"]
        node.inputs["targets"].data = [0.1, 0.2, 0.3, 0.4, 0.5]
        node.inputs["target_metadata"].data = {"kind": "continuous"}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["train_data"].data, ["a", "b", "c"])
        self.assertEqual(node.outputs["train_targets"].data, [0.1, 0.2, 0.3])
        self.assertEqual(node.outputs["val_targets"].data, [])
        self.assertEqual(node.outputs["test_data"].data, ["d", "e"])
        self.assertEqual(node.outputs["test_targets"].data, [0.4, 0.5])

    def test_continuous_targets_do_not_use_direct_stratify_by_default(self):
        node = create_node("split")
        self.assertIsNotNone(node)
        node.set_parameter("train_ratio", 0.8)
        node.set_parameter("val_ratio", 0.0)
        node.set_parameter("test_ratio", 0.2)
        node.set_parameter("shuffle", True)
        node.set_parameter("stratify", True)
        node.inputs["data"].data = list(range(10))
        node.inputs["targets"].data = [float(i) + 0.5 for i in range(10)]
        node.inputs["target_metadata"].data = {"kind": "continuous"}

        self.assertTrue(node.execute(), msg=node.error_message)

        train_data = node.outputs["train_data"].data
        train_targets = node.outputs["train_targets"].data
        test_data = node.outputs["test_data"].data
        test_targets = node.outputs["test_targets"].data
        self.assertEqual(len(train_data), 8)
        self.assertEqual(len(test_data), 2)
        self.assertEqual(train_targets, [float(item) + 0.5 for item in train_data])
        self.assertEqual(test_targets, [float(item) + 0.5 for item in test_data])

    def test_categorical_targets_can_use_metadata_guided_stratify(self):
        node = create_node("split")
        self.assertIsNotNone(node)
        node.set_parameter("train_ratio", 0.8)
        node.set_parameter("val_ratio", 0.0)
        node.set_parameter("test_ratio", 0.2)
        node.set_parameter("shuffle", True)
        node.set_parameter("stratify", True)
        node.inputs["data"].data = list(range(20))
        node.inputs["targets"].data = [0] * 10 + [1] * 10
        node.inputs["target_metadata"].data = {"kind": "categorical"}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["test_targets"].data.count(0), 2)
        self.assertEqual(node.outputs["test_targets"].data.count(1), 2)

    def test_categorical_targets_with_shuffle_false_do_not_request_stratify(self):
        node = create_node("split")
        self.assertIsNotNone(node)
        node.set_parameter("train_ratio", 0.6)
        node.set_parameter("val_ratio", 0.0)
        node.set_parameter("test_ratio", 0.4)
        node.set_parameter("shuffle", False)
        node.set_parameter("stratify", True)
        node.inputs["data"].data = ["a", "b", "c", "d", "e"]
        node.inputs["targets"].data = [0, 0, 1, 1, 1]
        node.inputs["target_metadata"].data = {"kind": "categorical"}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(node.outputs["train_targets"].data, [0, 0, 1])
        self.assertEqual(node.outputs["test_targets"].data, [1, 1])

    def test_categorical_targets_with_rare_classes_fall_back_from_stratify(self):
        node = create_node("split")
        self.assertIsNotNone(node)
        node.set_parameter("train_ratio", 0.8)
        node.set_parameter("val_ratio", 0.0)
        node.set_parameter("test_ratio", 0.2)
        node.set_parameter("shuffle", True)
        node.set_parameter("stratify", True)
        node.inputs["data"].data = list(range(10))
        node.inputs["targets"].data = [0] * 9 + [1]
        node.inputs["target_metadata"].data = {"kind": "categorical"}

        self.assertTrue(node.execute(), msg=node.error_message)

        self.assertEqual(len(node.outputs["train_targets"].data), 8)
        self.assertEqual(len(node.outputs["test_targets"].data), 2)

    def test_dict_targets_fail_clearly_instead_of_splitting_keys(self):
        node = self._node()
        node.inputs["data"].data = ["a", "b"]
        node.inputs["targets"].data = {"a.wav": 1.0, "b.wav": 2.0}

        self.assertFalse(node.execute())
        self.assertIn("ordered target list", node.error_message)

    def test_target_length_mismatch_fails_clearly(self):
        node = self._node()
        node.inputs["data"].data = ["a", "b", "c"]
        node.inputs["targets"].data = [1.0, 2.0]
        node.inputs["target_metadata"].data = {"kind": "continuous"}

        self.assertFalse(node.execute())
        self.assertIn("Target count", node.error_message)

    def test_split_node_no_longer_exposes_legacy_label_ports(self):
        node = self._node()

        self.assertNotIn("labels", node.inputs)
        self.assertNotIn("train_labels", node.outputs)
        self.assertNotIn("val_labels", node.outputs)
        self.assertNotIn("test_labels", node.outputs)


if __name__ == "__main__":
    unittest.main()
