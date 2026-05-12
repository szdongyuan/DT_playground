import unittest

import numpy as np

from src.workflow.node_base import NodeCategory, create_node, get_nodes_by_category
from src.workflow.nodes.data_source import AudioData
from src.workflow.port import DataType


class DataAugmentationNodeDefinitionTests(unittest.TestCase):
    def test_add_noise_only_outputs_noisy_audio(self):
        node = create_node("add_noise")

        self.assertIsNotNone(node)
        self.assertEqual(list(node.inputs), ["audio"])
        self.assertEqual(list(node.outputs), ["audio"])
        self.assertEqual(node.outputs["audio"].display_name, "Noisy audio")
        self.assertEqual(node.outputs["audio"].data_type, DataType.AUDIO)

    def test_random_augment_is_not_registered(self):
        self.assertIsNone(create_node("random_augment"))

        augmentation_node_types = [
            node_class.node_type
            for node_class in get_nodes_by_category(NodeCategory.AUGMENTATION)
        ]
        self.assertNotIn("random_augment", augmentation_node_types)

    def test_audio_slice_only_exposes_audio_ports(self):
        node = create_node("audio_slice")

        self.assertIsNotNone(node)
        self.assertEqual(list(node.inputs), ["audio"])
        self.assertEqual(list(node.outputs), ["audio"])
        self.assertEqual(node.outputs["audio"].display_name, "Sliced audio")
        self.assertEqual(node.outputs["audio"].data_type, DataType.AUDIO)

    def test_sample_expansion_is_first_augmentation_node(self):
        augmentation_node_classes = get_nodes_by_category(NodeCategory.AUGMENTATION)

        self.assertGreater(len(augmentation_node_classes), 0)
        self.assertEqual(augmentation_node_classes[0].node_type, "sample_expansion")
        self.assertEqual(augmentation_node_classes[0].display_name, "Sample expansion")

    def test_sample_expansion_duplicates_audio_and_labels_together(self):
        node = create_node("sample_expansion")

        self.assertIsNotNone(node)
        self.assertEqual(list(node.inputs), ["audio", "labels"])
        self.assertEqual(list(node.outputs), ["audio", "labels"])
        self.assertEqual(node.inputs["audio"].data_type, DataType.AUDIO)
        self.assertEqual(node.inputs["labels"].data_type, DataType.LABEL)
        self.assertEqual(node.outputs["audio"].data_type, DataType.AUDIO)
        self.assertEqual(node.outputs["labels"].data_type, DataType.LABEL)

        ok, _ = node.set_parameter("multiplier", 3)
        self.assertTrue(ok)

        first = AudioData(
            data=np.array([[0.1, 0.2]], dtype=np.float32),
            sample_rate=16000,
            file_path="first.wav",
        )
        second = AudioData(
            data=np.array([[0.3, 0.4]], dtype=np.float32),
            sample_rate=16000,
            file_path="second.wav",
        )
        node.inputs["audio"].data = [first, second]
        node.inputs["labels"].data = ["cat", "dog"]

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        output_labels = node.outputs["labels"].data
        self.assertEqual(output_audio, [first, first, first, second, second, second])
        self.assertEqual(output_labels, ["cat", "cat", "cat", "dog", "dog", "dog"])


if __name__ == "__main__":
    unittest.main()
