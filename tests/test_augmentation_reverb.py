import unittest

import numpy as np

from src.workflow.node_base import NodeCategory, create_node
from src.workflow.nodes.data_source import AudioData


class ReverbNodeTests(unittest.TestCase):
    def _create_ir_audio(self, sample_rate: int = 16000) -> AudioData:
        ir_data = np.array([[1.0, 0.5, 0.25, 0.125]], dtype=np.float32)
        return AudioData(data=ir_data, sample_rate=sample_rate)

    def test_reverb_node_is_registered_under_augmentation(self):
        node = create_node("reverb")

        self.assertIsNotNone(node)
        self.assertEqual(node.category, NodeCategory.AUGMENTATION)
        self.assertIn("audio", node.inputs)
        self.assertIn("ir", node.inputs)
        self.assertIn("audio", node.outputs)
        self.assertFalse(node.inputs["ir"].required)
        self.assertIn("decay", node.parameters)
        self.assertIn("wet_mix", node.parameters)
        self.assertIn("random_wet_mix", node.parameters)
        self.assertIn("wet_mix_min", node.parameters)
        self.assertIn("wet_mix_max", node.parameters)
        self.assertNotIn("random_decay", node.parameters)
        self.assertNotIn("decay_min", node.parameters)
        self.assertNotIn("decay_max", node.parameters)
        self.assertNotIn("ir_file", node.parameters)
        self.assertNotIn("room_size", node.parameters)
        self.assertNotIn("pre_delay_ms", node.parameters)

    def test_reverb_node_preserves_shape_and_changes_signal_without_ir_input(self):
        node = create_node("reverb")
        self.assertIsNotNone(node)

        for name, value in {
            "decay": 0.4,
            "wet_mix": 0.35,
        }.items():
            ok, _ = node.set_parameter(name, value)
            self.assertTrue(ok, msg=name)

        samples = 1600
        impulse = np.zeros((2, samples), dtype=np.float32)
        impulse[:, 0] = 1.0
        node.inputs["audio"].data = AudioData(data=impulse, sample_rate=16000)

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        self.assertIsInstance(output_audio, AudioData)
        self.assertEqual(output_audio.sample_rate, 16000)
        self.assertEqual(output_audio.data.shape, impulse.shape)
        self.assertFalse(np.allclose(output_audio.data, impulse))

    def test_reverb_node_accepts_external_ir_audio_input(self):
        node = create_node("reverb")
        self.assertIsNotNone(node)

        for name, value in {
            "decay": 0.25,
            "wet_mix": 0.3,
        }.items():
            ok, _ = node.set_parameter(name, value)
            self.assertTrue(ok, msg=name)

        node.inputs["ir"].data = self._create_ir_audio(sample_rate=8000)
        impulse = np.zeros((1, 800), dtype=np.float32)
        impulse[:, 0] = 1.0
        node.inputs["audio"].data = AudioData(data=impulse, sample_rate=8000)

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        self.assertIsInstance(output_audio, AudioData)
        self.assertEqual(output_audio.data.shape, impulse.shape)
        self.assertFalse(np.allclose(output_audio.data, impulse))

    def test_reverb_node_supports_random_wet_mix_ranges(self):
        node = create_node("reverb")
        self.assertIsNotNone(node)

        for name, value in {
            "decay": 0.25,
            "random_wet_mix": True,
            "wet_mix_min": 0.3,
            "wet_mix_max": 0.3,
        }.items():
            ok, _ = node.set_parameter(name, value)
            self.assertTrue(ok, msg=name)

        impulse = np.zeros((1, 800), dtype=np.float32)
        impulse[:, 0] = 1.0
        node.inputs["audio"].data = AudioData(data=impulse, sample_rate=8000)

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        self.assertIsInstance(output_audio, AudioData)
        self.assertEqual(output_audio.data.shape, impulse.shape)
        self.assertFalse(np.allclose(output_audio.data, impulse))

    def test_reverb_node_supports_audio_lists_and_ir_lists(self):
        node = create_node("reverb")
        self.assertIsNotNone(node)

        ok, _ = node.set_parameter("wet_mix", 0.25)
        self.assertTrue(ok)
        node.inputs["ir"].data = [
            self._create_ir_audio(sample_rate=8000),
            AudioData(data=np.array([[1.0, 0.2, 0.05, 0.01]], dtype=np.float32), sample_rate=8000),
        ]

        first = AudioData(
            data=np.array([[1.0] + [0.0] * 511], dtype=np.float32),
            sample_rate=8000,
        )
        second = AudioData(
            data=np.array([[0.5] + [0.0] * 511], dtype=np.float32),
            sample_rate=8000,
        )
        node.inputs["audio"].data = [first, second]

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        outputs = node.outputs["audio"].data
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        for original, processed in zip([first, second], outputs):
            self.assertIsInstance(processed, AudioData)
            self.assertEqual(processed.sample_rate, original.sample_rate)
            self.assertEqual(processed.data.shape, original.data.shape)


if __name__ == "__main__":
    unittest.main()
