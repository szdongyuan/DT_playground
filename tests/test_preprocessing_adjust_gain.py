import unittest

import numpy as np

from src.workflow.node_base import NodeCategory, create_node
from src.workflow.nodes.data_source import AudioData


class AdjustGainNodeTests(unittest.TestCase):
    def test_adjust_gain_node_is_registered_under_preprocessing(self):
        node = create_node("adjust_gain")

        self.assertIsNotNone(node)
        self.assertEqual(node.category, NodeCategory.PREPROCESSING)
        self.assertIn("audio", node.inputs)
        self.assertIn("audio", node.outputs)

    def test_adjust_gain_without_clip_protection_clips_and_logs_warning(self):
        node = create_node("adjust_gain")
        self.assertIsNotNone(node)

        ok, _ = node.set_parameter("gain_db", 6.0)
        self.assertTrue(ok)
        ok, _ = node.set_parameter("clip_protection", False)
        self.assertTrue(ok)

        node.inputs["audio"].data = AudioData(
            data=np.array([[0.8, 0.4, -0.8]], dtype=np.float32),
            sample_rate=16000,
        )

        with self.assertLogs("src.workflow.nodes.preprocessing", level="WARNING") as captured:
            self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        self.assertIsInstance(output_audio, AudioData)
        self.assertTrue(np.max(np.abs(output_audio.data)) <= 1.0)
        self.assertAlmostEqual(output_audio.data[0, 0], 1.0, places=6)
        self.assertAlmostEqual(output_audio.data[0, 1], 0.4 * (10 ** (6.0 / 20)), places=6)
        self.assertAlmostEqual(output_audio.data[0, 2], -1.0, places=6)
        self.assertTrue(any("clipp" in message.lower() for message in captured.output))

    def test_adjust_gain_with_clip_protection_rescales_audio(self):
        node = create_node("adjust_gain")
        self.assertIsNotNone(node)

        ok, _ = node.set_parameter("gain_db", 6.0)
        self.assertTrue(ok)
        ok, _ = node.set_parameter("clip_protection", True)
        self.assertTrue(ok)

        node.inputs["audio"].data = AudioData(
            data=np.array([[0.8, 0.4, -0.8]], dtype=np.float32),
            sample_rate=22050,
        )

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        self.assertIsInstance(output_audio, AudioData)
        self.assertEqual(output_audio.sample_rate, 22050)
        self.assertAlmostEqual(np.max(np.abs(output_audio.data)), 1.0, places=6)
        self.assertAlmostEqual(output_audio.data[0, 0], 1.0, places=6)
        self.assertAlmostEqual(output_audio.data[0, 1], 0.5, places=6)
        self.assertAlmostEqual(output_audio.data[0, 2], -1.0, places=6)

    def test_adjust_gain_supports_audio_lists(self):
        node = create_node("adjust_gain")
        self.assertIsNotNone(node)

        ok, _ = node.set_parameter("gain_db", -6.0)
        self.assertTrue(ok)

        first = AudioData(data=np.array([[0.8, -0.8]], dtype=np.float32), sample_rate=8000)
        second = AudioData(data=np.array([[0.4, -0.4]], dtype=np.float32), sample_rate=8000)
        node.inputs["audio"].data = [first, second]

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        outputs = node.outputs["audio"].data
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        expected_scale = 10 ** (-6.0 / 20)
        self.assertAlmostEqual(outputs[0].data[0, 0], 0.8 * expected_scale, places=6)
        self.assertAlmostEqual(outputs[1].data[0, 0], 0.4 * expected_scale, places=6)


if __name__ == "__main__":
    unittest.main()
