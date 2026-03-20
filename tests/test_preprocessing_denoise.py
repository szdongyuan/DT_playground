import unittest

import numpy as np

from src.ui.i18n import tr_
from src.workflow.node_base import NodeCategory, create_node
from src.workflow.nodes.data_source import AudioData


class SpectralSubtractionNodeTests(unittest.TestCase):
    def test_spectral_subtraction_node_is_registered_under_denoise_subcategory(self):
        node = create_node("spectral_subtraction")

        self.assertIsNotNone(node)
        self.assertEqual(node.category, NodeCategory.PREPROCESSING)
        self.assertEqual(node.subcategory, tr_("Denoise"))
        self.assertIn("audio", node.inputs)
        self.assertIn("audio", node.outputs)
        self.assertIn("noise", node.inputs)

    def test_spectral_subtraction_with_noise_reference_reduces_noise_energy(self):
        node = create_node("spectral_subtraction")
        self.assertIsNotNone(node)

        sample_rate = 16000
        duration_s = 0.5
        samples = int(sample_rate * duration_s)
        time_axis = np.arange(samples) / sample_rate

        clean = 0.35 * np.sin(2 * np.pi * 440.0 * time_axis)
        noise = 0.12 * np.random.RandomState(0).normal(size=samples)
        noisy_audio = AudioData(data=clean + noise, sample_rate=sample_rate)
        noise_reference = AudioData(data=noise, sample_rate=sample_rate)

        node.inputs["audio"].data = noisy_audio
        node.inputs["noise"].data = noise_reference

        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        output_audio = node.outputs["audio"].data
        self.assertIsInstance(output_audio, AudioData)
        self.assertEqual(output_audio.sample_rate, sample_rate)
        self.assertEqual(output_audio.data.shape, noisy_audio.data.shape)

        input_residual = noisy_audio.data - clean.reshape(1, -1)
        output_residual = output_audio.data - clean.reshape(1, -1)
        self.assertLess(
            np.sqrt(np.mean(output_residual ** 2)),
            np.sqrt(np.mean(input_residual ** 2)),
        )

    def test_spectral_subtraction_without_noise_reference_supports_audio_lists(self):
        node = create_node("spectral_subtraction")
        self.assertIsNotNone(node)

        sample_rate = 8000
        samples = sample_rate // 2
        time_axis = np.arange(samples) / sample_rate
        tone = 0.2 * np.sin(2 * np.pi * 220.0 * time_axis)
        head_noise = 0.06 * np.random.RandomState(1).normal(size=samples)

        first = AudioData(data=tone + head_noise, sample_rate=sample_rate)
        second = AudioData(data=0.5 * tone + head_noise, sample_rate=sample_rate)

        node.inputs["audio"].data = [first, second]

        success = node.set_parameter("noise_duration", 0.1)[0] if node is not None else False
        self.assertTrue(success)
        self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

        outputs = node.outputs["audio"].data
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        for original, processed in zip([first, second], outputs):
            self.assertIsInstance(processed, AudioData)
            self.assertEqual(processed.sample_rate, original.sample_rate)
            self.assertEqual(processed.data.shape, original.data.shape)

    def test_spectral_subtraction_rejects_noise_lists_for_single_audio_input(self):
        node = create_node("spectral_subtraction")
        self.assertIsNotNone(node)

        sample_rate = 16000
        samples = sample_rate // 4
        tone = 0.1 * np.sin(2 * np.pi * 220.0 * np.arange(samples) / sample_rate)

        node.inputs["audio"].data = AudioData(data=tone, sample_rate=sample_rate)
        node.inputs["noise"].data = [
            AudioData(data=0.02 * np.ones(samples), sample_rate=sample_rate),
            AudioData(data=0.03 * np.ones(samples), sample_rate=sample_rate),
        ]

        self.assertFalse(node.execute())
        self.assertIn("Noise reference list", node.error_message)


if __name__ == "__main__":
    unittest.main()
