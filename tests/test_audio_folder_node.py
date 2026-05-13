import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.workflow.node_base import create_node


class AudioFolderNodeTests(unittest.TestCase):
    def _create_audio_tree(self, root: Path) -> list[Path]:
        files = [
            root / "cat" / "a.wav",
            root / "cat" / "b.wav",
            root / "dog" / "c.wav",
            root / "dog" / "d.wav",
        ]
        for file_path in files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"placeholder")
        return files

    def _fake_librosa_load(self, file_path: str, sr: int, mono: bool):
        sample_value = float(sum(ord(char) for char in Path(file_path).name))
        return np.array([[sample_value, sample_value + 1.0]], dtype=np.float32), sr

    def test_audio_folder_exposes_selection_mode_with_existing_max_files_default(self):
        node = create_node("audio_folder")

        self.assertIsNotNone(node)
        self.assertIn("max_files", node.parameters)
        self.assertEqual(node.get_parameter("max_files"), 0)
        self.assertIn("selection_mode", node.parameters)
        self.assertEqual(node.get_parameter("selection_mode"), "first_n")
        self.assertEqual(node.parameters["selection_mode"].choices, ["first_n", "random_n"])

    def test_max_files_zero_loads_all_audio_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected_files = self._create_audio_tree(root)
            node = create_node("audio_folder")
            node.set_parameter("folder_path", str(root))
            node.set_parameter("max_files", 0)

            with patch("src.workflow.nodes.data_source.librosa.load", side_effect=self._fake_librosa_load):
                self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

            file_paths = node.outputs["file_paths"].data
            self.assertEqual(file_paths, [str(path) for path in expected_files])
            self.assertEqual(len(node.outputs["audio"].data), len(expected_files))
            self.assertEqual(len(node.outputs["labels"].data), len(expected_files))

    def test_first_n_selection_loads_first_files_and_keeps_outputs_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected_files = self._create_audio_tree(root)[:2]
            node = create_node("audio_folder")
            node.set_parameter("folder_path", str(root))
            node.set_parameter("max_files", 2)
            node.set_parameter("selection_mode", "first_n")

            with patch("src.workflow.nodes.data_source.librosa.load", side_effect=self._fake_librosa_load):
                self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

            audio = node.outputs["audio"].data
            labels = node.outputs["labels"].data
            file_paths = node.outputs["file_paths"].data

            self.assertEqual(file_paths, [str(path) for path in expected_files])
            self.assertEqual([item.file_path for item in audio], file_paths)
            self.assertEqual(labels, [0, 0])

    def test_random_selection_samples_from_all_files_and_keeps_outputs_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            all_files = self._create_audio_tree(root)
            selected_files = [all_files[2], all_files[0]]
            node = create_node("audio_folder")
            node.set_parameter("folder_path", str(root))
            node.set_parameter("max_files", 2)
            node.set_parameter("selection_mode", "random_n")

            with (
                patch("src.workflow.nodes.data_source.random.sample", return_value=selected_files) as sample,
                patch("src.workflow.nodes.data_source.librosa.load", side_effect=self._fake_librosa_load),
            ):
                self.assertTrue(node.execute(), msg=getattr(node, "error_message", ""))

            file_paths = node.outputs["file_paths"].data
            audio = node.outputs["audio"].data
            labels = node.outputs["labels"].data

            sample.assert_called_once_with(all_files, 2)
            self.assertEqual(file_paths, [str(path) for path in selected_files])
            self.assertEqual([item.file_path for item in audio], file_paths)
            self.assertEqual(labels, [0, 1])


if __name__ == "__main__":
    unittest.main()
