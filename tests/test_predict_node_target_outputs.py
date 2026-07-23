import unittest

import numpy as np

from src.workflow.node_base import create_node


class PredictNodeTargetOutputTests(unittest.TestCase):
    def setUp(self):
        self.node = create_node("predict")

    def test_output_type_choices_include_raw_regression_and_vector(self):
        choices = self.node.parameters["output_type"].choices

        self.assertIn("raw", choices)
        self.assertIn("regression", choices)
        self.assertIn("vector", choices)

    def test_raw_output_returns_predictions_without_argmax_or_rounding(self):
        predictions = np.array([[0.25, 0.75], [0.6, 0.4]], dtype=np.float32)

        result = self.node._process_output(predictions, [], "raw")

        self.assertIs(result, predictions)

    def test_regression_output_returns_scalar_list_for_single_output(self):
        predictions = np.array([[1.25], [2.5]], dtype=np.float32)

        result = self.node._process_output(predictions, [], "regression")

        self.assertEqual(result, [1.25, 2.5])

    def test_regression_output_returns_multi_output_rows(self):
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result = self.node._process_output(predictions, [], "regression")

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_label_output_keeps_existing_argmax_behavior(self):
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)

        result = self.node._process_output(predictions, [], "label")

        self.assertEqual(result, [1, 0])


if __name__ == "__main__":
    unittest.main()
