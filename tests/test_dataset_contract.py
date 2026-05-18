import unittest

from src.workflow.dataset import DataRecord, DatasetBundle, TaskSpec, TaskType
from src.workflow.port import DataType


class DatasetContractTests(unittest.TestCase):
    def test_task_type_values_cover_supported_ml_tasks(self):
        self.assertEqual(TaskType.CLASSIFICATION.value, "classification")
        self.assertEqual(TaskType.REGRESSION.value, "regression")
        self.assertEqual(TaskType.CLUSTERING.value, "clustering")
        self.assertEqual(TaskType.DIMENSIONALITY_REDUCTION.value, "dimensionality_reduction")
        self.assertEqual(TaskType.GENERATION.value, "generation")
        self.assertEqual(TaskType.DECISION.value, "decision")

    def test_data_record_defaults_allow_unlabeled_samples(self):
        record = DataRecord(record_id="sample-001", input="payload")

        self.assertEqual(record.record_id, "sample-001")
        self.assertEqual(record.input, "payload")
        self.assertIsNone(record.target)
        self.assertEqual(record.annotations, {})
        self.assertEqual(record.metadata, {})
        self.assertEqual(record.quality, {})

    def test_dataset_bundle_defaults_to_classification_task(self):
        bundle = DatasetBundle(records=[DataRecord(record_id="sample-001", input="payload")])

        self.assertEqual(len(bundle.records), 1)
        self.assertIsInstance(bundle.task_spec, TaskSpec)
        self.assertEqual(bundle.task_spec.task_type, TaskType.CLASSIFICATION)
        self.assertTrue(bundle.task_spec.target_required)
        self.assertEqual(bundle.schema.input_schema, {})
        self.assertEqual(bundle.stats.total_records, 1)
        self.assertEqual(bundle.lineage.sources, [])

    def test_dataset_data_type_connects_to_dataset_and_any_only(self):
        self.assertTrue(DataType.is_compatible(DataType.DATASET, DataType.DATASET))
        self.assertTrue(DataType.is_compatible(DataType.DATASET, DataType.ANY))
        self.assertTrue(DataType.is_compatible(DataType.ANY, DataType.DATASET))
        self.assertFalse(DataType.is_compatible(DataType.DATASET, DataType.AUDIO))
        self.assertFalse(DataType.is_compatible(DataType.AUDIO, DataType.DATASET))
        self.assertFalse(DataType.is_compatible(DataType.DATASET, DataType.LABEL))


if __name__ == "__main__":
    unittest.main()
