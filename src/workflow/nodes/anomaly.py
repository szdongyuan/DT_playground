# -*- coding: utf-8 -*-
"""Workflow nodes and runtime contracts for unsupervised anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.ui.i18n import tr_

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType


logger = logging.getLogger(__name__)


@dataclass
class FeatureMatrixData:
    """Fixed-length sample matrix with stable source alignment."""

    matrix: np.ndarray
    sample_ids: List[str]
    source_items: List[Any] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.ndim != 2:
            raise ValueError(tr_("Feature matrix must be two-dimensional"))
        if len(self.sample_ids) != len(self.matrix):
            raise ValueError(tr_("Sample ID count must match feature matrix rows"))
        if self.source_items and len(self.source_items) != len(self.matrix):
            raise ValueError(tr_("Source item count must match feature matrix rows"))
        if self.provenance and len(self.provenance) != len(self.matrix):
            raise ValueError(tr_("Provenance count must match feature matrix rows"))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(tr_("Sample IDs must be unique"))
        if not self.matrix.size or not np.isfinite(self.matrix).all():
            raise ValueError(tr_("Feature matrix contains empty, NaN, or infinite values"))

    @property
    def parent_ids(self):
        return [row["parent_sample_id"] for row in self.provenance] if self.provenance else list(self.sample_ids)


@dataclass
class AnomalyModelArtifact:
    """Serializable detector, fitted preprocessing, and score calibration."""

    algorithm: str
    estimator: Any
    scaler: Any
    feature_count: int
    score_bounds: Tuple[float, float]
    reference_scores: np.ndarray
    training_summary: Dict[str, Any] = field(default_factory=dict)
    feature_schema: Dict[str, Any] = field(default_factory=dict)
    preprocessing_state: Any = None
    reference_parent_ids: List[str] = field(default_factory=list)
    reference_sample_ids: List[str] = field(default_factory=list)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Validate and transform a matrix exactly as during fitting."""
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(tr_("Feature matrix must be two-dimensional"))
        if values.shape[1] != self.feature_count:
            raise ValueError(
                tr_("Feature count mismatch: expected {expected}, got {actual}").format(
                    expected=self.feature_count,
                    actual=values.shape[1],
                )
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(tr_("Feature matrix contains NaN or infinite values"))
        return self.scaler.transform(values) if self.scaler is not None else values

    def score(self, matrix: np.ndarray, sample_ids=None, parent_ids=None, batch_size=256) -> np.ndarray:
        """Return raw scores where larger values are always more anomalous."""
        transformed = self.transform(matrix)
        if self.algorithm == "knn":
            return self.estimator.score(transformed, sample_ids, parent_ids, batch_size)
        if self.algorithm == "autoencoder":
            from ..anomaly_backends import reconstruction_scores
            return reconstruction_scores(self.estimator, transformed, batch_size)
        return -np.asarray(self.estimator.score_samples(transformed), dtype=np.float64)

    def normalize_scores(self, scores: Sequence[float]) -> np.ndarray:
        """Map raw scores to a stable, clipped 0..100 display scale."""
        values = np.asarray(scores, dtype=np.float64)
        low, high = self.score_bounds
        span = max(float(high - low), abs(float(high)) * 1e-6, 1e-9)
        return np.clip((values - low) / span * 100.0, 0.0, 100.0)

    def save(self, path: str):
        """Persist a versioned bundle without overwriting an existing file."""
        from ..anomaly_io import save_model

        return save_model(self, path)

    @classmethod
    def load(cls, path: str, *, trusted=False, format="anomaly_zip") -> "AnomalyModelArtifact":
        """Load explicitly trusted state; legacy files require an explicit format."""
        from ..anomaly_io import load_model

        return load_model(path, trusted=trusted, format=format)


@dataclass
class AnomalyScoresData:
    """Aligned raw and normalized anomaly scores."""

    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    sample_ids: List[str]
    source_items: List[Any] = field(default_factory=list)
    reference_normalized_scores: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.raw_scores = np.asarray(self.raw_scores, dtype=np.float64)
        self.normalized_scores = np.asarray(self.normalized_scores, dtype=np.float64)
        self.reference_normalized_scores = np.asarray(
            self.reference_normalized_scores,
            dtype=np.float64,
        )
        count = len(self.raw_scores)
        if len(self.normalized_scores) != count or len(self.sample_ids) != count:
            raise ValueError(tr_("Score arrays and sample IDs must have the same length"))
        if self.source_items and len(self.source_items) != count:
            raise ValueError(tr_("Source item count must match score count"))


@dataclass
class AnomalyResultData:
    """Thresholded anomaly decisions aligned to original samples."""

    scores: AnomalyScoresData
    threshold: float
    attention_threshold: float
    severities: List[str]
    is_anomaly: np.ndarray
    strategy: str

    def __post_init__(self):
        self.is_anomaly = np.asarray(self.is_anomaly, dtype=bool)
        count = len(self.scores.raw_scores)
        if len(self.severities) != count or len(self.is_anomaly) != count:
            raise ValueError(tr_("Decision count must match score count"))


@register_node
class FeatureVectorizerNode(BaseNode):
    """Convert per-sample workflow features into fixed-length vectors."""

    node_type = "feature_vectorizer"
    display_name = tr_("Feature vectorizer")
    category = NodeCategory.FEATURE
    subcategory = tr_("Feature post-processing")
    subcategory_order = 40
    palette_order = 10
    description = tr_("Convert features into aligned fixed-length vectors for classical machine learning")
    icon = "🧮"

    def _setup_ports(self):
        self.add_input("features", DataType.ANY, tr_("Features"))
        self.add_output("feature_matrix", DataType.FEATURE_MATRIX, tr_("Feature matrix"))

    def _setup_parameters(self):
        self.add_parameter("mode", "choice", "whole_sample", display_name=tr_("Vectorization mode"),
                           choices=["whole_sample", "sliding_window"])
        self.add_parameter("window_length", "int", 5, display_name=tr_("Window length (frames)"), min_value=1)
        self.add_parameter("window_stride", "int", 5, display_name=tr_("Window stride (frames)"), min_value=1)
        self.add_parameter("tail_policy", "choice", "include_last", display_name=tr_("Tail policy"),
                           choices=["include_last", "drop"])
        self.add_parameter("short_policy", "choice", "error", display_name=tr_("Short sample policy"),
                           choices=["error", "pad_edge"])
        self.add_parameter(
            "aggregation",
            "choice",
            "mean_std",
            display_name=tr_("Aggregation"),
            choices=["mean_std", "mean", "global_max", "flatten"],
            description=tr_("How to reduce each sample to a fixed-length vector"),
        )

    def execute(self) -> bool:
        self.outputs["feature_matrix"].clear()
        try:
            samples = self._as_samples(self.get_input_data("features"))
            if not samples:
                self.error_message = tr_("No features provided")
                return False

            aggregation = self.get_parameter("aggregation")
            from ..feature_contract import temporal_layout
            mode = self.get_parameter("mode")
            vectors, sample_ids, sources, provenance = [], [], [], []
            layout = None
            for index, item in enumerate(samples):
                parent = self._sample_id(item, index)
                array = self._array_for_item(item)
                if hasattr(item, "feature_type"):
                    current = temporal_layout(item)
                    if layout is not None and current != layout:
                        raise ValueError(tr_("Temporal feature layouts must match"))
                    layout = current
                if mode == "whole_sample":
                    vectors.append(self._vectorize(item, aggregation))
                    sample_ids.append(parent)
                    sources.append(item)
                    continue
                if array.ndim < 2:
                    raise ValueError(tr_("Windowing requires a temporal feature axis"))
                width, stride = int(self.get_parameter("window_length")), int(self.get_parameter("window_stride"))
                if width < 1 or stride < 1:
                    raise ValueError(tr_("Window length and stride must be positive"))
                frames = array.shape[-1]
                if frames < width:
                    if self.get_parameter("short_policy") == "error" or frames == 0:
                        raise ValueError(tr_("Feature sample is shorter than the window"))
                    array = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, width - frames)], mode="edge")
                starts = list(range(0, array.shape[-1] - width + 1, stride))
                last = array.shape[-1] - width
                if self.get_parameter("tail_policy") == "include_last" and starts[-1] != last:
                    starts.append(last)
                for window_index, start in enumerate(starts):
                    vectors.append(self._vectorize(array[..., start:start + width], aggregation))
                    sample_ids.append(f"{parent}::window:{start}:{width}")
                    sources.append(item)
                    row = {"parent_sample_id": parent, "window_index": window_index,
                           "start_frame": start, "end_frame": min(start + width, frames),
                           "padded_frames": max(0, start + width - frames)}
                    if getattr(item, "sample_rate", 0) and getattr(item, "hop_length", 0):
                        seconds = item.hop_length / item.sample_rate
                        row.update(start_seconds=start * seconds, end_seconds=min(start + width, frames) * seconds)
                    provenance.append(row)
            vector_sizes = {vector.size for vector in vectors}
            if len(vector_sizes) != 1:
                self.error_message = tr_("Vectorized samples have inconsistent feature counts")
                return False

            matrix = np.vstack(vectors).astype(np.float64, copy=False)
            if matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
                self.error_message = tr_("Feature matrix contains empty, NaN, or infinite values")
                return False

            schema = {
                "aggregation": aggregation,
                "feature_count": int(matrix.shape[1]),
            }
            if layout is not None:
                schema["layout"] = layout
            if mode == "sliding_window":
                schema.update(mode=mode, window_length=width, window_stride=stride,
                              tail_policy=self.get_parameter("tail_policy"),
                              short_policy=self.get_parameter("short_policy"), flatten_order="C")
            result = FeatureMatrixData(matrix, sample_ids, sources, schema, provenance)
            self.set_output_data("feature_matrix", result)
            self.report_status(
                tr_("Feature matrix ready: {samples} samples, {features} features").format(
                    samples=matrix.shape[0],
                    features=matrix.shape[1],
                )
            )
            return True
        except Exception as exc:
            self.error_message = tr_("Feature vectorization failed: {error}").format(error=str(exc))
            logger.exception("Feature vectorization failed")
            return False

    @staticmethod
    def _as_samples(data: Any) -> List[Any]:
        if data is None:
            return []
        if isinstance(data, (list, tuple)):
            return list(data)
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            return [row for row in data]
        return [data]

    @staticmethod
    def _array_for_item(item: Any) -> np.ndarray:
        value = item if isinstance(item, np.ndarray) else getattr(item, "data", item)
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            raise ValueError(tr_("Feature sample is empty"))
        return array

    @classmethod
    def _vectorize(cls, item: Any, aggregation: str) -> np.ndarray:
        array = cls._array_for_item(item)
        if array.ndim <= 1 or aggregation == "flatten":
            return array.reshape(-1)
        if aggregation == "mean":
            return np.mean(array, axis=-1).reshape(-1)
        if aggregation == "global_max":
            return np.max(array, axis=-1).reshape(-1)
        means = np.mean(array, axis=-1).reshape(-1)
        stds = np.std(array, axis=-1).reshape(-1)
        return np.concatenate([means, stds])

    @staticmethod
    def _sample_id(item: Any, index: int) -> str:
        source = getattr(item, "source_file", "") or getattr(item, "file_path", "")
        return str(source) if source else f"sample_{index:04d}"


@register_node
class AnomalyDetectorTrainerNode(BaseNode):
    """Fit an Isolation Forest from mostly-normal unlabeled samples."""

    node_type = "anomaly_detector_trainer"
    display_name = tr_("Anomaly detector trainer")
    category = NodeCategory.TRAINING
    subcategory = tr_("Model training")
    subcategory_order = 10
    palette_order = 30
    description = tr_("Fit an unsupervised anomaly detector on mostly-normal reference data")
    icon = "🌲"

    def _setup_ports(self):
        self.add_input("feature_matrix", DataType.FEATURE_MATRIX, tr_("Reference feature matrix"))
        self.add_input("calibration_features", DataType.FEATURE_MATRIX, tr_("Independent normal calibration features"), required=False)
        self.add_input("preprocessing_state", DataType.ANY, tr_("Preprocessing state"), required=False)
        self.add_output("anomaly_model", DataType.ANOMALY_MODEL, tr_("Anomaly model"))
        self.add_output("reference_scores", DataType.ANOMALY_SCORES, tr_("Reference scores"))
        self.add_output("training_summary", DataType.METRICS, tr_("Training summary"))

    def _setup_parameters(self):
        self.add_parameter(
            "algorithm",
            "choice",
            "isolation_forest",
            display_name=tr_("Algorithm"),
            choices=["isolation_forest", "knn"],
        )
        self.add_parameter("n_neighbors", "int", 5, display_name=tr_("Number of neighbors"), min_value=1)
        self.add_parameter("distance", "choice", "euclidean", display_name=tr_("Distance metric"),
                           choices=["euclidean", "cosine"])
        self.add_parameter("reference_exclusion", "choice", "parent", display_name=tr_("Reference exclusion"),
                           choices=["parent", "sample"])
        self.add_parameter(
            "scaling",
            "choice",
            "standard",
            display_name=tr_("Feature scaling"),
            choices=["standard", "robust", "none"],
        )
        self.add_parameter(
            "contamination",
            "float",
            0.01,
            display_name=tr_("Expected anomaly ratio"),
            min_value=0.001,
            max_value=0.5,
        )
        self.add_parameter(
            "n_estimators",
            "int",
            200,
            display_name=tr_("Number of trees"),
            min_value=10,
            max_value=2000,
        )
        self.add_parameter(
            "max_samples",
            "float",
            1.0,
            display_name=tr_("Sample fraction per tree"),
            min_value=0.01,
            max_value=1.0,
        )
        self.add_parameter(
            "random_seed",
            "int",
            42,
            display_name=tr_("Random seed"),
            min_value=0,
        )

    def execute(self) -> bool:
        for port in self.outputs.values():
            port.clear()
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import RobustScaler, StandardScaler

            features = self.get_input_data("feature_matrix")
            if not isinstance(features, FeatureMatrixData):
                self.error_message = tr_("Anomaly trainer requires feature matrix data")
                return False
            if len(features.matrix) < 2:
                self.error_message = tr_("At least two reference samples are required")
                return False
            if not np.all(np.isfinite(features.matrix)):
                self.error_message = tr_("Feature matrix contains NaN or infinite values")
                return False

            scaling = self.get_parameter("scaling")
            fingerprint = features.schema.get("preprocessing") or features.schema.get("layout", {}).get("preprocessing")
            if fingerprint and scaling != "none":
                raise ValueError(tr_("Already standardized features require scaling=none"))
            state = self.get_input_data("preprocessing_state")
            if state is not None and state.fingerprint != fingerprint:
                raise ValueError(tr_("Preprocessing state differs from the feature schema"))
            scaler = None
            if scaling == "standard":
                scaler = StandardScaler()
            elif scaling == "robust":
                scaler = RobustScaler()

            transformed = scaler.fit_transform(features.matrix) if scaler is not None else features.matrix
            algorithm = self.get_parameter("algorithm")
            estimator = IsolationForest(
                n_estimators=self.get_parameter("n_estimators"),
                max_samples=self.get_parameter("max_samples"),
                contamination=self.get_parameter("contamination"),
                random_state=self.get_parameter("random_seed"),
                n_jobs=-1,
            )
            if algorithm == "knn":
                from ..anomaly_backends import KNNReference
                estimator = KNNReference(transformed, list(features.sample_ids), features.parent_ids,
                                         int(self.get_parameter("n_neighbors")), self.get_parameter("distance"),
                                         self.get_parameter("reference_exclusion"))
                estimator.validate()
            else:
                estimator.fit(transformed)
            reference = self.get_input_data("calibration_features")
            if reference is not None:
                if not isinstance(reference, FeatureMatrixData) or reference.schema != features.schema:
                    raise ValueError(tr_("Feature schema differs from the model training schema"))
                if set(reference.parent_ids) & set(features.parent_ids):
                    raise ValueError(tr_("Independent calibration must not overlap training files"))
            else:
                reference = features
            if len(reference.matrix) < 2:
                raise ValueError(tr_("At least two reference samples are required"))
            reference_matrix = scaler.transform(reference.matrix) if scaler is not None else reference.matrix
            raw_scores = (estimator.score(reference_matrix, reference.sample_ids, reference.parent_ids)
                          if algorithm == "knn" else -np.asarray(estimator.score_samples(reference_matrix), dtype=float))
            low, high = np.quantile(raw_scores, [0.01, 0.99])
            contamination = float(self.get_parameter("contamination"))
            suspected_threshold = float(np.quantile(raw_scores, 1.0 - contamination))
            suspected_count = int(np.count_nonzero(raw_scores >= suspected_threshold))
            summary = {
                "algorithm": algorithm,
                "sample_count": int(features.matrix.shape[0]),
                "feature_count": int(features.matrix.shape[1]),
                "contamination": contamination,
                "suspected_reference_count": suspected_count,
                "raw_score_min": float(np.min(raw_scores)),
                "raw_score_max": float(np.max(raw_scores)),
                "reference_role": "training_exclusion" if reference is features and algorithm == "knn" else
                                  "training" if reference is features else "independent_normal_calibration",
            }
            artifact = AnomalyModelArtifact(
                algorithm=algorithm,
                estimator=estimator,
                scaler=scaler,
                feature_count=features.matrix.shape[1],
                score_bounds=(float(low), float(high)),
                reference_scores=raw_scores,
                training_summary=summary,
                feature_schema=dict(features.schema),
                preprocessing_state=state,
                reference_parent_ids=reference.parent_ids,
                reference_sample_ids=list(reference.sample_ids),
            )
            normalized = artifact.normalize_scores(raw_scores)
            reference_scores = AnomalyScoresData(
                raw_scores=raw_scores,
                normalized_scores=normalized,
                sample_ids=list(reference.sample_ids),
                source_items=list(reference.source_items),
                reference_normalized_scores=normalized,
                metadata={"source": "reference", "algorithm": artifact.algorithm,
                          "granularity": "window" if reference.provenance else "sample",
                          "provenance": reference.provenance},
            )
            self.set_output_data("anomaly_model", artifact)
            self.set_output_data("reference_scores", reference_scores)
            self.set_output_data("training_summary", summary)
            self.report_status(
                tr_("Anomaly detector fitted: {samples} reference samples").format(
                    samples=features.matrix.shape[0]
                )
            )
            return True
        except Exception as exc:
            self.error_message = tr_("Anomaly detector training failed: {error}").format(error=str(exc))
            logger.exception("Anomaly detector training failed")
            return False


@register_node
class AnomalyScorerNode(BaseNode):
    """Score samples without applying a decision threshold."""

    node_type = "anomaly_scorer"
    display_name = tr_("Anomaly scorer")
    category = NodeCategory.TRAINING
    subcategory = tr_("Inference and decision")
    subcategory_order = 20
    palette_order = 30
    description = tr_("Calculate raw and normalized anomaly scores for each sample")
    icon = "🎚️"

    def _setup_ports(self):
        self.add_input("anomaly_model", DataType.ANOMALY_MODEL, tr_("Anomaly model"))
        self.add_input("feature_matrix", DataType.FEATURE_MATRIX, tr_("Feature matrix"))
        self.add_output("anomaly_scores", DataType.ANOMALY_SCORES, tr_("Anomaly scores"))
        self.add_output("detailed_scores", DataType.ANOMALY_SCORES, tr_("Detailed scores"))

    def _setup_parameters(self):
        self.add_parameter("aggregation", "choice", "none", display_name=tr_("Score aggregation"),
                           choices=["none", "mean", "max", "top_fraction_mean"])
        self.add_parameter("top_fraction", "float", 0.1, display_name=tr_("Top fraction"),
                           min_value=0.000001, max_value=1.0)
        self.add_parameter("batch_size", "int", 256, display_name=tr_("Batch size"), min_value=1)

    def execute(self) -> bool:
        for port in self.outputs.values():
            port.clear()
        try:
            artifact = self.get_input_data("anomaly_model")
            features = self.get_input_data("feature_matrix")
            if not isinstance(artifact, AnomalyModelArtifact):
                self.error_message = tr_("No valid anomaly model provided")
                return False
            if not isinstance(features, FeatureMatrixData):
                self.error_message = tr_("Anomaly scorer requires feature matrix data")
                return False

            legacy_schema = (artifact.algorithm == "isolation_forest"
                             and set(artifact.feature_schema) <= {"aggregation", "feature_count"}
                             and not features.schema.get("mode")
                             and not features.schema.get("preprocessing")
                             and not features.schema.get("layout", {}).get("preprocessing"))
            schema_matches = (all(features.schema.get(key) == value for key, value in artifact.feature_schema.items())
                              if legacy_schema else artifact.feature_schema == features.schema)
            if not schema_matches:
                raise ValueError(tr_("Feature schema differs from the model training schema"))
            raw_scores = artifact.score(features.matrix, features.sample_ids, features.parent_ids,
                                        int(self.get_parameter("batch_size")))
            normalized = artifact.normalize_scores(raw_scores)
            reference_normalized = artifact.normalize_scores(artifact.reference_scores)
            result = AnomalyScoresData(
                raw_scores=raw_scores,
                normalized_scores=normalized,
                sample_ids=list(features.sample_ids),
                source_items=list(features.source_items),
                reference_normalized_scores=reference_normalized,
                metadata={"algorithm": artifact.algorithm, "granularity": "window" if features.provenance else "sample",
                          "provenance": features.provenance},
            )
            self.set_output_data("detailed_scores", result)
            method = self.get_parameter("aggregation")
            if method != "none":
                from ..feature_contract import aggregate_scores
                parents = getattr(artifact, "reference_parent_ids", [])
                if not parents:
                    raise ValueError(tr_("Score aggregation requires normal reference parent IDs"))
                fraction = float(self.get_parameter("top_fraction"))
                raw, ids, first = aggregate_scores(raw_scores, features.parent_ids, method, fraction)
                reference, _, _ = aggregate_scores(artifact.reference_scores, parents, method, fraction)
                low, high = np.quantile(reference, [0.01, 0.99])
                span = max(float(high - low), abs(float(high)) * 1e-6, 1e-9)
                result = AnomalyScoresData(raw, np.clip((raw - low) / span * 100, 0, 100), ids,
                                           [features.source_items[i] for i in first] if features.source_items else [],
                                           np.clip((reference - low) / span * 100, 0, 100),
                                           {"algorithm": artifact.algorithm, "granularity": "sample",
                                            "aggregation": method, "top_fraction": fraction,
                                            "score_bounds": [float(low), float(high)]})
            self.set_output_data("anomaly_scores", result)
            self.report_status(
                tr_("Anomaly scoring finished: {samples} samples").format(samples=len(result.raw_scores))
            )
            return True
        except Exception as exc:
            self.error_message = tr_("Anomaly scoring failed: {error}").format(error=str(exc))
            logger.exception("Anomaly scoring failed")
            return False


@register_node
class AnomalyDecisionNode(BaseNode):
    """Convert continuous anomaly scores into severity decisions."""

    node_type = "anomaly_decision"
    display_name = tr_("Anomaly decision")
    category = NodeCategory.TRAINING
    subcategory = tr_("Inference and decision")
    subcategory_order = 20
    palette_order = 40
    description = tr_("Apply a configurable threshold without retraining the detector")
    icon = "🚦"

    def _setup_ports(self):
        self.add_input("anomaly_scores", DataType.ANOMALY_SCORES, tr_("Anomaly scores"))
        self.add_output("anomaly_result", DataType.ANOMALY_RESULT, tr_("Anomaly result"))

    def _setup_parameters(self):
        self.add_parameter(
            "strategy",
            "choice",
            "reference_quantile",
            display_name=tr_("Threshold strategy"),
            choices=["reference_quantile", "mean_std", "median_mad", "manual"],
        )
        self.add_parameter(
            "expected_anomaly_ratio",
            "float",
            0.01,
            display_name=tr_("Expected anomaly ratio"),
            min_value=0.001,
            max_value=0.5,
        )
        self.add_parameter(
            "std_multiplier",
            "float",
            3.0,
            display_name=tr_("Standard deviation multiplier"),
            min_value=0.1,
            max_value=20.0,
        )
        self.add_parameter(
            "mad_multiplier",
            "float",
            3.5,
            display_name=tr_("MAD multiplier"),
            min_value=0.1,
            max_value=20.0,
        )
        self.add_parameter(
            "manual_threshold",
            "float",
            80.0,
            display_name=tr_("Manual threshold (0-100)"),
            min_value=0.0,
            max_value=100.0,
        )
        self.add_parameter(
            "attention_ratio",
            "float",
            0.8,
            display_name=tr_("Attention threshold ratio"),
            min_value=0.0,
            max_value=1.0,
        )

    def execute(self) -> bool:
        try:
            scores = self.get_input_data("anomaly_scores")
            if not isinstance(scores, AnomalyScoresData):
                self.error_message = tr_("No valid anomaly scores provided")
                return False
            if len(scores.normalized_scores) == 0:
                self.error_message = tr_("Anomaly scores are empty")
                return False

            strategy = self.get_parameter("strategy")
            reference = scores.reference_normalized_scores
            if strategy != "manual" and len(reference) == 0:
                self.error_message = tr_("Reference scores are required for the selected threshold strategy")
                return False

            threshold = self._threshold(reference, strategy)
            attention_threshold = threshold * float(self.get_parameter("attention_ratio"))
            normalized = scores.normalized_scores
            is_anomaly = normalized >= threshold
            severities = [
                "anomaly" if value >= threshold else "attention" if value >= attention_threshold else "normal"
                for value in normalized
            ]
            result = AnomalyResultData(
                scores=scores,
                threshold=float(threshold),
                attention_threshold=float(attention_threshold),
                severities=severities,
                is_anomaly=is_anomaly,
                strategy=strategy,
            )
            self.set_output_data("anomaly_result", result)
            self.report_status(
                tr_("Anomaly decision finished: {anomalies} anomalies from {samples} samples").format(
                    anomalies=int(np.count_nonzero(is_anomaly)),
                    samples=len(normalized),
                )
            )
            return True
        except Exception as exc:
            self.error_message = tr_("Anomaly decision failed: {error}").format(error=str(exc))
            logger.exception("Anomaly decision failed")
            return False

    def _threshold(self, reference: np.ndarray, strategy: str) -> float:
        if strategy == "manual":
            return float(self.get_parameter("manual_threshold"))
        if strategy == "reference_quantile":
            ratio = float(self.get_parameter("expected_anomaly_ratio"))
            threshold = float(np.quantile(reference, 1.0 - ratio))
        elif strategy == "mean_std":
            threshold = float(
                np.mean(reference)
                + float(self.get_parameter("std_multiplier")) * np.std(reference)
            )
        elif strategy == "median_mad":
            median = float(np.median(reference))
            mad = float(np.median(np.abs(reference - median)))
            threshold = median + float(self.get_parameter("mad_multiplier")) * 1.4826 * mad
        else:
            raise ValueError(tr_("Unsupported threshold strategy: {strategy}").format(strategy=strategy))

        if np.allclose(reference, reference[0]):
            threshold = float(reference[0]) + 1e-6
        return float(np.clip(threshold, 0.0, 100.0))


@register_node
class AnomalyExplorerNode(BaseNode):
    """Expose anomaly results through the dedicated interactive preview."""

    node_type = "anomaly_explorer"
    display_name = tr_("Anomaly explorer")
    category = NodeCategory.OUTPUT
    subcategory = tr_("Result viewing")
    subcategory_order = 30
    palette_order = 10
    description = tr_("Inspect ranked anomaly results, thresholds, and source samples")
    icon = "🔎"

    def _setup_ports(self):
        self.add_input("anomaly_result", DataType.ANOMALY_RESULT, tr_("Anomaly result"))
        self.add_output("anomaly_result", DataType.ANOMALY_RESULT, tr_("Anomaly result"))

    def execute(self) -> bool:
        result = self.get_input_data("anomaly_result")
        if not isinstance(result, AnomalyResultData):
            self.error_message = tr_("No valid anomaly result provided")
            return False
        self.set_output_data("anomaly_result", result)
        self.report_status(
            tr_("Anomaly explorer ready: {samples} samples").format(
                samples=len(result.scores.normalized_scores)
            )
        )
        return True


__all__ = [
    "FeatureMatrixData",
    "AnomalyModelArtifact",
    "AnomalyScoresData",
    "AnomalyResultData",
    "FeatureVectorizerNode",
    "AnomalyDetectorTrainerNode",
    "AnomalyScorerNode",
    "AnomalyDecisionNode",
    "AnomalyExplorerNode",
]
