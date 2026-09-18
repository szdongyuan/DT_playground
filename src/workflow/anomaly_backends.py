"""Exact nearest-neighbor and batched reconstruction anomaly scoring."""

from src.ui.i18n import tr_
from dataclasses import dataclass
import numpy as np


@dataclass
class KNNReference:
    matrix: np.ndarray
    sample_ids: list
    parent_ids: list
    k: int = 5
    metric: str = "euclidean"
    exclusion: str = "parent"

    def validate(self):
        if (self.matrix.ndim != 2 or not self.matrix.size or not np.isfinite(self.matrix).all()
                or len(self.sample_ids) != len(self.matrix) or len(self.parent_ids) != len(self.matrix)
                or len(set(self.sample_ids)) != len(self.sample_ids)
                or type(self.k) is not int or self.k < 1 or self.k > len(self.matrix)
                or self.metric not in ("euclidean", "cosine") or self.exclusion not in ("sample", "parent")):
            raise ValueError(tr_("Invalid nearest-neighbor reference state"))
        if self.metric == "cosine" and np.any(np.linalg.norm(self.matrix, axis=1) == 0):
            raise ValueError(tr_("Cosine distance does not accept zero vectors"))

    def score(self, matrix, sample_ids=None, parent_ids=None, batch_size=256):
        from sklearn.metrics import pairwise_distances
        self.validate()
        values = np.asarray(matrix, dtype=float)
        if self.metric == "cosine" and np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError(tr_("Cosine distance does not accept zero vectors"))
        if sample_ids is not None and (len(sample_ids) != len(values) or len(parent_ids) != len(values)):
            raise ValueError(tr_("Nearest-neighbor query identities must be aligned"))
        result = []
        bank_ids = np.asarray(self.parent_ids if self.exclusion == "parent" else self.sample_ids)
        query_ids = parent_ids if self.exclusion == "parent" else sample_ids
        for start in range(0, len(values), batch_size):
            distances = pairwise_distances(values[start:start + batch_size], self.matrix, metric=self.metric)
            if query_ids is not None:
                excluded = np.asarray(query_ids[start:start + batch_size])[:, None] == bank_ids[None, :]
                distances[excluded] = np.inf
            if np.any(np.isfinite(distances).sum(axis=1) < self.k):
                raise ValueError(tr_("Insufficient neighbors after identity exclusion; reduce k or add reference files"))
            result.extend(np.partition(distances, self.k - 1, axis=1)[:, :self.k].mean(axis=1))
        return np.asarray(result)


def reconstruction_scores(model, matrix, batch_size=256):
    values = np.asarray(matrix, dtype=float)
    if batch_size < 1:
        raise ValueError(tr_("Batch size must be positive"))
    result = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        prediction = np.asarray(model(batch.astype(np.float32), training=False))
        if prediction.shape != batch.shape or not np.isfinite(prediction).all():
            raise ValueError(tr_("Reconstruction output must be finite and match input shape"))
        result.extend(np.mean((prediction.astype(float) - batch) ** 2, axis=1))
    return np.asarray(result)
