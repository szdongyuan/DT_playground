"""Identity-aligned anomaly metrics; labels are consumed only for evaluation."""

import re
import numpy as np
from src.ui.i18n import tr_
from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from .anomaly import AnomalyScoresData, AnomalyResultData


def evaluate_anomalies(scores, labels, protocol="generic", max_fpr=0.1, result=None):
    from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
    from scipy.stats import hmean
    if not isinstance(scores, AnomalyScoresData) or not isinstance(labels, dict):
        raise ValueError(tr_("Evaluation requires anomaly scores and an identity-to-label map"))
    if scores.metadata.get("granularity") == "window":
        raise ValueError(tr_("Aggregate window scores to files before evaluation"))
    if len(set(scores.sample_ids)) != len(scores.sample_ids) or set(scores.sample_ids) != set(labels):
        raise ValueError(tr_("Label identities must match scored sample identities exactly"))
    if not 0 < max_fpr <= 1:
        raise ValueError(tr_("Maximum false positive rate must be in (0, 1]"))
    records = [labels[key] if isinstance(labels[key], dict) else {"label": labels[key]} for key in scores.sample_ids]
    y = np.asarray([row.get("label") for row in records])
    raw = scores.raw_scores
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2 or not np.isfinite(raw).all():
        raise ValueError(tr_("Evaluation requires finite scores and both normal (0) and anomaly (1) labels"))
    metrics = {"sample_count": len(y), "protocol": protocol, "max_fpr": max_fpr,
               "roc_auc": float(roc_auc_score(y, raw)),
               "standardized_pauc": float(roc_auc_score(y, raw, max_fpr=max_fpr))}
    if protocol == "dcase":
        groups = {}
        for index, (key, row) in enumerate(zip(scores.sample_ids, records)):
            match = re.search(r"section_(\d+)_(source|target)_", key.replace("\\", "/").rsplit("/", 1)[-1])
            section = row.get("section", match.group(1) if match else None)
            domain = row.get("domain", match.group(2) if match else None)
            if section is None or domain not in ("source", "target"):
                raise ValueError(tr_("DCASE evaluation requires section and source/target domain for every file"))
            groups.setdefault(str(section), []).append((index, domain))
        sections, components = {}, []
        for section, items in groups.items():
            indices = np.array([i for i, _ in items])
            section_metrics = {}
            for domain in ("source", "target"):
                selected = [i for i, d in items if y[i] == 1 or d == domain]
                if np.unique(y[selected]).size != 2:
                    raise ValueError(tr_("Each DCASE domain needs normal samples and section anomalies"))
                section_metrics[f"auc_{domain}"] = float(roc_auc_score(y[selected], raw[selected]))
            section_metrics["pauc"] = float(roc_auc_score(y[indices], raw[indices], max_fpr=max_fpr))
            components.extend(section_metrics.values())
            sections[section] = section_metrics
        metrics.update(sections=sections, harmonic_mean=float(hmean(components)))
    elif protocol != "generic":
        raise ValueError(tr_("Unknown anomaly evaluation protocol"))
    if result is not None:
        if result.scores is not scores:
            raise ValueError(tr_("Decision and score inputs must reference the same scores"))
        precision, recall, f1, _ = precision_recall_fscore_support(y, result.is_anomaly, average="binary", zero_division=0)
        metrics.update(precision=float(precision), recall=float(recall), f1=float(f1),
                       false_positive_rate=float(np.mean(result.is_anomaly[y == 0])),
                       false_negative_rate=float(np.mean(~result.is_anomaly[y == 1])),
                       confusion_matrix=confusion_matrix(y, result.is_anomaly, labels=[0, 1]).tolist())
    return metrics


@register_node
class AnomalyEvaluationNode(BaseNode):
    node_type = "anomaly_evaluation"
    display_name = tr_("Anomaly evaluation")
    category = NodeCategory.TRAINING
    subcategory = tr_("Model evaluation")
    description = tr_("Evaluate raw anomaly rankings with strictly aligned file labels")
    icon = "📊"

    def _setup_ports(self):
        self.add_input("anomaly_scores", DataType.ANOMALY_SCORES, tr_("Anomaly scores"), required=False)
        self.add_input("anomaly_result", DataType.ANOMALY_RESULT, tr_("Anomaly result"), required=False)
        self.add_input("labels", DataType.ANY, tr_("Target map"))
        self.add_output("metrics", DataType.METRICS, tr_("Metrics"))

    def _setup_parameters(self):
        self.add_parameter("protocol", "choice", "generic", display_name=tr_("Evaluation protocol"),
                           choices=["generic", "dcase"])
        self.add_parameter("max_fpr", "float", 0.1, display_name=tr_("Maximum false positive rate"),
                           min_value=0.000001, max_value=1.0)

    def validate(self):
        valid, message = super().validate()
        if valid and sum(self.inputs[key].is_connected for key in ("anomaly_scores", "anomaly_result")) != 1:
            return False, tr_("Connect exactly one anomaly score or result input")
        return valid, message

    def execute(self):
        self.outputs["metrics"].clear()
        try:
            result = self.get_input_data("anomaly_result")
            scores = self.get_input_data("anomaly_scores")
            if result is not None:
                if scores is not None or not isinstance(result, AnomalyResultData):
                    raise ValueError(tr_("Connect exactly one anomaly score or result input"))
                scores = result.scores
            metrics = evaluate_anomalies(scores, self.get_input_data("labels"), self.get_parameter("protocol"),
                                         float(self.get_parameter("max_fpr")), result)
            self.set_output_data("metrics", metrics)
            return True
        except Exception as exc:
            self.error_message = tr_("Anomaly evaluation failed: {error}").format(error=str(exc))
            return False
