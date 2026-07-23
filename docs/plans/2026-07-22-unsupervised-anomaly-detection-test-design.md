# Unsupervised Anomaly Detection Test Design

## Scope

This plan covers task-specific supervised trainers and the first Isolation Forest anomaly-detection workflow defined by `FEAT2026072201`.

## Code Analysis

- Inputs: categorical or continuous targets, per-sample audio features, fixed-length feature matrices, detector artifacts, anomaly scores, and threshold parameters.
- Outputs: trained Keras models, feature matrices, serializable anomaly artifacts, aligned scores, severity decisions, and interactive previews.
- Dependencies: TensorFlow/Keras for supervised training, scikit-learn for Isolation Forest and scaling, joblib for artifact persistence, and PySide6/pyqtgraph for previews.
- Side effects: model fitting, optional artifact file persistence, and UI state updates.

## Test Matrix

| ID | Category | Scenario | Expected result |
|---|---|---|---|
| TR-01 | Registration | Load legacy and new trainer types | Legacy loads but is hidden; new trainers are visible |
| TR-02 | Equivalence | Integer, float-encoded, and one-hot classification targets | Classification accepts all categorical representations |
| TR-03 | Validation | Continuous targets sent to classification | Clear validation failure |
| TR-04 | Validation | Categorical or nonnumeric targets sent to regression | Clear validation failure |
| TR-05 | Boundary | Empty targets | Training is rejected before Keras execution |
| TR-06 | Configuration | Incompatible built-in model loss | Task-specific trainer rejects the model configuration |
| FV-01 | Happy path | FeatureData list with stable source paths | Fixed matrix and aligned sample IDs are produced |
| FV-02 | Boundary | Inconsistent flattened feature sizes | Vectorization fails without partial output |
| FV-03 | Error | Empty, NaN, or infinite features | Vectorization or training fails clearly |
| AD-01 | Happy path | Mostly-normal reference data with a distant outlier | Outlier receives a higher raw anomaly score |
| AD-02 | Reproducibility | Fixed random seed | Detector scores remain deterministic |
| AD-03 | Persistence | Save and load detector artifact | Loaded artifact returns matching scores |
| AD-04 | Contract | Score matrix feature count differs from training | Scoring fails with expected/actual dimensions |
| AD-05 | Separation | Score node followed by decision node | Threshold changes require no refit |
| AD-06 | Boundary | Manual normal/attention/anomaly cutoffs | Values map to the correct three severities |
| AD-07 | Degenerate | Identical reference scores | Normal zero score is not marked anomalous |
| UI-01 | Routing | PreviewView receives AnomalyResultData | Dedicated anomaly explorer is selected |
| UI-02 | Interaction | Select ranked feature or audio sample | Matching source preview is displayed |
| UI-03 | Summary | Mixed severity result | Counts, threshold, ranking, and distribution are shown |

## Mock Strategy

- Supervised trainer unit tests exercise validation helpers without running TensorFlow training.
- Isolation Forest tests use small deterministic matrices and the real scikit-learn implementation.
- Artifact tests use pytest `tmp_path` and real joblib serialization.
- UI tests use the Qt offscreen platform and small in-memory feature objects.

## AI/Model Catalog Refactor

| ID | Category | Description | Expected result |
|---|---|---|---|
| CAT-01 | Compatibility | Load legacy trainer, evaluator, and prediction node types | Nodes deserialize but remain hidden from the palette |
| CAT-02 | Catalog | Inspect AI/Model stage metadata | Training, inference/decision, evaluation, and management groups use explicit ordering |
| CAT-03 | UI integration | Render the node palette | Groups and nodes appear in the agreed workflow order |
| CAT-04 | Classification inference | Predict two-class outputs | Class labels and the unchanged raw model output are emitted separately |
| CAT-05 | Regression inference | Predict scalar continuous outputs | A flat list of continuous values is emitted |
| CAT-06 | Evaluation validation | Evaluate task-specific targets | Classification rejects continuous targets and regression accepts them |
| CAT-07 | Visualization | Inspect output-category nodes | Training history, metrics, and anomaly explorer appear under Output/Visualization |

The task-specific nodes use lightweight fake model objects so the tests verify workflow contracts without loading TensorFlow.
