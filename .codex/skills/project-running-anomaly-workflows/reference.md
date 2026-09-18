# Contracts and Recovery Reference

Treat `capabilities --json` and current source as authoritative if these details
change. Relevant code is in `src/cli/anomaly_generation.py`,
`src/workflow/nodes/anomaly.py`, `anomaly_files.py`, `anomaly_evaluation.py`,
`feature_standardization.py`, and `src/workflow/nodes/training.py`.

## Connections

| Purpose | Node | Inputs | Outputs / settings |
| --- | --- | --- | --- |
| Temporal/vector normalization | `feature_standardization` | `features`, optional `state` | `features`, `state`; `mode=fit_transform` or `transform` |
| Fixed vectors/windows | `feature_vectorizer` | `features` | `feature_matrix`; `mode=whole_sample/sliding_window`, `aggregation=flatten/mean/mean_std/global_max` |
| IF or exact kNN fitting | `anomaly_detector_trainer` | `feature_matrix`, optional `preprocessing_state`, `calibration_features` | `anomaly_model`, `reference_scores`, `training_summary` |
| AE fitting | `regression_trainer` | `model`, `x_train`, `y_train`, optional `x_val`, `y_val` | `trained_model`, `history`, `training_summary` |
| Ordinary Keras saving | `save_model` | `model` | `model_path`; `save_mode=new_file`, `save_dir`, `file_name`, `save_format=keras` |
| Package detector | `save_anomaly_model` | `anomaly_model`, optional `preprocessing_state` | `anomaly_model`, `model_path`; `mode=artifact` |
| Package AE | `save_anomaly_model` | `model`, `reference_features`, optional `preprocessing_state` | same outputs; `mode=autoencoder` |
| Reload scoring state | `load_anomaly_model` | none | `anomaly_model`, `preprocessing_state`; `model_path`, `trusted` |
| Raw scoring/aggregation | `anomaly_scorer` | `anomaly_model`, `feature_matrix` | `detailed_scores`, `anomaly_scores` |
| Threshold decisions | `anomaly_decision` | `anomaly_scores` | `anomaly_result` |
| Export | `export_anomaly_results` | exactly one of `anomaly_scores`, `anomaly_result` | complete CSV and metadata; `output_folder`, `directory_name` |
| Load evaluation labels | `target_file` | none | `target_map`; `target_kind=continuous`, `dtype=int64`, `output_format=map` |
| Evaluate | `anomaly_evaluation` | `labels`, exactly one of `anomaly_scores`, `anomaly_result` | `metrics`; `protocol=generic/dcase`, `max_fpr` |

## Features and identities

- `FeatureData` temporal layout is conventionally `(channels, bands, frames)`.
  Standardization fits each channel/band across original fitting frames, before
  windows overlap. Matrix input is standardized per column. Population standard
  deviation uses a configurable positive `epsilon` floor for constant dimensions.
- `FeatureMatrixData` retains matrix, unique sample IDs, source items, schema, and
  optional window provenance. A window has its own ID plus `parent_sample_id`,
  index, frame bounds, padding count, and available frame-hop time bounds.
- Preserve full resolved file identity. Identical basenames in different directories
  are distinct. Unnamed numeric arrays have positional fallback IDs; do not assume
  those IDs distinguish independently created training and validation arrays.
- `window_length` and `window_stride` are measured in frames. `include_last` adds
  a final complete window, potentially overlapping its predecessor; `drop` omits
  the remaining incomplete tail. Short samples use `error` or `pad_edge`.
- Match extraction layout, reduction, window settings, flatten order, and fitted
  preprocessing signature, not just column count. Set detector `scaling=none`
  when external standardization has already been applied.
- Typed X/Y row IDs must be identically ordered. Training and validation schemas
  must match and parent identities must be disjoint. The agent must additionally
  audit hashes/group identity because the runtime cannot detect all copied content.

## Scoring

- IF defaults remain available with its internal standard/robust/no scaling.
- kNN: `algorithm=knn`, `n_neighbors`, `distance=euclidean/cosine`,
  `reference_exclusion=parent/sample`. Parent exclusion removes every window from
  the same file; sample exclusion removes only the identical row ID. Duplicate
  values with different identities are still neighbors. Cosine rejects zero vectors.
  Too few eligible neighbors is an error; never silently shrink k.
- Optional independent normal `calibration_features` must match the fitting schema
  and not overlap fitting parents. They set the reference distribution without
  becoming the kNN reference bank. Record this distinction.
- AE scoring is per-row reconstruction MSE on a rank-two matrix. Input/output
  dimensions must agree; inference is batched and uses inference mode.
- `aggregation=none/mean/max/top_fraction_mean`; `top_fraction` defaults to 0.1.
  The upper fraction uses `ceil(window_count * fraction)` windows per parent.
  Aggregate raw scores before clipping. Reference scores receive the same grouping
  and aggregation, followed by recalculation of the display bounds.
- The 0-100 normalized scale is for display/decisions, not an anomaly probability.
  Use raw scores for AUC, pAUC, and ranking comparisons. Export `detailed_scores`
  separately when temporal localization is needed. File evaluation rejects windows.

## Evaluation labels and protocol

For the anomaly `target_file` branch, use a flat JSON object, for example:

```json
{
  "D:/data/test/section_00_source_test_normal_0000.wav": 0,
  "D:/data/test/section_00_source_test_anomaly_0000.wav": 1
}
```

These example strings must be replaced with exact sample IDs emitted by the local
loader. Windows IDs may use backslashes; serialize the actual Python path strings
with a JSON library rather than manually escaping them. Do not change the CSV IDs
or normalize labels independently and assume they still match.

This is not the classification `label_file` wrapper with a top-level `labels`
object. Duplicate keys/rows, missing/extra IDs, non-binary labels, and single-class
evaluation must be addressed before interpreting metrics. DCASE requires source
and target normal examples as well as section anomalies; the two-row example above
illustrates mapping syntax, not a complete DCASE fixture.

- Generic: pooled ROC AUC and standardized pAUC; default `max_fpr=0.1`.
- DCASE: within each section, each domain AUC uses that domain's normal files plus
  all anomalous files in the section. Section pAUC uses all section samples. The
  combined result is the harmonic mean of section/domain components.
- Section/domain are parsed from conventional filenames or supplied as structured
  records to the evaluation API. Do not claim the generic target loader supports
  arbitrary nested records without checking its current conversion behavior.
- Passing decisions also yields precision, recall, F1, false-positive/negative
  rates, and a confusion matrix. The threshold must be selected independently of
  the evaluation labels for an unsupervised operating-point claim.

## Persistence and continuation

Anomaly bundles carry scoring state, reference scores/parents, schemas, and optional
external standardization. v2 stores the Keras network separately; v1 IF is readable
with only the legacy contract guarantees actually stored. Keep trust gates, version
checks, hashes, path restrictions, and exclusive publication. A checksum is not
proof that a joblib payload is safe. Explicitly trust only an established local
artifact or a source authorized by the user; never blindly enable trust for an
unknown downloaded model to make validation pass.

AE anomaly loading uses inference semantics, not optimizer restoration. Ordinary
Keras saves from the trainer can preserve matching best-epoch weights/optimizer;
stop checkpoints preserve the last state. Inspect `training_summary` and
`.training.json` sidecars when present. Older files may not have matching best
weights and optimizer slots. Never invent metadata for them.

On continuation, `use_model_config=False` recompiles and replaces the optimizer;
use the compiled configuration to preserve optimizer state. Reuse the original
standardization state (e.g. from the prior trusted anomaly bundle) for both splits.
Do not regenerate an inference-only template and call it a continuation workflow.
Callbacks/random state are not fully restored; report the supported `model_optimizer`
level and run-relative epoch counts, not bitwise uninterrupted equivalence.

## Failure routing

| Symptom | Next action |
| --- | --- |
| Unknown CLI flag/node/port | Read current help/capabilities; edit supported JSON fields rather than inventing an API |
| Feature schema mismatch | Compare full layout, extraction/window settings, and preprocessing fingerprint; do not erase schema checks |
| kNN has insufficient neighbors | Count eligible distinct parents/rows after exclusion; adjust the experimental k or reference set explicitly |
| Label mismatch | Compare exact ID sets and duplicates against exported file scores; do not truncate, reorder blindly, or match ambiguous basenames |
| Invalid shape/non-finite features | Inspect a small sample's axes, channels, dtype, padding, and scale state before a full rerun |
| Long quiet kNN stage | Check the specific process and events; avoid duplicate work; report missing progress as a platform limitation |
| Existing output/run-directory conflict | Choose a fresh run directory; inspect an active run lock instead of deleting it |
| Trust/version/hash failure | Establish artifact provenance/compatibility or rebuild from a trusted source; do not bypass validation |
| Missing resume checkpoint | Explain inference-only state and choose an explicit warm start or retraining, not a false exact-resume claim |
