# Anomaly model persistence and complete result export

GUI and CLI use the same three nodes, property definitions and execution services.
Algorithms and decision thresholds are chosen by the workflow author. Existing
Keras save/load nodes and anomaly previews are unchanged.

## Nodes

| Node type | Inputs | Outputs | Parameters |
| --- | --- | --- | --- |
| `save_anomaly_model` | `anomaly_model: ANOMALY_MODEL` | `model_path` (path string) | `output_folder="."`, `file_name="anomaly_model.anomaly.zip"` |
| `load_anomaly_model` | none | `anomaly_model: ANOMALY_MODEL` | `model_path`, `format="anomaly_zip"`, `trusted=false` |
| `export_anomaly_results` | exactly one of `anomaly_scores: ANOMALY_SCORES`, `anomaly_result: ANOMALY_RESULT` | `export_summary: METRICS` | `output_folder="."`, `directory_name="anomaly_results"`, `spreadsheet_safe=false` |

In the GUI, model nodes appear under AI / Model → Model management; the export
node appears under Output / Visualization → Result export. Configure them with
the existing property editor. Connect a decision output to both the existing
anomaly explorer and the export node to preview and export the same result.

## Trust and compatibility

Joblib can execute arbitrary code. `trusted=true` explicitly authorizes loading
code from the selected model. The setting is saved in the workflow: review both
the workflow and the model source before execution. ZIP wrapping, checksums and
post-load type checks do not make an untrusted model safe. There is no sandbox or
automatic trust inferred from a filename, location, or hash.

The v1 ZIP contains exactly `metadata.json` and `model.joblib`. Before loading,
the reader checks format, runtime fingerprint and payload hash without invoking
joblib. CLI `validate` (including `--check-data`) never deserializes the model,
even when the workflow contains `trusted=true`. Passing these checks says nothing
about the safety or validity of executable model contents.

Python major/minor must match; NumPy, SciPy, scikit-learn and joblib versions must
match exactly. Unsupported versions are rejected, without an ignore option.
The reader limits metadata to 1 MiB and the payload to 1 GiB. It does not extract
archive members into the filesystem.

For an existing bare joblib artifact, explicitly select `format="legacy_joblib"`
and `trusted=true`. Its runtime compatibility cannot be checked; a warning is
reported, and scoring-state checks happen only after deserialization. Input files
are never converted or rewritten. The Python API follows the same rules:

```python
artifact.save("new.anomaly.zip")
restored = AnomalyModelArtifact.load("new.anomaly.zip", trusted=True)
legacy = AnomalyModelArtifact.load("old.joblib", trusted=True, format="legacy_joblib")
```

Calls to the old Python `load(path)` without explicit trust now fail. `save(path)`
writes the versioned ZIP and refuses an existing target. Node filenames must end
with `.anomaly.zip`; existing workflow node types are unchanged.

## Reproduction and export contract

The model preserves the fitted estimator and scaler, feature schema, score
calibration bounds, raw reference scores and training summary. It excludes audio
and decision policy. Retain the feature/preprocessing and decision workflow along
with the model: model state alone is insufficient for full reproduction.

`results.csv` is UTF-8 and contains every sample in input order, without ranking
or truncation. Columns are `row_index` (zero-based), `sample_id`, `raw_score`,
`normalized_score`; decision exports add `is_anomaly` (0/1) and `severity`
(`normal`, `attention`, `anomaly`). Scores must be finite one-dimensional arrays
with matching lengths. Empty score exports produce a header and zero count.

`metadata.json` declares `format="dt-anomaly-results"`, `format_version=1`,
sample count, columns, score direction, ID encoding, score metadata and CSV
SHA-256. Decisions additionally include strategy, threshold and attention
threshold, in normalized-score units. Full strategy parameters remain in the
workflow. Raw scores increase with anomalousness: use **raw_score for ROC AUC**;
normalized scores are clipped to 0–100 and serve display and existing decisions.

Sample IDs preserve full source paths, including distinct paths with the same
basename; do not join evaluation labels by basename alone. Dataset-specific
domain/label parsing belongs to external evaluation. The default CSV keeps IDs
verbatim. For spreadsheet viewing, `spreadsheet_safe=true` adds an apostrophe to
every ID and records `sample_id_encoding="apostrophe_prefix_all"`. A machine
reader removes exactly one leading apostrophe to reconstruct the original ID,
including IDs that originally began with an apostrophe. CSV quoting alone is
not formula protection; spreadsheet import behavior should be checked separately.

## Paths and writes

Names are single portable basenames: separators, traversal, drive prefixes,
Windows reserved names, trailing dots/spaces and alternate data streams are
rejected. GUI relative directories resolve from the process working directory.
CLI input model paths resolve relative to the workflow file; output directories
resolve inside `--run-dir`, and escaping paths are rejected.

Existing targets, including empty result directories, are never overwritten by
these nodes. Model files are staged beside the target and published with an
exclusive hard link (the destination filesystem must support hard links).
Result directories are completely staged before publication; on Windows,
directory rename refuses an existing destination. Failed operations clean only
their own staging paths. The existing CLI `--overwrite` remains an explicit
authorization to clear a previously managed run directory; omit it to preserve
all prior run data, and use a fresh run directory for each experiment.

The CLI manifest stays a bounded summary. `artifacts.produced_files` and the
additive `artifacts.sha256` map identify full output files (keys follow the host
path separator). Consumers must open the result files for per-sample evaluation.
