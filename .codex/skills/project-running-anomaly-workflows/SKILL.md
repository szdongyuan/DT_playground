---
name: project-running-anomaly-workflows
description: Build, train, resume, and evaluate native unsupervised audio anomaly workflows through this project's CLI. Use when an AI agent runs local anomaly experiments, DCASE validation, Isolation Forest, kNN, or autoencoder reconstruction workflows, or compares their features and scores.
---

# Run Native Anomaly Workflows

## Scope

The CLI consumer is an AI agent. Generate ordinary model/workflow JSON as needed;
do not add product commands or nodes merely to avoid writing JSON. Prefer native
nodes for feature extraction, standardization, windowing, training, scoring, and
evaluation. Small orchestration scripts may prepare manifests, organize files,
invoke CLI commands, and summarize recorded outputs. Do not silently replace
missing platform behavior with custom ML code and call it a native workflow.

This skill coordinates experiments; executable contracts, validation, persistence,
and numerical correctness remain responsibilities of the software.

## Read on demand

- [reference.md](reference.md): exact node connections, data contracts, evaluation,
  persistence, and recovery rules. Read before building or changing a graph.
- [examples.md](examples.md): CLI recipes, workflow editing, and completion evidence.
- [templates/dense-ae.model.json](templates/dense-ae.model.json): editable 320-input
  AE model definition, not pretrained weights or a universal architecture.

## 1. Establish the local contract

1. Work from the repository root. Follow the repository's Python environment and
   PowerShell skills; use `./.venv/Scripts/python.exe` explicitly. Print and verify
   the interpreter before execution. Do not install packages into global Python.
2. Locate `AGENTS.md` and applicable local instructions. Inspect existing results
   and the user's accepted configuration before selecting defaults.
3. Run `cli_main.py capabilities --json` and the relevant command's `--help`.
   Inspect current node definitions if a port or parameter is unclear. The
   examples describe the known contract, not permission to invent missing flags.
4. Create a new experiment directory, leaving earlier runs intact. Record the
   repository revision and relevant uncommitted changes, CLI commands, versions,
   seeds, thread settings, input hashes, split roles, and output locations.

Ask only when information materially changes the experiment and cannot be inferred:
which data are normal, the independent evaluation set, an ambiguous split/domain
policy, or an unbounded compute commitment. Do not ask the user to design JSON or
confirm routine node wiring. Continue independent preparation while awaiting answers.
Do not infer that every request to train authorizes an exhaustive parameter search.

## 2. Prepare leakage-safe data

1. Inventory actual files, sample rates, channels, duration, and available labels.
   `inspect` is useful for audio profiling, but not a substitute for role validation.
2. Establish disjoint normal fitting and normal validation sets before generating
   windows. Split by original file or a stronger recording/session group, never by
   individual windows. Preserve domain/section coverage where the data allow it.
3. Audit resolved paths and content hashes across fitting, validation, and test
   sets. Resolve duplicates before training; never silently move duplicate test
   content into fitting. Record exceptions explicitly if the dataset requires them.
4. The AE generator expects separate normal fitting/validation directories. Avoid
   identical or nested directories because audio loading is recursive. Prefer
   existing split folders; otherwise create reproducible copies or read-only-use
   hard links under the experiment directory, with a mapping to original files.
   Hard links share content: never modify linked audio. Do not move source data.
5. Fit preprocessing on fitting data only. Use normal validation for early stopping.
   Test labels belong exclusively to evaluation; never connect them to fitting,
   reference calibration, or model-selection callbacks.
6. Prepare anomaly labels as the exact scored sample ID to numeric 0/1 mapping
   described in reference.md. Do not reuse the supervised classification label
   wrapper or assume basename matching. Verify IDs after path resolution.

## 3. Select a bounded comparison

- Preserve an existing baseline and accepted budget. For a fresh deep-learning
  comparison, use a modest dense AE trained with the existing regression trainer.
- Isolation Forest remains available through the existing trainer; construct its
  workflow JSON directly if the generation command lacks that template.
- kNN stores an exact reference bank. Estimate reference/window counts first;
  exhaustive reference scoring scales quadratically in reference count. Prototype
  with an explicitly reported subset if needed, then run the agreed full dataset.
  Do not silently substitute a subset for full validation.
- Keep data splits, feature definitions, seed policy, score direction, aggregation,
  and evaluation protocol comparable. Change one factor at a time where possible.
- Compare raw ranking metrics before threshold calibration unless the user requests
  operating-point optimization. Treat comparisons on reused test data as development
  validation, not independent generalization evidence.

## 4. Construct the model and graph

1. Derive the model dimension from the actual feature layout and reduction.
   For flattened `(channels, bands, window_frames)`, dimension is their product;
   `mean_std` instead produces twice the non-time dimension product.
2. For an ordinary AE, edit the supplied model definition, including both input and
   reconstruction output sizes. Build it with `build-model`, compiled with MSE and
   a compatible optimizer. Do not invent an AE-specific training node.
3. Generate an initial graph with `create-anomaly-workflow` when supported, then
   edit its JSON if required. Query capabilities for supported fields and choices.
   Changing windowing or extraction requires corresponding changes to both training
   and inference graphs and, where applicable, the model dimension.
4. Wire normal validation through the *training* standardization state. Wire each
   AE feature matrix to both X and Y inputs. Keep identities and schemas intact.
5. If later extension/resumption is likely or explicitly requested, add the existing
   `save_model` node alongside anomaly packaging before starting training. The
   default AE anomaly template saves an inference bundle, not a resumable model.
6. Inspect the final JSON: required connections, independent splits, preprocessing
   reuse, X/Y alignment, exports, and evaluation-only label branches.

## 5. Validate and execute

1. Run `validate --check-data --json`; include model-definition validation/building
   where useful. Parse errors and warnings, not only the exit code. Current data
   preflight checks raw sources and some alignment, not all transformed shapes.
2. For new wiring/layouts, run a small isolated smoke workflow first. Include enough
   distinct normal parents for the selected kNN exclusion and k. Keep smoke output
   separate from the full experiment and retain the full intended split.
3. Launch `run` with an explicit seed and a fresh run directory. Capture stdout,
   stderr, exit status, events, and elapsed time. Do not rely on shell activation.
4. Monitor training loss, normal validation loss, epoch count, and process state.
   Some exact kNN stages do not emit batch progress; a quiet log alone is not proof
   of a hang. Do not start duplicate expensive jobs or kill unrelated processes.
5. On failure, retain evidence and fix the concrete contract violation. Revalidate
   only what changed, then use a new run directory. Do not use overwrite or erase
   a prior run merely to bypass errors. See reference.md for diagnostic routing.

## 6. Verify the result, including reload

1. Require successful process exit and `manifest.success`, and inspect failed-node
   events if they disagree. Check actual files, not just output path messages.
2. Generate a separate scoring workflow loading the saved trusted local bundle and
   its preprocessing state. Do not refit standardization. Match its feature schema;
   the current generator does not automatically reconstruct all settings from a bundle.
3. Verify score CSV count and unique IDs against the expected file set, finite raw
   scores, correct score direction, and intended granularity. A truncated manifest
   array preview is not the complete data; use CSV and full event logs.
4. Run native anomaly evaluation using exact ID alignment and the declared protocol.
   For changed aggregation, recompute matching normal reference aggregation before
   decisions. Never reuse an incompatible display scale or threshold.
5. When checking persistence or migrating an existing implementation, compare raw
   scores on identical inputs/settings. Declare dtype-sensitive tolerances in
   advance (e.g. `atol=1e-6, rtol=1e-5` for float32 neural inference), and investigate
   violations. Do not loosen tolerances silently or require freshly retrained models
   to produce identical scores.
6. Summarize ranking performance, runtime, artifact size, and operational friction
   separately. Multiple seeds are needed for robust model comparisons; a successful
   single-seed execution proves operability, not general superiority.

## 7. Resume and report

For resumption, load a compiled ordinary Keras model/checkpoint, preserve optimizer
state with `use_model_config=True`, and reuse the original split and fitted
preprocessing state. Declare whether the starting weights are best-epoch or
last-epoch. Record additional epochs separately from prior epochs. Optimizer/weights
consistency does not imply restored random-number, data-order, or callback history.
If only an inference bundle exists, do not describe a new optimizer as exact resume;
explain the available warm-start option or the missing checkpoint.

Deliver a short conclusion plus links to the final graphs, manifest/metrics,
complete score files, and a reproducibility report. Include the actual interpreter,
dataset roles, model/feature/scoring configuration, seeds, completed/best epochs,
normal-validation selection rule, evaluation scope, limitations, and next useful
comparison. Distinguish software capability improvements from detection gains.
Do not commit datasets, model binaries, experiment outputs, or push code unless
the user has requested it. Do not create issues or send external messages solely
because this skill was invoked.
