# Audio Training Platform CLI

The CLI is a deterministic, headless interface for audio and acoustic workflows. It does not choose features, model architectures, or hyperparameters. Automation clients must provide explicit workflow and model definitions.

## Entry point

During development, always use the repository virtual environment:

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py --help
```

The packaged console executable is `AudioTrainingPlatformCLI.exe`.

## Commands

### Discover capabilities

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py capabilities --json
```

The result contains the CLI, workflow, and model schema versions plus every supported node, port, parameter, constraint, and model layer. Hidden legacy nodes are omitted unless `--include-hidden` is supplied.

### Inspect a dataset

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py inspect D:/datasets/fan --labels D:/datasets/fan/labels.csv --output profile.json
```

Inspection is read-only. It reports readable and unreadable files, formats, sample rates, channel counts, duration statistics, folder-derived class distribution, and optional CSV/JSON filename-label alignment. CSV mappings require `filename` and `label` columns.

### Validate definitions

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py validate workflow.json --model classifier.model.json --build-model --json
```

Validation checks JSON structure, node and layer types, parameters, ports, connections, required inputs, input paths, model topology, and headless compatibility. `--build-model` also builds the model in memory to detect Keras shape errors. Validation does not train or save a model.

Enabled breakpoint nodes and the GUI-based training-history display node are rejected in headless mode.

#### Explicit data preflight

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py validate workflow.json --check-data --json
```

Without `--check-data`, validation does not scan or decode datasets. The flag
adds read-only checks after structural validation succeeds:

- `audio_folder` and `audio_file`: fully decode each selected file, reject empty,
  unreadable or non-finite audio, and report counts, raw shapes and sample rates.
  Folder recursion and `first_n` limits match the node parameters. For `random_n`,
  all candidates are checked with a warning; the runtime subset is not predicted.
- `label_file` and `target_file`: parse using their existing formats and parameters.
- `align_targets`: check direct source connections using the existing matching,
  duplicate and missing-target policies. An empty aligned result fails. Unused
  targets and configured drop/fill behavior produce warnings.

The workflow report includes `data_checks` with source statistics, alignment
results and explicit scope. Unsupported sources or transformed alignment paths
produce warnings, not claims of successful checking. Raw shapes are reported
before resampling. Feature transformations and model input compatibility are
**not checked**; `--build-model` separately validates model-internal shapes only.
Preflight never runs the workflow engine, loads models, trains or saves artifacts.
Audio is streamed in blocks where supported; fallback decoding may load one full
file into memory. Any data error returns exit code `3`. Like existing validation,
JSON results use stdout on success and stderr on failure.

### Build a model

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py build-model classifier.model.json --output classifier.keras
```

Existing output files are preserved unless `--overwrite` is explicitly supplied. Use `--no-compile` for inference-only models.

`build-model --seed 42` seeds Python, NumPy and Keras before model initialization.
The default is 42; accepted values are integers from 0 through 4294967295. The
JSON build result includes `seed`. Same definition and seed reproduce initial
weights in the same software environment, not necessarily identical archive
bytes or cross-platform training results. Model construction does not build an
extra throwaway model before initializing the artifact.

New compiled models initialize optimizer slots before saving, avoiding an
incomplete fresh optimizer checkpoint. This uses additional optimizer-state
memory at build time. `--no-compile` skips this; existing loaded checkpoints are
not reset or rewritten. Old artifacts with incomplete optimizer state must be
rebuilt explicitly to benefit; their warnings are not hidden.

Evaluation nodes return semantic metric keys such as `loss`, `accuracy`, `mae`
and `mse` using Keras' named evaluation results, instead of `compile_metrics`.
Clients consuming that previous aggregate key should use the configured metric
names instead. The same correction applies to GUI and CLI evaluation nodes.

### Run a workflow

```powershell
& './.venv/Scripts/python.exe' ./cli_main.py run workflow.json --run-dir outputs/run-001 --events jsonl --seed 42
```

Relative input paths are resolved against the workflow file directory. Relative `save_audio` and `save_model` output paths are resolved inside the run directory; absolute outputs outside the run directory are rejected. A non-empty run directory is never replaced unless `--overwrite` is supplied, and even then it must contain a compatible CLI manifest.

Press `Ctrl+C` to request cooperative cancellation. For training workflows, `--checkpoint checkpoints/interrupted.keras` requests a checkpoint inside the run directory before stopping.

Each run directory contains:

- `workflow.json`: the original workflow definition.
- `workflow.resolved.json`: the executed definition with resolved paths.
- `events.jsonl`: versioned progress and result events.
- `manifest.json`: input hash, seed, runtime versions, timings, validation results, summarized outputs, and artifact paths.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Success |
| `2` | Command-line usage error |
| `3` | Definition or dataset validation failed |
| `4` | Workflow or model execution failed |
| `5` | File, directory, or input/output contract error |
| `130` | Interrupted |

## JSON Lines events

`--events jsonl` writes one JSON object per line to stdout and to `events.jsonl`. Every record contains `schema_version`, `timestamp`, `event`, `message`, and `data`. Runtime arrays and models are summarized by type and shape; raw datasets are never written to stdout.

During JSONL execution, Keras progress bars are disabled; epoch metrics still
arrive as `node_progress` events. Ordinary Python output and stdout logging are
redirected to stderr. With console file descriptors, low-level stdout writes
are also redirected while events use a separate handle to the original stdout.
Output routing and verbosity are restored on exit, including failure or
interruption. GUI and text-mode defaults are unchanged. Do not merge stderr into
stdout (`2>&1`) when parsing JSONL. Routing is process-wide and intended for the
synchronous CLI process, not concurrent workflows embedded in a GUI process.

The first contract version supports local audio/acoustic classification, regression, anomaly detection, feature processing, evaluation, and prediction. Network execution, multi-user scheduling, AutoML, and non-audio datasets are outside its scope.
