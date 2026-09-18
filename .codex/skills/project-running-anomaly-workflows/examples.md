# CLI Recipes for Agents

Run from the repository root. Replace dataset/experiment paths with the actual
resolved locations; keep each run directory new. Check `$LASTEXITCODE` after each
CLI call when orchestrating in PowerShell, or use a subprocess call that checks
the return code. Do not continue a dependent command after failure.

## Preflight and discovery

```powershell
& './.venv/Scripts/python.exe' -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"
& './.venv/Scripts/python.exe' cli_main.py capabilities --json
& './.venv/Scripts/python.exe' cli_main.py create-anomaly-workflow --help
& './.venv/Scripts/python.exe' cli_main.py inspect 'D:/data/normal_fit' --json
```

Capture structured output in the experiment directory when needed. Do not assume
all diagnostics are JSON: stderr/runtime messages and structured stdout have
separate roles. `run --help` describes event-output options in the active version.

## Dense AE from scratch

Copy `templates/dense-ae.model.json` from this skill into a new experiment directory
as `model.json`, then edit the ordinary JSON to match the actual features. The
example's 320 dimensions assume one channel, 64 Mel bands, and five flattened frames.
The generation command's known defaults are 16 kHz, FFT 1024, hop 512, per-file peak
dB, window length/stride five, and top-10% aggregation. These are example settings,
not defaults to impose on an already agreed experiment.

```powershell
& './.venv/Scripts/python.exe' cli_main.py build-model 'outputs/experiment/model.json' --output 'outputs/experiment/initial.keras' --seed 42 --json
& './.venv/Scripts/python.exe' cli_main.py create-anomaly-workflow 'outputs/experiment/train.workflow.json' --dataset 'D:/data/normal_fit' --validation-dataset 'D:/data/normal_val' --algorithm autoencoder --model 'outputs/experiment/initial.keras' --epochs 40 --json
```

When a continuation artifact is needed, add this node to the generated `nodes`
array, and the connection below to `connections`, before validation/execution:

```json
{
  "id": "save_training_model",
  "type": "save_model",
  "position": [1800, 240],
  "parameters": {
    "save_mode": "new_file",
    "save_dir": "models",
    "file_name": "ae_training.keras",
    "save_format": "keras"
  }
}
```

```json
{
  "source": {"node": "trainer", "port": "trained_model"},
  "target": {"node": "save_training_model", "port": "model"}
}
```

Check that `trainer` is the actual generated trainer ID. In CLI runs, the save
directory is resolved through the runner's output policy; verify its location in
the manifest rather than guessing a working-directory-relative result.

```powershell
& './.venv/Scripts/python.exe' cli_main.py validate 'outputs/experiment/train.workflow.json' --model 'outputs/experiment/model.json' --build-model --check-data --json
& './.venv/Scripts/python.exe' cli_main.py run 'outputs/experiment/train.workflow.json' --run-dir 'outputs/experiment/train' --seed 42
```

## Exact kNN

Use the same fitting files as a comparison AE where appropriate. The template
uses external normalization and no second detector scaling. It computes reference
scores with same-parent exclusion; do not interpret them as independent test scores.

```powershell
& './.venv/Scripts/python.exe' cli_main.py create-anomaly-workflow 'outputs/knn_experiment/train.workflow.json' --dataset 'D:/data/normal_fit' --algorithm knn --k 5 --json
& './.venv/Scripts/python.exe' cli_main.py validate 'outputs/knn_experiment/train.workflow.json' --check-data --json
& './.venv/Scripts/python.exe' cli_main.py run 'outputs/knn_experiment/train.workflow.json' --run-dir 'outputs/knn_experiment/train' --seed 42
```

If the template recomputes training scores after reference fitting, account for
that cost. Do not delete required score aggregation or change reference semantics
to hide the overhead. Changing the implementation is a separate software task.

## Reload, score, and evaluate

Use the inference bundle produced by the corresponding training run. The following
trust flag is appropriate for that known local artifact, not an arbitrary file.
`labels.json` must be the flat exact-ID map described in reference.md.

```powershell
& './.venv/Scripts/python.exe' cli_main.py create-anomaly-workflow 'outputs/experiment/test.workflow.json' --phase score --algorithm autoencoder --dataset 'D:/data/test' --model 'outputs/experiment/train/anomaly_model.anomaly.zip' --trust-model --labels 'outputs/experiment/labels.json' --protocol dcase --aggregation top_fraction_mean --json
& './.venv/Scripts/python.exe' cli_main.py validate 'outputs/experiment/test.workflow.json' --check-data --json
& './.venv/Scripts/python.exe' cli_main.py run 'outputs/experiment/test.workflow.json' --run-dir 'outputs/experiment/test' --seed 42
```

For a non-DCASE dataset, select `--protocol generic`. For a kNN bundle, use
`--algorithm knn` in generation. The scorer dispatches from the saved artifact.
If training used nondefault feature/window settings, update the generated scoring
graph to match before execution; passing a bundle does not yet infer those settings.

To compare mean aggregation, generate a *new* scoring graph with `--aggregation
mean` using the same bundle and input files, then run it in a new directory. Do not
retrain. To change the top fraction, distance, batch size, or window parameters
when no CLI flag exists, edit the registered node parameters in JSON and validate.

## Continuation graph

Start from the prior AE training graph, not the scoring template:

1. Point `load_model` at the prior ordinary compiled Keras checkpoint; retain
   `regression_trainer.use_model_config=True` and set the requested additional epochs.
2. Add `load_anomaly_model` for the matching trusted prior preprocessing state.
3. Change the fitting standardization node from `fit_transform` to `transform`,
   connect the loaded `preprocessing_state` to its `state` input, and continue
   feeding its resulting state to normal-validation standardization.
4. Keep original input role assignments and matching feature parameters. Preserve
   both anomaly packaging and ordinary Keras saving in the resumed graph.
5. Validate and execute in a new directory. Compare optimizer iterations before
   and after; inspect the best-epoch metadata rather than assuming the final epoch
   is the saved state. State the lack of complete RNG/callback continuation.

## Completion evidence

Record and check:

- Every command's arguments, exit code, duration, stdout/stderr locations.
- `manifest.json`: `success`, `node_outputs`, artifacts and hashes; source/graph
  configuration snapshots are preserved by the CLI runner.
- Generated trainer ID's `training_summary` and full epoch events. The manifest
  may truncate long history lists; do not treat a preview as all epochs.
- Generated evaluator ID's `metrics`: protocol, source/target section AUC where
  applicable, standardized pAUC, harmonic mean, and any decision metrics.
- `anomaly_results/results.csv`: all expected unique file IDs and finite scores.
- Normal-fitting/validation/test counts, content-overlap audit, model dimension,
  preprocessing signature, aggregation, seed and completed/best epochs.

Report script-based vs native-workflow migration separately from detector quality.
Do not attribute stochastic retraining variation or additional tuning to software
correctness improvements. Preserve unsuccessful runs for diagnosis.
