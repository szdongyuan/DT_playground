"""Command-line entry point for deterministic acoustic workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

# Keep third-party runtime diagnostics out of normal CLI output unless the
# caller explicitly requests a more verbose TensorFlow log level.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.cli.capabilities import get_capabilities
from src.cli.classifier import ClassifierTrainingRequest, train_classifier
from src.cli.contracts import EventWriter, ExitCode, print_payload
from src.cli.dataset import inspect_dataset
from src.cli.generation import (
    ModelGenerationRequest,
    WorkflowGenerationRequest,
    create_model_definition,
    create_workflow_definition,
)
from src.cli.runner import execute_workflow
from src.cli.validation import validate_model_file, validate_workflow_file


def _seed_value(value: str) -> int:
    """Accept seeds supported by NumPy and Keras without implicit wrapping."""
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed must be an integer.") from exc
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("Seed must be between 0 and 4294967295.")
    return seed


def build_parser() -> argparse.ArgumentParser:
    """Create the public CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="audio-platform",
        description="Deterministic headless tools for audio/acoustic workflows.",
    )
    parser.add_argument("--version", action="version", version="audio-platform CLI 1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities = subparsers.add_parser("capabilities", help="Describe nodes and model layers.")
    capabilities.add_argument("--json", action="store_true", help="Emit the complete JSON contract.")
    capabilities.add_argument("--include-hidden", action="store_true", help="Include legacy hidden nodes.")

    inspect = subparsers.add_parser("inspect", help="Profile an audio dataset.")
    inspect.add_argument("dataset", help="Dataset directory.")
    inspect.add_argument("--labels", help="Optional CSV or JSON filename-to-label mapping.")
    inspect.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories.")
    inspect.add_argument("--output", help="Write the JSON profile to this file.")
    inspect.add_argument("--json", action="store_true", help="Print JSON to stdout.")

    validate = subparsers.add_parser("validate", help="Validate workflow and model definitions.")
    validate.add_argument("workflow", help="Workflow JSON file.")
    validate.add_argument("--model", help="Optional model definition JSON file.")
    validate.add_argument("--build-model", action="store_true", help="Build the model in memory to validate shapes.")
    validate.add_argument("--check-data", action="store_true", help="Read audio and check directly connected target alignment without running the workflow.")
    validate.add_argument("--json", action="store_true", help="Print JSON to stdout.")

    build = subparsers.add_parser("build-model", help="Build a Keras model from a model definition.")
    build.add_argument("model", help="Model definition JSON file.")
    build.add_argument("--output", required=True, help="Destination .keras or .h5 file.")
    build.add_argument("--no-compile", action="store_true", help="Do not compile the model.")
    build.add_argument("--seed", type=_seed_value, default=42, help="Model initialization seed (default: 42).")
    build.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    build.add_argument("--json", action="store_true", help="Print JSON to stdout.")

    create_model = subparsers.add_parser(
        "create-model", help="Generate a validated model definition."
    )
    create_model.add_argument("output", help="Destination model definition JSON file.")
    create_model.add_argument("--task", choices=("classification",), default="classification")
    create_model.add_argument("--template", choices=("mel-cnn",), default="mel-cnn")
    create_model.add_argument("--classes", type=int, required=True, help="Number of classes.")
    create_model.add_argument("--input-shape", default="(64, 94, 1)")
    create_model.add_argument("--learning-rate", type=float, default=0.001)
    create_model.add_argument("--dropout", type=float, default=0.4)
    create_model.add_argument("--name", default="mel_cnn_classifier")
    create_model.add_argument("--overwrite", action="store_true")
    create_model.add_argument("--json", action="store_true", help="Print JSON to stdout.")

    anomaly_workflow = subparsers.add_parser("create-anomaly-workflow", help="Generate a native anomaly training or scoring workflow.")
    anomaly_workflow.add_argument("output")
    anomaly_workflow.add_argument("--dataset", required=True)
    anomaly_workflow.add_argument("--phase", choices=("train", "score"), default="train")
    anomaly_workflow.add_argument("--algorithm", choices=("knn", "autoencoder"), default="knn")
    anomaly_workflow.add_argument("--model")
    anomaly_workflow.add_argument("--validation-dataset")
    anomaly_workflow.add_argument("--labels", help="Exact scored sample ID to binary label mapping; evaluation only.")
    anomaly_workflow.add_argument("--protocol", choices=("generic", "dcase"), default="generic")
    anomaly_workflow.add_argument("--aggregation", choices=("none", "mean", "max", "top_fraction_mean"), default="top_fraction_mean")
    anomaly_workflow.add_argument("--k", type=int, default=5)
    anomaly_workflow.add_argument("--epochs", type=int, default=40)
    anomaly_workflow.add_argument("--trust-model", action="store_true")
    anomaly_workflow.add_argument("--overwrite", action="store_true")
    anomaly_workflow.add_argument("--json", action="store_true")

    create_workflow = subparsers.add_parser(
        "create-workflow", help="Generate a validated classification workflow."
    )
    create_workflow.add_argument("output", help="Destination workflow JSON file.")
    create_workflow.add_argument("--task", choices=("classification",), default="classification")
    create_workflow.add_argument("--dataset", required=True, help="Audio dataset directory.")
    create_workflow.add_argument("--labels", required=True, help="CSV or JSON label mapping.")
    create_workflow.add_argument("--model", required=True, help="Built .keras or .h5 model artifact.")
    create_workflow.add_argument("--output-model", required=True, help="Saved model file name.")
    _add_workflow_options(create_workflow)
    create_workflow.add_argument("--overwrite", action="store_true")
    create_workflow.add_argument("--json", action="store_true", help="Print JSON to stdout.")

    train_classifier_parser = subparsers.add_parser(
        "train-classifier", help="Generate and run a classification experiment."
    )
    train_classifier_parser.add_argument("--dataset", required=True, help="Audio dataset directory.")
    train_classifier_parser.add_argument("--labels", required=True, help="CSV or JSON label mapping.")
    train_classifier_parser.add_argument("--experiment-dir", required=True)
    train_classifier_parser.add_argument("--template", choices=("mel-cnn",), default="mel-cnn")
    _add_workflow_options(train_classifier_parser)
    train_classifier_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_classifier_parser.add_argument("--dropout", type=float, default=0.4)
    train_classifier_parser.add_argument("--model-name", default="classifier")
    train_classifier_parser.add_argument("--overwrite", action="store_true")
    train_classifier_parser.add_argument("--events", choices=("text", "jsonl"), default="text")
    train_classifier_parser.add_argument("--checkpoint")
    train_classifier_parser.add_argument("--json", action="store_true", help="Print the final result as JSON.")

    run = subparsers.add_parser("run", help="Execute a validated workflow.")
    run.add_argument("workflow", help="Workflow JSON file.")
    run.add_argument("--run-dir", required=True, help="Directory for manifest and event artifacts.")
    run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    run.add_argument("--overwrite", action="store_true", help="Clear an existing non-empty run directory.")
    run.add_argument("--events", choices=("text", "jsonl"), default="text", help="Progress output format.")
    run.add_argument("--checkpoint", help="Checkpoint path, relative to the run directory, when interrupted.")
    run.add_argument("--json", action="store_true", help="Print the final manifest as JSON.")
    return parser


def _add_workflow_options(parser: argparse.ArgumentParser) -> None:
    """Add shared classification feature, split, and training options."""
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=_seed_value, default=42)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capabilities":
            payload = get_capabilities(include_hidden=args.include_hidden)
            if args.json:
                print_payload(payload, True)
            else:
                print(f"{len(payload['nodes'])} nodes, {len(payload['layers'])} model layers")
            return int(ExitCode.SUCCESS)

        if args.command == "inspect":
            payload = inspect_dataset(
                args.dataset,
                recursive=not args.no_recursive,
                labels_path=args.labels,
            )
            if args.output:
                output = Path(args.output).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.json or not args.output:
                print_payload(payload, args.json)
                if not args.json:
                    summary = payload["summary"]
                    print(f"Valid audio: {summary['valid_files']}; invalid audio: {summary['invalid_files']}")
            return int(ExitCode.SUCCESS if payload["summary"]["valid_files"] else ExitCode.VALIDATION_ERROR)

        if args.command == "validate":
            workflow_report, _ = validate_workflow_file(args.workflow, headless=True, check_data=args.check_data)
            payload = {"workflow": workflow_report.to_dict()}
            valid = workflow_report.valid
            if args.model:
                model_report, _ = validate_model_file(args.model, build=args.build_model)
                payload["model"] = model_report.to_dict()
                valid = valid and model_report.valid
            payload["valid"] = valid
            payload["summary"] = "Definitions are valid." if valid else "Definition validation failed."
            if args.check_data:
                payload["summary"] = "Validation passed for the reported data-check scope." if valid else "Definition or data validation failed."
            print_payload(payload, args.json, sys.stdout if valid else sys.stderr)
            if args.check_data and not args.json:
                for issue in workflow_report.errors + workflow_report.warnings:
                    print(f"{issue['code']}: {issue['message']} ({issue['context']})", file=sys.stdout if valid else sys.stderr)
            return int(ExitCode.SUCCESS if valid else ExitCode.VALIDATION_ERROR)

        if args.command == "build-model":
            report, graph = validate_model_file(args.model)
            if not report.valid or graph is None:
                print_payload({"summary": "Model validation failed.", "validation": report.to_dict()}, args.json, sys.stderr)
                return int(ExitCode.VALIDATION_ERROR)
            output = Path(args.output).expanduser().resolve()
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Model output already exists: {output}")
            from tensorflow import keras

            keras.utils.set_random_seed(args.seed)
            try:
                model = graph.build_keras_model(compile_model=not args.no_compile)
            except Exception as exc:
                report.error("model.build", str(exc))
                print_payload({"summary": "Model validation failed.", "validation": report.to_dict()}, args.json, sys.stderr)
                return int(ExitCode.VALIDATION_ERROR)
            output.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output))
            payload = {
                "summary": f"Model built: {output}",
                "model_definition": str(Path(args.model).resolve()),
                "output": str(output),
                "compiled": not args.no_compile,
                "seed": args.seed,
            }
            print_payload(payload, args.json)
            return int(ExitCode.SUCCESS)

        if args.command == "create-model":
            try:
                payload = create_model_definition(
                    ModelGenerationRequest(
                        output=args.output,
                        classes=args.classes,
                        task=args.task,
                        template=args.template,
                        input_shape=args.input_shape,
                        learning_rate=args.learning_rate,
                        dropout=args.dropout,
                        name=args.name,
                        overwrite=args.overwrite,
                    )
                )
            except ValueError as exc:
                print_payload({"summary": str(exc), "valid": False}, args.json, sys.stderr)
                return int(ExitCode.VALIDATION_ERROR)
            print_payload(payload, args.json)
            return int(ExitCode.SUCCESS)

        if args.command == "create-anomaly-workflow":
            from .anomaly_generation import create_anomaly_workflow
            try:
                payload = create_anomaly_workflow(args.output, args.dataset, phase=args.phase, algorithm=args.algorithm,
                    model=args.model, validation_dataset=args.validation_dataset, aggregation=args.aggregation,
                    k=args.k, epochs=args.epochs, overwrite=args.overwrite, trust_model=args.trust_model,
                    labels=args.labels, protocol=args.protocol)
            except ValueError as exc:
                print_payload({"summary": str(exc), "valid": False}, args.json, sys.stderr)
                return int(ExitCode.VALIDATION_ERROR)
            print_payload(payload, args.json)
            return int(ExitCode.SUCCESS)

        if args.command == "create-workflow":
            try:
                payload = create_workflow_definition(
                    WorkflowGenerationRequest(
                        output=args.output,
                        dataset=args.dataset,
                        labels=args.labels,
                        model=args.model,
                        output_model=args.output_model,
                        task=args.task,
                        sample_rate=args.sample_rate,
                        duration=args.duration,
                        n_mels=args.n_mels,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        train_ratio=args.train_ratio,
                        val_ratio=args.val_ratio,
                        test_ratio=args.test_ratio,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        patience=args.patience,
                        seed=args.seed,
                        overwrite=args.overwrite,
                    )
                )
            except ValueError as exc:
                print_payload({"summary": str(exc), "valid": False}, args.json, sys.stderr)
                return int(ExitCode.VALIDATION_ERROR)
            print_payload(payload, args.json)
            return int(ExitCode.SUCCESS)

        if args.command == "train-classifier":
            try:
                success, payload = train_classifier(
                    ClassifierTrainingRequest(
                        dataset=args.dataset,
                        labels=args.labels,
                        experiment_dir=args.experiment_dir,
                        template=args.template,
                        sample_rate=args.sample_rate,
                        duration=args.duration,
                        n_mels=args.n_mels,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        train_ratio=args.train_ratio,
                        val_ratio=args.val_ratio,
                        test_ratio=args.test_ratio,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        dropout=args.dropout,
                        model_name=args.model_name,
                        seed=args.seed,
                        overwrite=args.overwrite,
                        checkpoint=args.checkpoint,
                    ),
                    writer=EventWriter(args.events),
                )
            except ValueError as exc:
                print_payload({"summary": str(exc), "valid": False}, args.json, sys.stderr)
                return int(ExitCode.VALIDATION_ERROR)
            if args.json and args.events != "jsonl":
                print_payload(payload, True)
            if payload.get("status") == "validation_failed":
                return int(ExitCode.VALIDATION_ERROR)
            return int(ExitCode.SUCCESS if success else ExitCode.EXECUTION_ERROR)

        if args.command == "run":
            success, manifest = execute_workflow(
                args.workflow,
                args.run_dir,
                writer=EventWriter(args.events),
                seed=args.seed,
                overwrite=args.overwrite,
                checkpoint=args.checkpoint,
            )
            if args.json and args.events != "jsonl":
                print_payload(manifest, True)
            if manifest.get("status") == "validation_failed":
                return int(ExitCode.VALIDATION_ERROR)
            return int(ExitCode.SUCCESS if success else ExitCode.EXECUTION_ERROR)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return int(ExitCode.IO_ERROR)
    except Exception as exc:
        print(f"Unexpected CLI error: {exc}", file=sys.stderr)
        return int(ExitCode.EXECUTION_ERROR)
    return int(ExitCode.USAGE_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
