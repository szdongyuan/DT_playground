"""Command-line entry point for deterministic acoustic workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.cli.capabilities import get_capabilities
from src.cli.contracts import EventWriter, ExitCode, print_payload
from src.cli.dataset import inspect_dataset
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

    run = subparsers.add_parser("run", help="Execute a validated workflow.")
    run.add_argument("workflow", help="Workflow JSON file.")
    run.add_argument("--run-dir", required=True, help="Directory for manifest and event artifacts.")
    run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    run.add_argument("--overwrite", action="store_true", help="Clear an existing non-empty run directory.")
    run.add_argument("--events", choices=("text", "jsonl"), default="text", help="Progress output format.")
    run.add_argument("--checkpoint", help="Checkpoint path, relative to the run directory, when interrupted.")
    run.add_argument("--json", action="store_true", help="Print the final manifest as JSON.")
    return parser


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
