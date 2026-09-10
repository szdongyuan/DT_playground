"""Stable output contracts shared by CLI commands."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, TextIO


CLI_SCHEMA_VERSION = "1.0"


class ExitCode(IntEnum):
    """Process exit codes exposed by the CLI."""

    SUCCESS = 0
    USAGE_ERROR = 2
    VALIDATION_ERROR = 3
    EXECUTION_ERROR = 4
    IO_ERROR = 5
    INTERRUPTED = 130


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common runtime values into bounded JSON-safe summaries."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            return {
                "type": type(value).__name__,
                "length": len(value),
                "preview": [json_safe(item) for item in value[:5]],
            }
        return [json_safe(item) for item in value]
    if is_dataclass(value):
        return json_safe(asdict(value))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return json_safe(value.item())
    name = getattr(value, "name", None)
    return {
        "type": type(value).__name__,
        **({"name": str(name)} if name else {}),
    }


class EventWriter:
    """Write human-readable messages or versioned JSON Lines events."""

    def __init__(self, mode: str = "text", stream: TextIO | None = None):
        self.mode = mode
        self.stream = stream or sys.stdout

    def emit(self, event: str, message: str = "", **data: Any) -> dict[str, Any]:
        """Emit one event and return the normalized event record."""
        record = {
            "schema_version": CLI_SCHEMA_VERSION,
            "timestamp": utc_now(),
            "event": event,
            "message": message,
            "data": json_safe(data),
        }
        if self.mode == "jsonl":
            print(json.dumps(record, ensure_ascii=False), file=self.stream, flush=True)
        elif message:
            print(message, file=self.stream, flush=True)
        return record


def print_payload(payload: dict[str, Any], as_json: bool, stream: TextIO | None = None) -> None:
    """Print a command result in JSON or concise human-readable form."""
    target = stream or sys.stdout
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), file=target)
        return
    summary = payload.get("summary") or payload.get("message")
    if isinstance(summary, str):
        print(summary, file=target)
