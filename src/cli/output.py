"""Keep machine events separate from incidental runtime output."""

import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout

from src.utils.runtime import quiet_training_output


@contextmanager
def isolated_event_output(writer):
    """Reserve stdout for JSONL during synchronous CLI execution.

    Real console descriptors also capture low-level writes and cached streams.
    In-memory streams use Python redirection only. All routing is restored on
    failure or cancellation; this process-wide routing is CLI-only.
    """
    if writer.mode != "jsonl":
        yield
        return
    original_stdout = sys.stdout
    original_writer = writer.stream
    saved_fd = None
    event_stream = None
    handlers = []
    try:
        try:
            stdout_fd = original_stdout.fileno()
            stderr_fd = sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            stdout_fd = stderr_fd = None
        if stdout_fd is not None and stdout_fd != stderr_fd:
            original_stdout.flush()
            saved_fd = os.dup(stdout_fd)
            if original_writer is original_stdout:
                event_stream = os.fdopen(os.dup(saved_fd), "w", encoding=original_stdout.encoding or "utf-8")
                writer.stream = event_stream
            os.dup2(stderr_fd, stdout_fd)

        # Existing logging handlers may retain the old Python stdout object.
        loggers = [logging.getLogger()] + [
            item for item in logging.Logger.manager.loggerDict.values()
            if isinstance(item, logging.Logger)
        ]
        seen = set()
        for logger in loggers:
            for handler in logger.handlers:
                if id(handler) not in seen and getattr(handler, "stream", None) is original_stdout:
                    seen.add(id(handler))
                    handlers.append((handler, original_stdout))
                    handler.setStream(sys.stderr)
        with quiet_training_output(), redirect_stdout(sys.stderr):
            yield
    finally:
        try:
            if saved_fd is not None:
                try:
                    original_stdout.flush()
                finally:
                    try:
                        os.dup2(saved_fd, stdout_fd)
                    finally:
                        os.close(saved_fd)
        finally:
            writer.stream = original_writer
            try:
                if event_stream is not None:
                    event_stream.close()
            finally:
                for handler, stream in handlers:
                    handler.setStream(stream)
