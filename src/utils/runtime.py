"""Runtime environment helpers."""

import sys
from contextlib import contextmanager
from contextvars import ContextVar


_quiet_training = ContextVar("quiet_training", default=False)


@contextmanager
def quiet_training_output():
    """Temporarily suppress Keras bars without changing GUI defaults."""
    token = _quiet_training.set(True)
    try:
        yield
    finally:
        _quiet_training.reset(token)


def get_training_verbose() -> int:
    """Enable Keras progress output only when running from source."""
    return 0 if _quiet_training.get() or getattr(sys, "frozen", False) else 1
