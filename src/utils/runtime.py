"""Runtime environment helpers."""

import sys


def get_training_verbose() -> int:
    """Enable Keras progress output only when running from source."""
    return 0 if getattr(sys, "frozen", False) else 1
