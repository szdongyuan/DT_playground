from enum import IntEnum


class ViewType(IntEnum):
    """View type enum shared across controllers and UI."""

    WORKFLOW = 0
    MODEL = 1
    PREVIEW = 2
    TRAINING = 3

