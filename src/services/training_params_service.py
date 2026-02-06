from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TrainingParamsService:
    """Derive training-related parameters from a workflow (UI-agnostic)."""

    @staticmethod
    def derive_total_epochs(workflow: Any, *, default_epochs: int = 20) -> int:
        """
        Derive total epochs from workflow nodes.

        Current rule (kept consistent with legacy UI logic):
        - Start from `default_epochs`
        - Iterate nodes in insertion order
        - Use the first truthy `node.get_parameter(\"epochs\")` value if present
        - Fall back to `default_epochs` on any error
        """
        total_epochs = default_epochs
        nodes = getattr(workflow, "nodes", None)
        if not isinstance(nodes, dict):
            return total_epochs

        for node in nodes.values():
            get_parameter = getattr(node, "get_parameter", None)
            if not callable(get_parameter):
                continue
            try:
                epochs = get_parameter("epochs")
            except Exception:
                continue

            if not epochs:
                continue

            try:
                epochs_int = int(epochs)
            except Exception:
                logger.debug("Invalid epochs value: %r", epochs)
                continue

            if epochs_int > 0:
                total_epochs = epochs_int
                break

        return total_epochs

