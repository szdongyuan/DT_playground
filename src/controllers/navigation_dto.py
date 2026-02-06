from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .view_types import ViewType


@dataclass(frozen=True)
class NodeDoubleClickContext:
    """Context required to decide what to do on node double-click."""

    node_id: str
    node_type: str
    category: str
    display_name: str
    has_output: bool
    outputs: Dict[str, Any]


@dataclass(frozen=True)
class MessageBoxSpec:
    """
    UI message box request produced by a controller.

    `title` and `message` should already be translated via `tr_()` by the controller.
    """

    level: str  # "info" | "warning"
    title: str
    message: str


@dataclass(frozen=True)
class NodeDoubleClickDecision:
    """Decision for node double-click handling."""

    handled: bool = False
    switch_to: Optional[ViewType] = None
    preview_payload: Optional[Tuple[str, str, Dict[str, Any]]] = None  # (node_id, node_name, outputs)
    training_history: Optional[Any] = None
    message_box: Optional[MessageBoxSpec] = None
    show_not_run_tip: bool = False

