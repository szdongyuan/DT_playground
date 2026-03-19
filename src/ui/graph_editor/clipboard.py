# -*- coding: utf-8 -*-
"""
Shared graph clipboard helpers.

Stores structured payloads in the system clipboard as JSON text.
"""

import json
from typing import Any, Dict, Optional

from PyQt6.QtWidgets import QApplication


GRAPH_CLIPBOARD_VERSION = 1
GRAPH_CLIPBOARD_MARKER = "dt_graph_clipboard"


def set_graph_clipboard_data(kind: str, payload: Dict[str, Any]) -> None:
    clipboard_payload = {
        "marker": GRAPH_CLIPBOARD_MARKER,
        "version": GRAPH_CLIPBOARD_VERSION,
        "kind": kind,
        "payload": payload,
    }
    QApplication.clipboard().setText(json.dumps(clipboard_payload, ensure_ascii=False))


def get_graph_clipboard_data(expected_kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    text = QApplication.clipboard().text() or ""
    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if data.get("marker") != GRAPH_CLIPBOARD_MARKER:
        return None

    if expected_kind is not None and data.get("kind") != expected_kind:
        return None

    payload = data.get("payload")
    if not isinstance(payload, dict):
        return None

    return payload


def has_graph_clipboard_data(expected_kind: Optional[str] = None) -> bool:
    return get_graph_clipboard_data(expected_kind) is not None
