# -*- coding: utf-8 -*-
"""Declarative specifications for grouped command toolbars."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolbarActionSpec:
    """Describes one toolbar action button."""

    key: str
    text: str
    object_name: str
    variant: str = "normal"
    enabled: bool = True
    tooltip: str | None = None


@dataclass(frozen=True)
class ToolbarGroupSpec:
    """Describes one titled toolbar group."""

    key: str
    title: str
    actions: tuple[ToolbarActionSpec, ...]
