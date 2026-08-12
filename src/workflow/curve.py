# -*- coding: utf-8 -*-
"""Physical-coordinate curve data shared by analysis and visualization nodes."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from src.ui.i18n import tr_


@dataclass
class CurveData:
    """A multi-channel curve with explicit physical coordinates and metadata."""

    data: np.ndarray
    x: np.ndarray
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    curve_type: str
    channel_names: List[str] = field(default_factory=list)
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self.data)
        coordinates = np.asarray(self.x)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(tr_("Curve data must have shape (channels, points)"))
        if coordinates.ndim != 1:
            raise ValueError(tr_("Curve coordinates must be one-dimensional"))
        if values.shape[1] != coordinates.size:
            raise ValueError(tr_("Curve coordinates must match the number of points"))
        if values.shape[1] == 0:
            raise ValueError(tr_("Curve data cannot be empty"))
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(tr_("Curve coordinates must be finite"))
        if np.any(np.diff(coordinates) <= 0):
            raise ValueError(tr_("Curve coordinates must be strictly increasing"))

        self.data = values.astype(np.float64, copy=False)
        self.x = coordinates.astype(np.float64, copy=False)
        if not self.channel_names:
            self.channel_names = [f"Ch {index + 1}" for index in range(values.shape[0])]
        if len(self.channel_names) != values.shape[0]:
            raise ValueError(tr_("Channel names must match the curve channel count"))

    @property
    def channels(self) -> int:
        """Return the number of curve channels."""
        return self.data.shape[0]

    @property
    def points(self) -> int:
        """Return the number of points per channel."""
        return self.data.shape[1]

    def get_channel(self, channel_index: int) -> np.ndarray:
        """Return one curve channel."""
        return self.data[channel_index]
