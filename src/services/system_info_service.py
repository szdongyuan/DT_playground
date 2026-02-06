from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    """GPU detection result with a stable shape."""

    available: bool
    device_count: int
    device_names: List[str]
    backend: str  # e.g. "tensorflow", "missing", "error"
    error: Optional[str] = None


@dataclass(frozen=True)
class MemoryInfo:
    """Memory information with a stable shape."""

    total_gb: float
    used_gb: float
    backend: str  # e.g. "psutil", "missing", "error"
    error: Optional[str] = None


class SystemInfoService:
    """
    Provide system/environment information for the application.

    Notes:
    - This module must remain UI-agnostic.
    - Heavy dependencies (e.g. TensorFlow) are imported lazily and results are cached.
    """

    def __init__(self) -> None:
        self._gpu_info_cache: Optional[GPUInfo] = None

    def get_gpu_info(self, *, force_refresh: bool = False) -> GPUInfo:
        """
        Detect GPU availability via TensorFlow if installed.

        Returns a stable `GPUInfo` shape even when TensorFlow is missing or errors.
        """
        if self._gpu_info_cache is not None and not force_refresh:
            return self._gpu_info_cache

        try:
            tf = import_module("tensorflow")
        except Exception as e:
            info = GPUInfo(
                available=False,
                device_count=0,
                device_names=[],
                backend="missing",
                error=str(e),
            )
            self._gpu_info_cache = info
            return info

        try:
            gpus = tf.config.list_physical_devices("GPU")
            names: List[str] = []
            for gpu in gpus:
                name = getattr(gpu, "name", None)
                names.append(name if name else str(gpu))
            info = GPUInfo(
                available=len(gpus) > 0,
                device_count=len(gpus),
                device_names=names,
                backend="tensorflow",
                error=None,
            )
            self._gpu_info_cache = info
            return info
        except Exception as e:
            logger.exception("GPU detection failed.")
            info = GPUInfo(
                available=False,
                device_count=0,
                device_names=[],
                backend="error",
                error=str(e),
            )
            self._gpu_info_cache = info
            return info

    def get_memory_info(self) -> MemoryInfo:
        """
        Return memory usage via psutil if installed.

        If psutil is missing or errors, returns a stable `MemoryInfo` shape.
        """
        try:
            psutil = import_module("psutil")
        except Exception as e:
            return MemoryInfo(total_gb=0.0, used_gb=0.0, backend="missing", error=str(e))

        try:
            memory = psutil.virtual_memory()
            used_gb = float(memory.used) / (1024.0**3)
            total_gb = float(memory.total) / (1024.0**3)
            return MemoryInfo(total_gb=total_gb, used_gb=used_gb, backend="psutil", error=None)
        except Exception as e:
            logger.exception("Memory detection failed.")
            return MemoryInfo(total_gb=0.0, used_gb=0.0, backend="error", error=str(e))

