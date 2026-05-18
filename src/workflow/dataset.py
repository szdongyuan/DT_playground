# -*- coding: utf-8 -*-
"""
Workflow dataset contract.

These structures describe source data without requiring it to be immediately
usable as model training arrays. Downstream nodes can progressively transform
the same records while preserving identity, task metadata, and lineage.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(Enum):
    """Supported machine learning task families."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    GENERATION = "generation"
    DECISION = "decision"


@dataclass
class DataRecord:
    """Single dataset sample with optional target and metadata."""

    record_id: str
    input: Any
    target: Optional[Any] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """Task-level expectations for dataset consumers."""

    task_type: TaskType = TaskType.CLASSIFICATION
    target_required: bool = True
    input_schema: Dict[str, Any] = field(default_factory=dict)
    target_schema: Dict[str, Any] = field(default_factory=dict)
    metric_hints: List[str] = field(default_factory=list)


@dataclass
class DatasetSchema:
    """Schema hints for source records."""

    input_schema: Dict[str, Any] = field(default_factory=dict)
    target_schema: Dict[str, Any] = field(default_factory=dict)
    metadata_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetStats:
    """Lightweight summary of a dataset bundle."""

    total_records: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    target_distribution: Dict[str, Any] = field(default_factory=dict)
    quality_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetLineage:
    """Data source and transformation provenance."""

    sources: List[str] = field(default_factory=list)
    version: str = "1.0"
    transforms: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """Runtime bundle emitted by data source nodes."""

    records: List[DataRecord] = field(default_factory=list)
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    task_spec: TaskSpec = field(default_factory=TaskSpec)
    splits: Dict[str, List[str]] = field(default_factory=dict)
    stats: DatasetStats = field(default_factory=DatasetStats)
    lineage: DatasetLineage = field(default_factory=DatasetLineage)

    def __post_init__(self):
        if self.stats.total_records == 0:
            self.stats.total_records = len(self.records)
