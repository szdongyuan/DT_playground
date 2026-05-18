# -*- coding: utf-8 -*-
"""
Port and Data Type Definitions

Defines data types and port structures for data transfer between nodes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DataType(Enum):
    """
    Data type enumeration
    
    Note: All data types support single or batch (list) form, determined at runtime via isinstance.
    """
    
    # Audio data - supports AudioData or List[AudioData]
    AUDIO = "audio"
    
    # Feature data
    FEATURE_1D = "feature_1d"     # 1D features (e.g., statistical features)
    FEATURE_2D = "feature_2d"     # 2D features (e.g., Mel spectrogram)
    FEATURE = "feature"           # Generic feature type (compatible with 1D and 2D)
    
    # Label data - supports single or list
    LABEL = "label"

    # Dataset bundle - supports structured records, task specs, and lineage
    DATASET = "dataset"
    
    # Model related
    MODEL = "model"               # Keras/TensorFlow model
    METRICS = "metrics"           # Training/evaluation metrics
    
    # Generic
    ANY = "any"                   # Any type (for generic nodes)
    TRIGGER = "trigger"           # Trigger signal (for control flow)
    
    @classmethod
    def is_compatible(cls, source: 'DataType', target: 'DataType') -> bool:
        """
        Check if two data types are compatible
        
        Note: ANY type is bidirectionally compatible, allowing flexible connections.
        Actual type validation is performed at runtime in the node's execute() method.
        """
        # ANY type is bidirectionally compatible (runtime data type validation)
        if target == cls.ANY or source == cls.ANY:
            return True
        if source == target:
            return True
        # Feature type compatibility: specific feature types can connect to generic FEATURE type
        if target == cls.FEATURE and source in (cls.FEATURE_1D, cls.FEATURE_2D):
            return True
        if source == cls.FEATURE and target in (cls.FEATURE_1D, cls.FEATURE_2D):
            return True
        return False


@dataclass
class Port:
    """Node port definition"""
    
    name: str                           # Port name (unique identifier)
    display_name: str                   # Display name
    data_type: DataType                 # Data type
    is_input: bool                      # True=input port, False=output port
    required: bool = True               # Whether connection is required (only for input ports)
    multi_connection: bool = False      # Whether multiple connections are allowed
    default_value: Any = None           # Default value (only for input ports)
    description: str = ""               # Port description
    
    # Runtime data
    _data: Any = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)
    
    @property
    def data(self) -> Any:
        """Get port data"""
        return self._data
    
    @data.setter
    def data(self, value: Any):
        """Set port data"""
        self._data = value
    
    @property
    def is_connected(self) -> bool:
        """Whether port is connected"""
        return self._connected
    
    def set_connected(self, connected: bool):
        """Set connection status"""
        self._connected = connected
    
    def clear(self):
        """Clear port data"""
        self._data = None
    
    def can_connect_to(self, other: 'Port') -> bool:
        """Check if can connect to another port"""
        # Must be one input and one output
        if self.is_input == other.is_input:
            return False
        # Check data type compatibility
        if self.is_input:
            return DataType.is_compatible(other.data_type, self.data_type)
        else:
            return DataType.is_compatible(self.data_type, other.data_type)


def create_input_port(
    name: str,
    data_type: DataType,
    display_name: str = None,
    required: bool = True,
    default_value: Any = None,
    description: str = ""
) -> Port:
    """Convenience function to create input port"""
    return Port(
        name=name,
        display_name=display_name or name,
        data_type=data_type,
        is_input=True,
        required=required,
        default_value=default_value,
        description=description
    )


def create_output_port(
    name: str,
    data_type: DataType,
    display_name: str = None,
    multi_connection: bool = True,
    description: str = ""
) -> Port:
    """Convenience function to create output port"""
    return Port(
        name=name,
        display_name=display_name or name,
        data_type=data_type,
        is_input=False,
        multi_connection=multi_connection,
        description=description
    )

