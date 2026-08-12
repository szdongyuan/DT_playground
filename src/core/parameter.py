# -*- coding: utf-8 -*-
"""
Unified Parameter Definitions

Provides common parameter type definitions for nodes and layers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from src.ui.i18n import tr_


class ParamType(Enum):
    """Parameter type enumeration"""
    INT = "int"
    FLOAT = "float"
    STRING = "str"
    BOOL = "bool"
    CHOICE = "choice"
    FILE = "file"
    FOLDER = "folder"
    TUPLE = "tuple"
    LIST = "list"


@dataclass
class Parameter:
    """
    Universal parameter definition
    
    Used for unified representation of node parameters and layer parameters.
    Supports multiple parameter types including numeric, string, choice, file path, etc.
    """
    name: str                               # Parameter name (unique identifier)
    display_name: str                       # Display name
    param_type: "ParamType | str"           # Parameter type (supports enum or string)
    default_value: Any                      # Default value
    description: str = ""                   # Description
    min_value: Optional[Any] = None         # Minimum value (for numeric types)
    max_value: Optional[Any] = None         # Maximum value (for numeric types)
    choices: Optional[List[Any]] = None     # Options list (for choice type)
    file_filter: str = ""                   # File filter (for file type)
    default_directory: str = ""             # File dialog default directory (for file/folder types)
    required: bool = True                   # Whether required
    visible_when: Optional[Any] = None            # Parameter dependency rules
    
    def _get_param_type_str(self) -> str:
        """Get parameter type string (compatible with enum and string)"""
        if isinstance(self.param_type, ParamType):
            return self.param_type.value
        return str(self.param_type)
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """
        Validate parameter value
        
        Args:
            value: Value to validate
            
        Returns:
            (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, tr_("Parameter '{name}' is required").format(
                    name=self.display_name
                )
            return True, ""
        
        # Get type string, compatible with enum and string
        ptype = self._get_param_type_str()
        
        if ptype == "int":
            if not isinstance(value, int):
                return False, tr_("Parameter '{name}' must be an integer").format(
                    name=self.display_name
                )
            if self.min_value is not None and value < self.min_value:
                return False, tr_("Parameter '{name}' must be >= {min_value}").format(
                    name=self.display_name,
                    min_value=self.min_value,
                )
            if self.max_value is not None and value > self.max_value:
                return False, tr_("Parameter '{name}' must be <= {max_value}").format(
                    name=self.display_name,
                    max_value=self.max_value,
                )
                
        elif ptype == "float":
            if not isinstance(value, (int, float)):
                return False, tr_("Parameter '{name}' must be a number").format(
                    name=self.display_name
                )
            if self.min_value is not None and value < self.min_value:
                return False, tr_("Parameter '{name}' must be >= {min_value}").format(
                    name=self.display_name,
                    min_value=self.min_value,
                )
            if self.max_value is not None and value > self.max_value:
                return False, tr_("Parameter '{name}' must be <= {max_value}").format(
                    name=self.display_name,
                    max_value=self.max_value,
                )
                
        elif ptype == "bool":
            if not isinstance(value, bool):
                return False, tr_("Parameter '{name}' must be a boolean").format(
                    name=self.display_name
                )
                
        elif ptype == "choice":
            if self.choices and value not in self.choices:
                return False, tr_("Parameter '{name}' must be one of {choices}").format(
                    name=self.display_name,
                    choices=self.choices,
                )
                
        elif ptype == "str":
            if not isinstance(value, str):
                return False, tr_("Parameter '{name}' must be a string").format(
                    name=self.display_name
                )
        
        return True, ""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "param_type": self._get_param_type_str(),
            "default_value": self.default_value,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "choices": self.choices,
            "file_filter": self.file_filter,
            "default_directory": self.default_directory,
            "required": self.required,
            "visible_when": self.visible_when,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Parameter':
        """Deserialize from dictionary"""
        # Compatible with enum value and string
        param_type_raw = data["param_type"]
        try:
            param_type = ParamType(param_type_raw)
        except (ValueError, KeyError):
            param_type = param_type_raw  # Keep as string
        
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            param_type=param_type,
            default_value=data["default_value"],
            description=data.get("description", ""),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            choices=data.get("choices"),
            file_filter=data.get("file_filter", ""),
            default_directory=data.get("default_directory", ""),
            required=data.get("required", True),
            visible_when=data.get("visible_when"),
        )


def create_parameter(
    name: str,
    param_type: "ParamType | str",
    default_value: Any,
    display_name: str = None,
    description: str = "",
    min_value: Any = None,
    max_value: Any = None,
    choices: List[Any] = None,
    file_filter: str = "",
    default_directory: str = "",
    required: bool = True,
    visible_when: Optional[Any] = None,
) -> Parameter:
    """
    Convenience function to create parameters
    
    Args:
        name: Parameter name
        param_type: Parameter type (ParamType enum or string)
        default_value: Default value
        display_name: Display name (defaults to name)
        description: Description
        min_value: Minimum value
        max_value: Maximum value
        choices: Options list
        file_filter: File filter
        default_directory: File dialog default directory
        required: Whether required
        
    Returns:
        Parameter instance
    """
    return Parameter(
        name=name,
        display_name=display_name or name,
        param_type=param_type,
        default_value=default_value,
        description=description,
        min_value=min_value,
        max_value=max_value,
        choices=choices,
        file_filter=file_filter,
        default_directory=default_directory,
        required=required,
        visible_when=visible_when,
    )


# Backward compatibility type aliases
# NodeParameter and LayerParameter now uniformly use Parameter
NodeParameter = Parameter
LayerParameter = Parameter

