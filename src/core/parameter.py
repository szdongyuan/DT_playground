# -*- coding: utf-8 -*-
"""
统一参数定义

为节点和层提供通用的参数类型定义。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class ParamType(Enum):
    """参数类型枚举"""
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
    通用参数定义
    
    用于节点参数和层参数的统一表示。
    """
    name: str                           # 参数名称（唯一标识）
    display_name: str                   # 显示名称
    param_type: ParamType               # 参数类型
    default_value: Any                  # 默认值
    description: str = ""               # 描述
    min_value: Optional[Any] = None     # 最小值（数值类型）
    max_value: Optional[Any] = None     # 最大值（数值类型）
    choices: Optional[List[Any]] = None # 选项列表（choice类型）
    file_filter: str = ""               # 文件过滤器（file类型）
    required: bool = True               # 是否必需
    
    def validate(self, value: Any) -> Tuple[bool, str]:
        """
        验证参数值
        
        Args:
            value: 待验证的值
            
        Returns:
            (is_valid, error_message)
        """
        if value is None:
            if self.required:
                return False, f"参数 {self.display_name} 是必需的"
            return True, ""
        
        if self.param_type == ParamType.INT:
            if not isinstance(value, int):
                return False, f"参数 {self.display_name} 必须是整数"
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 {self.display_name} 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 {self.display_name} 不能大于 {self.max_value}"
                
        elif self.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"参数 {self.display_name} 必须是数值"
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 {self.display_name} 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 {self.display_name} 不能大于 {self.max_value}"
                
        elif self.param_type == ParamType.BOOL:
            if not isinstance(value, bool):
                return False, f"参数 {self.display_name} 必须是布尔值"
                
        elif self.param_type == ParamType.CHOICE:
            if self.choices and value not in self.choices:
                return False, f"参数 {self.display_name} 必须是 {self.choices} 之一"
                
        elif self.param_type == ParamType.STRING:
            if not isinstance(value, str):
                return False, f"参数 {self.display_name} 必须是字符串"
        
        return True, ""
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "param_type": self.param_type.value,
            "default_value": self.default_value,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "choices": self.choices,
            "file_filter": self.file_filter,
            "required": self.required,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Parameter':
        """从字典反序列化"""
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            param_type=ParamType(data["param_type"]),
            default_value=data["default_value"],
            description=data.get("description", ""),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            choices=data.get("choices"),
            file_filter=data.get("file_filter", ""),
            required=data.get("required", True),
        )


def create_parameter(
    name: str,
    param_type: ParamType,
    default_value: Any,
    display_name: str = None,
    description: str = "",
    min_value: Any = None,
    max_value: Any = None,
    choices: List[Any] = None,
    file_filter: str = "",
    required: bool = True
) -> Parameter:
    """
    创建参数的便捷函数
    
    Args:
        name: 参数名称
        param_type: 参数类型
        default_value: 默认值
        display_name: 显示名称（默认使用name）
        description: 描述
        min_value: 最小值
        max_value: 最大值
        choices: 选项列表
        file_filter: 文件过滤器
        required: 是否必需
        
    Returns:
        Parameter 实例
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
        required=required,
    )


