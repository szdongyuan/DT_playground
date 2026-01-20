# -*- coding: utf-8 -*-
"""
端口和数据类型定义

定义节点之间传递数据的类型和端口结构。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DataType(Enum):
    """
    数据类型枚举
    
    注意：所有数据类型均支持单个或批量（列表）形式，运行时通过 isinstance 判断。
    """
    
    # 音频数据 - 支持 AudioData 或 List[AudioData]
    AUDIO = "audio"
    
    # 特征数据
    FEATURE_1D = "feature_1d"     # 1D特征 (如统计特征)
    FEATURE_2D = "feature_2d"     # 2D特征 (如Mel频谱图)
    FEATURE = "feature"           # 通用特征类型（兼容1D和2D）
    
    # 标签数据 - 支持单个或列表
    LABEL = "label"
    
    # 模型相关
    MODEL = "model"               # Keras/TensorFlow 模型
    METRICS = "metrics"           # 训练/评估指标
    
    # 通用
    ANY = "any"                   # 任意类型（用于通用节点）
    TRIGGER = "trigger"           # 触发信号（用于控制流）
    
    @classmethod
    def is_compatible(cls, source: 'DataType', target: 'DataType') -> bool:
        """
        检查两个数据类型是否兼容
        
        注意：ANY 类型双向兼容，允许灵活连接。
        实际类型验证在节点的 execute() 方法中进行运行时检查。
        """
        # ANY 类型双向兼容（运行时验证数据类型）
        if target == cls.ANY or source == cls.ANY:
            return True
        if source == target:
            return True
        # 特征类型兼容：具体特征类型可以连接到通用 FEATURE 类型
        if target == cls.FEATURE and source in (cls.FEATURE_1D, cls.FEATURE_2D):
            return True
        if source == cls.FEATURE and target in (cls.FEATURE_1D, cls.FEATURE_2D):
            return True
        return False


@dataclass
class Port:
    """节点端口定义"""
    
    name: str                           # 端口名称（唯一标识）
    display_name: str                   # 显示名称
    data_type: DataType                 # 数据类型
    is_input: bool                      # True=输入端口, False=输出端口
    required: bool = True               # 是否必须连接（仅输入端口有效）
    multi_connection: bool = False      # 是否允许多连接
    default_value: Any = None           # 默认值（仅输入端口有效）
    description: str = ""               # 端口描述
    
    # 运行时数据
    _data: Any = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)
    
    @property
    def data(self) -> Any:
        """获取端口数据"""
        return self._data
    
    @data.setter
    def data(self, value: Any):
        """设置端口数据"""
        self._data = value
    
    @property
    def is_connected(self) -> bool:
        """端口是否已连接"""
        return self._connected
    
    def set_connected(self, connected: bool):
        """设置连接状态"""
        self._connected = connected
    
    def clear(self):
        """清除端口数据"""
        self._data = None
    
    def can_connect_to(self, other: 'Port') -> bool:
        """检查是否可以连接到另一个端口"""
        # 必须一个输入一个输出
        if self.is_input == other.is_input:
            return False
        # 检查数据类型兼容性
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
    """创建输入端口的便捷函数"""
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
    """创建输出端口的便捷函数"""
    return Port(
        name=name,
        display_name=display_name or name,
        data_type=data_type,
        is_input=False,
        multi_connection=multi_connection,
        description=description
    )

