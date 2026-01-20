# -*- coding: utf-8 -*-
"""
控制节点

提供工作流控制功能：循环、数据分割等。
"""

import logging
from typing import Any, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType


logger = logging.getLogger(__name__)


@register_node
class LoopNode(BaseNode):
    """
    循环节点
    
    遍历数据列表，逐个输出。
    用于批量处理场景。
    """
    
    node_type = "loop"
    display_name = "循环"
    category = NodeCategory.CONTROL
    description = "遍历数据列表"
    icon = "🔄"
    
    def _setup_ports(self):
        self.add_input("data_list", DataType.ANY, "数据列表")
        self.add_output("item", DataType.ANY, "当前项")
        self.add_output("index", DataType.ANY, "当前索引")
        self.add_output("total", DataType.ANY, "总数")
    
    def _setup_parameters(self):
        self.add_parameter(
            "max_iterations", "int", 0,
            display_name="最大迭代次数",
            description="0表示不限制",
            min_value=0
        )
    
    # 循环状态
    _current_index: int = 0
    _data_list: List = None
    
    def execute(self) -> bool:
        data_list = self.get_input_data("data_list")
        
        if data_list is None:
            self.error_message = "未提供数据列表"
            return False
        
        if not isinstance(data_list, (list, tuple, np.ndarray)):
            # 单个元素包装为列表
            data_list = [data_list]
        
        max_iterations = self.get_parameter("max_iterations")
        
        # 限制迭代次数
        if max_iterations > 0:
            data_list = data_list[:max_iterations]
        
        # 对于首次执行，输出第一个元素
        # 后续由引擎控制循环
        if len(data_list) > 0:
            self.set_output_data("item", data_list[0])
            self.set_output_data("index", 0)
            self.set_output_data("total", len(data_list))
            
            # 保存列表供循环使用
            self._data_list = list(data_list)
            self._current_index = 0
            
            return True
        else:
            self.error_message = "数据列表为空"
            return False
    
    def has_more(self) -> bool:
        """检查是否还有更多数据"""
        if self._data_list is None:
            return False
        return self._current_index < len(self._data_list) - 1
    
    def next_item(self) -> bool:
        """移动到下一个元素"""
        if not self.has_more():
            return False
        
        self._current_index += 1
        self.set_output_data("item", self._data_list[self._current_index])
        self.set_output_data("index", self._current_index)
        return True
    
    def reset_loop(self):
        """重置循环状态"""
        self._current_index = 0
        self._data_list = None


@register_node
class SplitNode(BaseNode):
    """
    数据分割节点
    
    将数据集分割为训练集、验证集、测试集。
    """
    
    node_type = "split"
    display_name = "数据分割"
    category = NodeCategory.CONTROL
    description = "分割数据集"
    icon = "✂️"
    
    def _setup_ports(self):
        self.add_input("data", DataType.ANY, "数据")
        self.add_input("labels", DataType.LABEL, "标签", required=False)
        self.add_output("train_data", DataType.ANY, "训练数据")
        self.add_output("train_labels", DataType.LABEL, "训练标签")
        self.add_output("val_data", DataType.ANY, "验证数据")
        self.add_output("val_labels", DataType.LABEL, "验证标签")
        self.add_output("test_data", DataType.ANY, "测试数据")
        self.add_output("test_labels", DataType.LABEL, "测试标签")
    
    def _setup_parameters(self):
        self.add_parameter(
            "train_ratio", "float", 0.7,
            display_name="训练集比例",
            min_value=0.1, max_value=0.9
        )
        self.add_parameter(
            "val_ratio", "float", 0.15,
            display_name="验证集比例",
            min_value=0.0, max_value=0.4
        )
        self.add_parameter(
            "test_ratio", "float", 0.15,
            display_name="测试集比例",
            min_value=0.0, max_value=0.4
        )
        self.add_parameter(
            "shuffle", "bool", True,
            display_name="随机打乱"
        )
        self.add_parameter(
            "stratify", "bool", True,
            display_name="分层抽样",
            description="保持各类别比例"
        )
        self.add_parameter(
            "random_seed", "int", 42,
            display_name="随机种子",
            min_value=0
        )
    
    def execute(self) -> bool:
        data = self.get_input_data("data")
        labels = self.get_input_data("labels")
        
        if data is None:
            self.error_message = "未提供数据"
            return False
        
        train_ratio = self.get_parameter("train_ratio")
        val_ratio = self.get_parameter("val_ratio")
        test_ratio = self.get_parameter("test_ratio")
        shuffle = self.get_parameter("shuffle")
        stratify = self.get_parameter("stratify")
        random_seed = self.get_parameter("random_seed")
        
        # 归一化比例
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        # 转换为列表
        if not isinstance(data, list):
            data = list(data) if hasattr(data, '__iter__') else [data]
        
        n_samples = len(data)
        indices = np.arange(n_samples)
        
        if labels is not None:
            if not isinstance(labels, (list, np.ndarray)):
                labels = list(labels)
            labels = np.array(labels)
        
        try:
            # 第一次分割：分出测试集
            if test_ratio > 0:
                stratify_arr = labels if (stratify and labels is not None) else None
                train_val_idx, test_idx = train_test_split(
                    indices,
                    test_size=test_ratio,
                    shuffle=shuffle,
                    stratify=stratify_arr,
                    random_state=random_seed
                )
            else:
                train_val_idx = indices
                test_idx = np.array([], dtype=int)
            
            # 第二次分割：分出验证集
            if val_ratio > 0:
                # 调整验证集比例（相对于剩余数据）
                adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
                stratify_arr = labels[train_val_idx] if (stratify and labels is not None) else None
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=adjusted_val_ratio,
                    shuffle=shuffle,
                    stratify=stratify_arr,
                    random_state=random_seed
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([], dtype=int)
            
            # 提取数据
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx] if len(val_idx) > 0 else []
            test_data = [data[i] for i in test_idx] if len(test_idx) > 0 else []
            
            # 提取标签
            if labels is not None:
                train_labels = labels[train_idx].tolist()
                val_labels = labels[val_idx].tolist() if len(val_idx) > 0 else []
                test_labels = labels[test_idx].tolist() if len(test_idx) > 0 else []
            else:
                train_labels = []
                val_labels = []
                test_labels = []
            
            # 设置输出
            self.set_output_data("train_data", train_data)
            self.set_output_data("train_labels", train_labels)
            self.set_output_data("val_data", val_data)
            self.set_output_data("val_labels", val_labels)
            self.set_output_data("test_data", test_data)
            self.set_output_data("test_labels", test_labels)
            
            msg = f"数据分割完成: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}"
            self.report_status(msg)
            logger.info(msg)
            return True
            
        except Exception as e:
            self.error_message = f"分割失败: {str(e)}"
            logger.exception("数据分割异常")
            return False


@register_node
class MergeNode(BaseNode):
    """
    合并节点
    
    将多个数据源合并为一个。
    """
    
    node_type = "merge"
    display_name = "合并"
    category = NodeCategory.CONTROL
    description = "合并多个数据源"
    icon = "🔗"
    
    def _setup_ports(self):
        self.add_input("data_1", DataType.ANY, "数据1")
        self.add_input("data_2", DataType.ANY, "数据2")
        self.add_input("data_3", DataType.ANY, "数据3", required=False)
        self.add_input("data_4", DataType.ANY, "数据4", required=False)
        self.add_output("merged", DataType.ANY, "合并数据")
    
    def _setup_parameters(self):
        self.add_parameter(
            "merge_mode", "choice", "concat",
            display_name="合并模式",
            choices=["concat", "stack", "zip"]
        )
    
    def execute(self) -> bool:
        data_1 = self.get_input_data("data_1")
        data_2 = self.get_input_data("data_2")
        data_3 = self.get_input_data("data_3")
        data_4 = self.get_input_data("data_4")
        
        # 收集非空数据
        all_data = []
        for d in [data_1, data_2, data_3, data_4]:
            if d is not None:
                if isinstance(d, list):
                    all_data.extend(d)
                else:
                    all_data.append(d)
        
        if not all_data:
            self.error_message = "未提供有效数据"
            return False
        
        merge_mode = self.get_parameter("merge_mode")
        
        if merge_mode == "concat":
            merged = all_data
        elif merge_mode == "stack":
            try:
                if hasattr(all_data[0], 'data'):
                    merged = np.stack([d.data for d in all_data])
                else:
                    merged = np.stack(all_data)
            except Exception:
                merged = all_data
        else:  # zip
            # 对于zip模式，返回配对列表
            merged = list(zip(*[d if isinstance(d, list) else [d] for d in [data_1, data_2, data_3, data_4] if d is not None]))
        
        self.set_output_data("merged", merged)
        logger.info(f"合并完成: {len(all_data)} 项")
        return True


@register_node
class PassthroughNode(BaseNode):
    """
    空节点 / 透传节点
    
    一个最小化的透传节点，仅有一个输入一个输出。
    数据不做任何处理直接透传。
    用于工作流布线整理或占位。
    """
    
    node_type = "passthrough"
    display_name = "透传"
    category = NodeCategory.CONTROL
    description = "透传数据，不做处理"
    icon = "◇"
    
    # 紧凑节点标志
    compact_mode = True
    
    def _setup_ports(self):
        self.add_input("in", DataType.ANY, "入")
        self.add_output("out", DataType.ANY, "出")
    
    def execute(self) -> bool:
        """透传数据"""
        data = self.get_input_data("in")
        self.set_output_data("out", data)
        return True


@register_node
class BreakpointNode(BaseNode):
    """
    断点节点
    
    工作流调试节点，执行到此处时暂停工作流。
    用户可以在预览视图中查看数据，确认后继续执行。
    数据透传：输入什么就输出什么。
    """
    
    node_type = "breakpoint"
    display_name = "断点"
    category = NodeCategory.CONTROL
    description = "暂停工作流执行，查看数据"
    icon = "🔴"
    
    # 标识这是一个断点节点
    is_breakpoint = True
    
    def _setup_ports(self):
        # 通用数据透传端口
        self.add_input("data", DataType.ANY, "数据")
        self.add_output("data", DataType.ANY, "数据")
        
        # 可选的辅助端口（用于查看多个数据流）
        self.add_input("data_2", DataType.ANY, "数据2", required=False)
        self.add_output("data_2", DataType.ANY, "数据2")
    
    def _setup_parameters(self):
        self.add_parameter(
            "enabled", "bool", True,
            display_name="启用断点",
            description="是否在此处暂停"
        )
        self.add_parameter(
            "description", "str", "",
            display_name="断点描述",
            description="描述此断点的用途"
        )
    
    def execute(self) -> bool:
        """执行断点节点 - 透传数据"""
        # 透传主数据
        data = self.get_input_data("data")
        self.set_output_data("data", data)
        
        # 透传辅助数据
        data_2 = self.get_input_data("data_2")
        if data_2 is not None:
            self.set_output_data("data_2", data_2)
        
        # 报告状态
        desc = self.get_parameter("description")
        if desc:
            self.report_status(f"断点: {desc}")
        else:
            self.report_status("断点已触发")
        
        logger.info(f"断点节点执行完成: {self.node_id}")
        return True
    
    def is_breakpoint_enabled(self) -> bool:
        """检查断点是否启用"""
        return self.get_parameter("enabled")


@register_node
class ValidateShapeNode(BaseNode):
    """
    校验数据Shape节点
    
    检查一批数据中所有数据的 shape 是否一致。
    - 如果所有数据 shape 一致，数据透传通过
    - 如果 shape 不一致，打印错误信息并可选择是否中断工作流
    """
    
    node_type = "validate_shape"
    display_name = "校验Shape"
    category = NodeCategory.CONTROL
    description = "校验批量数据的shape一致性"
    icon = "✅"
    
    def _setup_ports(self):
        self.add_input("data", DataType.ANY, "数据批次")
        self.add_output("data", DataType.ANY, "验证通过的数据")
        self.add_output("is_valid", DataType.ANY, "是否通过")
        self.add_output("shape_info", DataType.ANY, "Shape信息")
    
    def _setup_parameters(self):
        self.add_parameter(
            "strict_mode", "bool", True,
            display_name="严格模式",
            description="Shape不一致时是否中断工作流（否则仅警告）"
        )
        self.add_parameter(
            "ignore_batch_dim", "bool", False,
            display_name="忽略批次维度",
            description="是否忽略第一个维度（批次维度）的差异"
        )
    
    def _get_shape(self, item: Any) -> Tuple[int, ...]:
        """
        获取数据的 shape
        
        支持多种数据类型：
        - numpy array
        - AudioData 对象
        - list/tuple
        """
        # numpy 数组
        if isinstance(item, np.ndarray):
            return item.shape
        
        # 有 shape 属性的对象（如 AudioData）
        if hasattr(item, 'shape'):
            shape = item.shape
            if callable(shape):
                shape = shape()
            return tuple(shape) if hasattr(shape, '__iter__') else (shape,)
        
        # 有 data 属性且 data 是 numpy 数组的对象
        if hasattr(item, 'data') and isinstance(item.data, np.ndarray):
            return item.data.shape
        
        # list 或 tuple
        if isinstance(item, (list, tuple)):
            return (len(item),)
        
        # 标量或其他
        return ()
    
    def execute(self) -> bool:
        data = self.get_input_data("data")
        
        if data is None:
            self.error_message = "未提供数据"
            return False
        
        # 确保是列表形式
        if not isinstance(data, (list, tuple)):
            data = [data]
        
        if len(data) == 0:
            self.error_message = "数据列表为空"
            return False
        
        strict_mode = self.get_parameter("strict_mode")
        ignore_batch_dim = self.get_parameter("ignore_batch_dim")
        
        # 收集所有 shape
        shapes = []
        shape_details = []
        
        for i, item in enumerate(data):
            shape = self._get_shape(item)
            shapes.append(shape)
            shape_details.append({
                "index": i,
                "shape": shape,
                "type": type(item).__name__
            })
        
        # 根据是否忽略批次维度来比较
        if ignore_batch_dim:
            # 忽略第一个维度进行比较
            compare_shapes = [s[1:] if len(s) > 1 else s for s in shapes]
        else:
            compare_shapes = shapes
        
        # 检查一致性
        first_shape = compare_shapes[0]
        inconsistent_items = []
        
        for i, shape in enumerate(compare_shapes):
            if shape != first_shape:
                inconsistent_items.append({
                    "index": i,
                    "expected": first_shape,
                    "actual": shape,
                    "full_shape": shapes[i]
                })
        
        is_valid = len(inconsistent_items) == 0
        
        # 构建 shape 信息
        shape_info = {
            "total_count": len(data),
            "first_shape": shapes[0],
            "is_consistent": is_valid,
            "details": shape_details
        }
        
        if is_valid:
            # 所有 shape 一致，数据通过
            msg = f"✅ Shape校验通过: {len(data)} 个数据, 统一shape: {shapes[0]}"
            logger.info(msg)
            self.report_status(msg)
            
            self.set_output_data("data", list(data))
            self.set_output_data("is_valid", True)
            self.set_output_data("shape_info", shape_info)
            return True
        else:
            # Shape 不一致
            shape_info["inconsistent_items"] = inconsistent_items
            
            # 统计各种 shape 的数量
            shape_counter = {}
            for s in shapes:
                key = str(s)
                shape_counter[key] = shape_counter.get(key, 0) + 1
            shape_info["shape_distribution"] = shape_counter
            
            # 打印详细错误信息
            error_lines = [
                f"❌ Shape校验失败: 发现 {len(inconsistent_items)} 个不一致项",
                f"   预期shape: {first_shape}",
                f"   Shape分布: {shape_counter}"
            ]
            
            # 显示前5个不一致项
            for item in inconsistent_items[:5]:
                error_lines.append(
                    f"   - 索引 {item['index']}: 期望 {item['expected']}, 实际 {item['actual']}"
                )
            
            if len(inconsistent_items) > 5:
                error_lines.append(f"   ... 还有 {len(inconsistent_items) - 5} 个不一致项")
            
            error_msg = "\n".join(error_lines)
            logger.error(error_msg)
            print(error_msg)  # 打印到控制台
            
            self.set_output_data("is_valid", False)
            self.set_output_data("shape_info", shape_info)
            
            if strict_mode:
                # 严格模式：中断工作流
                self.error_message = f"Shape不一致: 发现 {len(inconsistent_items)} 个不一致项"
                return False
            else:
                # 非严格模式：警告但继续，输出原始数据
                self.report_status(f"⚠️ Shape校验警告: {len(inconsistent_items)} 个不一致项")
                self.set_output_data("data", list(data))
                return True