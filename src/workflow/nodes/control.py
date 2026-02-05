# -*- coding: utf-8 -*-
"""
Control Nodes

Provides workflow control functionality: loops, data splitting, etc.
"""

import logging
from typing import Any, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from src.ui.i18n import tr_
logger = logging.getLogger(__name__)


@register_node
class LoopNode(BaseNode):
    """
    循环节点
    
    遍历数据列表，逐个输出。
    用于批量处理场景。
    """
    
    node_type = "loop"
    display_name = tr_("Loop")
    category = NodeCategory.CONTROL
    description = tr_("Iterate over a list and output items one by one")
    icon = "🔄"
    
    def _setup_ports(self):
        self.add_input("data_list", DataType.ANY, tr_("Data list"))
        self.add_output("item", DataType.ANY, tr_("Current item"))
        self.add_output("index", DataType.ANY, tr_("Current index"))
        self.add_output("total", DataType.ANY, tr_("Total"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "max_iterations", "int", 0,
            display_name=tr_("Max iterations"),
            description=tr_("0 means no limit"),
            min_value=0
        )
    
    # 循环状态
    _current_index: int = 0
    _data_list: List = None
    
    def execute(self) -> bool:
        data_list = self.get_input_data("data_list")
        
        if data_list is None:
            self.error_message = tr_("No data list provided")
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
            self.error_message = tr_("Data list is empty")
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
    display_name = tr_("Split dataset")
    category = NodeCategory.CONTROL
    description = tr_("Split data into train/val/test sets")
    icon = "✂️"
    
    def _setup_ports(self):
        self.add_input("data", DataType.ANY, tr_("Data"))
        self.add_input("labels", DataType.LABEL, tr_("Labels"), required=False)
        self.add_output("train_data", DataType.ANY, tr_("Train data"))
        self.add_output("train_labels", DataType.LABEL, tr_("Train labels"))
        self.add_output("val_data", DataType.ANY, tr_("Validation data"))
        self.add_output("val_labels", DataType.LABEL, tr_("Validation labels"))
        self.add_output("test_data", DataType.ANY, tr_("Test data"))
        self.add_output("test_labels", DataType.LABEL, tr_("Test labels"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "train_ratio", "float", 0.7,
            display_name=tr_("Train ratio"),
            min_value=0.1, max_value=0.9
        )
        self.add_parameter(
            "val_ratio", "float", 0.15,
            display_name=tr_("Validation ratio"),
            min_value=0.0, max_value=0.4
        )
        self.add_parameter(
            "test_ratio", "float", 0.15,
            display_name=tr_("Test ratio"),
            min_value=0.0, max_value=0.4
        )
        self.add_parameter(
            "shuffle", "bool", True,
            display_name=tr_("Shuffle")
        )
        self.add_parameter(
            "stratify", "bool", True,
            display_name=tr_("Stratify"),
            description=tr_("Keep class proportions")
        )
        self.add_parameter(
            "random_seed", "int", 42,
            display_name=tr_("Random seed"),
            min_value=0
        )
    
    def execute(self) -> bool:
        data = self.get_input_data("data")
        labels = self.get_input_data("labels")
        
        if data is None:
            self.error_message = tr_("No data provided")
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
            
            msg = tr_("Split complete: train={train}, val={val}, test={test}").format(
                train=len(train_data),
                val=len(val_data),
                test=len(test_data),
            )
            self.report_status(msg)
            logger.info(msg)
            return True
            
        except Exception as e:
            self.error_message = tr_("Split failed: {error}").format(error=str(e))
            logger.exception("数据分割异常")
            return False


@register_node
class MergeNode(BaseNode):
    """
    合并节点
    
    将多个数据源合并为一个。
    """
    
    node_type = "merge"
    display_name = tr_("Merge")
    category = NodeCategory.CONTROL
    description = tr_("Merge multiple data sources")
    icon = "🔗"
    
    def _setup_ports(self):
        self.add_input("data_1", DataType.ANY, tr_("Data 1"))
        self.add_input("data_2", DataType.ANY, tr_("Data 2"))
        self.add_input("data_3", DataType.ANY, tr_("Data 3"), required=False)
        self.add_input("data_4", DataType.ANY, tr_("Data 4"), required=False)
        self.add_output("merged", DataType.ANY, tr_("Merged data"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "merge_mode", "choice", "concat",
            display_name=tr_("Merge mode"),
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
            self.error_message = tr_("No valid data provided")
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
    display_name = tr_("Passthrough")
    category = NodeCategory.CONTROL
    description = tr_("Pass data through without modification")
    icon = "◇"
    
    # 紧凑节点标志
    compact_mode = True
    
    def _setup_ports(self):
        self.add_input("in", DataType.ANY, tr_("In"))
        self.add_output("out", DataType.ANY, tr_("Out"))
    
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
    display_name = tr_("Breakpoint")
    category = NodeCategory.CONTROL
    description = tr_("Pause workflow execution to inspect data")
    icon = "🔴"
    
    # 标识这是一个断点节点
    is_breakpoint = True
    
    def _setup_ports(self):
        # 通用数据透传端口
        self.add_input("data", DataType.ANY, tr_("Data"))
        self.add_output("data", DataType.ANY, tr_("Data"))
        
        # 可选的辅助端口（用于查看多个数据流）
        self.add_input("data_2", DataType.ANY, tr_("Data 2"), required=False)
        self.add_output("data_2", DataType.ANY, tr_("Data 2"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "enabled", "bool", True,
            display_name=tr_("Enable breakpoint"),
            description=tr_("Pause execution at this node")
        )
        self.add_parameter(
            "description", "str", "",
            display_name=tr_("Breakpoint description"),
            description=tr_("Describe the purpose of this breakpoint")
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
            self.report_status(tr_("Breakpoint: {desc}").format(desc=desc))
        else:
            self.report_status(tr_("Breakpoint triggered"))
        
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
    display_name = tr_("Validate shape")
    category = NodeCategory.CONTROL
    description = tr_("Validate shape consistency in a batch")
    icon = "✅"
    
    def _setup_ports(self):
        self.add_input("data", DataType.ANY, tr_("Batch data"))
        self.add_output("data", DataType.ANY, tr_("Validated data"))
        self.add_output("is_valid", DataType.ANY, tr_("Is valid"))
        self.add_output("shape_info", DataType.ANY, tr_("Shape info"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "strict_mode", "bool", True,
            display_name=tr_("Strict mode"),
            description=tr_("Stop workflow on inconsistency (otherwise warn only)")
        )
        self.add_parameter(
            "ignore_batch_dim", "bool", False,
            display_name=tr_("Ignore batch dimension"),
            description=tr_("Ignore differences in the first dimension (batch dimension)")
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
            self.error_message = tr_("No data provided")
            return False
        
        # 确保是列表形式
        if not isinstance(data, (list, tuple)):
            data = [data]
        
        if len(data) == 0:
            self.error_message = tr_("Data list is empty")
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
            msg = tr_("✅ Shape validation passed: {count} items, shape: {shape}").format(
                count=len(data),
                shape=shapes[0],
            )
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
                tr_("❌ Shape validation failed: {count} inconsistent items").format(
                    count=len(inconsistent_items)
                ),
                tr_("   Expected shape: {shape}").format(shape=first_shape),
                tr_("   Shape distribution: {dist}").format(dist=shape_counter),
            ]
            
            # 显示前5个不一致项
            for item in inconsistent_items[:5]:
                error_lines.append(
                    tr_("   - Index {index}: expected {expected}, got {actual}").format(
                        index=item["index"],
                        expected=item["expected"],
                        actual=item["actual"],
                    )
                )
            
            if len(inconsistent_items) > 5:
                error_lines.append(
                    tr_("   ... and {count} more").format(
                        count=len(inconsistent_items) - 5
                    )
                )
            
            error_msg = "\n".join(error_lines)
            logger.error(error_msg)
            print(error_msg)  # 打印到控制台
            
            self.set_output_data("is_valid", False)
            self.set_output_data("shape_info", shape_info)
            
            if strict_mode:
                # 严格模式：中断工作流
                self.error_message = tr_("Shape inconsistent: {count} inconsistent items").format(
                    count=len(inconsistent_items)
                )
                return False
            else:
                # 非严格模式：警告但继续，输出原始数据
                self.report_status(
                    tr_("⚠️ Shape validation warning: {count} inconsistent items").format(
                        count=len(inconsistent_items)
                    )
                )
                self.set_output_data("data", list(data))
                return True