# -*- coding: utf-8 -*-
"""
Data Source Nodes

Provides audio file, label file, and other data loading nodes.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType


logger = logging.getLogger(__name__)


@dataclass
class AudioData:
    """
    音频数据包装
    
    数据格式统一为 (channels, samples)：
    - 多通道音频: (n_channels, samples)
    - 单通道音频: (1, samples)
    """
    data: np.ndarray        # 音频数据，形状为 (channels, samples)
    sample_rate: int        # 采样率
    file_path: str = ""     # 源文件路径
    duration: float = 0.0   # 时长（秒）
    
    def __post_init__(self):
        # 确保数据格式为 (channels, samples)
        if self.data.ndim == 1:
            # 单声道 (N,) -> (1, N)
            self.data = self.data.reshape(1, -1)
        
        # 计算时长
        if self.duration == 0.0 and self.data.size > 0:
            self.duration = self.samples / self.sample_rate
    
    @property
    def channels(self) -> int:
        """返回通道数"""
        return self.data.shape[0]
    
    @property
    def samples(self) -> int:
        """返回样本数"""
        return self.data.shape[1]
    
    @property
    def is_mono(self) -> bool:
        """是否为单声道"""
        return self.channels == 1
    
    @property
    def is_stereo(self) -> bool:
        """是否为立体声"""
        return self.channels == 2
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的数据 (返回 1D 数组)"""
        if channel_idx >= self.channels:
            raise ValueError(f"通道索引 {channel_idx} 超出范围 (共 {self.channels} 通道)")
        return self.data[channel_idx]
    
    def to_mono(self, method: str = "mean") -> 'AudioData':
        """
        转换为单声道
        
        Args:
            method: 混合方法 - "mean" (平均), "left" (左声道), "right" (右声道)
        """
        if self.is_mono:
            return self
        
        if method == "mean":
            mono_data = np.mean(self.data, axis=0, keepdims=True)
        elif method == "left":
            mono_data = self.data[0:1]
        elif method == "right":
            mono_data = self.data[-1:]
        else:
            mono_data = np.mean(self.data, axis=0, keepdims=True)
        
        return AudioData(
            data=mono_data,
            sample_rate=self.sample_rate,
            file_path=self.file_path
        )


@register_node
class AudioFolderNode(BaseNode):
    """
    音频文件夹节点
    
    加载指定目录下的所有音频文件。
    支持按子目录自动生成标签。
    """
    
    node_type = "audio_folder"
    display_name = "音频文件夹"
    category = NodeCategory.DATA_SOURCE
    description = "加载文件夹中的音频文件，支持按子目录生成标签"
    icon = "📂"
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    def _setup_ports(self):
        self.add_output("audio", DataType.AUDIO, "音频列表")
        self.add_output("labels", DataType.LABEL, "标签列表")
        self.add_output("file_paths", DataType.ANY, "文件路径列表")
    
    def _setup_parameters(self):
        self.add_parameter(
            "folder_path", "folder", "",
            display_name="文件夹路径",
            description="包含音频文件的目录",
            default_directory="audio_data"
        )
        self.add_parameter(
            "recursive", "bool", True,
            display_name="递归扫描",
            description="是否扫描子目录"
        )
        self.add_parameter(
            "auto_label", "bool", True,
            display_name="自动标签",
            description="根据子目录名称自动生成标签"
        )
        self.add_parameter(
            "target_sr", "int", 22050,
            display_name="目标采样率",
            description="加载时统一的采样率",
            min_value=8000, max_value=48000
        )
        self.add_parameter(
            "max_files", "int", 0,
            display_name="最大文件数",
            description="最多加载的文件数，0表示不限制",
            min_value=0
        )
    
    def execute(self) -> bool:
        folder_path = self.get_parameter("folder_path")
        if not folder_path or not os.path.isdir(folder_path):
            self.error_message = f"目录不存在: {folder_path}"
            return False
        
        recursive = self.get_parameter("recursive")
        auto_label = self.get_parameter("auto_label")
        target_sr = self.get_parameter("target_sr")
        max_files = self.get_parameter("max_files")
        
        audio_list = []
        labels = []
        file_paths = []
        label_map = {}  # 子目录名 -> 标签索引
        
        # 扫描文件
        root_path = Path(folder_path)
        if recursive:
            files = list(root_path.rglob("*"))
        else:
            files = list(root_path.glob("*"))
        
        # 过滤音频文件
        audio_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        # 限制数量
        if max_files > 0:
            audio_files = audio_files[:max_files]
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        self.report_status(f"正在加载 {len(audio_files)} 个音频文件...")
        
        for file_path in audio_files:
            try:
                # 加载音频 (保留原始通道)
                audio_data, sr = librosa.load(str(file_path), sr=target_sr, mono=False)
                
                # AudioData.__post_init__ 会自动将 1D 转为 (1, N)
                audio = AudioData(
                    data=audio_data,
                    sample_rate=sr,
                    file_path=str(file_path)
                )
                audio_list.append(audio)
                file_paths.append(str(file_path))
                
                # 生成标签
                if auto_label:
                    # 使用相对于根目录的父目录名作为标签
                    rel_path = file_path.relative_to(root_path)
                    if len(rel_path.parts) > 1:
                        label_name = rel_path.parts[0]
                    else:
                        label_name = "default"
                    
                    if label_name not in label_map:
                        label_map[label_name] = len(label_map)
                    labels.append(label_map[label_name])
                else:
                    labels.append(0)
                    
            except Exception as e:
                logger.warning(f"加载音频失败 {file_path}: {e}")
                continue
        
        if not audio_list:
            self.error_message = "未找到有效的音频文件"
            return False
        
        # 设置输出
        self.set_output_data("audio", audio_list)
        self.set_output_data("labels", labels)
        self.set_output_data("file_paths", file_paths)
        
        logger.info(f"加载完成: {len(audio_list)} 个音频, {len(label_map)} 个类别")
        self.report_status(f"音频加载完成: {len(audio_list)} 个文件, {len(label_map)} 个类别")
        return True


@register_node
class AudioFileNode(BaseNode):
    """
    单个音频文件节点
    
    加载单个音频文件，用于测试和预览。
    """
    
    node_type = "audio_file"
    display_name = "音频文件"
    category = NodeCategory.DATA_SOURCE
    description = "加载单个音频文件"
    icon = "🎵"
    
    def _setup_ports(self):
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "file_path", "file", "",
            display_name="文件路径",
            description="音频文件路径",
            file_filter="Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)",
            default_directory="audio_data"
        )
        self.add_parameter(
            "target_sr", "int", 22050,
            display_name="目标采样率",
            min_value=8000, max_value=48000
        )
    
    def execute(self) -> bool:
        file_path = self.get_parameter("file_path")
        target_sr = self.get_parameter("target_sr")
        
        if not file_path or not os.path.isfile(file_path):
            self.error_message = f"文件不存在: {file_path}"
            return False
        
        try:
            # 加载音频 (保留原始通道)
            audio_data, sr = librosa.load(file_path, sr=target_sr, mono=False)
            
            # AudioData.__post_init__ 会自动将 1D 转为 (1, N)
            audio = AudioData(
                data=audio_data,
                sample_rate=sr,
                file_path=file_path
            )
            self.set_output_data("audio", audio)
            return True
        except Exception as e:
            self.error_message = f"加载音频失败: {e}"
            return False


@register_node
class LabelFileNode(BaseNode):
    """
    标签文件节点
    
    从CSV或JSON文件加载标签数据。
    """
    
    node_type = "label_file"
    display_name = "标签文件"
    category = NodeCategory.DATA_SOURCE
    description = "从CSV/JSON文件加载标签"
    icon = "📄"
    
    def _setup_ports(self):
        self.add_output("labels", DataType.LABEL, "标签列表")
        self.add_output("label_map", DataType.ANY, "标签映射")
    
    def _setup_parameters(self):
        self.add_parameter(
            "file_path", "file", "",
            display_name="文件路径",
            file_filter="Label Files (*.csv *.json *.txt)",
            default_directory="audio_data"
        )
        self.add_parameter(
            "format", "choice", "auto",
            display_name="文件格式",
            choices=["auto", "csv", "json", "txt"]
        )
        self.add_parameter(
            "filename_column", "str", "filename",
            display_name="文件名列",
            description="CSV中文件名所在列名"
        )
        self.add_parameter(
            "label_column", "str", "label",
            display_name="标签列",
            description="CSV中标签所在列名"
        )
    
    def execute(self) -> bool:
        file_path = self.get_parameter("file_path")
        file_format = self.get_parameter("format")
        
        if not file_path or not os.path.isfile(file_path):
            self.error_message = f"文件不存在: {file_path}"
            return False
        
        # 自动检测格式
        if file_format == "auto":
            ext = Path(file_path).suffix.lower()
            if ext == ".csv":
                file_format = "csv"
            elif ext == ".json":
                file_format = "json"
            else:
                file_format = "txt"
        
        try:
            labels = []
            label_map = {}
            
            if file_format == "csv":
                labels, label_map = self._load_csv(file_path)
            elif file_format == "json":
                labels, label_map = self._load_json(file_path)
            else:
                labels, label_map = self._load_txt(file_path)
            
            self.set_output_data("labels", labels)
            self.set_output_data("label_map", label_map)
            
            logger.info(f"加载标签完成: {len(labels)} 个, {len(set(labels))} 个类别")
            return True
            
        except Exception as e:
            self.error_message = f"加载标签失败: {e}"
            return False
    
    def _load_csv(self, file_path: str) -> Tuple[List, Dict]:
        """加载CSV格式标签"""
        filename_col = self.get_parameter("filename_column")
        label_col = self.get_parameter("label_column")
        
        labels = []
        label_map = {}  # filename -> label
        unique_labels = {}  # label_name -> label_id
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get(filename_col, "")
                label_name = row.get(label_col, "")
                
                if label_name not in unique_labels:
                    unique_labels[label_name] = len(unique_labels)
                
                label_id = unique_labels[label_name]
                labels.append(label_id)
                label_map[filename] = label_id
        
        return labels, {"filename_map": label_map, "label_names": unique_labels}
    
    def _load_json(self, file_path: str) -> Tuple[List, Dict]:
        """加载JSON格式标签
        
        支持的格式：
        1. 简单列表: [0, 1, 0, 1]
        2. 简单字典: {"filename": label}
        3. 嵌套格式 (SaveAudioNode保存的格式): 
           {"version": "1.0", "labels": {"filename": label}, ...}
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # 简单列表格式: [0, 1, 0, 1]
            labels = data
            label_map = {str(i): l for i, l in enumerate(labels)}
        elif isinstance(data, dict):
            # 检查是否是 SaveAudioNode 保存的嵌套格式
            if "labels" in data and isinstance(data["labels"], dict):
                # 嵌套格式: {"version": "1.0", "labels": {...}, ...}
                label_map = data["labels"]
                labels = list(label_map.values())
            else:
                # 简单字典格式: {"filename": label}
                label_map = data
                labels = list(data.values())
        else:
            raise ValueError("不支持的JSON格式")
        
        return labels, label_map
    
    def _load_txt(self, file_path: str) -> Tuple[List, Dict]:
        """加载TXT格式标签（每行一个标签）"""
        labels = []
        unique_labels = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label_name = line.strip()
                if label_name:
                    if label_name not in unique_labels:
                        unique_labels[label_name] = len(unique_labels)
                    labels.append(unique_labels[label_name])
        
        return labels, {"label_names": unique_labels}


@register_node
class SaveAudioNode(BaseNode):
    """
    保存音频节点
    
    将音频数据保存为WAV文件，可选保存对应的标签JSON文件。
    标签JSON文件格式与LabelFileNode兼容，可用于后续加载。
    """
    
    node_type = "save_audio"
    display_name = "保存音频"
    category = NodeCategory.DATA_SOURCE
    description = "将音频保存为WAV文件，可选保存标签JSON文件"
    icon = "💾"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频", required=True,
                      description="要保存的音频数据（单个或列表）")
        self.add_input("labels", DataType.LABEL, "标签", required=False,
                      description="可选的标签数据，与音频一一对应")
    
    def _setup_parameters(self):
        self.add_parameter(
            "output_folder", "folder", "",
            display_name="输出文件夹",
            description="保存音频文件的目标文件夹",
            default_directory="audio_data"
        )
        self.add_parameter(
            "filename_prefix", "str", "audio",
            display_name="文件名前缀",
            description="保存的音频文件名前缀"
        )
        self.add_parameter(
            "save_labels", "bool", True,
            display_name="保存标签文件",
            description="当有标签输入时，是否保存labels.json文件"
        )
        self.add_parameter(
            "overwrite", "bool", False,
            display_name="覆盖已有文件",
            description="如果文件已存在，是否覆盖"
        )
    
    def execute(self) -> bool:
        # 获取输入
        audio_input = self.get_input_data("audio")
        labels_input = self.get_input_data("labels")
        
        # 获取参数
        output_folder = self.get_parameter("output_folder")
        filename_prefix = self.get_parameter("filename_prefix")
        save_labels = self.get_parameter("save_labels")
        overwrite = self.get_parameter("overwrite")
        
        # 验证输出文件夹
        if not output_folder:
            self.error_message = "请选择输出文件夹"
            return False
        
        output_path = Path(output_folder)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.error_message = f"无法创建输出文件夹: {e}"
            return False
        
        # 处理音频输入 - 统一为列表
        if audio_input is None:
            self.error_message = "没有输入音频数据"
            return False
        
        if isinstance(audio_input, AudioData):
            audio_list = [audio_input]
        elif isinstance(audio_input, list):
            audio_list = audio_input
        else:
            self.error_message = f"不支持的音频数据类型: {type(audio_input)}"
            return False
        
        # 处理标签输入 - 统一为列表
        labels_list = None
        if labels_input is not None:
            if isinstance(labels_input, list):
                labels_list = labels_input
            else:
                labels_list = [labels_input]
            
            # 验证标签数量匹配
            if len(labels_list) != len(audio_list):
                logger.warning(
                    f"标签数量({len(labels_list)})与音频数量({len(audio_list)})不匹配，"
                    "将按顺序对应，多余的将被忽略"
                )
        
        self.report_status(f"正在保存 {len(audio_list)} 个音频文件...")
        
        saved_files = []
        saved_labels = {}
        
        for i, audio in enumerate(audio_list):
            if not isinstance(audio, AudioData):
                logger.warning(f"跳过非AudioData类型的数据: {type(audio)}")
                continue
            
            # 生成文件名
            filename = f"{filename_prefix}_{i:04d}.wav"
            filepath = output_path / filename
            
            # 检查是否覆盖
            if filepath.exists() and not overwrite:
                logger.warning(f"文件已存在，跳过: {filepath}")
                continue
            
            try:
                # 保存WAV文件
                # soundfile期望数据格式为 (samples, channels)
                audio_data_to_save = audio.data.T if audio.data.ndim > 1 else audio.data
                sf.write(str(filepath), audio_data_to_save, audio.sample_rate)
                saved_files.append(str(filepath))
                
                # 记录标签
                if labels_list is not None and i < len(labels_list):
                    saved_labels[filename] = labels_list[i]
                    
            except Exception as e:
                logger.error(f"保存音频失败 {filepath}: {e}")
                continue
        
        # 保存标签JSON文件
        if save_labels and saved_labels:
            labels_filepath = output_path / "labels.json"
            try:
                label_data = {
                    "version": "1.0",
                    "description": "Audio labels saved by SaveAudioNode",
                    "labels": saved_labels,
                    "label_count": len(saved_labels)
                }
                with open(labels_filepath, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                logger.info(f"标签文件已保存: {labels_filepath}")
            except Exception as e:
                logger.error(f"保存标签文件失败: {e}")
        
        if not saved_files:
            self.error_message = "没有成功保存任何音频文件"
            return False
        
        logger.info(f"保存完成: {len(saved_files)} 个音频文件")
        self.report_status(
            f"音频保存完成: {len(saved_files)} 个文件" + 
            (f", 标签: {len(saved_labels)} 个" if saved_labels else "")
        )
        return True
