# -*- coding: utf-8 -*-
"""
Preprocessing Nodes

Provides audio preprocessing functionality: resampling, trimming, padding, normalization, etc.
"""

import logging
from typing import List, Union

import librosa
import numpy as np

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from .data_source import AudioData


logger = logging.getLogger(__name__)


def validate_audio_input(data, node_name: str = "节点") -> Union[AudioData, List[AudioData]]:
    """
    验证输入数据是否为有效的音频数据
    
    Args:
        data: 输入数据
        node_name: 节点名称（用于错误信息）
    
    Returns:
        验证通过的音频数据
    
    Raises:
        ValueError: 数据为空
        TypeError: 数据类型不匹配
    """
    if data is None:
        raise ValueError(f"{node_name}: 未提供输入音频")
    
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{node_name}: 音频列表为空")
        if not isinstance(data[0], AudioData):
            raise TypeError(
                f"{node_name}: 期望 AudioData 类型，"
                f"但收到 {type(data[0]).__name__}。"
                f"请检查上游节点的输出类型是否正确。"
            )
    else:
        if not isinstance(data, AudioData):
            raise TypeError(
                f"{node_name}: 期望 AudioData 类型，"
                f"但收到 {type(data).__name__}。"
                f"请检查上游节点的输出类型是否正确。"
            )
    return data


def process_audio_or_list(audio_input, process_func) -> Union[AudioData, List[AudioData]]:
    """
    处理单个音频或音频列表的通用函数
    
    Args:
        audio_input: AudioData 或 List[AudioData]（已通过 validate_audio_input 验证）
        process_func: 处理函数，接受 AudioData 返回 AudioData
    
    Returns:
        处理后的音频（同输入类型）
    """
    if isinstance(audio_input, list):
        return [process_func(audio) for audio in audio_input]
    else:
        return process_func(audio_input)


def process_channels(data: np.ndarray, channel_func) -> np.ndarray:
    """
    对每个通道分别应用处理函数
    
    Args:
        data: 音频数据，形状为 (channels, samples)
        channel_func: 处理单个通道的函数，接受 1D 数组返回 1D 数组
    
    Returns:
        处理后的数据，形状为 (channels, new_samples)
    """
    processed_channels = []
    for ch in range(data.shape[0]):
        processed = channel_func(data[ch])
        processed_channels.append(processed)
    
    # 确保所有通道长度一致（取最短的）
    min_len = min(len(ch) for ch in processed_channels)
    processed_channels = [ch[:min_len] for ch in processed_channels]
    
    return np.stack(processed_channels, axis=0)


@register_node
class ResampleNode(BaseNode):
    """
    重采样节点
    
    将音频重采样到目标采样率。
    """
    
    node_type = "resample"
    display_name = "重采样"
    category = NodeCategory.PREPROCESSING
    description = "将音频重采样到指定采样率"
    icon = "📏"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频", description="音频或音频列表")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "target_sr", "int", 22050,
            display_name="目标采样率",
            min_value=8000, max_value=48000
        )
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        target_sr = self.get_parameter("target_sr")
        
        def resample(audio: AudioData) -> AudioData:
            if audio.sample_rate == target_sr:
                return audio
            
            # 对每个通道分别重采样
            def resample_channel(ch_data: np.ndarray) -> np.ndarray:
                return librosa.resample(
                    ch_data, 
                    orig_sr=audio.sample_rate, 
                    target_sr=target_sr
                )
            
            resampled = process_channels(audio.data, resample_channel)
            return AudioData(
                data=resampled,
                sample_rate=target_sr,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, resample)
        self.set_output_data("audio", result)
        return True


@register_node
class TrimPadNode(BaseNode):
    """
    裁剪/填充节点
    
    将音频统一到指定长度。
    """
    
    node_type = "trim_pad"
    display_name = "裁剪/填充"
    category = NodeCategory.PREPROCESSING
    description = "将音频统一到指定长度"
    icon = "✂️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "duration", "float", 3.0,
            display_name="目标时长(秒)",
            min_value=0.1, max_value=60.0
        )
        self.add_parameter(
            "mode", "choice", "pad_trim",
            display_name="处理模式",
            choices=["pad_trim", "pad_only", "trim_only", "loop"]
        )
        self.add_parameter(
            "pad_mode", "choice", "constant",
            display_name="填充模式",
            choices=["constant", "edge", "reflect", "wrap"]
        )
        self.add_parameter(
            "position", "choice", "center",
            display_name="对齐位置",
            choices=["start", "center", "end"]
        )
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        duration = self.get_parameter("duration")
        mode = self.get_parameter("mode")
        pad_mode = self.get_parameter("pad_mode")
        position = self.get_parameter("position")
        
        def trim_pad(audio: AudioData) -> AudioData:
            target_length = int(duration * audio.sample_rate)
            current_length = audio.samples  # 使用 samples 属性
            
            if current_length == target_length:
                return audio
            
            if current_length > target_length:
                # 需要裁剪
                if mode == "pad_only":
                    return audio
                
                if position == "start":
                    trimmed = audio.data[:, :target_length]
                elif position == "end":
                    trimmed = audio.data[:, -target_length:]
                else:  # center
                    start = (current_length - target_length) // 2
                    trimmed = audio.data[:, start:start + target_length]
                
                return AudioData(
                    data=trimmed,
                    sample_rate=audio.sample_rate,
                    file_path=audio.file_path
                )
            else:
                # 需要填充
                if mode == "trim_only":
                    return audio
                
                if mode == "loop":
                    # 循环填充 - 对每个通道分别处理
                    def loop_channel(ch_data: np.ndarray) -> np.ndarray:
                        repeats = (target_length // len(ch_data)) + 1
                        return np.tile(ch_data, repeats)[:target_length]
                    
                    looped = process_channels(audio.data, loop_channel)
                    return AudioData(
                        data=looped,
                        sample_rate=audio.sample_rate,
                        file_path=audio.file_path
                    )
                
                pad_total = target_length - current_length
                if position == "start":
                    pad_before, pad_after = 0, pad_total
                elif position == "end":
                    pad_before, pad_after = pad_total, 0
                else:  # center
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                
                # 对 2D 数组进行填充：只在第二维（样本维度）填充
                padded = np.pad(audio.data, ((0, 0), (pad_before, pad_after)), mode=pad_mode)
                return AudioData(
                    data=padded,
                    sample_rate=audio.sample_rate,
                    file_path=audio.file_path
                )
        
        result = process_audio_or_list(audio_input, trim_pad)
        self.set_output_data("audio", result)
        return True


@register_node
class NormalizeNode(BaseNode):
    """
    归一化节点
    
    对音频进行振幅归一化。
    """
    
    node_type = "normalize"
    display_name = "归一化"
    category = NodeCategory.PREPROCESSING
    description = "对音频进行振幅归一化"
    icon = "📊"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "method", "choice", "peak",
            display_name="归一化方法",
            choices=["peak", "rms", "lufs"]
        )
        self.add_parameter(
            "target_level", "float", -3.0,
            display_name="目标电平(dB)",
            min_value=-60.0, max_value=0.0
        )
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        method = self.get_parameter("method")
        target_level = self.get_parameter("target_level")
        target_amplitude = 10 ** (target_level / 20)
        
        def normalize(audio: AudioData) -> AudioData:
            # 对每个通道分别归一化
            def normalize_channel(ch_data: np.ndarray) -> np.ndarray:
                data = ch_data.copy()
                
                if method == "peak":
                    peak = np.max(np.abs(data))
                    if peak > 0:
                        data = data * (target_amplitude / peak)
                elif method == "rms":
                    rms = np.sqrt(np.mean(data ** 2))
                    if rms > 0:
                        data = data * (target_amplitude / rms)
                else:  # lufs (简化版)
                    rms = np.sqrt(np.mean(data ** 2))
                    if rms > 0:
                        data = data * (target_amplitude / rms)
                
                # 防止削波
                if np.max(np.abs(data)) > 1.0:
                    data = data / np.max(np.abs(data))
                
                return data
            
            normalized = process_channels(audio.data, normalize_channel)
            return AudioData(
                data=normalized,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, normalize)
        self.set_output_data("audio", result)
        return True


@register_node
class SilenceTrimNode(BaseNode):
    """
    静音裁剪节点
    
    去除音频首尾的静音部分。
    """
    
    node_type = "silence_trim"
    display_name = "静音裁剪"
    category = NodeCategory.PREPROCESSING
    description = "去除音频首尾的静音"
    icon = "🎚️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "top_db", "int", 30,
            display_name="阈值(dB)",
            description="低于此阈值视为静音",
            min_value=10, max_value=80
        )
        self.add_parameter(
            "frame_length", "int", 2048,
            display_name="帧长度",
            min_value=256, max_value=8192
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name="跳跃长度",
            min_value=64, max_value=2048
        )
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        top_db = self.get_parameter("top_db")
        frame_length = self.get_parameter("frame_length")
        hop_length = self.get_parameter("hop_length")
        
        def trim_silence(audio: AudioData) -> AudioData:
            # 对多通道音频，使用混合后的信号来检测静音边界
            # 然后对所有通道应用相同的裁剪
            mono_mix = np.mean(audio.data, axis=0)
            _, trim_indices = librosa.effects.trim(
                mono_mix,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            # 对所有通道应用相同的裁剪
            start, end = trim_indices
            trimmed = audio.data[:, start:end]
            
            return AudioData(
                data=trimmed,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, trim_silence)
        self.set_output_data("audio", result)
        return True


@register_node
class ChannelMapperNode(BaseNode):
    """
    通道映射节点
    
    将多通道音频的指定通道映射到不同输出端口。
    支持单通道或多通道选择。
    """
    
    node_type = "channel_mapper"
    display_name = "通道映射"
    category = NodeCategory.PREPROCESSING
    description = "将指定通道映射到不同输出"
    icon = "🔀"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频", description="多通道音频输入")
        self.add_output("out1", DataType.AUDIO, "输出1")
        self.add_output("out2", DataType.AUDIO, "输出2")
        self.add_output("out3", DataType.AUDIO, "输出3")
        self.add_output("out4", DataType.AUDIO, "输出4")
    
    def _setup_parameters(self):
        self.add_parameter(
            "map1", "str", "0",
            display_name="输出1通道",
            description="通道索引(从0开始)，多个用逗号分隔，如: 0 或 0,1,2"
        )
        self.add_parameter(
            "map2", "str", "1",
            display_name="输出2通道",
            description="通道索引，留空则不输出"
        )
        self.add_parameter(
            "map3", "str", "",
            display_name="输出3通道",
            description="通道索引，留空则不输出"
        )
        self.add_parameter(
            "map4", "str", "",
            display_name="输出4通道",
            description="通道索引，留空则不输出"
        )
    
    def _parse_channel_indices(self, map_str: str, max_channels: int) -> List[int]:
        """
        解析通道映射字符串
        
        Args:
            map_str: 通道映射字符串，如 "0" 或 "0,1,2"
            max_channels: 最大通道数
        
        Returns:
            有效的通道索引列表
        """
        if not map_str or not map_str.strip():
            return []
        
        indices = []
        for part in map_str.split(','):
            part = part.strip()
            if part:
                try:
                    idx = int(part)
                    # 如果索引超过总通道数，映射到最后一个通道
                    if idx >= max_channels:
                        idx = max_channels - 1
                    if idx < 0:
                        idx = 0
                    indices.append(idx)
                except ValueError:
                    logger.warning(f"无效的通道索引: {part}")
                    continue
        
        return indices
    
    def _map_channels(self, audio: AudioData, channel_indices: List[int]) -> AudioData:
        """
        根据通道索引创建新的AudioData
        
        Args:
            audio: 输入音频
            channel_indices: 要提取的通道索引列表
        
        Returns:
            包含指定通道的新AudioData
        """
        if not channel_indices:
            return None
        
        # 提取指定通道
        selected_channels = []
        for idx in channel_indices:
            selected_channels.append(audio.data[idx])
        
        # 堆叠为 (n_selected_channels, samples)
        mapped_data = np.stack(selected_channels, axis=0)
        
        return AudioData(
            data=mapped_data,
            sample_rate=audio.sample_rate,
            file_path=audio.file_path
        )
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        # 获取映射配置
        map_configs = [
            ("out1", self.get_parameter("map1")),
            ("out2", self.get_parameter("map2")),
            ("out3", self.get_parameter("map3")),
            ("out4", self.get_parameter("map4")),
        ]
        
        def map_audio(audio: AudioData):
            """对单个音频进行通道映射，返回映射结果字典"""
            max_channels = audio.channels
            results = {}
            
            for port_name, map_str in map_configs:
                indices = self._parse_channel_indices(map_str, max_channels)
                if indices:
                    results[port_name] = self._map_channels(audio, indices)
                else:
                    results[port_name] = None
            
            return results
        
        # 处理输入（单个或列表）
        if isinstance(audio_input, list):
            # 对列表中的每个音频进行映射
            all_results = [map_audio(audio) for audio in audio_input]
            
            # 按端口重组结果
            for port_name, _ in map_configs:
                port_results = [r[port_name] for r in all_results if r[port_name] is not None]
                if port_results:
                    self.set_output_data(port_name, port_results)
                else:
                    self.set_output_data(port_name, None)
        else:
            # 单个音频
            results = map_audio(audio_input)
            for port_name, data in results.items():
                self.set_output_data(port_name, data)
        
        # 记录映射信息
        if isinstance(audio_input, list):
            sample_audio = audio_input[0]
        else:
            sample_audio = audio_input
        
        logger.info(
            f"通道映射: 输入 {sample_audio.channels} 通道, "
            f"映射配置: out1={self.get_parameter('map1')}, "
            f"out2={self.get_parameter('map2')}, "
            f"out3={self.get_parameter('map3')}, "
            f"out4={self.get_parameter('map4')}"
        )
        
        return True


@register_node
class ChannelMergeNode(BaseNode):
    """
    通道合并节点
    
    将多个音频的通道合并为一个多通道音频。
    支持最多4个输入，每个输入可以是单通道或多通道。
    """
    
    node_type = "channel_merge"
    display_name = "通道合并"
    category = NodeCategory.PREPROCESSING
    description = "合并多个音频的通道"
    icon = "🔗"
    
    def _setup_ports(self):
        self.add_input("in1", DataType.AUDIO, "输入1", description="音频输入1")
        self.add_input("in2", DataType.AUDIO, "输入2", description="音频输入2（可选）", required=False)
        self.add_input("in3", DataType.AUDIO, "输入3", description="音频输入3（可选）", required=False)
        self.add_input("in4", DataType.AUDIO, "输入4", description="音频输入4（可选）", required=False)
        self.add_output("audio", DataType.AUDIO, "合并音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "length_mode", "choice", "min",
            display_name="长度处理",
            description="当输入长度不一致时的处理方式",
            choices=["min", "max", "first"]
        )
        self.add_parameter(
            "pad_mode", "choice", "constant",
            display_name="填充模式",
            description="当使用max长度时，短音频的填充方式",
            choices=["constant", "edge", "reflect", "wrap"]
        )
    
    def execute(self) -> bool:
        # 收集所有非空输入
        inputs = []
        input_names = ["in1", "in2", "in3", "in4"]
        
        for name in input_names:
            data = self.get_input_data(name)
            if data is not None:
                inputs.append(data)
        
        if not inputs:
            self.error_message = "至少需要一个有效的音频输入"
            return False
        
        # 检查是否是列表输入
        is_list_input = isinstance(inputs[0], list)
        
        # 如果是列表，需要确保所有输入都是列表且长度相同
        if is_list_input:
            list_lengths = []
            for inp in inputs:
                if isinstance(inp, list):
                    list_lengths.append(len(inp))
                else:
                    self.error_message = "所有输入必须是相同类型（都是单个音频或都是音频列表）"
                    return False
            
            if len(set(list_lengths)) > 1:
                self.error_message = f"音频列表长度不一致: {list_lengths}"
                return False
            
            # 逐个处理列表中的音频
            results = []
            for i in range(list_lengths[0]):
                audios_to_merge = [inp[i] for inp in inputs]
                merged = self._merge_audios(audios_to_merge)
                if merged is None:
                    return False
                results.append(merged)
            
            self.set_output_data("audio", results)
        else:
            # 单个音频处理
            # 验证所有输入都是 AudioData
            for inp in inputs:
                if not isinstance(inp, AudioData):
                    self.error_message = f"期望 AudioData 类型，但收到 {type(inp).__name__}"
                    return False
            
            merged = self._merge_audios(inputs)
            if merged is None:
                return False
            
            self.set_output_data("audio", merged)
        
        return True
    
    def _merge_audios(self, audios: List[AudioData]) -> AudioData:
        """
        合并多个音频的通道
        
        Args:
            audios: AudioData 列表
        
        Returns:
            合并后的 AudioData
        """
        if not audios:
            self.error_message = "没有要合并的音频"
            return None
        
        # 检查采样率是否一致
        sample_rates = [a.sample_rate for a in audios]
        if len(set(sample_rates)) > 1:
            self.error_message = f"采样率不一致: {sample_rates}，请先使用重采样节点统一采样率"
            return None
        
        sample_rate = sample_rates[0]
        
        # 获取各音频的长度
        lengths = [a.samples for a in audios]
        
        # 确定目标长度
        length_mode = self.get_parameter("length_mode")
        if length_mode == "min":
            target_length = min(lengths)
        elif length_mode == "max":
            target_length = max(lengths)
        else:  # first
            target_length = lengths[0]
        
        pad_mode = self.get_parameter("pad_mode")
        
        # 收集所有通道
        all_channels = []
        for audio in audios:
            data = audio.data
            current_length = audio.samples
            
            # 调整长度
            if current_length > target_length:
                # 裁剪
                data = data[:, :target_length]
            elif current_length < target_length:
                # 填充
                pad_width = target_length - current_length
                data = np.pad(data, ((0, 0), (0, pad_width)), mode=pad_mode)
            
            # 添加所有通道
            for ch in range(data.shape[0]):
                all_channels.append(data[ch])
        
        # 堆叠所有通道
        merged_data = np.stack(all_channels, axis=0)
        
        # 使用第一个音频的文件路径
        file_path = audios[0].file_path
        
        logger.info(
            f"通道合并: {len(audios)} 个输入, "
            f"通道数 {[a.channels for a in audios]} -> {merged_data.shape[0]}, "
            f"长度模式={length_mode}"
        )
        
        return AudioData(
            data=merged_data,
            sample_rate=sample_rate,
            file_path=file_path
        )


@register_node
class FilterNode(BaseNode):
    """
    滤波器节点
    
    对音频应用数字滤波器，支持低通、高通、带通、带阻四种类型。
    使用 Butterworth 滤波器设计，采用零相位滤波避免相位失真。
    """
    
    node_type = "filter"
    display_name = "滤波器"
    category = NodeCategory.PREPROCESSING
    description = "对音频应用数字滤波（低通/高通/带通/带阻）"
    icon = "🎛️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频", description="音频或音频列表")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "filter_type", "choice", "lowpass",
            display_name="滤波类型",
            description="lowpass=低通, highpass=高通, bandpass=带通, bandstop=带阻",
            choices=["lowpass", "highpass", "bandpass", "bandstop"]
        )
        self.add_parameter(
            "cutoff", "float", 1000.0,
            display_name="截止频率(Hz)",
            description="低通/高通使用此值；带通/带阻时为低频边界",
            min_value=1.0, max_value=22000.0
        )
        self.add_parameter(
            "cutoff_high", "float", 5000.0,
            display_name="高频截止(Hz)",
            description="仅带通/带阻使用，设置高频边界（低通/高通时忽略）",
            min_value=1.0, max_value=22000.0
        )
        self.add_parameter(
            "order", "int", 4,
            display_name="滤波器阶数",
            description="Butterworth滤波器阶数，越高衰减越陡峭",
            min_value=1, max_value=10
        )
    
    def execute(self) -> bool:
        from scipy.signal import butter, filtfilt
        
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        filter_type = self.get_parameter("filter_type")
        cutoff = self.get_parameter("cutoff")
        cutoff_high = self.get_parameter("cutoff_high")
        order = self.get_parameter("order")
        
        def apply_filter(audio: AudioData) -> AudioData:
            # 计算 Nyquist 频率
            nyquist = audio.sample_rate / 2.0
            
            # 归一化截止频率（相对于 Nyquist 频率）
            # 确保不超过 Nyquist 频率
            low_norm = min(cutoff / nyquist, 0.99)
            high_norm = min(cutoff_high / nyquist, 0.99)
            
            # 确保频率有效
            if low_norm <= 0:
                low_norm = 0.01
            if high_norm <= 0:
                high_norm = 0.01
            
            # 根据滤波类型设计滤波器
            try:
                if filter_type == "lowpass":
                    b, a = butter(order, low_norm, btype='low')
                elif filter_type == "highpass":
                    b, a = butter(order, low_norm, btype='high')
                elif filter_type == "bandpass":
                    # 确保 low < high
                    if low_norm >= high_norm:
                        low_norm, high_norm = high_norm, low_norm
                    b, a = butter(order, [low_norm, high_norm], btype='band')
                elif filter_type == "bandstop":
                    # 确保 low < high
                    if low_norm >= high_norm:
                        low_norm, high_norm = high_norm, low_norm
                    b, a = butter(order, [low_norm, high_norm], btype='bandstop')
                else:
                    raise ValueError(f"未知的滤波类型: {filter_type}")
            except Exception as e:
                raise ValueError(f"滤波器设计失败: {e}")
            
            # 对每个通道应用滤波
            def filter_channel(ch_data: np.ndarray) -> np.ndarray:
                # 使用 filtfilt 进行零相位滤波
                # padlen 设置为确保足够的边缘填充
                padlen = min(3 * max(len(b), len(a)), len(ch_data) - 1)
                if padlen < 1:
                    padlen = 1
                return filtfilt(b, a, ch_data, padlen=padlen)
            
            filtered = process_channels(audio.data, filter_channel)
            
            return AudioData(
                data=filtered,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        try:
            result = process_audio_or_list(audio_input, apply_filter)
        except Exception as e:
            self.error_message = f"滤波处理失败: {e}"
            return False
        
        self.set_output_data("audio", result)
        
        # 记录滤波信息
        if filter_type in ["lowpass", "highpass"]:
            logger.info(
                f"滤波器: {filter_type}, 截止频率={cutoff}Hz, 阶数={order}"
            )
        else:
            logger.info(
                f"滤波器: {filter_type}, 频率范围={cutoff}-{cutoff_high}Hz, 阶数={order}"
            )
        
        return True

