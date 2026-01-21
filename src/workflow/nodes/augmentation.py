# -*- coding: utf-8 -*-
"""
Data Augmentation Nodes

Provides audio data augmentation functionality: adding noise, time stretching, pitch shifting, data slicing, etc.
"""

import logging
from typing import List, Union

import librosa
import numpy as np

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from .data_source import AudioData
from .preprocessing import process_audio_or_list, process_channels, validate_audio_input


logger = logging.getLogger(__name__)


@register_node
class AddNoiseNode(BaseNode):
    """
    添加噪声节点
    
    向音频添加各类噪声。
    """
    
    node_type = "add_noise"
    display_name = "添加噪声"
    category = NodeCategory.AUGMENTATION
    description = "向音频添加噪声"
    icon = "🔊"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "加噪音频")
        self.add_output("original", DataType.AUDIO, "原始音频", 
                       description="用于训练降噪模型时作为目标")
    
    def _setup_parameters(self):
        self.add_parameter(
            "noise_type", "choice", "gaussian",
            display_name="噪声类型",
            choices=["gaussian", "uniform", "pink", "brown"]
        )
        self.add_parameter(
            "snr_db", "float", 20.0,
            display_name="信噪比(dB)",
            min_value=-10.0, max_value=60.0
        )
        self.add_parameter(
            "random_snr", "bool", False,
            display_name="随机信噪比",
            description="在指定范围内随机选择SNR"
        )
        self.add_parameter(
            "snr_min", "float", 10.0,
            display_name="最小SNR(dB)",
            min_value=-10.0, max_value=60.0
        )
        self.add_parameter(
            "snr_max", "float", 30.0,
            display_name="最大SNR(dB)",
            min_value=-10.0, max_value=60.0
        )
    
    def _generate_noise(self, length: int, noise_type: str) -> np.ndarray:
        """生成指定类型的噪声"""
        if noise_type == "gaussian":
            return np.random.randn(length)
        elif noise_type == "uniform":
            return np.random.uniform(-1, 1, length)
        elif noise_type == "pink":
            # 粉红噪声 (1/f)
            white = np.random.randn(length)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(length)
            freqs[0] = 1  # 避免除零
            fft = fft / np.sqrt(freqs)
            return np.fft.irfft(fft, length)
        elif noise_type == "brown":
            # 布朗噪声（积分白噪声）
            white = np.random.randn(length)
            brown = np.cumsum(white)
            return brown / np.max(np.abs(brown))
        else:
            return np.random.randn(length)
    
    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        
        noise_type = self.get_parameter("noise_type")
        random_snr = self.get_parameter("random_snr")
        
        # 保存原始音频
        self.set_output_data("original", audio_input)
        
        def add_noise(audio: AudioData) -> AudioData:
            # 确定SNR
            if random_snr:
                snr_min = self.get_parameter("snr_min")
                snr_max = self.get_parameter("snr_max")
                snr_db = np.random.uniform(snr_min, snr_max)
            else:
                snr_db = self.get_parameter("snr_db")
            
            # 对每个通道分别添加噪声
            def add_noise_channel(ch_data: np.ndarray) -> np.ndarray:
                # 生成噪声
                noise = self._generate_noise(len(ch_data), noise_type)
                
                # 计算信号和噪声功率
                signal_power = np.mean(ch_data ** 2)
                noise_power = np.mean(noise ** 2)
                
                # 根据SNR调整噪声强度
                target_noise_power = signal_power / (10 ** (snr_db / 10))
                if noise_power > 0:
                    noise = noise * np.sqrt(target_noise_power / noise_power)
                
                # 混合
                noisy = ch_data + noise
                
                # 归一化防止削波
                max_val = np.max(np.abs(noisy))
                if max_val > 1.0:
                    noisy = noisy / max_val
                
                return noisy.astype(np.float32)
            
            noisy_data = process_channels(audio.data, add_noise_channel)
            return AudioData(
                data=noisy_data,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, add_noise)
        self.set_output_data("audio", result)
        return True


@register_node
class TimeStretchNode(BaseNode):
    """
    时间拉伸节点
    
    改变音频速度但保持音高不变。
    """
    
    node_type = "time_stretch"
    display_name = "时间拉伸"
    category = NodeCategory.AUGMENTATION
    description = "改变音频速度（保持音高）"
    icon = "⏱️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "rate", "float", 1.0,
            display_name="拉伸比例",
            description="<1减慢, >1加快",
            min_value=0.5, max_value=2.0
        )
        self.add_parameter(
            "random_rate", "bool", False,
            display_name="随机比例"
        )
        self.add_parameter(
            "rate_min", "float", 0.8,
            display_name="最小比例",
            min_value=0.5, max_value=1.0
        )
        self.add_parameter(
            "rate_max", "float", 1.2,
            display_name="最大比例",
            min_value=1.0, max_value=2.0
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
        
        random_rate = self.get_parameter("random_rate")
        
        def time_stretch(audio: AudioData) -> AudioData:
            if random_rate:
                rate_min = self.get_parameter("rate_min")
                rate_max = self.get_parameter("rate_max")
                rate = np.random.uniform(rate_min, rate_max)
            else:
                rate = self.get_parameter("rate")
            
            if rate == 1.0:
                return audio
            
            # 对每个通道分别进行时间拉伸
            def stretch_channel(ch_data: np.ndarray) -> np.ndarray:
                return librosa.effects.time_stretch(ch_data, rate=rate)
            
            stretched = process_channels(audio.data, stretch_channel)
            return AudioData(
                data=stretched,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, time_stretch)
        self.set_output_data("audio", result)
        return True


@register_node
class PitchShiftNode(BaseNode):
    """
    音高偏移节点
    
    改变音频音高但保持速度不变。
    """
    
    node_type = "pitch_shift"
    display_name = "音高偏移"
    category = NodeCategory.AUGMENTATION
    description = "改变音频音高（保持速度）"
    icon = "🎵"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "semitones", "float", 0.0,
            display_name="半音数",
            description="正数升调，负数降调",
            min_value=-12.0, max_value=12.0
        )
        self.add_parameter(
            "random_shift", "bool", False,
            display_name="随机偏移"
        )
        self.add_parameter(
            "shift_min", "float", -4.0,
            display_name="最小偏移(半音)",
            min_value=-12.0, max_value=0.0
        )
        self.add_parameter(
            "shift_max", "float", 4.0,
            display_name="最大偏移(半音)",
            min_value=0.0, max_value=12.0
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
        
        random_shift = self.get_parameter("random_shift")
        
        def pitch_shift(audio: AudioData) -> AudioData:
            if random_shift:
                shift_min = self.get_parameter("shift_min")
                shift_max = self.get_parameter("shift_max")
                semitones = np.random.uniform(shift_min, shift_max)
            else:
                semitones = self.get_parameter("semitones")
            
            if semitones == 0.0:
                return audio
            
            # 对每个通道分别进行音高偏移
            def shift_channel(ch_data: np.ndarray) -> np.ndarray:
                return librosa.effects.pitch_shift(
                    ch_data, 
                    sr=audio.sample_rate, 
                    n_steps=semitones
                )
            
            shifted = process_channels(audio.data, shift_channel)
            return AudioData(
                data=shifted,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, pitch_shift)
        self.set_output_data("audio", result)
        return True


@register_node
class RandomAugmentNode(BaseNode):
    """
    随机增强节点
    
    随机应用多种增强效果。
    """
    
    node_type = "random_augment"
    display_name = "随机增强"
    category = NodeCategory.AUGMENTATION
    description = "随机应用多种增强效果"
    icon = "🔀"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_output("audio", DataType.AUDIO, "音频")
        self.add_output("original", DataType.AUDIO, "原始音频")
    
    def _setup_parameters(self):
        self.add_parameter(
            "enable_noise", "bool", True,
            display_name="启用噪声"
        )
        self.add_parameter(
            "enable_stretch", "bool", True,
            display_name="启用时间拉伸"
        )
        self.add_parameter(
            "enable_pitch", "bool", True,
            display_name="启用音高偏移"
        )
        self.add_parameter(
            "enable_gain", "bool", True,
            display_name="启用增益变化"
        )
        self.add_parameter(
            "probability", "float", 0.5,
            display_name="应用概率",
            description="每种效果被应用的概率",
            min_value=0.0, max_value=1.0
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
        
        # 保存原始音频
        self.set_output_data("original", audio_input)
        
        enable_noise = self.get_parameter("enable_noise")
        enable_stretch = self.get_parameter("enable_stretch")
        enable_pitch = self.get_parameter("enable_pitch")
        enable_gain = self.get_parameter("enable_gain")
        probability = self.get_parameter("probability")
        
        def random_augment(audio: AudioData) -> AudioData:
            sr = audio.sample_rate
            
            # 预先确定哪些增强会被应用（所有通道使用相同的决策）
            apply_noise = enable_noise and np.random.random() < probability
            apply_stretch = enable_stretch and np.random.random() < probability
            apply_pitch = enable_pitch and np.random.random() < probability
            apply_gain = enable_gain and np.random.random() < probability
            
            # 预先生成随机参数（所有通道使用相同参数）
            snr_db = np.random.uniform(15, 35) if apply_noise else 0
            stretch_rate = np.random.uniform(0.9, 1.1) if apply_stretch else 1.0
            pitch_semitones = np.random.uniform(-3, 3) if apply_pitch else 0
            gain = 10 ** (np.random.uniform(-6, 6) / 20) if apply_gain else 1.0
            
            # 对每个通道分别处理
            def augment_channel(ch_data: np.ndarray) -> np.ndarray:
                data = ch_data.copy()
                
                # 添加噪声
                if apply_noise:
                    noise = np.random.randn(len(data))
                    signal_power = np.mean(data ** 2)
                    noise_power = np.mean(noise ** 2)
                    target_noise_power = signal_power / (10 ** (snr_db / 10))
                    if noise_power > 0:
                        noise = noise * np.sqrt(target_noise_power / noise_power)
                    data = data + noise
                
                # 时间拉伸
                if apply_stretch:
                    data = librosa.effects.time_stretch(data, rate=stretch_rate)
                
                # 音高偏移
                if apply_pitch:
                    data = librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_semitones)
                
                # 增益变化
                if apply_gain:
                    data = data * gain
                
                # 归一化防止削波
                max_val = np.max(np.abs(data))
                if max_val > 1.0:
                    data = data / max_val
                
                return data.astype(np.float32)
            
            augmented = process_channels(audio.data, augment_channel)
            return AudioData(
                data=augmented,
                sample_rate=sr,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, random_augment)
        self.set_output_data("audio", result)
        return True


@register_node
class AudioSliceNode(BaseNode):
    """
    音频切片节点
    
    将长音频切片成多个短音频片段，以扩充数据量。
    
    例如：5秒音频 → 切片成1秒 × 10个 = 数据量扩充10倍
    
    支持两种切片模式：
    - 随机切片：随机起始位置，指定扩充倍数
    - 滑动窗口：固定步长滑动，可设置重叠率
    
    标签处理：
    - 如果提供了标签输入，每个音频对应的标签会被复制到其所有切片上
    - 确保输出的标签数量与切片数量一致
    """
    
    node_type = "audio_slice"
    display_name = "数据切片"
    category = NodeCategory.AUGMENTATION
    description = "将音频切片成多个短片段扩充数据"
    icon = "✂️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, "音频")
        self.add_input("labels", DataType.LABEL, "标签", required=False,
                      description="可选标签输入，将随切片同步扩充")
        self.add_output("audio", DataType.AUDIO, "切片音频列表")
        self.add_output("labels", DataType.LABEL, "切片标签列表",
                       description="与切片音频一一对应的标签")
    
    def _setup_parameters(self):
        self.add_parameter(
            "slice_duration", "float", 1.0,
            display_name="切片长度(秒)",
            description="每个切片的时长",
            min_value=0.1, max_value=30.0
        )
        self.add_parameter(
            "slice_mode", "choice", "random",
            display_name="切片模式",
            choices=["random", "sliding"],
            description="random: 随机起始位置; sliding: 滑动窗口"
        )
        self.add_parameter(
            "multiplier", "int", 10,
            display_name="扩充倍数",
            description="随机模式下生成的切片数量",
            min_value=1, max_value=100
        )
        self.add_parameter(
            "overlap", "float", 0.5,
            display_name="重叠率",
            description="滑动窗口模式的重叠比例 (0-0.9)",
            min_value=0.0, max_value=0.9
        )
        self.add_parameter(
            "include_remainder", "bool", False,
            display_name="包含尾部残余",
            description="滑动窗口模式下，是否包含不足一个切片长度的尾部"
        )
        self.add_parameter(
            "pad_mode", "choice", "zero",
            display_name="填充模式",
            choices=["zero", "reflect", "wrap"],
            description="尾部残余的填充方式"
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
        
        # 获取标签输入（可选）
        labels_input = self.get_input_data("labels")
        
        slice_duration = self.get_parameter("slice_duration")
        slice_mode = self.get_parameter("slice_mode")
        multiplier = self.get_parameter("multiplier")
        overlap = self.get_parameter("overlap")
        include_remainder = self.get_parameter("include_remainder")
        pad_mode = self.get_parameter("pad_mode")
        
        def slice_audio(audio: AudioData) -> List[AudioData]:
            """对单个音频进行切片"""
            sr = audio.sample_rate
            slice_samples = int(slice_duration * sr)
            total_samples = audio.data.shape[-1]
            
            # 如果音频比切片还短，直接返回填充后的版本
            if total_samples <= slice_samples:
                if total_samples < slice_samples:
                    # 需要填充
                    pad_width = slice_samples - total_samples
                    if audio.data.ndim == 1:
                        if pad_mode == "zero":
                            padded = np.pad(audio.data, (0, pad_width), mode='constant')
                        elif pad_mode == "reflect":
                            padded = np.pad(audio.data, (0, pad_width), mode='reflect')
                        else:  # wrap
                            padded = np.pad(audio.data, (0, pad_width), mode='wrap')
                    else:
                        if pad_mode == "zero":
                            padded = np.pad(audio.data, ((0, 0), (0, pad_width)), mode='constant')
                        elif pad_mode == "reflect":
                            padded = np.pad(audio.data, ((0, 0), (0, pad_width)), mode='reflect')
                        else:  # wrap
                            padded = np.pad(audio.data, ((0, 0), (0, pad_width)), mode='wrap')
                    return [AudioData(
                        data=padded.astype(np.float32),
                        sample_rate=sr,
                        file_path=audio.file_path
                    )]
                return [audio]
            
            slices = []
            
            if slice_mode == "random":
                # 随机切片模式
                max_start = total_samples - slice_samples
                for _ in range(multiplier):
                    start = np.random.randint(0, max_start + 1)
                    if audio.data.ndim == 1:
                        slice_data = audio.data[start:start + slice_samples]
                    else:
                        slice_data = audio.data[:, start:start + slice_samples]
                    
                    slices.append(AudioData(
                        data=slice_data.astype(np.float32),
                        sample_rate=sr,
                        file_path=audio.file_path
                    ))
            
            else:  # sliding window
                # 滑动窗口模式
                step_samples = int(slice_samples * (1 - overlap))
                step_samples = max(step_samples, 1)  # 至少移动1个样本
                
                start = 0
                while start + slice_samples <= total_samples:
                    if audio.data.ndim == 1:
                        slice_data = audio.data[start:start + slice_samples]
                    else:
                        slice_data = audio.data[:, start:start + slice_samples]
                    
                    slices.append(AudioData(
                        data=slice_data.astype(np.float32),
                        sample_rate=sr,
                        file_path=audio.file_path
                    ))
                    start += step_samples
                
                # 处理尾部残余
                if include_remainder and start < total_samples:
                    remaining = total_samples - start
                    if audio.data.ndim == 1:
                        remainder_data = audio.data[start:]
                    else:
                        remainder_data = audio.data[:, start:]
                    
                    # 填充到完整切片长度
                    pad_width = slice_samples - remaining
                    if audio.data.ndim == 1:
                        if pad_mode == "zero":
                            padded = np.pad(remainder_data, (0, pad_width), mode='constant')
                        elif pad_mode == "reflect":
                            padded = np.pad(remainder_data, (0, pad_width), mode='reflect')
                        else:  # wrap
                            padded = np.pad(remainder_data, (0, pad_width), mode='wrap')
                    else:
                        if pad_mode == "zero":
                            padded = np.pad(remainder_data, ((0, 0), (0, pad_width)), mode='constant')
                        elif pad_mode == "reflect":
                            padded = np.pad(remainder_data, ((0, 0), (0, pad_width)), mode='reflect')
                        else:  # wrap
                            padded = np.pad(remainder_data, ((0, 0), (0, pad_width)), mode='wrap')
                    
                    slices.append(AudioData(
                        data=padded.astype(np.float32),
                        sample_rate=sr,
                        file_path=audio.file_path
                    ))
            
            return slices
        
        # 处理单个音频或音频列表
        all_slices = []
        slice_counts = []  # 记录每个原始音频生成的切片数量
        
        if isinstance(audio_input, list):
            # 输入是列表，对每个音频切片后展平
            for audio in audio_input:
                slices = slice_audio(audio)
                slice_counts.append(len(slices))
                all_slices.extend(slices)
            result = all_slices
        else:
            result = slice_audio(audio_input)
            slice_counts.append(len(result))
        
        # 处理标签同步扩充
        output_labels = None
        if labels_input is not None:
            # 将标签转换为列表形式
            if not isinstance(labels_input, list):
                labels_list = [labels_input]
            else:
                labels_list = labels_input
            
            # 验证标签数量与音频数量匹配
            expected_audio_count = len(audio_input) if isinstance(audio_input, list) else 1
            if len(labels_list) != expected_audio_count:
                logger.warning(
                    f"标签数量 ({len(labels_list)}) 与音频数量 ({expected_audio_count}) 不匹配，"
                    f"将尝试按比例对应"
                )
                # 如果标签数量与音频不匹配但比例正确，尝试调整
                if len(labels_list) == 1:
                    # 单个标签应用到所有切片
                    labels_list = labels_list * expected_audio_count
            
            # 按切片数量复制标签
            output_labels = []
            for i, count in enumerate(slice_counts):
                if i < len(labels_list):
                    label = labels_list[i]
                    output_labels.extend([label] * count)
                else:
                    # 标签不足时使用最后一个标签
                    output_labels.extend([labels_list[-1]] * count)
            
            logger.info(f"标签同步扩充完成: {len(labels_list)} 个标签 → {len(output_labels)} 个标签")
        
        logger.info(f"数据切片完成: 生成 {len(result)} 个切片 (每个 {slice_duration}s)")
        self.set_output_data("audio", result)
        self.set_output_data("labels", output_labels)
        return True

