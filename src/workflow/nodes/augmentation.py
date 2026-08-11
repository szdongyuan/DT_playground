# -*- coding: utf-8 -*-
"""
Data Augmentation Nodes

Provides audio data augmentation functionality: adding noise, time stretching, pitch shifting, data slicing, etc.
"""

import logging
from typing import List, Union

import librosa
import numpy as np
from scipy.signal import fftconvolve

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from .data_source import AudioData
from .preprocessing import process_audio_or_list, process_channels, validate_audio_input
from src.ui.i18n import tr_
logger = logging.getLogger(__name__)


@register_node
class SampleExpansionNode(BaseNode):
    """
    Sample expansion node.

    Duplicates audio and labels together so their item order remains aligned.
    """

    node_type = "sample_expansion"
    display_name = tr_("Sample expansion")
    category = NodeCategory.AUGMENTATION
    palette_order = 20
    description = tr_("Duplicate audio and labels together for sample expansion")
    icon = "📈"

    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_input("labels", DataType.LABEL, tr_("Labels"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("labels", DataType.LABEL, tr_("Labels"))

    def _setup_parameters(self):
        self.add_parameter(
            "multiplier", "int", 2,
            display_name=tr_("Multiplier"),
            description=tr_("Number of duplicated samples to create per input item"),
            min_value=1, max_value=100
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

        labels_input = self.get_input_data("labels")
        if labels_input is None:
            self.error_message = tr_("{node}: labels input is required").format(
                node=self.display_name
            )
            return False

        multiplier = self.get_parameter("multiplier")
        if not isinstance(multiplier, int) or multiplier < 1:
            self.error_message = tr_("Multiplier must be a positive integer")
            return False

        audio_items = audio_input if isinstance(audio_input, list) else [audio_input]
        label_items = labels_input if isinstance(labels_input, list) else [labels_input]

        if len(audio_items) != len(label_items):
            self.error_message = tr_(
                "Audio and label counts must match for sample expansion"
            )
            return False

        expanded_audio = []
        expanded_labels = []
        for audio, label in zip(audio_items, label_items):
            expanded_audio.extend([audio] * multiplier)
            expanded_labels.extend([label] * multiplier)

        self.set_output_data("audio", expanded_audio)
        self.set_output_data("labels", expanded_labels)
        return True


@register_node
class AddNoiseNode(BaseNode):
    """
    添加噪声节点
    
    向音频添加各类噪声。
    """
    
    node_type = "add_noise"
    display_name = tr_("Add noise")
    category = NodeCategory.AUGMENTATION
    palette_order = 30
    description = tr_("Add noise to audio")
    icon = "🔊"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Noisy audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "noise_type", "choice", "gaussian",
            display_name=tr_("Noise type"),
            choices=["gaussian", "uniform", "pink", "brown"]
        )
        self.add_parameter(
            "snr_db", "float", 20.0,
            display_name=tr_("SNR (dB)"),
            min_value=-10.0, max_value=60.0
        )
        self.add_parameter(
            "random_snr", "bool", False,
            display_name=tr_("Random SNR"),
            description=tr_("Randomly choose SNR within the specified range")
        )
        self.add_parameter(
            "snr_min", "float", 10.0,
            display_name=tr_("Min SNR (dB)"),
            min_value=-10.0, max_value=60.0
        )
        self.add_parameter(
            "snr_max", "float", 30.0,
            display_name=tr_("Max SNR (dB)"),
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
class AdjustGainNode(BaseNode):
    """
    Gain adjustment node.

    Applies a fixed or random relative gain change in decibels.
    """

    node_type = "adjust_gain"
    display_name = tr_("Adjust gain")
    category = NodeCategory.AUGMENTATION
    palette_order = 40
    description = tr_("Increase or decrease audio level by a fixed or random dB amount")
    icon = "🔊"

    def _setup_ports(self):
        self.add_input(
            "audio",
            DataType.AUDIO,
            tr_("Audio"),
            description=tr_("Audio or audio list"),
        )
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))

    def _setup_parameters(self):
        self.add_parameter(
            "gain_db", "float", 0.0,
            display_name=tr_("Gain (dB)"),
            min_value=-60.0, max_value=60.0
        )
        self.add_parameter(
            "random_gain", "bool", False,
            display_name=tr_("Random gain"),
            description=tr_("Randomly choose gain within the specified range")
        )
        self.add_parameter(
            "gain_db_min", "float", -6.0,
            display_name=tr_("Minimum gain (dB)"),
            min_value=-60.0, max_value=60.0
        )
        self.add_parameter(
            "gain_db_max", "float", 6.0,
            display_name=tr_("Maximum gain (dB)"),
            min_value=-60.0, max_value=60.0
        )
        self.add_parameter(
            "clip_protection", "bool", False,
            display_name=tr_("Clip protection"),
            description=tr_("Automatically rescale audio to avoid clipping")
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

        gain_db = self.get_parameter("gain_db")
        random_gain = self.get_parameter("random_gain")
        gain_db_min = self.get_parameter("gain_db_min")
        gain_db_max = self.get_parameter("gain_db_max")
        clip_protection = self.get_parameter("clip_protection")

        if random_gain:
            if not np.isfinite(gain_db_min) or not np.isfinite(gain_db_max):
                self.error_message = str(
                    tr_("Random gain bounds must be finite values")
                )
                return False
            if gain_db_min > gain_db_max:
                self.error_message = str(
                    tr_("Minimum gain must be less than or equal to maximum gain")
                )
                return False
        elif not np.isfinite(gain_db):
            self.error_message = str(
                tr_("Gain must be a finite value")
            )
            return False

        def adjust_gain(audio: AudioData) -> AudioData:
            selected_gain_db = (
                np.random.uniform(gain_db_min, gain_db_max)
                if random_gain else gain_db
            )
            gain_scale = 10 ** (selected_gain_db / 20)
            adjusted = audio.data.astype(np.float32, copy=True) * gain_scale
            peak = float(np.max(np.abs(adjusted))) if adjusted.size else 0.0

            if peak > 1.0:
                if clip_protection:
                    adjusted = adjusted / peak
                else:
                    logger.warning(
                        "%s clipped audio after gain adjustment (gain_db=%s, peak=%s, file=%s)",
                        self.node_type,
                        selected_gain_db,
                        peak,
                        audio.file_path or "<memory>",
                    )
                    adjusted = np.clip(adjusted, -1.0, 1.0)

            return AudioData(
                data=adjusted.astype(audio.data.dtype, copy=False),
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )

        result = process_audio_or_list(audio_input, adjust_gain)
        self.set_output_data("audio", result)
        return True


@register_node
class TimeStretchNode(BaseNode):
    """
    时间拉伸节点
    
    改变音频速度但保持音高不变。
    """
    
    node_type = "time_stretch"
    display_name = tr_("Time stretch")
    category = NodeCategory.AUGMENTATION
    palette_order = 50
    description = tr_("Change speed while preserving pitch")
    icon = "⏱️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "rate", "float", 1.0,
            display_name=tr_("Stretch rate"),
            description=tr_("<1 slower, >1 faster"),
            min_value=0.5, max_value=2.0
        )
        self.add_parameter(
            "random_rate", "bool", False,
            display_name=tr_("Random rate")
        )
        self.add_parameter(
            "rate_min", "float", 0.8,
            display_name=tr_("Min rate"),
            min_value=0.5, max_value=1.0
        )
        self.add_parameter(
            "rate_max", "float", 1.2,
            display_name=tr_("Max rate"),
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
    display_name = tr_("Pitch shift")
    category = NodeCategory.AUGMENTATION
    palette_order = 60
    description = tr_("Change pitch while preserving speed")
    icon = "🎵"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "semitones", "float", 0.0,
            display_name=tr_("Semitones"),
            description=tr_("Positive = higher, negative = lower"),
            min_value=-12.0, max_value=12.0
        )
        self.add_parameter(
            "random_shift", "bool", False,
            display_name=tr_("Random shift")
        )
        self.add_parameter(
            "shift_min", "float", -4.0,
            display_name=tr_("Min shift (semitones)"),
            min_value=-12.0, max_value=0.0
        )
        self.add_parameter(
            "shift_max", "float", 4.0,
            display_name=tr_("Max shift (semitones)"),
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
class ReverbNode(BaseNode):
    node_type = "reverb"
    display_name = tr_("Reverb")
    category = NodeCategory.AUGMENTATION
    palette_order = 70
    description = tr_("Apply synthetic room reverb to audio")
    icon = "🏛️"

    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_input(
            "ir",
            DataType.AUDIO,
            tr_("Impulse response"),
            required=False,
            description=tr_("Optional impulse response audio or audio list"),
        )
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))

    def _setup_parameters(self):
        self.add_parameter(
            "decay", "float", 0.4,
            display_name=tr_("Decay"),
            description=tr_("Controls how long the reverb tail lasts"),
            min_value=0.05, max_value=1.0
        )
        self.add_parameter(
            "wet_mix", "float", 0.3,
            display_name=tr_("Wet mix"),
            description=tr_("Blend between dry input and reverberated output"),
            min_value=0.0, max_value=1.0
        )
        self.add_parameter(
            "random_wet_mix", "bool", False,
            display_name=tr_("Random wet mix"),
            description=tr_("Randomly choose wet mix within the specified range")
        )
        self.add_parameter(
            "wet_mix_min", "float", 0.2,
            display_name=tr_("Min wet mix"),
            min_value=0.0, max_value=1.0
        )
        self.add_parameter(
            "wet_mix_max", "float", 0.7,
            display_name=tr_("Max wet mix"),
            min_value=0.0, max_value=1.0
        )

    def _create_synthetic_impulse_response(
        self,
        sample_rate: int,
        decay: float,
    ) -> np.ndarray:
        pre_delay_samples = int(sample_rate * 0.012)
        density = 0.7 + (0.5 * decay)
        tail_seconds = 0.18 + (decay * 0.95)
        tail_samples = max(1, int(sample_rate * tail_seconds))
        ir = np.zeros(pre_delay_samples + tail_samples, dtype=np.float32)

        early_reflections = np.array([0.0, 8.0, 16.0, 26.0, 38.0, 53.0], dtype=np.float32)
        early_reflections *= density
        early_gains = np.array([0.9, 0.65, 0.5, 0.38, 0.26, 0.18], dtype=np.float32) * decay

        for reflection_ms, gain in zip(early_reflections, early_gains):
            reflection_idx = pre_delay_samples + int(sample_rate * reflection_ms / 1000.0)
            if reflection_idx < ir.size:
                ir[reflection_idx] += gain

        tail_time = np.arange(tail_samples, dtype=np.float32) / float(sample_rate)
        decay_time = 0.08 + (decay * 1.35)
        envelope = np.exp(-tail_time / max(decay_time, 1e-4)).astype(np.float32)
        diffusion = (
            0.55 * np.sin(2 * np.pi * 41.0 * tail_time)
            + 0.30 * np.sin(2 * np.pi * 89.0 * tail_time + 0.7)
            + 0.15 * np.sin(2 * np.pi * 173.0 * tail_time + 1.3)
        ).astype(np.float32)
        ir[pre_delay_samples:] += 0.12 * decay * envelope * diffusion

        peak = float(np.max(np.abs(ir)))
        if peak > 0.0:
            ir /= peak

        return ir

    def _prepare_impulse_response(
        self,
        ir_input: Union[AudioData, List[AudioData], None],
        sample_rate: int,
        decay: float,
    ) -> np.ndarray:
        if ir_input is None:
            return self._create_synthetic_impulse_response(sample_rate, decay)

        selected_ir = ir_input[np.random.randint(0, len(ir_input))] if isinstance(ir_input, list) else ir_input
        ir = np.mean(selected_ir.data, axis=0)

        if selected_ir.sample_rate != sample_rate:
            ir = librosa.resample(ir, orig_sr=selected_ir.sample_rate, target_sr=sample_rate)

        peak = float(np.max(np.abs(ir)))
        if peak <= 0.0:
            raise ValueError(tr_("Reverb: impulse response audio contains only silence"))

        ir = (ir / peak).astype(np.float32)
        tail_time = np.arange(ir.shape[0], dtype=np.float32) / float(sample_rate)
        decay_time = 0.08 + (decay * 1.35)
        envelope = np.exp(-tail_time / max(decay_time, 1e-4)).astype(np.float32)
        return (ir * envelope).astype(np.float32)

    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name
            )
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False

        ir_input = self.get_input_data("ir")
        if ir_input is not None:
            try:
                ir_input = validate_audio_input(ir_input, tr_("Reverb impulse response"))
            except (ValueError, TypeError) as e:
                self.error_message = str(e)
                return False

        random_wet_mix = self.get_parameter("random_wet_mix")

        def apply_reverb(audio: AudioData) -> AudioData:
            if random_wet_mix:
                wet_mix = np.random.uniform(
                    self.get_parameter("wet_mix_min"),
                    self.get_parameter("wet_mix_max"),
                )
            else:
                wet_mix = self.get_parameter("wet_mix")

            if wet_mix <= 0.0:
                return audio

            decay = self.get_parameter("decay")

            impulse_response = self._prepare_impulse_response(
                ir_input,
                audio.sample_rate,
                decay,
            )

            def reverb_channel(ch_data: np.ndarray) -> np.ndarray:
                wet_signal = fftconvolve(ch_data, impulse_response, mode="full")[: len(ch_data)]
                mixed = ((1.0 - wet_mix) * ch_data) + (wet_mix * wet_signal)

                peak = np.max(np.abs(mixed))
                if peak > 1.0:
                    mixed = mixed / peak

                return mixed.astype(np.float32)

            reverberated = process_channels(audio.data, reverb_channel)
            return AudioData(
                data=reverberated,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )

        result = process_audio_or_list(audio_input, apply_reverb)
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
    
    """
    
    node_type = "audio_slice"
    display_name = tr_("Audio slicing")
    category = NodeCategory.AUGMENTATION
    palette_order = 10
    description = tr_("Slice audio into multiple short segments to augment data")
    icon = "✂️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Sliced audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "slice_duration", "float", 1.0,
            display_name=tr_("Slice duration (s)"),
            description=tr_("Duration of each slice"),
            min_value=0.1, max_value=30.0
        )
        self.add_parameter(
            "slice_mode", "choice", "random",
            display_name=tr_("Slice mode"),
            choices=["random", "sliding"],
            description=tr_("random: random start; sliding: sliding window")
        )
        self.add_parameter(
            "multiplier", "int", 10,
            display_name=tr_("Multiplier"),
            description=tr_("Number of slices to generate in random mode"),
            min_value=1, max_value=100
        )
        self.add_parameter(
            "overlap", "float", 0.5,
            display_name=tr_("Overlap"),
            description=tr_("Overlap ratio in sliding window mode (0-0.9)"),
            min_value=0.0, max_value=0.9
        )
        self.add_parameter(
            "include_remainder", "bool", False,
            display_name=tr_("Include remainder"),
            description=tr_("In sliding window mode, include the final remainder shorter than a full slice")
        )
        self.add_parameter(
            "pad_mode", "choice", "zero",
            display_name=tr_("Pad mode"),
            choices=["zero", "reflect", "wrap"],
            description=tr_("Padding strategy for the remainder segment")
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
        
        if isinstance(audio_input, list):
            # 输入是列表，对每个音频切片后展平
            for audio in audio_input:
                slices = slice_audio(audio)
                all_slices.extend(slices)
            result = all_slices
        else:
            result = slice_audio(audio_input)
        
        logger.info(f"数据切片完成: 生成 {len(result)} 个切片 (每个 {slice_duration}s)")
        self.set_output_data("audio", result)
        return True

