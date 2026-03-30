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
from src.ui.i18n import tr_
logger = logging.getLogger(__name__)


def validate_audio_input(data, node_name: str = "Node") -> Union[AudioData, List[AudioData]]:
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
        raise ValueError(tr_("{node}: No input audio provided").format(node=node_name))
    
    if isinstance(data, list):
        if not data:
            raise ValueError(tr_("{node}: Audio list is empty").format(node=node_name))
        if not isinstance(data[0], AudioData):
            raise TypeError(
                tr_(
                    "{node}: Expected AudioData, but got {type}. "
                    "Please check whether the upstream node output type is correct."
                ).format(
                    node=node_name,
                    type=type(data[0]).__name__,
                )
            )
    else:
        if not isinstance(data, AudioData):
            raise TypeError(
                tr_(
                    "{node}: Expected AudioData, but got {type}. "
                    "Please check whether the upstream node output type is correct."
                ).format(
                    node=node_name,
                    type=type(data).__name__,
                )
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
    display_name = tr_("Resample")
    category = NodeCategory.PREPROCESSING
    description = tr_("Resample audio to the target sample rate")
    icon = "📏"
    
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
            "target_sr", "int", 22050,
            display_name=tr_("Target sample rate"),
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
    display_name = tr_("Trim / pad")
    category = NodeCategory.PREPROCESSING
    description = tr_("Make audio a fixed duration by trimming and/or padding")
    icon = "✂️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "duration", "float", 3.0,
            display_name=tr_("Target duration (s)"),
            min_value=0.1, max_value=60.0
        )
        self.add_parameter(
            "mode", "choice", "pad_trim",
            display_name=tr_("Mode"),
            choices=["pad_trim", "pad_only", "trim_only", "loop"]
        )
        self.add_parameter(
            "pad_mode", "choice", "constant",
            display_name=tr_("Pad mode"),
            choices=["constant", "edge", "reflect", "wrap"]
        )
        self.add_parameter(
            "position", "choice", "center",
            display_name=tr_("Alignment"),
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
    display_name = tr_("Normalize")
    category = NodeCategory.PREPROCESSING
    description = tr_("Normalize audio amplitude")
    icon = "📊"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "method", "choice", "peak",
            display_name=tr_("Method"),
            choices=["peak", "rms", "lufs"]
        )
        self.add_parameter(
            "target_level", "float", -3.0,
            display_name=tr_("Target level (dB)"),
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
class AdjustGainNode(BaseNode):
    """
    Gain adjustment node.

    Applies a relative gain change in decibels to the incoming audio.
    """

    node_type = "adjust_gain"
    display_name = tr_("Adjust gain")
    category = NodeCategory.PREPROCESSING
    description = tr_("Increase or decrease audio level by a relative dB amount")
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
        clip_protection = self.get_parameter("clip_protection")
        gain_scale = 10 ** (gain_db / 20)

        def adjust_gain(audio: AudioData) -> AudioData:
            adjusted = audio.data.astype(np.float32, copy=True) * gain_scale
            peak = float(np.max(np.abs(adjusted))) if adjusted.size else 0.0

            if peak > 1.0:
                if clip_protection:
                    adjusted = adjusted / peak
                else:
                    logger.warning(
                        "%s clipped audio after gain adjustment (gain_db=%s, peak=%s, file=%s)",
                        self.node_type,
                        gain_db,
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
class WindowingNode(BaseNode):
    """
    加窗节点
    
    对整段波形应用窗函数以减少频谱泄漏等边缘效应。
    输入输出均为 AudioData（或 AudioData 列表），不改变数据形状。
    """
    
    node_type = "windowing"
    display_name = tr_("Windowing")
    category = NodeCategory.PREPROCESSING
    description = tr_("Apply a window function to the full waveform")
    icon = "🪟"
    
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
            "window_type",
            "choice",
            "hann",
            display_name=tr_("Window type"),
            description=tr_("Window function applied to the full waveform"),
            choices=["hann", "hamming", "blackman", "boxcar"],
        )
        self.add_parameter(
            "window_len", "int", 0,
            display_name=tr_("Window length"),
            min_value=0
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
        
        window_type = self.get_parameter("window_type")
        
        def apply_window(audio: AudioData) -> AudioData:
            # boxcar is effectively "no window" (identity)
            if window_type == "boxcar" or audio.samples <= 0:
                return audio
            
            from scipy.signal import get_window
            
            data = audio.data
            target_len = audio.samples

            win_len = int(self.get_parameter("window_len") or 0)
            if win_len <= 0:
                win_len = target_len  # default: full length

            dtype = data.dtype if data.dtype.kind == "f" else np.float32

            win = get_window(window_type, win_len).astype(dtype, copy=False)
            effective_window = np.ones(target_len, dtype=dtype)

            if win_len == target_len:
                effective_window = win
            elif win_len > target_len:
                offset = (win_len - target_len) // 2
                effective_window = win[offset:offset + target_len]
            else:
                split_idx = win_len // 2
                left_half = win[:split_idx]
                right_half = win[split_idx:]
                effective_window[:len(left_half)] = left_half
                effective_window[-len(right_half):] = right_half

            if data.dtype.kind != "f":
                # Ensure numeric stability and avoid integer truncation.
                data = data.astype(np.float32, copy=False)

            windowed = data * effective_window[None, :]
            
            return AudioData(
                data=windowed,
                sample_rate=audio.sample_rate,
                file_path=audio.file_path
            )
        
        result = process_audio_or_list(audio_input, apply_window)
        self.set_output_data("audio", result)
        return True


@register_node
class SpectralSubtractionNode(BaseNode):
    """
    Spectral subtraction denoise node.

    Performs frequency-domain denoising on waveform audio while preserving the
    input/output `AudioData` shape semantics used by other preprocessing nodes.
    """

    node_type = "spectral_subtraction"
    display_name = tr_("Spectral subtraction")
    category = NodeCategory.PREPROCESSING
    subcategory = tr_("Denoise")
    description = tr_("Reduce stationary noise using spectral subtraction")
    icon = "🔉"

    def _setup_ports(self):
        self.add_input(
            "audio",
            DataType.AUDIO,
            tr_("Audio"),
            description=tr_("Audio or audio list"),
        )
        self.add_input(
            "noise",
            DataType.AUDIO,
            tr_("Noise reference"),
            required=False,
            description=tr_("Optional noise-only audio or audio list"),
        )
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))

    def _setup_parameters(self):
        self.add_parameter(
            "noise_duration",
            "float",
            0.25,
            display_name=tr_("Noise duration (s)"),
            description=tr_("Head duration used for noise estimation when no reference is connected"),
            min_value=0.01,
            max_value=10.0,
        )
        self.add_parameter(
            "subtract_alpha",
            "float",
            1.0,
            display_name=tr_("Subtract alpha"),
            description=tr_("Strength of spectral subtraction"),
            min_value=0.1,
            max_value=5.0,
        )
        self.add_parameter(
            "spectral_floor",
            "float",
            0.02,
            display_name=tr_("Spectral floor"),
            description=tr_("Residual floor factor to avoid musical noise"),
            min_value=0.0,
            max_value=1.0,
        )

    def execute(self) -> bool:
        try:
            audio_input = validate_audio_input(
                self.get_input_data("audio"),
                self.display_name,
            )
            noise_input = self._validate_noise_input(
                self.get_input_data("noise"),
                self.display_name,
            )
            result = self._process_audio_input(audio_input, noise_input)
        except (ValueError, TypeError) as e:
            self.error_message = str(e)
            return False
        except Exception as e:
            self.error_message = tr_("Spectral subtraction failed: {error}").format(
                error=str(e)
            )
            return False

        self.set_output_data("audio", result)
        return True

    def _validate_noise_input(self, data, node_name: str):
        if data is None:
            return None
        return validate_audio_input(data, node_name)

    def _process_audio_input(self, audio_input, noise_input):
        if isinstance(audio_input, list):
            if isinstance(noise_input, list) and len(noise_input) != len(audio_input):
                raise ValueError(
                    tr_("{node}: Noise reference list length must match audio list length").format(
                        node=self.display_name
                    )
                )

            results = []
            for index, audio in enumerate(audio_input):
                noise_reference = None
                if isinstance(noise_input, list):
                    noise_reference = noise_input[index]
                elif noise_input is not None:
                    noise_reference = noise_input
                results.append(self._denoise_audio(audio, noise_reference))
            return results

        if isinstance(noise_input, list):
            raise ValueError(
                tr_("{node}: Noise reference list requires audio list input").format(
                    node=self.display_name
                )
            )
        else:
            noise_reference = noise_input
        return self._denoise_audio(audio_input, noise_reference)

    def _denoise_audio(
        self,
        audio: AudioData,
        noise_reference: AudioData | None,
    ) -> AudioData:
        noise_waveform = self._resolve_noise_waveform(audio, noise_reference)

        def subtract_channel(channel_data: np.ndarray) -> np.ndarray:
            return self._spectral_subtract_channel(channel_data, noise_waveform)

        denoised = process_channels(audio.data, subtract_channel)
        return AudioData(
            data=denoised,
            sample_rate=audio.sample_rate,
            file_path=audio.file_path,
        )

    def _resolve_noise_waveform(
        self,
        audio: AudioData,
        noise_reference: AudioData | None,
    ) -> np.ndarray:
        if audio.samples <= 0:
            raise ValueError(
                tr_("{node}: Input audio is empty").format(node=self.display_name)
            )

        if noise_reference is not None:
            if noise_reference.sample_rate != audio.sample_rate:
                raise ValueError(
                    tr_("{node}: Noise reference sample rate must match input audio").format(
                        node=self.display_name
                    )
                )
            noise_data = noise_reference.data
        else:
            duration_s = float(self.get_parameter("noise_duration"))
            noise_samples = max(1, int(duration_s * audio.sample_rate))
            noise_data = audio.data[:, : min(noise_samples, audio.samples)]

        if noise_data.size == 0:
            raise ValueError(
                tr_("{node}: Noise estimation segment is empty").format(
                    node=self.display_name
                )
            )

        mono_noise = np.mean(noise_data, axis=0)
        return mono_noise.astype(np.float32, copy=False)

    def _spectral_subtract_channel(
        self,
        channel_data: np.ndarray,
        noise_waveform: np.ndarray,
    ) -> np.ndarray:
        alpha = float(self.get_parameter("subtract_alpha"))
        spectral_floor = float(self.get_parameter("spectral_floor"))

        channel_data = np.asarray(channel_data, dtype=np.float32)
        n_fft = self._choose_fft_size(channel_data.shape[0], noise_waveform.shape[0])
        hop_length = max(8, n_fft // 4)

        signal_stft = librosa.stft(channel_data, n_fft=n_fft, hop_length=hop_length)
        noise_stft = librosa.stft(noise_waveform, n_fft=n_fft, hop_length=hop_length)

        signal_mag = np.abs(signal_stft)
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        phase = np.angle(signal_stft)

        cleaned_mag = np.maximum(signal_mag - (alpha * noise_mag), spectral_floor * noise_mag)
        cleaned_stft = cleaned_mag * np.exp(1j * phase)

        return librosa.istft(cleaned_stft, hop_length=hop_length, length=channel_data.shape[0])

    def _choose_fft_size(self, signal_len: int, noise_len: int) -> int:
        base_length = max(32, min(signal_len, max(32, noise_len), 512))
        fft_size = 1 << int(np.floor(np.log2(base_length)))
        return max(32, int(fft_size))


@register_node
class SilenceTrimNode(BaseNode):
    """
    静音裁剪节点
    
    去除音频首尾的静音部分。
    """
    
    node_type = "silence_trim"
    display_name = tr_("Trim silence")
    category = NodeCategory.PREPROCESSING
    description = tr_("Remove leading and trailing silence")
    icon = "🎚️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "top_db", "int", 30,
            display_name=tr_("Threshold (dB)"),
            description=tr_("Below this threshold is considered silence"),
            min_value=10, max_value=80
        )
        self.add_parameter(
            "frame_length", "int", 2048,
            display_name=tr_("Frame length"),
            min_value=256, max_value=8192
        )
        self.add_parameter(
            "hop_length", "int", 512,
            display_name=tr_("Hop length"),
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
    display_name = tr_("Channel mapper")
    category = NodeCategory.PREPROCESSING
    description = tr_("Map selected channels to different outputs")
    icon = "🔀"
    
    def _setup_ports(self):
        self.add_input(
            "audio",
            DataType.AUDIO,
            tr_("Audio"),
            description=tr_("Multichannel audio input"),
        )
        self.add_output("out1", DataType.AUDIO, tr_("Output 1"))
        self.add_output("out2", DataType.AUDIO, tr_("Output 2"))
        self.add_output("out3", DataType.AUDIO, tr_("Output 3"))
        self.add_output("out4", DataType.AUDIO, tr_("Output 4"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "map1", "str", "0",
            display_name=tr_("Output 1 channels"),
            description=tr_("Channel indices (0-based), comma-separated, e.g. 0 or 0,1,2"),
        )
        self.add_parameter(
            "map2", "str", "1",
            display_name=tr_("Output 2 channels"),
            description=tr_("Channel indices; leave blank to disable output"),
        )
        self.add_parameter(
            "map3", "str", "",
            display_name=tr_("Output 3 channels"),
            description=tr_("Channel indices; leave blank to disable output"),
        )
        self.add_parameter(
            "map4", "str", "",
            display_name=tr_("Output 4 channels"),
            description=tr_("Channel indices; leave blank to disable output"),
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
    display_name = tr_("Channel merge")
    category = NodeCategory.PREPROCESSING
    description = tr_("Merge channels from multiple audio inputs")
    icon = "🔗"
    
    def _setup_ports(self):
        self.add_input("in1", DataType.AUDIO, tr_("Input 1"), description=tr_("Audio input 1"))
        self.add_input(
            "in2",
            DataType.AUDIO,
            tr_("Input 2"),
            description=tr_("Audio input 2 (optional)"),
            required=False,
        )
        self.add_input(
            "in3",
            DataType.AUDIO,
            tr_("Input 3"),
            description=tr_("Audio input 3 (optional)"),
            required=False,
        )
        self.add_input(
            "in4",
            DataType.AUDIO,
            tr_("Input 4"),
            description=tr_("Audio input 4 (optional)"),
            required=False,
        )
        self.add_output("audio", DataType.AUDIO, tr_("Merged audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "length_mode", "choice", "min",
            display_name=tr_("Length handling"),
            description=tr_("How to handle different input lengths"),
            choices=["min", "max", "first"]
        )
        self.add_parameter(
            "pad_mode", "choice", "constant",
            display_name=tr_("Pad mode"),
            description=tr_("Padding strategy for shorter audio when using max length"),
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
            self.error_message = tr_("At least one valid audio input is required")
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
                    self.error_message = tr_(
                        "All inputs must be the same type (all single audio or all audio lists)"
                    )
                    return False
            
            if len(set(list_lengths)) > 1:
                self.error_message = tr_("Audio list lengths differ: {lengths}").format(
                    lengths=list_lengths
                )
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
                    self.error_message = tr_("Expected AudioData, but got {type}").format(
                        type=type(inp).__name__
                    )
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
            self.error_message = tr_("No audio to merge")
            return None
        
        # 检查采样率是否一致
        sample_rates = [a.sample_rate for a in audios]
        if len(set(sample_rates)) > 1:
            self.error_message = tr_(
                "Sample rates differ: {rates}. Please resample first."
            ).format(rates=sample_rates)
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
    display_name = tr_("Filter")
    category = NodeCategory.PREPROCESSING
    description = tr_("Apply digital filtering (lowpass/highpass/bandpass/bandstop)")
    icon = "🎛️"
    
    def _setup_ports(self):
        self.add_input("audio", DataType.AUDIO, tr_("Audio"), description=tr_("Audio or audio list"))
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "filter_type", "choice", "lowpass",
            display_name=tr_("Filter type"),
            description=tr_("lowpass, highpass, bandpass, bandstop"),
            choices=["lowpass", "highpass", "bandpass", "bandstop"]
        )
        self.add_parameter(
            "cutoff", "float", 1000.0,
            display_name=tr_("Cutoff (Hz)"),
            description=tr_("Used by lowpass/highpass; low bound for bandpass/bandstop"),
            min_value=1.0, max_value=22000.0
        )
        self.add_parameter(
            "cutoff_high", "float", 5000.0,
            display_name=tr_("High cutoff (Hz)"),
            description=tr_("Only for bandpass/bandstop; ignored for lowpass/highpass"),
            min_value=1.0, max_value=22000.0
        )
        self.add_parameter(
            "order", "int", 4,
            display_name=tr_("Filter order"),
            description=tr_("Butterworth order; higher means steeper rolloff"),
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
                    raise ValueError(tr_("Unknown filter type: {type}").format(type=filter_type))
            except Exception as e:
                raise ValueError(tr_("Filter design failed: {error}").format(error=str(e)))
            
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
            self.error_message = tr_("Filtering failed: {error}").format(error=str(e))
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

