"""
Audio Utility Functions
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    获取音频文件信息
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        音频信息字典
    """
    import librosa
    
    duration = librosa.get_duration(path=file_path)
    y, sr = librosa.load(file_path, sr=None, duration=0.1)
    
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'duration': duration,
        'sample_rate': sr,
        'format': os.path.splitext(file_path)[1].lower(),
        'size_mb': os.path.getsize(file_path) / (1024 * 1024)
    }


def convert_audio(input_path: str, output_path: str,
                  target_sr: int = 44100,
                  mono: bool = True) -> bool:
    """
    转换音频格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        target_sr: 目标采样率
        mono: 是否转换为单声道
        
    Returns:
        是否成功
    """
    try:
        import librosa
        import soundfile as sf
        
        y, sr = librosa.load(input_path, sr=target_sr, mono=mono)
        sf.write(output_path, y, target_sr)
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False


def split_audio(audio: np.ndarray, sample_rate: int,
                segment_duration: float = 3.0,
                overlap: float = 0.0) -> List[np.ndarray]:
    """
    将音频分割为固定长度的片段
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        segment_duration: 片段时长（秒）
        overlap: 重叠比例 (0-1)
        
    Returns:
        音频片段列表
    """
    segment_length = int(segment_duration * sample_rate)
    hop_length = int(segment_length * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_length <= len(audio):
        segments.append(audio[start:start + segment_length])
        start += hop_length
    
    # 处理最后一个不完整的片段
    if start < len(audio):
        last_segment = np.zeros(segment_length)
        remaining = audio[start:]
        last_segment[:len(remaining)] = remaining
        segments.append(last_segment)
    
    return segments


def compute_energy(audio: np.ndarray, frame_length: int = 2048,
                   hop_length: int = 512) -> np.ndarray:
    """
    计算音频能量
    
    Args:
        audio: 音频数据
        frame_length: 帧长度
        hop_length: 帧移
        
    Returns:
        能量数组
    """
    import librosa
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
    return rms[0]


def detect_silence(audio: np.ndarray, sample_rate: int,
                   threshold_db: float = -40) -> List[Tuple[float, float]]:
    """
    检测静音段
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        threshold_db: 静音阈值（dB）
        
    Returns:
        静音段列表 [(start_time, end_time), ...]
    """
    import librosa
    
    # 计算RMS能量
    rms = librosa.feature.rms(y=audio)[0]
    rms_db = librosa.amplitude_to_db(rms)
    
    # 找出低于阈值的帧
    silent_frames = rms_db < threshold_db
    
    # 转换为时间段
    silent_regions = []
    in_silence = False
    start_frame = 0
    
    hop_length = 512  # 默认hop_length
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not is_silent and in_silence:
            in_silence = False
            start_time = librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=hop_length)
            end_time = librosa.frames_to_time(i, sr=sample_rate, hop_length=hop_length)
            silent_regions.append((start_time, end_time))
    
    # 处理结尾的静音
    if in_silence:
        start_time = librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=hop_length)
        end_time = len(audio) / sample_rate
        silent_regions.append((start_time, end_time))
    
    return silent_regions


def mix_audio(audio1: np.ndarray, audio2: np.ndarray,
              ratio: float = 0.5) -> np.ndarray:
    """
    混合两个音频
    
    Args:
        audio1: 第一个音频
        audio2: 第二个音频
        ratio: 混合比例 (0-1, 0=全部audio1, 1=全部audio2)
        
    Returns:
        混合后的音频
    """
    # 对齐长度
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # 混合
    mixed = audio1 * (1 - ratio) + audio2 * ratio
    
    # 归一化防止裁剪
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


def fade_in_out(audio: np.ndarray, sample_rate: int,
                fade_in_duration: float = 0.1,
                fade_out_duration: float = 0.1) -> np.ndarray:
    """
    添加淡入淡出效果
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        fade_in_duration: 淡入时长（秒）
        fade_out_duration: 淡出时长（秒）
        
    Returns:
        处理后的音频
    """
    audio = audio.copy()
    
    # 淡入
    fade_in_samples = int(fade_in_duration * sample_rate)
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in
    
    # 淡出
    fade_out_samples = int(fade_out_duration * sample_rate)
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out
    
    return audio


def get_supported_formats() -> List[str]:
    """获取支持的音频格式"""
    return ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']


def is_audio_file(file_path: str) -> bool:
    """检查是否为音频文件"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in get_supported_formats()


def batch_process_audio(file_paths: List[str],
                        process_func,
                        output_dir: str = None,
                        **kwargs) -> List[str]:
    """
    批量处理音频文件
    
    Args:
        file_paths: 音频文件路径列表
        process_func: 处理函数
        output_dir: 输出目录
        **kwargs: 传递给处理函数的参数
        
    Returns:
        处理后的文件路径列表
    """
    import librosa
    import soundfile as sf
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_paths = []
    
    for file_path in file_paths:
        try:
            # 加载音频
            y, sr = librosa.load(file_path, sr=None)
            
            # 处理
            processed = process_func(y, sr, **kwargs)
            
            # 保存
            if output_dir:
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, filename)
            else:
                base, ext = os.path.splitext(file_path)
                output_path = f"{base}_processed{ext}"
            
            sf.write(output_path, processed, sr)
            output_paths.append(output_path)
            
        except Exception as e:
            print(f"处理失败 {file_path}: {e}")
    
    return output_paths

