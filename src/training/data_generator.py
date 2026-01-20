"""
音频数据生成器
用于从音频文件批量生成训练数据
支持组合特征提取
"""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras


class AudioDataGenerator(keras.utils.Sequence):
    """
    音频数据生成器
    继承自Keras Sequence，支持多进程数据加载
    支持多种特征类型及其组合
    """
    
    def __init__(self,
                 file_paths: List[str],
                 labels: List[int],
                 batch_size: int = 32,
                 sample_rate: int = 44100,
                 duration: float = 3.0,
                 feature_type: str = 'mel_spectrogram',
                 feature_types: List[str] = None,
                 feature_combine_mode: str = 'channel',
                 n_mels: int = 128,
                 n_mfcc: int = 20,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 num_classes: int = None,
                 shuffle: bool = True,
                 augment: bool = False,
                 normalize: bool = True):
        """
        初始化数据生成器
        
        Args:
            file_paths: 音频文件路径列表
            labels: 标签列表
            batch_size: 批大小
            sample_rate: 目标采样率
            duration: 音频时长（秒）
            feature_type: 单一特征类型 (兼容旧版API)
            feature_types: 特征类型列表 (新API，支持多特征组合)
            feature_combine_mode: 特征组合模式 ('channel', 'concat', 'flatten')
            n_mels: Mel滤波器数量
            n_mfcc: MFCC系数数量
            n_fft: FFT窗口大小
            hop_length: 帧移
            num_classes: 类别数（用于one-hot编码）
            shuffle: 是否打乱数据
            augment: 是否进行数据增强
            normalize: 是否归一化
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.duration = duration
        
        # 处理特征类型 - 支持新旧两种API
        if feature_types is not None and len(feature_types) > 0:
            self.feature_types = feature_types
        else:
            # 兼容旧版单一特征API
            self.feature_types = [feature_type]
        
        self.feature_combine_mode = feature_combine_mode
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_classes = num_classes or (max(labels) + 1 if labels else 2)
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        
        # 计算目标样本数
        self.target_length = int(sample_rate * duration)
        
        # 最小有效长度（至少需要n_fft长度）
        self.min_valid_length = max(n_fft, 1024)
        
        # 初始化特征提取器
        self._init_feature_extractor()
        
        # 索引
        self.indices = np.arange(len(file_paths))
        
        # 预先验证文件（可选，加速后续加载）
        self.valid_indices = self._validate_files()
        
        if shuffle:
            np.random.shuffle(self.valid_indices)
    
    def _init_feature_extractor(self):
        """初始化特征提取器"""
        from src.audio.features import FeatureExtractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc
        )
    
    def _validate_files(self) -> np.ndarray:
        """验证文件有效性，过滤掉无效文件"""
        valid = []
        for idx in self.indices:
            file_path = self.file_paths[idx]
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                valid.append(idx)
        
        if len(valid) < len(self.indices):
            invalid_count = len(self.indices) - len(valid)
            print(f"警告: 发现 {invalid_count} 个无效文件，已跳过")
        
        return np.array(valid)
    
    def __len__(self) -> int:
        """返回每个epoch的批次数"""
        return int(np.ceil(len(self.valid_indices) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取一个批次的数据"""
        # 获取当前批次的索引
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.valid_indices))
        batch_indices = self.valid_indices[start_idx:end_idx]
        
        # 加载和处理数据
        batch_x = []
        batch_y = []
        
        for idx in batch_indices:
            try:
                # 加载音频
                audio = self._load_audio(self.file_paths[idx])
                
                # 验证音频长度
                if audio is None or len(audio) < self.min_valid_length:
                    continue
                
                # 数据增强
                if self.augment:
                    audio = self._augment_audio(audio)
                
                # 提取特征
                features = self._extract_features(audio)
                
                if features is not None:
                    batch_x.append(features)
                    batch_y.append(self.labels[idx])
                
            except Exception as e:
                # 静默跳过错误文件
                continue
        
        if not batch_x:
            # 如果整个批次都失败，返回一个虚拟样本
            # 获取特征形状
            dummy_shape = self._get_feature_shape()
            return np.zeros((1,) + dummy_shape), np.zeros((1, self.num_classes))
        
        # 转换为numpy数组
        X = np.array(batch_x)
        y = np.array(batch_y)
        
        # One-hot编码
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        return X, y
    
    def on_epoch_end(self):
        """每个epoch结束时调用"""
        if self.shuffle:
            np.random.shuffle(self.valid_indices)
    
    def _load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """加载音频文件"""
        import librosa
        
        try:
            # 抑制librosa警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                audio, _ = librosa.load(
                    file_path,
                    sr=self.sample_rate,
                    duration=self.duration,
                    mono=True
                )
            
            # 检查音频是否有效
            if audio is None or len(audio) == 0:
                return None
            
            # 填充或裁剪到目标长度
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
            
            return audio
            
        except Exception as e:
            return None
    
    def _extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """提取音频特征 - 支持组合特征"""
        # 验证音频长度
        if len(audio) < self.n_fft:
            return None
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 使用特征提取器提取组合特征
                features = self.feature_extractor.extract_combined_features(
                    audio,
                    self.feature_types,
                    normalize=self.normalize,
                    combine_mode=self.feature_combine_mode
                )
                
                return features
                
        except Exception as e:
            return None
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """音频数据增强"""
        try:
            import librosa
            
            # 随机应用增强
            if np.random.random() < 0.3:
                # 时间拉伸
                rate = np.random.uniform(0.9, 1.1)
                audio = librosa.effects.time_stretch(audio, rate=rate)
            
            if np.random.random() < 0.3:
                # 音高偏移
                n_steps = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
            
            if np.random.random() < 0.5:
                # 添加噪声
                noise = np.random.randn(len(audio)) * 0.005
                audio = audio + noise
            
            if np.random.random() < 0.3:
                # 音量变化
                gain = np.random.uniform(0.8, 1.2)
                audio = audio * gain
            
            # 确保长度一致
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
            
            return audio
            
        except Exception:
            # 增强失败时返回原始音频
            return audio
    
    def _get_feature_shape(self) -> Tuple[int, ...]:
        """获取特征形状"""
        return self.feature_extractor.get_feature_shape(
            self.target_length,
            self.feature_types,
            self.feature_combine_mode
        )
    
    def get_feature_shape(self) -> Tuple[int, ...]:
        """获取特征形状（公共方法）"""
        # 尝试从实际数据获取形状
        for idx in self.valid_indices[:10]:
            try:
                audio = self._load_audio(self.file_paths[idx])
                if audio is not None and len(audio) >= self.min_valid_length:
                    features = self._extract_features(audio)
                    if features is not None:
                        return features.shape
            except:
                continue
        
        # 使用计算的形状作为后备
        return self._get_feature_shape()
    
    def get_feature_info(self) -> dict:
        """获取特征配置信息"""
        return {
            'feature_types': self.feature_types,
            'combine_mode': self.feature_combine_mode,
            'shape': self.get_feature_shape(),
            'n_features': len(self.feature_types)
        }


def create_data_generators(file_paths: List[str],
                           labels: List[int],
                           train_ratio: float = 0.8,
                           batch_size: int = 32,
                           **kwargs) -> Tuple[AudioDataGenerator, AudioDataGenerator]:
    """
    创建训练和验证数据生成器
    
    Args:
        file_paths: 音频文件路径
        labels: 标签
        train_ratio: 训练集比例
        batch_size: 批大小
        **kwargs: 其他参数（支持 feature_types, feature_combine_mode 等）
        
    Returns:
        (训练生成器, 验证生成器)
    """
    import logging
    logger = logging.getLogger('AudioTrainingApp')
    
    logger.info(f"create_data_generators: {len(file_paths)} 文件, {len(set(labels))} 类别")
    
    # 处理特征配置
    feature_types = kwargs.get('feature_types', None)
    feature_combine_mode = kwargs.get('feature_combine_mode', 'channel')
    
    if feature_types:
        logger.info(f"使用特征组合: {feature_types}, 模式: {feature_combine_mode}")
    
    # 确保输入是普通Python列表
    file_paths = list(file_paths)
    labels = [int(l) for l in labels]
    
    logger.info("开始数据划分...")
    
    # 手动实现分层划分以避免sklearn签名检查问题
    train_files, val_files, train_labels, val_labels = _stratified_split(
        file_paths, labels, train_ratio=train_ratio, random_seed=42
    )
    
    logger.info(f"划分完成: 训练集 {len(train_files)}, 验证集 {len(val_files)}")
    
    # 创建生成器
    train_gen = AudioDataGenerator(
        train_files, train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=kwargs.get('augment', True),
        feature_types=feature_types,
        feature_combine_mode=feature_combine_mode,
        **{k: v for k, v in kwargs.items() 
           if k not in ['augment', 'feature_types', 'feature_combine_mode']}
    )
    
    val_gen = AudioDataGenerator(
        val_files, val_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        feature_types=feature_types,
        feature_combine_mode=feature_combine_mode,
        **{k: v for k, v in kwargs.items() 
           if k not in ['augment', 'feature_types', 'feature_combine_mode']}
    )
    
    return train_gen, val_gen


def _stratified_split(file_paths: List[str], labels: List[int], 
                      train_ratio: float = 0.8, random_seed: int = 42):
    """
    手动实现分层划分，避免sklearn签名检查问题
    """
    np.random.seed(random_seed)
    
    # 按类别分组
    class_to_indices = {}
    for idx in range(len(labels)):
        label = labels[idx]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # 对每个类别进行划分
    for label in class_to_indices:
        indices = np.array(class_to_indices[label])
        np.random.shuffle(indices)
        
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point].tolist())
        val_indices.extend(indices[split_point:].tolist())
    
    # 打乱顺序
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # 构建结果
    train_files = [file_paths[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_files, val_files, train_labels, val_labels
