"""
Dataset Manager
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.ui.i18n import tr_
@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    path: str
    total_samples: int
    num_classes: int
    class_names: List[str]
    class_counts: Dict[str, int]
    sample_rate: int
    duration_stats: Dict[str, float]  # min, max, mean, total
    created_at: str


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, dataset_dir: str = None):
        """
        初始化数据集管理器
        
        Args:
            dataset_dir: 数据集根目录
        """
        self.dataset_dir = dataset_dir
        self.audio_files: List[str] = []
        self.labels: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
        # 划分后的数据
        self.train_files: List[str] = []
        self.train_labels: List[int] = []
        self.val_files: List[str] = []
        self.val_labels: List[int] = []
        self.test_files: List[str] = []
        self.test_labels: List[int] = []
    
    def load_from_directory(self, directory: str, 
                            class_mode: str = 'subdirectory') -> DatasetInfo:
        """
        从目录加载数据集
        
        Args:
            directory: 数据集目录
            class_mode: 分类模式
                - 'subdirectory': 每个子目录代表一个类别
                - 'filename': 从文件名解析类别
                
        Returns:
            数据集信息
        """
        self.dataset_dir = directory
        self.audio_files = []
        self.labels = []
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        if class_mode == 'subdirectory':
            # 每个子目录是一个类别
            class_names = sorted([d for d in os.listdir(directory) 
                                  if os.path.isdir(os.path.join(directory, d))])
            
            for class_name in class_names:
                class_dir = os.path.join(directory, class_name)
                for filename in os.listdir(class_dir):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in audio_extensions:
                        self.audio_files.append(os.path.join(class_dir, filename))
                        self.labels.append(class_name)
        
        elif class_mode == 'filename':
            # 从文件名解析类别 (假设格式: class_name_xxx.wav)
            for filename in os.listdir(directory):
                ext = os.path.splitext(filename)[1].lower()
                if ext in audio_extensions:
                    # 假设类别是文件名的第一部分
                    class_name = filename.split('_')[0]
                    self.audio_files.append(os.path.join(directory, filename))
                    self.labels.append(class_name)
        
        # 构建标签映射
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 统计信息
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # 计算时长统计
        duration_stats = self._compute_duration_stats()
        
        from datetime import datetime
        
        return DatasetInfo(
            name=os.path.basename(directory),
            path=directory,
            total_samples=len(self.audio_files),
            num_classes=len(unique_labels),
            class_names=unique_labels,
            class_counts=class_counts,
            sample_rate=44100,  # 默认值，可以从文件读取
            duration_stats=duration_stats,
            created_at=datetime.now().isoformat()
        )
    
    def _compute_duration_stats(self) -> Dict[str, float]:
        """计算时长统计"""
        try:
            import librosa
            
            durations = []
            for file_path in self.audio_files[:100]:  # 抽样计算
                try:
                    duration = librosa.get_duration(path=file_path)
                    durations.append(duration)
                except:
                    pass
            
            if durations:
                return {
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                    'mean': float(np.mean(durations)),
                    'total': float(np.sum(durations) * len(self.audio_files) / len(durations))
                }
        except:
            pass
        
        return {'min': 0, 'max': 0, 'mean': 0, 'total': 0}
    
    def split_dataset(self, train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      stratify: bool = True,
                      shuffle: bool = True,
                      random_seed: int = 42) -> Dict[str, int]:
        """
        划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify: 是否分层抽样
            shuffle: 是否打乱
            random_seed: 随机种子
            
        Returns:
            各集合的样本数量
        """
        if not self.audio_files:
            raise ValueError(tr_("No dataset loaded"))
        
        # 转换标签为索引
        label_indices = [self.label_to_idx[label] for label in self.labels]
        
        # 第一次划分：训练集 + (验证集+测试集)
        stratify_labels = label_indices if stratify else None
        
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            self.audio_files, label_indices,
            train_size=train_ratio,
            stratify=stratify_labels,
            shuffle=shuffle,
            random_state=random_seed
        )
        
        # 第二次划分：验证集 + 测试集
        val_size = val_ratio / (val_ratio + test_ratio)
        stratify_temp = temp_labels if stratify else None
        
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels,
            train_size=val_size,
            stratify=stratify_temp,
            shuffle=shuffle,
            random_state=random_seed
        )
        
        # 保存划分结果
        self.train_files = train_files
        self.train_labels = train_labels
        self.val_files = val_files
        self.val_labels = val_labels
        self.test_files = test_files
        self.test_labels = test_labels
        
        return {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
    
    def get_train_data(self) -> Tuple[List[str], List[int]]:
        """获取训练数据"""
        return self.train_files, self.train_labels
    
    def get_val_data(self) -> Tuple[List[str], List[int]]:
        """获取验证数据"""
        return self.val_files, self.val_labels
    
    def get_test_data(self) -> Tuple[List[str], List[int]]:
        """获取测试数据"""
        return self.test_files, self.test_labels
    
    def save_split(self, output_path: str):
        """
        保存数据划分结果
        
        Args:
            output_path: 输出JSON文件路径
        """
        split_data = {
            'dataset_dir': self.dataset_dir,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': {str(k): v for k, v in self.idx_to_label.items()},
            'train': {
                'files': self.train_files,
                'labels': self.train_labels
            },
            'val': {
                'files': self.val_files,
                'labels': self.val_labels
            },
            'test': {
                'files': self.test_files,
                'labels': self.test_labels
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
    
    def load_split(self, split_path: str):
        """
        加载数据划分结果
        
        Args:
            split_path: JSON文件路径
        """
        with open(split_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        self.dataset_dir = split_data['dataset_dir']
        self.label_to_idx = split_data['label_to_idx']
        self.idx_to_label = {int(k): v for k, v in split_data['idx_to_label'].items()}
        
        self.train_files = split_data['train']['files']
        self.train_labels = split_data['train']['labels']
        self.val_files = split_data['val']['files']
        self.val_labels = split_data['val']['labels']
        self.test_files = split_data['test']['files']
        self.test_labels = split_data['test']['labels']
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        计算类别权重（用于处理不平衡数据）
        
        Returns:
            类别权重字典
        """
        if not self.train_labels:
            return {}
        
        from collections import Counter
        
        label_counts = Counter(self.train_labels)
        total_samples = len(self.train_labels)
        n_classes = len(label_counts)
        
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total_samples / (n_classes * count)
        
        return weights
    
    @property
    def num_classes(self) -> int:
        """类别数量"""
        return len(self.label_to_idx)
    
    @property
    def class_names(self) -> List[str]:
        """类别名称列表"""
        return list(self.label_to_idx.keys())

