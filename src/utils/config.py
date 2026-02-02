"""
Configuration Manager
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """配置管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 音频设置
        'audio': {
            'sample_rate': 44100,
            'mono': True,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 128,
            'n_mfcc': 20,
            'normalize': True,
            'remove_silence': False,
            'silence_threshold': 20
        },
        
        # 模型设置
        'model': {
            'default_epochs': 50,
            'default_batch_size': 32,
            'default_learning_rate': 0.001,
            'save_path': '',
            'auto_save_best': True,
            'save_format': 'SavedModel',
            'early_stopping': True,
            'patience': 10,
            'min_delta': 0.0001
        },
        
        # 界面设置
        'ui': {
            'theme': '深色',
            'waveform_color': '蓝色',
            'spectrogram_cmap': 'viridis',
            'show_grid': True,
            'chart_update_interval': 500
        },
        
        # GPU设置
        'gpu': {
            'use_gpu': True,
            'memory_limit': 0,
            'mixed_precision': False
        },
        
        # 最近文件
        'recent': {
            'files': [],
            'projects': [],
            'max_recent': 10
        },

        # 会话/启动恢复
        'session': {
            # 上次打开/保存的工作流文件路径（存在且可读时启动自动加载）
            'last_workflow_path': None,
            # 上次打开/保存的模型定义文件路径（.model.json）
            'last_model_path': None,
            # 模型编辑器快照（用于无可用 last_model_path 时恢复）
            'last_model_graph_snapshot': None
        }
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ConfigManager._initialized:
            return
        
        self.config_dir = Path.home() / '.audio_training_platform'
        self.config_file = self.config_dir / 'config.json'
        self.config: Dict[str, Any] = {}
        
        self._ensure_config_dir()
        self._load_config()
        
        ConfigManager._initialized = True
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
    
    def _load_config(self):
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # 合并默认配置和加载的配置
                self.config = self._merge_config(self.DEFAULT_CONFIG, loaded_config)
            except Exception as e:
                print(f"加载配置失败: {e}")
                self.config = self.DEFAULT_CONFIG.copy()
        else:
            self.config = self.DEFAULT_CONFIG.copy()
            self.save()
    
    def _merge_config(self, default: dict, loaded: dict) -> dict:
        """合并配置，保留默认值中存在但loaded中不存在的键"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self._merge_config(result[key], value)
                else:
                    result[key] = value
        return result
    
    def save(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的路径 (如 'audio.sample_rate')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, save_immediately: bool = True):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔的路径
            value: 配置值
            save_immediately: 是否立即保存
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        if save_immediately:
            self.save()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段"""
        return self.config.get(section, {})
    
    def set_section(self, section: str, values: Dict[str, Any], save_immediately: bool = True):
        """设置配置段"""
        self.config[section] = values
        if save_immediately:
            self.save()
    
    def update(self, settings: Dict[str, Any], save_immediately: bool = True):
        """批量更新配置"""
        for key, value in settings.items():
            self.set(key, value, save_immediately=False)
        
        if save_immediately:
            self.save()
    
    def reset_to_default(self, section: Optional[str] = None):
        """重置为默认配置"""
        if section:
            if section in self.DEFAULT_CONFIG:
                self.config[section] = self.DEFAULT_CONFIG[section].copy()
        else:
            self.config = self.DEFAULT_CONFIG.copy()
        
        self.save()
    
    def add_recent_file(self, file_path: str):
        """添加最近文件"""
        recent_files = self.get('recent.files', [])
        
        # 移除重复项
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # 添加到开头
        recent_files.insert(0, file_path)
        
        # 限制数量
        max_recent = self.get('recent.max_recent', 10)
        recent_files = recent_files[:max_recent]
        
        self.set('recent.files', recent_files)
    
    def get_recent_files(self) -> list:
        """获取最近文件列表"""
        return self.get('recent.files', [])
    
    def clear_recent_files(self):
        """清除最近文件"""
        self.set('recent.files', [])
    
    @property
    def audio_config(self) -> Dict[str, Any]:
        """获取音频配置"""
        return self.get_section('audio')
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get_section('model')
    
    @property
    def ui_config(self) -> Dict[str, Any]:
        """获取界面配置"""
        return self.get_section('ui')
    
    @property
    def gpu_config(self) -> Dict[str, Any]:
        """获取GPU配置"""
        return self.get_section('gpu')


# 全局配置实例
config = ConfigManager()

