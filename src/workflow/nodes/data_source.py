# -*- coding: utf-8 -*-
"""
Data Source Nodes

Provides audio file, label file, and other data loading nodes.
"""

import csv
import json
import logging
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from src.ui.i18n import tr_
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
            raise ValueError(
                tr_("Channel index {idx} out of range (total {channels} channels)").format(
                    idx=channel_idx,
                    channels=self.channels,
                )
            )
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
    display_name = tr_("Audio folder")
    category = NodeCategory.DATA_SOURCE
    palette_order = 10
    description = tr_("Load audio files from a folder, optionally auto-label by subfolder")
    icon = "📂"
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    SELECTION_FIRST_N = "first_n"
    SELECTION_RANDOM_N = "random_n"
    
    def _setup_ports(self):
        self.add_output("audio", DataType.AUDIO, tr_("Audio list"))
        self.add_output("labels", DataType.LABEL, tr_("Label list"))
        self.add_output("file_paths", DataType.ANY, tr_("File path list"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "folder_path", "folder", "",
            display_name=tr_("Folder path"),
            description=tr_("Directory containing audio files"),
            default_directory="audio_data"
        )
        self.add_parameter(
            "recursive", "bool", True,
            display_name=tr_("Recursive scan"),
            description=tr_("Scan subfolders")
        )
        self.add_parameter(
            "auto_label", "bool", True,
            display_name=tr_("Auto label"),
            description=tr_("Auto-generate labels from subfolder names")
        )
        self.add_parameter(
            "target_sr", "int", 22050,
            display_name=tr_("Target sample rate"),
            description=tr_("Resample to this sample rate when loading"),
            min_value=8000, max_value=48000
        )
        self.add_parameter(
            "max_files", "int", 0,
            display_name=tr_("Max files"),
            description=tr_("Maximum number of files to load"),
            min_value=0
        )
        self.add_parameter(
            "selection_mode", "choice", self.SELECTION_FIRST_N,
            display_name=tr_("Selection mode"),
            description=tr_("Choose files by order or random sampling when max files is greater than 0"),
            choices=[self.SELECTION_FIRST_N, self.SELECTION_RANDOM_N]
        )
    
    def execute(self) -> bool:
        folder_path = self.get_parameter("folder_path")
        if not folder_path or not os.path.isdir(folder_path):
            self.error_message = tr_("Directory does not exist: {path}").format(
                path=folder_path
            )
            return False
        
        recursive = self.get_parameter("recursive")
        auto_label = self.get_parameter("auto_label")
        target_sr = self.get_parameter("target_sr")
        max_files = self.get_parameter("max_files")
        selection_mode = self.get_parameter("selection_mode")
        
        audio_list = []
        labels = []
        file_paths = []
        label_map = {}  # 子目录名 -> 标签索引
        
        # 扫描文件
        root_path = Path(folder_path)
        if recursive:
            files = sorted(root_path.rglob("*"))
        else:
            files = sorted(root_path.glob("*"))
        
        # 过滤音频文件
        audio_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        # 限制数量
        if max_files > 0:
            if selection_mode == self.SELECTION_RANDOM_N and max_files < len(audio_files):
                audio_files = random.sample(audio_files, max_files)
            else:
                audio_files = audio_files[:max_files]
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        self.report_status(tr_("Loading {count} audio files...").format(count=len(audio_files)))
        
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
            self.error_message = tr_("No valid audio files found")
            return False
        
        # 设置输出
        self.set_output_data("audio", audio_list)
        self.set_output_data("labels", labels)
        self.set_output_data("file_paths", file_paths)
        
        logger.info(f"加载完成: {len(audio_list)} 个音频, {len(label_map)} 个类别")
        self.report_status(
            tr_("Audio loaded: {files} files, {classes} classes").format(
                files=len(audio_list),
                classes=len(label_map),
            )
        )
        return True


@register_node
class SQLiteAudioDatabaseNode(BaseNode):
    """Load schema-aware audio records from a read-only SQLite database."""

    node_type = "sqlite_audio_database"
    display_name = tr_("SQLite audio database")
    category = NodeCategory.DATA_SOURCE
    palette_order = 20
    description = tr_("Load filtered audio records from an audio SQLite database")
    icon = "🗄️"

    TABLE_NAME = "audio_data_table"
    COLUMNS = (
        "audio_data_id",
        "file_path",
        "product_model",
        "sample_rate",
        "record_date",
        "labels",
        "barcode",
        "stimulus_id",
    )

    def _setup_ports(self):
        self.add_output("audio", DataType.AUDIO, tr_("Audio list"))
        self.add_output("labels", DataType.LABEL, tr_("Label list"))
        self.add_output("file_paths", DataType.ANY, tr_("File path list"))
        self.add_output("label_map", DataType.ANY, tr_("Label map"))
        self.add_output("metadata", DataType.ANY, tr_("Audio metadata"))

    def _setup_parameters(self):
        self.add_parameter(
            "database_path",
            "file",
            "",
            display_name=tr_("Database path"),
            description=tr_("SQLite database containing audio_data_table"),
            file_filter="SQLite Databases (*.db *.sqlite *.sqlite3)",
        )
        self.add_parameter(
            "audio_root",
            "folder",
            "",
            display_name=tr_("Audio root directory"),
            description=tr_(
                "Optional root for relative audio paths; defaults to the parent of the database directory"
            ),
        )
        self.add_parameter(
            "sample_rate",
            "int",
            44100,
            display_name=tr_("Sample rate filter"),
            description=tr_("Load records whose database sample rate exactly matches this value"),
            min_value=1,
        )
        self.add_parameter(
            "labels",
            "str",
            "",
            display_name=tr_("Labels"),
            description=tr_(
                "Optional comma-separated labels; leave empty to map all non-empty database labels"
            ),
        )
        self.add_parameter(
            "file_path_regex",
            "str",
            "",
            display_name=tr_("File path regular expression"),
            description=tr_("Optional case-sensitive regular expression matched against the stored file path"),
        )
        self.add_parameter(
            "max_files",
            "int",
            0,
            display_name=tr_("Max files"),
            description=tr_("Maximum number of files to load after sorting; 0 means unlimited"),
            min_value=0,
        )

    def execute(self) -> bool:
        database_path = str(self.get_parameter("database_path") or "").strip()
        audio_root = str(self.get_parameter("audio_root") or "").strip()
        sample_rate = self.get_parameter("sample_rate")
        labels_text = str(self.get_parameter("labels") or "")
        regex_text = str(self.get_parameter("file_path_regex") or "")
        max_files = self.get_parameter("max_files")

        db_path = Path(database_path)
        if not database_path or not db_path.is_file():
            self.error_message = tr_("Database file does not exist: {path}").format(
                path=database_path
            )
            return False
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            self.error_message = tr_("Sample rate filter must be a positive integer")
            return False
        if not isinstance(max_files, int) or isinstance(max_files, bool) or max_files < 0:
            self.error_message = tr_("Max files must be a non-negative integer")
            return False

        if audio_root:
            root_path = Path(audio_root)
            if not root_path.is_dir():
                self.error_message = tr_("Audio root directory does not exist: {path}").format(
                    path=audio_root
                )
                return False
            root_path = root_path.resolve()
        else:
            root_path = db_path.resolve().parent.parent

        try:
            path_pattern = re.compile(regex_text) if regex_text else None
        except re.error as exc:
            self.error_message = tr_("Invalid file path regular expression: {error}").format(
                error=str(exc)
            )
            return False

        requested_labels = self._parse_labels(labels_text)
        try:
            rows = self._query_rows(db_path, sample_rate, requested_labels)
        except (sqlite3.Error, OSError) as exc:
            self.error_message = tr_("Failed to read SQLite audio database: {error}").format(
                error=str(exc)
            )
            return False
        except ValueError as exc:
            self.error_message = str(exc)
            return False

        queried_count = len(rows)
        label_names = self._build_label_names(rows, requested_labels)
        if path_pattern is not None:
            rows = [row for row in rows if path_pattern.search(str(row["file_path"]))]
        rows.sort(key=lambda row: str(row["file_path"]))
        regex_count = len(rows)
        if max_files > 0:
            rows = rows[:max_files]

        if not rows:
            self.error_message = tr_("No database audio records matched the filters")
            return False

        self.report_status(
            tr_("Loading {count} database audio files...").format(count=len(rows))
        )

        audio_list = []
        encoded_labels = []
        file_paths = []
        metadata = []
        filename_map = {}
        missing_count = 0
        failed_count = 0
        mismatch_count = 0

        for row in rows:
            resolved_path = self._resolve_audio_path(str(row["file_path"]), root_path)
            if not resolved_path.is_file():
                missing_count += 1
                logger.warning("Database audio path does not exist: %s", resolved_path)
                continue

            try:
                audio_data, actual_sample_rate = librosa.load(
                    str(resolved_path),
                    sr=None,
                    mono=False,
                )
            except Exception as exc:
                failed_count += 1
                logger.warning("Failed to load database audio %s: %s", resolved_path, exc)
                continue

            if int(actual_sample_rate) != int(row["sample_rate"]):
                mismatch_count += 1
                logger.warning(
                    "Database sample rate mismatch for %s: stored=%s actual=%s",
                    resolved_path,
                    row["sample_rate"],
                    actual_sample_rate,
                )

            label_name = str(row["labels"])
            label_id = label_names[label_name]
            absolute_path = str(resolved_path)
            audio_list.append(
                AudioData(
                    data=audio_data,
                    sample_rate=int(actual_sample_rate),
                    file_path=absolute_path,
                )
            )
            encoded_labels.append(label_id)
            file_paths.append(absolute_path)
            filename_map[absolute_path] = label_id
            metadata.append({column: row[column] for column in self.COLUMNS})

        if not audio_list:
            self.error_message = tr_("No valid database audio files could be loaded")
            return False

        self.set_output_data("audio", audio_list)
        self.set_output_data("labels", encoded_labels)
        self.set_output_data("file_paths", file_paths)
        self.set_output_data(
            "label_map",
            {
                "filename_map": filename_map,
                "label_names": label_names,
            },
        )
        self.set_output_data("metadata", metadata)

        self.report_status(
            tr_(
                "SQLite audio import: queried {queried}, regex matched {matched}, "
                "loaded {loaded}, missing {missing}, failed {failed}, "
                "sample-rate mismatches {mismatches}"
            ).format(
                queried=queried_count,
                matched=regex_count,
                loaded=len(audio_list),
                missing=missing_count,
                failed=failed_count,
                mismatches=mismatch_count,
            )
        )
        return True

    @classmethod
    def _query_rows(cls, db_path: Path, sample_rate: int, labels: List[str]):
        """Return matching rows while keeping the SQLite connection read-only."""
        database_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(database_uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            table_row = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (cls.TABLE_NAME,),
            ).fetchone()
            if table_row is None:
                raise ValueError(
                    tr_("SQLite audio database schema is missing: {items}").format(
                        items=cls.TABLE_NAME
                    )
                )

            present_columns = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({cls.TABLE_NAME})")
            }
            missing_columns = [column for column in cls.COLUMNS if column not in present_columns]
            if missing_columns:
                raise ValueError(
                    tr_("SQLite audio database schema is missing: {items}").format(
                        items=", ".join(missing_columns)
                    )
                )

            selected_columns = ", ".join(cls.COLUMNS)
            query = (
                f"SELECT {selected_columns} FROM {cls.TABLE_NAME} "
                "WHERE sample_rate = ? AND labels IS NOT NULL AND TRIM(labels) <> ''"
            )
            params: List[Any] = [sample_rate]
            if labels:
                placeholders = ", ".join("?" for _ in labels)
                query += f" AND labels IN ({placeholders})"
                params.extend(labels)
            return connection.execute(query, params).fetchall()

    @staticmethod
    def _parse_labels(labels_text: str) -> List[str]:
        """Parse comma-separated labels while preserving the first occurrence."""
        labels = []
        seen = set()
        for part in labels_text.split(","):
            label = part.strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    @staticmethod
    def _build_label_names(rows, requested_labels: List[str]) -> Dict[str, int]:
        """Build either user-ordered or database-derived deterministic label IDs."""
        if requested_labels:
            return {label: index for index, label in enumerate(requested_labels)}
        labels = sorted({str(row["labels"]) for row in rows})
        return {label: index for index, label in enumerate(labels)}

    @staticmethod
    def _resolve_audio_path(stored_path: str, audio_root: Path) -> Path:
        """Resolve relative database paths without rebasing absolute Windows paths."""
        path = Path(stored_path)
        if path.is_absolute() or PureWindowsPath(stored_path).is_absolute():
            return path
        return (audio_root / path).resolve()


@register_node
class AudioFileNode(BaseNode):
    """
    单个音频文件节点
    
    加载单个音频文件，用于测试和预览。
    """
    
    node_type = "audio_file"
    display_name = tr_("Audio file")
    category = NodeCategory.DATA_SOURCE
    palette_order = 30
    description = tr_("Load a single audio file")
    icon = "🎵"
    
    def _setup_ports(self):
        self.add_output("audio", DataType.AUDIO, tr_("Audio"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "file_path", "file", "",
            display_name=tr_("File path"),
            description=tr_("Audio file path"),
            file_filter="Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)",
            default_directory="audio_data"
        )
        self.add_parameter(
            "target_sr", "int", 22050,
            display_name=tr_("Target sample rate"),
            min_value=8000, max_value=48000
        )
    
    def execute(self) -> bool:
        file_path = self.get_parameter("file_path")
        target_sr = self.get_parameter("target_sr")
        
        if not file_path or not os.path.isfile(file_path):
            self.error_message = tr_("File does not exist: {path}").format(path=file_path)
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
            self.error_message = tr_("Failed to load audio: {error}").format(error=str(e))
            return False


@register_node
class LabelFileNode(BaseNode):
    """
    标签文件节点
    
    从CSV或JSON文件加载标签数据。
    """
    
    node_type = "label_file"
    display_name = tr_("Label file")
    category = NodeCategory.DATA_SOURCE
    palette_order = 40
    description = tr_("Load labels from CSV/JSON/TXT")
    icon = "📄"
    
    def _setup_ports(self):
        self.add_output("labels", DataType.LABEL, tr_("Label list"))
        self.add_output("label_map", DataType.ANY, tr_("Label map"))
    
    def _setup_parameters(self):
        self.add_parameter(
            "file_path", "file", "",
            display_name=tr_("File path"),
            file_filter="Label Files (*.csv *.json *.txt)",
            default_directory="audio_data"
        )
        self.add_parameter(
            "format", "choice", "auto",
            display_name=tr_("File format"),
            choices=["auto", "csv", "json", "txt"]
        )
        self.add_parameter(
            "filename_column", "str", "filename",
            display_name=tr_("Filename column"),
            description=tr_("Column name for filename in CSV")
        )
        self.add_parameter(
            "label_column", "str", "label",
            display_name=tr_("Label column"),
            description=tr_("Column name for label in CSV")
        )
    
    def execute(self) -> bool:
        file_path = self.get_parameter("file_path")
        file_format = self.get_parameter("format")
        
        if not file_path or not os.path.isfile(file_path):
            self.error_message = tr_("File does not exist: {path}").format(path=file_path)
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
            self.error_message = tr_("Failed to load labels: {error}").format(error=str(e))
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
            raise ValueError(tr_("Unsupported JSON format"))
        
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
class TargetFileNode(BaseNode):
    """
    Generic supervised target file node.

    Loads target values without assuming classification labels. Business target
    semantics are intentionally left to downstream workflow nodes.
    """

    node_type = "target_file"
    display_name = tr_("Target file")
    category = NodeCategory.DATA_SOURCE
    palette_order = 50
    description = tr_("Load generic supervised targets from CSV/JSON/TXT")
    icon = "🎯"

    TARGET_KINDS = ["auto", "continuous", "categorical", "array", "sequence"]
    DTYPES = ["auto", "float32", "int64", "str"]
    OUTPUT_FORMATS = ["list", "numpy", "map"]

    def _setup_ports(self):
        self.add_output("targets", DataType.ANY, tr_("Target list"))
        self.add_output("target_map", DataType.ANY, tr_("Target map"))
        self.add_output("target_metadata", DataType.ANY, tr_("Target metadata"))

    def _setup_parameters(self):
        self.add_parameter(
            "file_path", "file", "",
            display_name=tr_("File path"),
            file_filter="Target Files (*.csv *.json *.txt)",
            default_directory="audio_data",
        )
        self.add_parameter(
            "format", "choice", "auto",
            display_name=tr_("File format"),
            choices=["auto", "csv", "json", "txt"],
        )
        self.add_parameter(
            "filename_column", "str", "filename",
            display_name=tr_("Filename column"),
            description=tr_("Column name for filename in CSV"),
        )
        self.add_parameter(
            "target_columns", "str", "target",
            display_name=tr_("Target columns"),
            description=tr_("Comma-separated target column names in CSV"),
        )
        self.add_parameter(
            "target_kind", "choice", "auto",
            display_name=tr_("Target kind"),
            choices=self.TARGET_KINDS,
            description=tr_("Machine-learning target shape, not business type"),
        )
        self.add_parameter(
            "dtype", "choice", "auto",
            display_name=tr_("Data type"),
            choices=self.DTYPES,
        )
        self.add_parameter(
            "output_format", "choice", "list",
            display_name=tr_("Output format"),
            choices=self.OUTPUT_FORMATS,
        )

    def execute(self) -> bool:
        file_path = self.get_parameter("file_path")
        file_format = self._resolve_format(file_path, self.get_parameter("format"))

        if not file_path or not os.path.isfile(file_path):
            self.error_message = tr_("File does not exist: {path}").format(path=file_path)
            return False

        try:
            if file_format == "csv":
                raw_targets, target_map, columns = self._load_csv(file_path)
            elif file_format == "json":
                raw_targets, target_map, columns = self._load_json(file_path)
            else:
                raw_targets, target_map, columns = self._load_txt(file_path)

            kind = self._resolve_kind(raw_targets)
            targets, target_map, dtype, extra_metadata = self._normalize_targets(
                raw_targets,
                target_map,
                kind,
            )
            metadata = {
                "kind": kind,
                "shape": self._target_shape(targets),
                "columns": columns,
                "dtype": dtype,
                "source": str(file_path),
                "count": len(targets),
            }
            metadata.update(extra_metadata)

            self.set_output_data("targets", self._format_targets(targets, target_map, dtype))
            self.set_output_data("target_map", target_map)
            self.set_output_data("target_metadata", metadata)

            logger.info(f"Loaded targets: {len(targets)} items, kind={kind}, dtype={dtype}")
            return True
        except Exception as e:
            self.error_message = tr_("Failed to load targets: {error}").format(error=str(e))
            return False

    def _resolve_format(self, file_path: str, file_format: str) -> str:
        if file_format != "auto":
            return file_format
        ext = Path(file_path).suffix.lower() if file_path else ""
        if ext == ".csv":
            return "csv"
        if ext == ".json":
            return "json"
        return "txt"

    def _load_csv(self, file_path: str) -> Tuple[List, Dict, List[str]]:
        filename_col = self.get_parameter("filename_column")
        columns = self._parse_target_columns()

        targets = []
        target_map = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = self._row_target(row, columns)
                targets.append(target)

                filename = row.get(filename_col, "")
                if filename:
                    target_map[filename] = target

        return targets, target_map, columns

    def _load_json(self, file_path: str) -> Tuple[List, Dict, List[str]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data, {}, []
        if isinstance(data, dict):
            if "targets" in data:
                targets = data["targets"]
                target_map = data.get("target_map", {})
                if not isinstance(targets, list):
                    raise ValueError(tr_("Structured JSON targets must be a list"))
                if not isinstance(target_map, dict):
                    raise ValueError(tr_("Structured JSON target_map must be an object"))
                columns = data.get("columns", [])
                if not isinstance(columns, list):
                    columns = []
                return targets, target_map, columns
            return list(data.values()), dict(data), []
        raise ValueError(tr_("Unsupported JSON format"))

    def _load_txt(self, file_path: str) -> Tuple[List, Dict, List[str]]:
        targets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                value = line.strip()
                if value:
                    targets.append(value)
        return targets, {}, []

    def _parse_target_columns(self) -> List[str]:
        value = self.get_parameter("target_columns") or "target"
        columns = [item.strip() for item in value.split(",") if item.strip()]
        if not columns:
            raise ValueError(tr_("At least one target column is required"))
        return columns

    def _row_target(self, row: Dict[str, str], columns: List[str]):
        values = []
        for column in columns:
            if column not in row:
                raise ValueError(tr_("Missing target column: {column}").format(column=column))
            values.append(row[column])
        return values[0] if len(values) == 1 else values

    def _resolve_kind(self, raw_targets: List) -> str:
        kind = self.get_parameter("target_kind")
        if kind != "auto":
            return kind

        if not raw_targets:
            return "continuous"
        first = raw_targets[0]
        if isinstance(first, (list, tuple, dict)):
            return "array"
        if self._is_numeric(first):
            return "continuous"
        return "categorical"

    def _normalize_targets(
        self,
        raw_targets: List,
        target_map: Dict,
        kind: str,
    ) -> Tuple[List, Dict, str, Dict]:
        if kind == "categorical":
            return self._normalize_categorical(raw_targets, target_map)

        dtype = self._resolve_dtype(kind)
        targets = [self._convert_value(value, dtype) for value in raw_targets]
        normalized_map = {
            key: self._convert_value(value, dtype)
            for key, value in target_map.items()
        }
        return targets, normalized_map, dtype, {}

    def _normalize_categorical(self, raw_targets: List, target_map: Dict) -> Tuple[List, Dict, str, Dict]:
        category_mapping = {}

        def category_id(value):
            key = self._category_key(value)
            if key not in category_mapping:
                category_mapping[key] = len(category_mapping)
            return category_mapping[key]

        targets = [category_id(value) for value in raw_targets]
        normalized_map = {
            key: category_id(value)
            for key, value in target_map.items()
        }
        return targets, normalized_map, "int64", {"category_mapping": category_mapping}

    def _resolve_dtype(self, kind: str) -> str:
        dtype = self.get_parameter("dtype")
        if kind == "continuous" and dtype == "str":
            raise ValueError(tr_("Continuous targets require numeric dtype"))
        if dtype != "auto":
            return dtype
        if kind in ("continuous", "array", "sequence"):
            return "float32"
        return "str"

    def _convert_value(self, value, dtype: str):
        if isinstance(value, list):
            return [self._convert_value(item, dtype) for item in value]
        if isinstance(value, tuple):
            return [self._convert_value(item, dtype) for item in value]
        if dtype == "float32":
            return float(value)
        if dtype == "int64":
            return int(value)
        if dtype == "str":
            return str(value)
        return value

    def _format_targets(self, targets: List, target_map: Dict, dtype: str):
        output_format = self.get_parameter("output_format")
        if output_format == "numpy":
            np_dtype = np.float32 if dtype == "float32" else np.int64 if dtype == "int64" else None
            return np.array(targets, dtype=np_dtype)
        if output_format == "map":
            return target_map
        return targets

    def _target_shape(self, targets: List) -> List[int]:
        if not targets:
            return [0]
        first = targets[0]
        if isinstance(first, list):
            return [len(targets), len(first)]
        return [len(targets)]

    def _is_numeric(self, value) -> bool:
        if isinstance(value, list):
            return all(self._is_numeric(item) for item in value)
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _category_key(self, value) -> str:
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        return str(value)


@register_node
class SaveAudioNode(BaseNode):
    """
    保存音频节点
    
    将音频数据保存为WAV文件，可选保存对应的标签JSON文件。
    标签JSON文件格式与LabelFileNode兼容，可用于后续加载。
    """
    
    node_type = "save_audio"
    display_name = tr_("Export audio")
    category = NodeCategory.OUTPUT
    subcategory = tr_("Data export")
    subcategory_order = 10
    palette_order = 10
    description = tr_("Save audio as WAV files, optionally save labels JSON")
    icon = "💾"
    
    def _setup_ports(self):
        self.add_input(
            "audio",
            DataType.AUDIO,
            tr_("Audio"),
            required=True,
            description=tr_("Audio data to save (single item or list)"),
        )
        self.add_input(
            "labels",
            DataType.LABEL,
            tr_("Labels"),
            required=False,
            description=tr_("Optional labels corresponding to audio items"),
        )
    
    def _setup_parameters(self):
        self.add_parameter(
            "output_folder", "folder", "",
            display_name=tr_("Output folder"),
            description=tr_("Target folder to save audio files"),
            default_directory="audio_data"
        )
        self.add_parameter(
            "filename_prefix", "str", "audio",
            display_name=tr_("Filename prefix"),
            description=tr_("Prefix for output audio filenames")
        )
        self.add_parameter(
            "save_labels", "bool", True,
            display_name=tr_("Save labels file"),
            description=tr_("When labels are provided, save labels.json")
        )
        self.add_parameter(
            "overwrite", "bool", False,
            display_name=tr_("Overwrite existing files"),
            description=tr_("Overwrite files if they already exist")
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
            self.error_message = tr_("Please select an output folder")
            return False
        
        output_path = Path(output_folder)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.error_message = tr_("Unable to create output folder: {error}").format(
                error=str(e)
            )
            return False
        
        # 处理音频输入 - 统一为列表
        if audio_input is None:
            self.error_message = tr_("No input audio data")
            return False
        
        if isinstance(audio_input, AudioData):
            audio_list = [audio_input]
        elif isinstance(audio_input, list):
            audio_list = audio_input
        else:
            self.error_message = tr_("Unsupported audio data type: {type}").format(
                type=str(type(audio_input))
            )
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
                    tr_(
                        "Label count ({labels}) does not match audio count ({audio}); "
                        "items will be paired in order and extras will be ignored"
                    ).format(labels=len(labels_list), audio=len(audio_list))
                )
        
        self.report_status(tr_("Saving {count} audio files...").format(count=len(audio_list)))
        
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
            self.error_message = tr_("No audio files were saved successfully")
            return False
        
        logger.info(f"保存完成: {len(saved_files)} 个音频文件")
        msg = tr_("Audio saved: {files} files").format(files=len(saved_files))
        if saved_labels:
            msg += tr_(", labels: {count}").format(count=len(saved_labels))
        self.report_status(msg)
        return True
