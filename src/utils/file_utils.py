"""
File Utility Functions
"""

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> str:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ('md5', 'sha256')
        
    Returns:
        哈希值字符串
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
    """
    stat = os.stat(file_path)
    
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'extension': os.path.splitext(file_path)[1],
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'is_file': os.path.isfile(file_path),
        'is_dir': os.path.isdir(file_path)
    }


def find_files(directory: str, 
               extensions: List[str] = None,
               recursive: bool = True) -> List[str]:
    """
    查找文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表
        recursive: 是否递归查找
        
    Returns:
        文件路径列表
    """
    files = []
    
    if extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in extensions]
    
    if recursive:
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if extensions is None or os.path.splitext(filename)[1].lower() in extensions:
                    files.append(file_path)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if extensions is None or os.path.splitext(filename)[1].lower() in extensions:
                    files.append(file_path)
    
    return sorted(files)


def copy_files(src_files: List[str], dest_dir: str,
               preserve_structure: bool = False,
               base_dir: str = None) -> List[str]:
    """
    复制文件
    
    Args:
        src_files: 源文件列表
        dest_dir: 目标目录
        preserve_structure: 是否保持目录结构
        base_dir: 基础目录（用于保持结构）
        
    Returns:
        目标文件路径列表
    """
    ensure_dir(dest_dir)
    dest_files = []
    
    for src_file in src_files:
        if preserve_structure and base_dir:
            rel_path = os.path.relpath(src_file, base_dir)
            dest_file = os.path.join(dest_dir, rel_path)
            ensure_dir(os.path.dirname(dest_file))
        else:
            dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        
        shutil.copy2(src_file, dest_file)
        dest_files.append(dest_file)
    
    return dest_files


def move_files(src_files: List[str], dest_dir: str) -> List[str]:
    """
    移动文件
    
    Args:
        src_files: 源文件列表
        dest_dir: 目标目录
        
    Returns:
        目标文件路径列表
    """
    ensure_dir(dest_dir)
    dest_files = []
    
    for src_file in src_files:
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        shutil.move(src_file, dest_file)
        dest_files.append(dest_file)
    
    return dest_files


def delete_files(file_paths: List[str], to_trash: bool = True) -> int:
    """
    删除文件
    
    Args:
        file_paths: 文件路径列表
        to_trash: 是否移动到回收站
        
    Returns:
        删除的文件数量
    """
    deleted = 0
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                if to_trash:
                    try:
                        from send2trash import send2trash
                        send2trash(file_path)
                    except ImportError:
                        os.remove(file_path)
                else:
                    os.remove(file_path)
                deleted += 1
        except Exception as e:
            print(f"删除失败 {file_path}: {e}")
    
    return deleted


def get_unique_filename(directory: str, filename: str) -> str:
    """
    获取唯一文件名
    
    Args:
        directory: 目录路径
        filename: 原始文件名
        
    Returns:
        唯一文件名
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    return new_filename


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_directory_size(directory: str) -> int:
    """
    获取目录大小
    
    Args:
        directory: 目录路径
        
    Returns:
        总大小（字节）
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def create_temp_dir(prefix: str = 'audio_training_') -> str:
    """
    创建临时目录
    
    Args:
        prefix: 目录前缀
        
    Returns:
        临时目录路径
    """
    import tempfile
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """
    清理临时文件
    
    Args:
        directory: 目录路径
        max_age_hours: 最大保留时间（小时）
    """
    now = datetime.now()
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_hours = (now - mtime).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    os.remove(file_path)
            except Exception:
                pass

