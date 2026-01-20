"""
训练模块
"""

from src.training.callbacks import EarlyStoppingWithUI
from src.training.callbacks import TrainingCallback as UICallback
from src.training.data_generator import AudioDataGenerator, create_data_generators
from src.training.evaluator import EvaluationResult, ModelEvaluator, quick_evaluate
from src.training.trainer import Trainer, TrainerWorker, TrainingCallback, TrainingPipeline

__all__ = [
    # 回调
    "EarlyStoppingWithUI",
    "UICallback",
    
    # 数据生成器
    "AudioDataGenerator",
    "create_data_generators",
    
    # 评估器
    "EvaluationResult",
    "ModelEvaluator",
    "quick_evaluate",
    
    # 训练器
    "Trainer",
    "TrainerWorker",
    "TrainingCallback",
    "TrainingPipeline",
]
