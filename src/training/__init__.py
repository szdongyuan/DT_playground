"""
Training Module
"""

from src.training.callbacks import EarlyStoppingWithUI
from src.training.callbacks import TrainingCallback as UICallback
from src.training.data_generator import AudioDataGenerator, create_data_generators
from src.training.evaluator import EvaluationResult, ModelEvaluator, quick_evaluate
from src.training.trainer import Trainer, TrainerWorker, TrainingCallback, TrainingPipeline

__all__ = [
    # Callbacks
    "EarlyStoppingWithUI",
    "UICallback",
    
    # Data generators
    "AudioDataGenerator",
    "create_data_generators",
    
    # Evaluator
    "EvaluationResult",
    "ModelEvaluator",
    "quick_evaluate",
    
    # Trainer
    "Trainer",
    "TrainerWorker",
    "TrainingCallback",
    "TrainingPipeline",
]
