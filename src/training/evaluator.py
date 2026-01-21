"""
Model Evaluation Module
Provides model evaluation, confusion matrix, classification report, and other functions
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


@dataclass
class EvaluationResult:
    """评估结果"""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    class_report: Dict[str, Dict[str, float]]
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: np.ndarray


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: keras.Model, class_names: List[str] = None):
        """
        初始化评估器
        
        Args:
            model: Keras模型
            class_names: 类别名称列表
        """
        self.model = model
        self.class_names = class_names or []
        self.num_classes = len(class_names) if class_names else None
    
    def evaluate(self, test_data, verbose: bool = True) -> EvaluationResult:
        """
        评估模型
        
        Args:
            test_data: 测试数据（生成器或元组）
            verbose: 是否打印结果
            
        Returns:
            EvaluationResult对象
        """
        from sklearn.metrics import (
            confusion_matrix, classification_report,
            precision_score, recall_score, f1_score
        )
        
        # 获取预测和真实标签
        if hasattr(test_data, '__getitem__'):
            # 数据生成器
            y_true = []
            y_pred_proba = []
            
            for i in range(len(test_data)):
                x_batch, y_batch = test_data[i]
                pred_batch = self.model.predict(x_batch, verbose=0)
                
                y_true.extend(np.argmax(y_batch, axis=1))
                y_pred_proba.extend(pred_batch)
            
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba)
        else:
            # 数组数据
            x_test, y_test = test_data
            y_pred_proba = self.model.predict(x_test, verbose=0)
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        
        # 获取预测标签
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算损失和准确率
        if hasattr(test_data, '__getitem__'):
            loss, accuracy = self.model.evaluate(test_data, verbose=0)
        else:
            x_test, y_test = test_data
            loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算精确率、召回率、F1
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 分类报告
        target_names = self.class_names if self.class_names else None
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        result = EvaluationResult(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            class_report=report,
            predictions=y_pred,
            true_labels=y_true,
            probabilities=y_pred_proba
        )
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: EvaluationResult):
        """打印评估结果"""
        print("\n" + "=" * 50)
        print("模型评估结果")
        print("=" * 50)
        print(f"Loss: {result.loss:.4f}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"Precision: {result.precision:.4f}")
        print(f"Recall: {result.recall:.4f}")
        print(f"F1 Score: {result.f1_score:.4f}")
        print("\n混淆矩阵:")
        print(result.confusion_matrix)
        print("=" * 50)
    
    def predict_single(self, audio_features: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        预测单个样本
        
        Args:
            audio_features: 音频特征
            
        Returns:
            (预测类别, 置信度, 所有类别概率)
        """
        # 添加批次维度
        if audio_features.ndim == 2:
            audio_features = audio_features[np.newaxis, ...]
        elif audio_features.ndim == 3:
            audio_features = audio_features[np.newaxis, ...]
        
        # 预测
        proba = self.model.predict(audio_features, verbose=0)[0]
        predicted_class = np.argmax(proba)
        confidence = proba[predicted_class]
        
        return predicted_class, confidence, proba
    
    def predict_with_label(self, audio_features: np.ndarray) -> Tuple[str, float]:
        """
        预测并返回类别名称
        
        Args:
            audio_features: 音频特征
            
        Returns:
            (类别名称, 置信度)
        """
        predicted_class, confidence, _ = self.predict_single(audio_features)
        
        if self.class_names and predicted_class < len(self.class_names):
            class_name = self.class_names[predicted_class]
        else:
            class_name = f"Class_{predicted_class}"
        
        return class_name, confidence
    
    def get_top_k_predictions(self, audio_features: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        获取Top-K预测结果
        
        Args:
            audio_features: 音频特征
            k: 返回的预测数量
            
        Returns:
            [(类别名称, 概率), ...]
        """
        _, _, proba = self.predict_single(audio_features)
        
        # 获取Top-K索引
        top_k_indices = np.argsort(proba)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            if self.class_names and idx < len(self.class_names):
                class_name = self.class_names[idx]
            else:
                class_name = f"Class_{idx}"
            results.append((class_name, float(proba[idx])))
        
        return results
    
    def get_roc_curve(self, test_data) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算ROC曲线（二分类）
        
        Args:
            test_data: 测试数据
            
        Returns:
            (fpr, tpr, auc_score)
        """
        from sklearn.metrics import roc_curve, auc
        
        # 获取预测概率
        if hasattr(test_data, '__getitem__'):
            y_true = []
            y_score = []
            
            for i in range(len(test_data)):
                x_batch, y_batch = test_data[i]
                pred_batch = self.model.predict(x_batch, verbose=0)
                
                y_true.extend(np.argmax(y_batch, axis=1))
                y_score.extend(pred_batch[:, 1] if pred_batch.shape[1] == 2 else pred_batch[:, 0])
            
            y_true = np.array(y_true)
            y_score = np.array(y_score)
        else:
            x_test, y_test = test_data
            y_score = self.model.predict(x_test, verbose=0)
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
            y_score = y_score[:, 1] if y_score.shape[1] == 2 else y_score[:, 0]
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score
    
    def export_results(self, result: EvaluationResult, output_path: str):
        """
        导出评估结果
        
        Args:
            result: 评估结果
            output_path: 输出路径
        """
        import json
        
        # 准备导出数据
        export_data = {
            'metrics': {
                'loss': float(result.loss),
                'accuracy': float(result.accuracy),
                'precision': float(result.precision),
                'recall': float(result.recall),
                'f1_score': float(result.f1_score)
            },
            'confusion_matrix': result.confusion_matrix.tolist(),
            'classification_report': result.class_report,
            'class_names': self.class_names
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已导出到: {output_path}")


def quick_evaluate(model: keras.Model, 
                   test_data,
                   class_names: List[str] = None) -> EvaluationResult:
    """
    快速评估模型
    
    Args:
        model: Keras模型
        test_data: 测试数据
        class_names: 类别名称
        
    Returns:
        EvaluationResult对象
    """
    evaluator = ModelEvaluator(model, class_names)
    return evaluator.evaluate(test_data)

