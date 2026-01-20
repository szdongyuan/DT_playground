"""
训练指标可视化工具
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MetricsPlotter:
    """训练指标绘制器"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 4), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
    
    def plot_training_history(self, history: Dict[str, List[float]],
                               metrics: List[str] = None) -> Figure:
        """
        绘制训练历史曲线
        
        Args:
            history: 训练历史字典
            metrics: 要绘制的指标列表
            
        Returns:
            matplotlib Figure对象
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        n_metrics = len(metrics)
        self.fig, axes = plt.subplots(1, n_metrics, figsize=self.figsize, dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = {'train': '#89b4fa', 'val': '#f38ba8'}
        
        for ax, metric in zip(axes, metrics):
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], color=colors['train'], 
                       linewidth=2, label=f'训练 {metric}')
            
            val_metric = f'val_{metric}'
            if val_metric in history:
                epochs = range(1, len(history[val_metric]) + 1)
                ax.plot(epochs, history[val_metric], color=colors['val'],
                       linewidth=2, linestyle='--', label=f'验证 {metric}')
            
            ax.set_facecolor('#181825')
            ax.set_title(metric.title(), color='#cdd6f4')
            ax.set_xlabel('Epoch', color='#cdd6f4')
            ax.set_ylabel(metric.title(), color='#cdd6f4')
            ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
            ax.grid(True, alpha=0.3, color='#45475a')
            ax.tick_params(colors='#a6adc8')
            
            for spine in ax.spines.values():
                spine.set_color('#45475a')
        
        self.fig.patch.set_facecolor('#1e1e2e')
        plt.tight_layout()
        
        return self.fig
    
    def plot_confusion_matrix(self, cm: np.ndarray,
                               labels: List[str] = None,
                               normalize: bool = True) -> Figure:
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            labels: 类别标签
            normalize: 是否归一化
            
        Returns:
            matplotlib Figure对象
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        n_classes = cm.shape[0]
        if labels is None:
            labels = [str(i) for i in range(n_classes)]
        
        # 绘制热力图
        im = self.ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # 颜色条
        cbar = self.fig.colorbar(im, ax=self.ax)
        cbar.ax.yaxis.set_tick_params(color='#cdd6f4')
        cbar.outline.set_edgecolor('#45475a')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a6adc8')
        
        # 标签
        self.ax.set_xticks(np.arange(n_classes))
        self.ax.set_yticks(np.arange(n_classes))
        self.ax.set_xticklabels(labels, color='#cdd6f4')
        self.ax.set_yticklabels(labels, color='#cdd6f4')
        
        # 旋转x轴标签
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在单元格中显示数值
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                value = f'{cm[i, j]:.2%}' if normalize else f'{cm[i, j]}'
                self.ax.text(j, i, value,
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        
        self.ax.set_title('混淆矩阵', color='#cdd6f4')
        self.ax.set_ylabel('真实标签', color='#cdd6f4')
        self.ax.set_xlabel('预测标签', color='#cdd6f4')
        
        self.fig.patch.set_facecolor('#1e1e2e')
        plt.tight_layout()
        
        return self.fig
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray,
                       auc_score: float = None) -> Figure:
        """
        绘制ROC曲线
        
        Args:
            fpr: 假正率
            tpr: 真正率
            auc_score: AUC分数
            
        Returns:
            matplotlib Figure对象
        """
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        
        label = f'ROC曲线 (AUC = {auc_score:.3f})' if auc_score else 'ROC曲线'
        self.ax.plot(fpr, tpr, color='#89b4fa', linewidth=2, label=label)
        self.ax.plot([0, 1], [0, 1], color='#6c7086', linestyle='--', label='随机猜测')
        
        self.ax.set_facecolor('#181825')
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.05])
        self.ax.set_xlabel('假正率 (FPR)', color='#cdd6f4')
        self.ax.set_ylabel('真正率 (TPR)', color='#cdd6f4')
        self.ax.set_title('ROC曲线', color='#cdd6f4')
        self.ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4', loc='lower right')
        self.ax.grid(True, alpha=0.3, color='#45475a')
        self.ax.tick_params(colors='#a6adc8')
        
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')
        
        self.fig.patch.set_facecolor('#1e1e2e')
        plt.tight_layout()
        
        return self.fig
    
    def plot_class_distribution(self, labels: List[str], 
                                 counts: List[int]) -> Figure:
        """
        绘制类别分布柱状图
        
        Args:
            labels: 类别标签
            counts: 各类别数量
            
        Returns:
            matplotlib Figure对象
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        
        bars = self.ax.bar(labels, counts, color=colors, edgecolor='#45475a')
        
        # 在柱子上显示数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', color='#cdd6f4')
        
        self.ax.set_facecolor('#181825')
        self.ax.set_xlabel('类别', color='#cdd6f4')
        self.ax.set_ylabel('样本数', color='#cdd6f4')
        self.ax.set_title('类别分布', color='#cdd6f4')
        self.ax.tick_params(colors='#a6adc8')
        
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')
        
        self.fig.patch.set_facecolor('#1e1e2e')
        plt.tight_layout()
        
        return self.fig
    
    def get_canvas(self) -> Optional[FigureCanvasQTAgg]:
        """获取Qt画布"""
        if self.fig is not None:
            return FigureCanvasQTAgg(self.fig)
        return None
    
    def save(self, path: str, dpi: int = 150):
        """保存图像"""
        if self.fig is not None:
            self.fig.savefig(path, dpi=dpi, facecolor='#1e1e2e', edgecolor='none')
    
    def close(self):
        """关闭图像"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

