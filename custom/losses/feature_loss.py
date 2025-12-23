import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
import math


class SingleFeatureLosses:
    """
    单特征回归损失函数集合
    
    专门为只有一个特征值的回归任务设计
    """
    
    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor, 
                reduction: str = 'mean') -> torch.Tensor:
        """
        均方误差损失（标准版）
        
        Args:
            pred: 预测值 (batch_size, 1)
            target: 真实值 (batch_size, 1)
            reduction: 约简方式 ('mean', 'sum', 'none')
        """
        return F.mse_loss(pred, target, reduction=reduction)
    
    @staticmethod
    def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                         weight: Optional[torch.Tensor] = None,
                         reduction: str = 'mean') -> torch.Tensor:
        """
        加权均方误差损失
        
        Args:
            pred: 预测值
            target: 真实值
            weight: 样本权重
            reduction: 约简方式
        """
        loss = (pred - target) ** 2
        if weight is not None:
            loss = loss * weight.view_as(loss)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def huber_loss(pred: torch.Tensor, target: torch.Tensor,
                  delta: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
        """
        Huber损失（平滑L1损失）
        对异常值比MSE更鲁棒
        
        Args:
            pred: 预测值
            target: 真实值
            delta: 阈值，误差大于delta时使用L1，小于等于时使用L2
            reduction: 约简方式
        """
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Huber损失公式
        loss = torch.where(abs_diff <= delta,
                          0.5 * diff ** 2,
                          delta * (abs_diff - 0.5 * delta))
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor,
                     reduction: str = 'mean') -> torch.Tensor:
        """
        对数双曲余弦损失
        结合了MSE和MAE的优点，对异常值鲁棒且处处可微
        
        Args:
            pred: 预测值
            target: 真实值
            reduction: 约简方式
        """
        diff = pred - target
        loss = torch.log(torch.cosh(diff))
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def cauchy_loss(pred: torch.Tensor, target: torch.Tensor,
                   c: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
        """
        柯西损失函数
        对异常值非常鲁棒
        
        Args:
            pred: 预测值
            target: 真实值
            c: 尺度参数
            reduction: 约简方式
        """
        diff = pred - target
        loss = torch.log(1 + (diff / c) ** 2)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def welsch_loss(pred: torch.Tensor, target: torch.Tensor,
                   c: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
        """
        Welsch损失函数
        对异常值高度鲁棒
        
        Args:
            pred: 预测值
            target: 真实值
            c: 尺度参数
            reduction: 约简方式
        """
        diff = pred - target
        loss = 1 - torch.exp(-0.5 * (diff / c) ** 2)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def tukey_loss(pred: torch.Tensor, target: torch.Tensor,
                  c: float = 4.685, reduction: str = 'mean') -> torch.Tensor:
        """
        Tukey双权损失函数
        完全忽略大误差的样本
        
        Args:
            pred: 预测值
            target: 真实值
            c: 截断参数
            reduction: 约简方式
        """
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Tukey双权函数
        mask = abs_diff <= c
        loss = torch.where(mask,
                          (c ** 2 / 6) * (1 - (1 - (diff / c) ** 2) ** 3),
                          torch.zeros_like(diff))
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                     quantile: float = 0.5, reduction: str = 'mean') -> torch.Tensor:
        """
        分位数损失
        用于估计条件分位数，对异常值鲁棒
        
        Args:
            pred: 预测值
            target: 真实值
            quantile: 分位数 (0到1之间)
            reduction: 约简方式
        """
        diff = pred - target
        loss = torch.where(diff >= 0,
                          quantile * diff,
                          (quantile - 1) * diff)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def elastic_net_loss(pred: torch.Tensor, target: torch.Tensor,
                        l1_ratio: float = 0.5, alpha: float = 1.0,
                        reduction: str = 'mean') -> torch.Tensor:
        """
        弹性网络损失
        结合L1和L2正则化
        
        Args:
            pred: 预测值
            target: 真实值
            l1_ratio: L1正则化比例
            alpha: 正则化强度
            reduction: 约简方式
        """
        mse = F.mse_loss(pred, target, reduction='none')
        l1 = torch.abs(pred - target)
        
        loss = mse + alpha * (l1_ratio * l1 + (1 - l1_ratio) * mse.sqrt())
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    @staticmethod
    def geman_mcclure_loss(pred: torch.Tensor, target: torch.Tensor,
                          reduction: str = 'mean') -> torch.Tensor:
        """
        Geman-McClure损失函数
        对异常值鲁棒，有重尾特性
        
        Args:
            pred: 预测值
            target: 真实值
            reduction: 约简方式
        """
        diff = pred - target
        loss = 2 * (diff ** 2) / (1 + diff ** 2)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
        
class AdaptiveSingleFeatureLoss(nn.Module):
    """
    自适应单特征回归损失函数
    根据误差分布自动调整损失函数特性
    """
    
    def __init__(self, 
                 loss_type: str = 'adaptive',
                 initial_params: Optional[Dict] = None,
                 device: torch.device = None):
        """
        Args:
            loss_type: 损失类型 ('adaptive', 'huber', 'cauchy', 'welsch', 'tukey')
            initial_params: 初始参数
            device: 设备
        """
        super().__init__()
        
        self.loss_type = loss_type
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # 可学习参数
        if loss_type == 'adaptive':
            self.scale = nn.Parameter(torch.tensor(1.0, device=device))
            self.robustness = nn.Parameter(torch.tensor(0.5, device=device))  # 鲁棒性参数
        elif loss_type in ['huber', 'cauchy', 'welsch', 'tukey']:
            if initial_params and 'scale' in initial_params:
                self.scale = nn.Parameter(torch.tensor(initial_params['scale'], device=device))
            else:
                self.scale = nn.Parameter(torch.tensor(1.0, device=device))
        
        # 误差统计
        self.error_history = []
        self.max_history_size = 1000
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算自适应损失
        
        Args:
            pred: 预测值 (batch_size, 1)
            target: 真实值 (batch_size, 1)
            
        Returns:
            loss: 损失值
        """
        errors = pred - target
        
        # 更新误差历史
        self.update_error_history(errors.detach())
        
        if self.loss_type == 'adaptive':
            return self.adaptive_loss(errors)
        elif self.loss_type == 'huber':
            return SingleFeatureLosses.huber_loss(pred, target, delta=self.scale.abs())
        elif self.loss_type == 'cauchy':
            return SingleFeatureLosses.cauchy_loss(pred, target, c=self.scale.abs())
        elif self.loss_type == 'welsch':
            return SingleFeatureLosses.welsch_loss(pred, target, c=self.scale.abs())
        elif self.loss_type == 'tukey':
            return SingleFeatureLosses.tukey_loss(pred, target, c=self.scale.abs())
        else:
            return SingleFeatureLosses.mse_loss(pred, target)
    
    def adaptive_loss(self, errors: torch.Tensor) -> torch.Tensor:
        """
        自适应损失函数
        根据误差分布自动调整鲁棒性
        """
        scale = self.scale.abs() + 1e-8
        robustness = torch.sigmoid(self.robustness)  # 限制在0-1之间
        
        # 计算误差统计
        if self.error_history:
            err_list = [item.cpu() for item in self.error_history[-100:]]
            error_array = torch.cat(err_list).numpy()
            if len(error_array) > 10:
                error_std = np.std(error_array)
                kurtosis = self._compute_kurtosis(error_array)
            else:
                error_std = 1.0
                kurtosis = 3.0
        else:
            error_std = 1.0
            kurtosis = 3.0
        
        # 根据峰度调整鲁棒性（峰度高表示异常值多，需要更鲁棒的损失）
        adaptive_robustness = robustness * (1.0 + torch.sigmoid(torch.tensor(kurtosis - 3.0).to(self.device)))
        
        # 组合多个损失函数
        mse_loss = 0.5 * (errors / scale) ** 2
        cauchy_loss = torch.log(1 + (errors / scale) ** 2)
        
        # 根据自适应鲁棒性加权组合
        loss = (1 - adaptive_robustness) * mse_loss + adaptive_robustness * cauchy_loss
        
        return loss.mean()
    
    def update_error_history(self, errors: torch.Tensor):
        """更新误差历史记录"""
        errors_flat = errors.view(-1)
        self.error_history.append(errors_flat)
        
        # 限制历史大小
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 3.0
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        
        # 计算峰度
        kurtosis = np.sum((data - mean) ** 4) / (n * std ** 4)
        return kurtosis
    
    def get_loss_info(self) -> Dict:
        """获取损失函数信息"""
        info = {
            'loss_type': self.loss_type,
            'scale': self.scale.item(),
        }
        
        if self.loss_type == 'adaptive':
            info['robustness'] = torch.sigmoid(self.robustness).item()
        
        # 误差统计
        if self.error_history:
            all_errors = torch.cat(self.error_history).cpu().numpy()
            info.update({
                'error_mean': np.mean(all_errors),
                'error_std': np.std(all_errors),
                'error_skew': self._compute_skewness(all_errors),
                'error_kurtosis': self._compute_kurtosis(all_errors),
                'num_samples': len(all_errors)
            })
        
        return info
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        
        # 计算偏度
        skewness = np.sum((data - mean) ** 3) / (n * std ** 3)
        return skewness
    
    