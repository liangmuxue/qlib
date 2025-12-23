import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Callable
import math
from cus_utils.common_compute import pairwise_distances,pairwise_compare
from .arc_loss import UniformityRegularizer,ContrastiveAugmentation

class ContrastiveRegressionLoss(nn.Module):
    """
    对比学习回归损失函数基础类
    """
    
    def __init__(self,
                 temperature: float = 0.5,
                 similarity_metric: str = 'cosine',
                 normalize: bool = True,
                 device: torch.device = None):
        """
        Args:
            temperature: 温度参数，控制对比损失的锐度
            similarity_metric: 相似度度量方法 ('cosine', 'euclidean', 'manhattan')
            normalize: 是否对特征进行L2归一化
            device: 设备
        """
        super().__init__()
        
        self.temperature = temperature
        self.similarity_metric = similarity_metric
        self.normalize = normalize
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
    
    def compute_similarity(self, 
                          features1: torch.Tensor, 
                          features2: torch.Tensor) -> torch.Tensor:
        """
        计算特征相似度
        
        Args:
            features1: 特征1，形状为 (batch_size, feature_dim)
            features2: 特征2，形状为 (batch_size, feature_dim)
            
        Returns:
            相似度矩阵，形状为 (batch_size, batch_size)
        """
        if self.normalize:
            features1 = F.normalize(features1, p=2, dim=1)
            features2 = F.normalize(features2, p=2, dim=1)
        
        if self.similarity_metric == 'cosine':
            # 余弦相似度
            similarity = torch.mm(features1, features2.t())
        elif self.similarity_metric == 'euclidean':
            # 欧氏距离转相似度
            distance = torch.cdist(features1, features2, p=2)
            similarity = 1.0 / (1.0 + distance)
        elif self.similarity_metric == 'manhattan':
            # 曼哈顿距离转相似度
            distance = torch.cdist(features1, features2, p=1)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"未知的相似度度量方法: {self.similarity_metric}")
        
        return similarity
    
    def compute_target_similarity(self,
                                 targets1: torch.Tensor,
                                 targets2: torch.Tensor,
                                 similarity_type: str = 'gaussian') -> torch.Tensor:
        """
        计算目标值之间的相似度
        
        Args:
            targets1: 目标值1，形状为 (batch_size, 1)
            targets2: 目标值2，形状为 (batch_size, 1)
            similarity_type: 相似度类型 ('gaussian', 'linear', 'threshold')
            
        Returns:
            目标相似度矩阵，形状为 (batch_size, batch_size)
        """
        # 确保目标是2D
        if targets1.dim() == 1:
            targets1 = targets1.unsqueeze(1)
        if targets2.dim() == 1:
            targets2 = targets2.unsqueeze(1)
        
        # 计算目标差异
        diff = targets1 - targets2.t()  # (batch_size, batch_size)
        
        if similarity_type == 'gaussian':
            # 高斯相似度
            sigma = torch.std(targets1).item() + 1e-8
            similarity = torch.exp(-diff ** 2 / (2 * sigma ** 2))
        
        elif similarity_type == 'linear':
            # 线性相似度（基于绝对差异）
            max_diff = torch.max(torch.abs(diff))
            similarity = 1.0 - torch.abs(diff) / (max_diff + 1e-8)
        
        elif similarity_type == 'threshold':
            # 阈值相似度（二值化）
            threshold = 0.1 * torch.std(targets1).item()
            similarity = (torch.abs(diff) < threshold).float()
        
        elif similarity_type == 'inverse':
            # 逆差异相似度
            similarity = 1.0 / (1.0 + torch.abs(diff))
        
        else:
            raise ValueError(f"未知的相似度类型: {similarity_type}")
        
        return similarity
    
class SupervisedContrastiveRegressionLoss(ContrastiveRegressionLoss):
    """
    监督对比回归损失函数
    使用目标值的相似度作为监督信号
    """
    
    def __init__(self,
                 contrast_weight: float = 0.5,
                 base_loss: str = 'mse',
                 temperature: float = 0.5,
                 similarity_metric: str = 'cosine',
                 device: torch.device = None):
        
        super().__init__(temperature, similarity_metric, normalize=True, device=device)
        
        self.contrast_weight = contrast_weight
        
        # 基础损失函数
        if base_loss == 'mse':
            self.base_loss_fn = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_loss_fn = nn.L1Loss()
        elif base_loss == 'huber':
            self.base_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"未知的基础损失函数: {base_loss}")
    
    def forward(self,
                features: torch.Tensor,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征向量，形状为 (batch_size, feature_dim)
            predictions: 预测值，形状为 (batch_size, 1)
            targets: 真实值，形状为 (batch_size, 1)
            
        Returns:
            总损失
        """
        batch_size = features.shape[0]
        
        # 基础回归损失
        base_loss = self.base_loss_fn(predictions, targets)
        
        # 计算特征相似度
        feature_similarity = self.compute_similarity(features, features)
        
        # 计算目标相似度（作为监督信号）
        target_similarity = self.compute_target_similarity(targets, targets, 'gaussian')
        
        # 对比损失：使特征相似度与目标相似度对齐
        contrast_loss = self._contrastive_loss(feature_similarity, target_similarity, batch_size)
        
        # 组合损失
        total_loss = (1 - self.contrast_weight) * base_loss + self.contrast_weight * contrast_loss
        
        return total_loss
    
    def _contrastive_loss(self,
                         feature_sim: torch.Tensor,
                         target_sim: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            feature_sim: 特征相似度矩阵
            target_sim: 目标相似度矩阵
            batch_size: 批次大小
            
        Returns:
            对比损失
        """
        # 将相似度转换为logits
        logits = feature_sim / self.temperature
        
        # 创建标签：目标相似度作为软标签
        labels = target_sim
        
        # 归一化标签（使其和为1）
        labels_sum = labels.sum(dim=1, keepdim=True)
        labels = labels / (labels_sum + 1e-8)
        
        # 交叉熵损失（使用软标签）
        loss = -torch.sum(labels * F.log_softmax(logits, dim=1)) / batch_size
    
class PairwiseContrastiveRegressionLoss(ContrastiveRegressionLoss):
    """
    成对对比回归损失函数
    直接优化正负样本对的相似度
    """
    
    def __init__(self,
                 margin: float = 1.0,
                 negative_weight: float = 0.5,
                 temperature: float = 0.5,
                 device: torch.device = None):
        
        super().__init__(temperature, similarity_metric='cosine', normalize=True, device=device)
        
        self.margin = margin
        self.negative_weight = negative_weight
    
    def forward(self,
                features: torch.Tensor,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征向量
            predictions: 预测值
            targets: 真实值
            
        Returns:
            总损失
        """
        batch_size = features.shape[0]
        
        # 计算特征相似度
        feature_sim = self.compute_similarity(features, features)
        
        # 计算目标差异
        targets_expanded = targets.repeat(1, batch_size).view(batch_size, batch_size)
        targets_transposed = targets_expanded.t()
        target_diff = torch.abs(targets_expanded - targets_transposed)
        target_diff = (target_diff - torch.min(target_diff)) / (torch.max(target_diff) - torch.min(target_diff))
        
        # 创建相似性掩码
        similarity_mask = self._create_similarity_mask(target_diff)
        
        # 计算对比损失
        contrast_loss = self._pairwise_contrastive_loss(feature_sim, similarity_mask)
        
        return contrast_loss
    
    def _create_similarity_mask(self, target_diff: torch.Tensor) -> torch.Tensor:
        """
        创建相似性掩码
        
        Args:
            target_diff: 目标差异矩阵
            
        Returns:
            相似性掩码矩阵（1表示相似，-1表示不相似，0表示忽略）
        """
        # 计算目标值的标准差作为阈值
        std = torch.std(target_diff[target_diff > 0]) if torch.any(target_diff > 0) else 1.0
        
        # 相似阈值（目标差异小于标准差的一半）
        similar_threshold = std * 0.5
        
        # 不相似阈值（目标差异大于标准差）
        dissimilar_threshold = std
        
        # 创建掩码
        mask = torch.zeros_like(target_diff)
        
        # 相似对（目标差异小）
        mask[target_diff < similar_threshold] = 1
        
        # 不相似对（目标差异大）
        mask[target_diff > dissimilar_threshold] = -1
        
        # 对角线设置为0（不与自己比较）
        mask.fill_diagonal_(0)
        
        return mask
    
    def _pairwise_contrastive_loss(self,
                                 feature_sim: torch.Tensor,
                                 similarity_mask: torch.Tensor) -> torch.Tensor:
        """
        计算成对对比损失
        
        Args:
            feature_sim: 特征相似度矩阵
            similarity_mask: 相似性掩码矩阵
            
        Returns:
            成对对比损失
        """
        # 提取正样本对（相似对）
        positive_mask = (similarity_mask == 1).float()
        positive_pairs = feature_sim * positive_mask
        
        # 提取负样本对（不相似对）
        negative_mask = (similarity_mask == -1).float()
        negative_pairs = feature_sim * negative_mask
        
        # 正样本损失：使相似度接近1
        positive_loss = torch.sum((1 - positive_pairs) ** 2) / (torch.sum(positive_mask) + 1e-8)
        
        # 负样本损失：使相似度远离（使用margin）
        negative_loss = torch.sum(F.relu(negative_pairs - self.margin) ** 2) / (torch.sum(negative_mask) + 1e-8)
        
        # 组合损失
        total_loss = positive_loss + self.negative_weight * negative_loss
        
        return total_loss    
    
class TripletContrastiveRegressionLoss(ContrastiveRegressionLoss):
    """
    三元组对比回归损失函数
    使用锚点、正样本、负样本的三元组
    """
    
    def __init__(self,
                 margin: float = 1.0,
                 hard_mining: bool = True,
                 temperature: float = 0.5,
                 distance_func = None,
                 diff_matrix_func = None,
                 lambda_reg=0.001, 
                 device: torch.device = None):
        
        super().__init__(temperature, similarity_metric='euclidean', normalize=True, device=device)
        
        self.margin = margin
        self.hard_mining = hard_mining
        self.distance_func = distance_func
        self.diff_matrix_func = diff_matrix_func
        self.lambda_reg = lambda_reg
    
    def forward(self,
                features: torch.Tensor,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 特征向量，形状为 (batch_size, feature_dim)
            predictions: 预测值
            targets: 真实值
            
        Returns:
            三元组损失
        """
        batch_size = features.shape[0]
        
        if batch_size < 3:
            return torch.tensor(0.0, device=self.device)
        
        # 生成三元组
        triplets = self._generate_triplets(features, targets)
        
        if len(triplets) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 计算三元组损失
        triplet_loss = self._compute_triplet_loss(features, triplets)
        
        return triplet_loss
    
    def _generate_triplets(self,
                          features: torch.Tensor,
                          targets: torch.Tensor) -> List[Tuple[int, int, int]]:
        """
        生成三元组（锚点，正样本，负样本）
        
        Args:
            features: 特征向量
            targets: 目标值
            
        Returns:
            三元组列表
        """
        batch_size = features.shape[0]
        triplets = []
        
        # 计算目标差异矩阵
        # target_diff = pairwise_distances(targets,distance_func=self.distance_func)
        target_diff = torch.abs(targets - targets.t())
        target_diff = (target_diff - torch.min(target_diff)) / (torch.max(target_diff) - torch.min(target_diff))
        
        for i in range(batch_size):
            # 找到正样本（目标值相近）
            positive_candidates = []
            for j in range(batch_size):
                if i != j and target_diff[i, j] < 0.2:  # 目标差异小于阈值
                    positive_candidates.append(j)
            
            if not positive_candidates:
                continue
            
            # 找到负样本（目标值相差大）
            negative_candidates = []
            for j in range(batch_size):
                if i != j and target_diff[i, j] > 0.5:  # 目标差异大于阈值
                    negative_candidates.append(j)
            
            if not negative_candidates:
                continue
            
            # 选择正样本和负样本
            if self.hard_mining:
                # 困难负样本挖掘：选择最难的正样本和最难的负样本
                pos_feat = features[positive_candidates]
                neg_feat = features[negative_candidates]
                anchor_feat = features[i].unsqueeze(0)
                
                # 计算距离
                pos_dist = pairwise_compare(anchor_feat, pos_feat,distance_func=self.distance_func).squeeze()
                neg_dist = pairwise_compare(anchor_feat, neg_feat,distance_func=self.distance_func).squeeze()                
                # pos_dist = torch.cdist(anchor_feat, pos_feat).squeeze()
                # neg_dist = torch.cdist(anchor_feat, neg_feat).squeeze()
                
                # 最难正样本（距离最远）
                hardest_pos_idx = torch.argmax(pos_dist)
                hardest_pos = positive_candidates[hardest_pos_idx]
                
                # 最难负样本（距离最近）
                hardest_neg_idx = torch.argmin(neg_dist)
                hardest_neg = negative_candidates[hardest_neg_idx]
                
                triplets.append((i, hardest_pos, hardest_neg))
            else:
                # 随机选择
                import random
                pos = random.choice(positive_candidates)
                neg = random.choice(negative_candidates)
                triplets.append((i, pos, neg))
        
        return triplets
    
    def _compute_triplet_loss(self,
                             features: torch.Tensor,
                             triplets: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        计算三元组损失
        
        Args:
            features: 特征向量
            triplets: 三元组列表
            
        Returns:
            三元组损失
        """
        if len(triplets) == 0:
            return torch.tensor(0.0, device=self.device)
        
        losses = []
        
        for anchor_idx, pos_idx, neg_idx in triplets:
            anchor = features[anchor_idx]
            positive = features[pos_idx]
            negative = features[neg_idx]
            
            # 计算距离
            pos_distance = self.distance_func(anchor, positive)
            neg_distance = self.distance_func(anchor, negative)
            
            # 三元组损失
            triplet_loss = F.relu(pos_distance - neg_distance + self.margin)
            # reg_loss = (anchor.unsqueeze(0).norm(p=2, dim=1).mean() - 1.0) ** 2 + \
            #            (positive.unsqueeze(0).norm(p=2, dim=1).mean() - 1.0) ** 2 + \
            #            (negative.unsqueeze(0).norm(p=2, dim=1).mean() - 1.0) ** 2        
            # loss = triplet_loss.mean() + self.lambda_reg * reg_loss    
            losses.append(triplet_loss)
        
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        
        return total_loss    

class AdaptiveMSRegressionLoss(nn.Module):
    """
    自适应多重相似度回归损失
    
    特点:
    1. 自适应相似度阈值
    2. 支持多维度回归
    3. 支持多种相似度度量
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        margin: float = 0.5,
        similarity_type: str = 'cosine',  # 'cosine', 'l2', 'l1', 'correlation'
        adaptive_threshold: bool = True,
        temperature: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-8
    ):
        super(AdaptiveMSRegressionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.similarity_type = similarity_type
        self.adaptive_threshold = adaptive_threshold
        self.temperature = temperature
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, predictions, targets):
        """
        参数:
            predictions: 预测值 [batch_size, output_dim]
            targets: 目标值 [batch_size, output_dim]
        """
        batch_size = predictions.size(0)
        
        # 计算预测和目标相似度矩阵
        pred_sim = self._compute_similarity(predictions, predictions)
        target_sim = self._compute_similarity(targets, targets)
        
        # 自适应阈值
        if self.adaptive_threshold:
            pos_thresh, neg_thresh = self._compute_adaptive_thresholds(target_sim)
        else:
            pos_thresh = 0.7
            neg_thresh = 0.3
        
        # 创建样本对掩码
        pos_mask = (target_sim > pos_thresh).float()
        neg_mask = (target_sim < neg_thresh).float()
        
        # 移除对角线
        eye = 1 - torch.eye(batch_size, device=predictions.device)
        pos_mask = pos_mask * eye
        neg_mask = neg_mask * eye
        
        # 计算损失
        loss = self._compute_ms_loss(pred_sim, target_sim, pos_mask, neg_mask)
        
        return loss * self.gamma
    
    def _compute_similarity(self, x, y):
        """计算相似度矩阵"""
        if self.similarity_type == 'cosine':
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            return torch.mm(x_norm, y_norm.t())
        
        elif self.similarity_type == 'l2':
            # 使用负L2距离作为相似度
            x_exp = x.unsqueeze(1)  # [batch, 1, dim]
            y_exp = y.unsqueeze(0)  # [1, batch, dim]
            dist = torch.sqrt(torch.sum((x_exp - y_exp)**2, dim=2) + self.eps)
            # 转换为相似度（距离越小，相似度越大）
            similarity = 1.0 / (1.0 + dist)
            return similarity
        
        elif self.similarity_type == 'l1':
            # 使用负L1距离
            x_exp = x.unsqueeze(1)
            y_exp = y.unsqueeze(0)
            dist = torch.sum(torch.abs(x_exp - y_exp), dim=2)
            similarity = 1.0 / (1.0 + dist)
            return similarity
        
        elif self.similarity_type == 'correlation':
            # Pearson相关系数
            x_mean = x - x.mean(dim=1, keepdim=True)
            y_mean = y - y.mean(dim=1, keepdim=True)
            x_std = torch.sqrt(torch.sum(x_mean**2, dim=1, keepdim=True) + self.eps)
            y_std = torch.sqrt(torch.sum(y_mean**2, dim=1, keepdim=True) + self.eps)
            
            x_norm = x_mean / x_std
            y_norm = y_mean / y_std
            
            return torch.mm(x_norm, y_norm.t())
        
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")
    
    def _compute_adaptive_thresholds(self, similarity_matrix):
        """计算自适应阈值"""
        # 基于相似度矩阵的统计特性计算阈值
        batch_size = similarity_matrix.size(0)
        
        # 移除对角线
        mask = 1 - torch.eye(batch_size, device=similarity_matrix.device)
        valid_sim = similarity_matrix * mask
        
        # 计算分位数作为阈值
        flat_sim = valid_sim.flatten()
        
        # 正样本阈值：相似度高的前30%
        pos_thresh = torch.quantile(flat_sim, 0.7)
        
        # 负样本阈值：相似度低的后30%
        neg_thresh = torch.quantile(flat_sim, 0.3)
        
        return pos_thresh, neg_thresh
    
    def _compute_ms_loss(self, pred_sim, target_sim, pos_mask, neg_mask):
        """计算多重相似度损失"""
        # 正样本损失：预测相似度应接近目标相似度
        pos_diff = torch.abs(pred_sim - target_sim)
        pos_weight = torch.exp(self.beta * (target_sim - pred_sim) / self.temperature)
        
        pos_loss = (pos_mask * pos_weight * 
                   torch.log(1 + torch.exp(-self.alpha * (pred_sim - self.margin))))
        
        # 负样本损失：预测相似度应低于目标相似度
        neg_diff = torch.abs(pred_sim - target_sim)
        neg_weight = torch.exp(self.alpha * (pred_sim - target_sim) / self.temperature)
        
        neg_loss = (neg_mask * neg_weight * 
                   torch.log(1 + torch.exp(self.beta * (pred_sim - self.margin))))
        
        # 归一化
        pos_norm = pos_mask.sum() + self.eps
        neg_norm = neg_mask.sum() + self.eps
        
        total_loss = pos_loss.sum() / pos_norm + neg_loss.sum() / neg_norm
        
        return total_loss

class TimeSeriesMSLoss(nn.Module):
    """
    时间序列回归的多重相似度损失
    
    考虑时间序列的时序关系
    """
    
    def __init__(self, alpha=1.5, beta=40.0, margin=0.3, distance_func=None,
                 temporal_weight=0.3, eps=1e-8):
        super(TimeSeriesMSLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.temporal_weight = temporal_weight
        self.eps = eps
        self.distance_func = distance_func
        
    def forward(self, predictions, targets, timestamps=None):
        """
        参数:
            predictions: [batch_size, seq_len, feature_dim]
            targets: [batch_size, seq_len, feature_dim]
            timestamps: [batch_size, seq_len] 可选，时间戳
        """
        feature_dim = predictions.shape[-1]
        
        # 重塑为 [batch_size * seq_len, feature_dim]
        pred_flat = predictions.reshape(-1, feature_dim)
        target_flat = targets.reshape(-1, feature_dim)
        
        # 计算全局相似度损失
        global_loss = self._compute_global_loss(pred_flat, target_flat)
        
        # 计算时序相似度损失
        # temporal_loss = self._compute_temporal_loss(predictions, targets)
        
        # 组合损失
        total_loss = global_loss # + self.temporal_weight * temporal_loss
        
        return total_loss
    
    def _compute_global_loss(self, predictions, targets):
        """计算全局相似度损失"""
        # 使用余弦相似度
        pred_norm = F.normalize(predictions, p=2, dim=1)
        target_norm = F.normalize(targets, p=2, dim=1)
        
        target_sim = pairwise_distances(targets,distance_func=self.distance_func)
        pred_sim = pairwise_distances(predictions,distance_func=self.distance_func)
        # pred_sim = torch.mm(pred_norm, pred_norm.t())
        # target_sim = torch.mm(target_norm, target_norm.t())
        
        # 创建掩码
        pos_mask = (target_sim > 0.7).float()
        neg_mask = (target_sim < 0.3).float()
        
        # 计算损失
        pos_loss = self._compute_positive_loss(pred_sim, target_sim, pos_mask)
        neg_loss = self._compute_negative_loss(pred_sim, target_sim, neg_mask)
        
        return pos_loss + neg_loss
    
    def _compute_temporal_loss(self, predictions, targets):
        """计算时序相似度损失"""
        batch_size, seq_len, feature_dim = predictions.shape
        
        # 计算相邻时间步的相似度
        temporal_loss = 0.0
        for t in range(seq_len - 1):
            # 当前时间步
            pred_t = predictions[:, t, :]
            target_t = targets[:, t, :]
            
            # 下一个时间步
            pred_next = predictions[:, t+1, :]
            target_next = targets[:, t+1, :]
            
            # 计算时间步间的相似度变化
            pred_sim = F.cosine_similarity(pred_t, pred_next, dim=1)
            target_sim = F.cosine_similarity(target_t, target_next, dim=1)
            
            # 计算相似度变化的损失
            loss = torch.mean((pred_sim - target_sim) ** 2)
            temporal_loss += loss
        
        return temporal_loss / (seq_len - 1)
    
    def _compute_positive_loss(self, pred_sim, target_sim, pos_mask):
        """正样本损失"""
        weight = torch.exp(self.beta * (target_sim - pred_sim)) * pos_mask
        loss = weight * torch.log(1 + torch.exp(-self.alpha * (pred_sim - self.margin)))
        return loss.sum() / (pos_mask.sum() + self.eps)
    
    def _compute_negative_loss(self, pred_sim, target_sim, neg_mask):
        """负样本损失"""
        weight = torch.exp(self.alpha * (pred_sim - target_sim)) * neg_mask
        loss = weight * torch.log(1 + torch.exp(self.beta * (pred_sim - self.margin)))
        return loss.sum() / (neg_mask.sum() + self.eps)

class ContinuousArcFace_ORI(nn.Module):
    def __init__(self, in_features, num_proxies=64, s=30.0):
        super().__init__()
        self.in_features = in_features
        self.num_proxies = num_proxies  # 代理数量
        self.s = s
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 创建连续分布的代理向量
        # 每个代理对应圆周上的一个特定角度
        self.proxies = nn.Parameter(torch.randn(num_proxies, 128))
        nn.init.xavier_normal_(self.proxies)
        
        # 代理对应的基准角度（均匀分布在[0, 2π]）
        self.proxy_angles = torch.linspace(0, 2*torch.pi, num_proxies)
        
    def forward(self, x, targets=None):
        # 提取并归一化特征
        features = self.feature_extractor(x)
        features = F.normalize(features, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # 计算特征与所有代理的余弦相似度
        cos_theta = F.linear(features, proxies)  # [batch, num_proxies]
        
        if targets is not None:
            # 将连续目标值映射到角度
            # 假设targets在[0, 1]，映射到[0, 2π]
            target_angles = targets * 2 * torch.pi
            
            # 为每个目标找到最近的两个代理（用于插值）
            batch_size = x.size(0)
            losses = []
            
            for i in range(batch_size):
                # 计算目标角度与所有代理角度的距离
                angle_diff = torch.abs(self.proxy_angles - target_angles[i])
                
                # 找到最近的两个代理
                k = 2  # 使用最近邻插值
                _, indices = torch.topk(-angle_diff, k)  # 距离最小的k个
                
                # 计算与这两个代理的相似度（带温度系数）
                cos_sim = cos_theta[i, indices]
                
                # 使用距离作为权重
                weights = 1.0 / (angle_diff[indices] + 1e-8)
                weights = weights / weights.sum()
                
                # 目标：鼓励特征靠近加权代理
                # 使用softmax交叉熵，但目标是连续的权重
                target_dist = F.softmax(weights * 10.0, dim=0)  # 温度系数
                
                # 计算KL散度损失
                pred_dist = F.softmax(self.s * cos_sim, dim=0)
                loss = F.kl_div(pred_dist.log(), target_dist, reduction='sum')
                losses.append(loss)
            
            total_loss = torch.stack(losses).mean()
            
            # 预测值：通过代理角度加权平均
            with torch.no_grad():
                proxy_weights = F.softmax(self.s * cos_theta, dim=1)
                pred_angles = (proxy_weights * self.proxy_angles).sum(dim=1)
                predictions = pred_angles / (2 * torch.pi)  # 回到[0, 1]
            
            return predictions, total_loss
        
        # 推理时：通过代理角度加权平均
        proxy_weights = F.softmax(self.s * cos_theta, dim=1)
        pred_angles = (proxy_weights * self.proxy_angles).sum(dim=1)
        return pred_angles / (2 * torch.pi)
  
