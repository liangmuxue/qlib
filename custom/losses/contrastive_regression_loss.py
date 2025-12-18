import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Callable
import math


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
        
        return loss


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
                 device: torch.device = None):
        
        super().__init__(temperature, similarity_metric='euclidean', normalize=True, device=device)
        
        self.margin = margin
        self.hard_mining = hard_mining
    
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
        target_diff = torch.abs(targets - targets.t())
        target_diff = (target_diff - torch.min(target_diff)) / (torch.max(target_diff) - torch.min(target_diff))
        
        for i in range(batch_size):
            # 找到正样本（目标值相近）
            positive_candidates = []
            for j in range(batch_size):
                if i != j and target_diff[i, j] < 0.1:  # 目标差异小于阈值
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
                pos_dist = torch.cdist(anchor_feat, pos_feat).squeeze()
                neg_dist = torch.cdist(anchor_feat, neg_feat).squeeze()
                
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
            pos_distance = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
            neg_distance = F.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
            
            # 三元组损失
            loss = F.relu(pos_distance - neg_distance + self.margin)
            losses.append(loss)
        
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        
        return total_loss    