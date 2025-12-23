import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Callable
import math
from cus_utils.common_compute import pairwise_distances,pairwise_compare

class SemiHardTripletLoss(nn.Module):
    """
    半软间隔三元组损失
    参考：FaceNet: A Unified Embedding for Face Recognition and Clustering
    """
    
    def __init__(self, margin=1.0, mining_strategy='semi-hard', 
                 squared=True, device='cuda'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy  # 'semi-hard', 'hard', 'all'
        self.squared = squared
        self.device = device
        
    def _pairwise_distance(self, embeddings):
        """计算成对距离矩阵"""
        # embeddings: [batch_size, embedding_dim]
        dot_product = torch.matmul(embeddings, embeddings.t())  # [batch_size, batch_size]
        
        # 提取对角线（每个向量的平方范数）
        square_norm = torch.diag(dot_product)  # [batch_size]
        
        # 计算距离平方：||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
        
        # 确保距离非负（数值稳定性）
        distances = F.relu(distances)
        
        if not self.squared:
            distances = torch.sqrt(distances + 1e-8)
            
        return distances
    
    def _get_triplet_mask(self, labels):
        """
        生成三元组mask，标识有效的(anchor, positive, negative)组合
        
        Args:
            labels: [batch_size] 标签
            
        Returns:
            mask: [batch_size, batch_size, batch_size] bool矩阵
                  其中mask[i, j, k] = True 当且仅当:
                  - i != j != k (三个索引不同)
                  - labels[i] == labels[j] (anchor和positive同标签)
                  - labels[i] != labels[k] (anchor和negative不同标签)
        """
        batch_size = labels.size(0)
        
        # 检查i != j
        indices_equal = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)  # [batch_size, batch_size, 1]
        i_not_equal_k = indices_not_equal.unsqueeze(1)  # [batch_size, 1, batch_size]
        j_not_equal_k = indices_not_equal.unsqueeze(0)  # [1, batch_size, batch_size]
        
        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
        
        # 检查标签条件
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))  # [batch_size, batch_size]
        labels_not_equal = ~labels_equal
        
        # anchor和positive同标签
        same_label = labels_equal.unsqueeze(2)  # [batch_size, batch_size, 1]
        
        # anchor和negative不同标签
        diff_label = labels_not_equal.unsqueeze(1)  # [batch_size, 1, batch_size]
        
        # 组合所有条件
        mask = distinct_indices & same_label & diff_label
        
        return mask
    
    def _get_semi_hard_mask(self, distances, labels, margin):
        """
        获取半软样本的mask
        
        Returns:
            semi_hard_mask: [batch_size, batch_size, batch_size] bool矩阵
        """
        batch_size = labels.size(0)
        
        # 获取所有有效三元组的mask
        triplet_mask = self._get_triplet_mask(labels)
        
        # 提取anchor-positive距离
        d_ap = distances.unsqueeze(2).expand(-1, -1, batch_size)  # [batch_size, batch_size, batch_size]
        
        # 提取anchor-negative距离
        d_an = distances.unsqueeze(1).expand(-1, batch_size, -1)  # [batch_size, batch_size, batch_size]
        
        # 半软条件：d_ap < d_an < d_ap + margin
        condition1 = d_ap < d_an  # negative比positive远
        condition2 = d_an < d_ap + margin  # negative不超过margin
        
        semi_hard_mask = triplet_mask & condition1 & condition2
        
        return semi_hard_mask
    
    def _batch_hard_triplet_loss(self, distances, labels):
        """批次硬样本三元组损失"""
        # 获取相同标签的距离
        mask_positive = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # 获取最远的正样本距离（对于每个anchor）
        hardest_positive = torch.max(distances * mask_positive.float(), dim=1)[0]
        
        # 获取不同标签的距离
        mask_negative = ~mask_positive
        # 设置对角线为inf（避免自己成为负样本）
        mask_negative.fill_diagonal_(False)
        
        # 设置正样本距离为inf，确保不会被选为负样本
        distances_with_inf = distances.clone()
        distances_with_inf[mask_positive] = float('inf')
        
        # 获取最近的负样本距离（对于每个anchor）
        hardest_negative = torch.min(distances_with_inf, dim=1)[0]
        
        # 计算损失
        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        
        return losses.mean()
    
    def forward(self, embeddings, labels):
        """
        embeddings: [batch_size, embedding_dim]
        labels: [batch_size]
        """
        # 计算成对距离矩阵
        distances = self._pairwise_distance(embeddings)
        
        if self.mining_strategy == 'batch-hard':
            # 批次硬样本挖掘
            return self._batch_hard_triplet_loss(distances, labels)
        
        elif self.mining_strategy == 'semi-hard':
            # 半软样本挖掘
            batch_size = labels.size(0)
            
            # 获取半软样本mask
            semi_hard_mask = self._get_semi_hard_mask(distances, labels, self.margin)
            
            # 提取anchor-positive和anchor-negative距离
            d_ap = distances.unsqueeze(2).expand(-1, -1, batch_size)
            d_an = distances.unsqueeze(1).expand(-1, batch_size, -1)
            
            # 计算所有有效三元组的损失
            triplet_loss = F.relu(d_ap - d_an + self.margin)
            
            # 只保留半软样本的损失
            semi_hard_losses = triplet_loss[semi_hard_mask]
            
            if semi_hard_losses.numel() > 0:
                # 计算半软样本的平均损失
                loss = semi_hard_losses.mean()
            else:
                # 如果没有半软样本，回退到批次硬样本
                loss = self._batch_hard_triplet_loss(distances, labels)
                
                # 或者返回0损失（根据需求选择）
                # loss = torch.tensor(0.0, device=self.device)
                
            return loss
        
        elif self.mining_strategy == 'all':
            # 使用所有有效三元组
            batch_size = labels.size(0)
            
            # 获取所有有效三元组mask
            triplet_mask = self._get_triplet_mask(labels)
            
            # 提取距离
            d_ap = distances.unsqueeze(2).expand(-1, -1, batch_size)
            d_an = distances.unsqueeze(1).expand(-1, batch_size, -1)
            
            # 计算损失
            triplet_loss = F.relu(d_ap - d_an + self.margin)
            
            # 计算所有有效三元组的平均损失
            valid_losses = triplet_loss[triplet_mask]
            
            if valid_losses.numel() > 0:
                loss = valid_losses.mean()
            else:
                loss = torch.tensor(0.0, device=self.device)
                
            return loss
        
        else:
            raise ValueError(f"未知的挖掘策略: {self.mining_strategy}")

class AdaptiveSemiHardTripletLoss(nn.Module):
    """
    自适应半软间隔三元组损失
    根据训练进度动态调整半软样本的选择条件
    """
    
    def __init__(self, base_margin=1.0, min_margin=0.5, max_margin=2.0,
                 mining_strategy='adaptive', 
                 semi_hard_ratio=0.3,  # 每批中半软样本的目标比例
                 pairwise_distance=None,
                 device='cuda'):
        super().__init__()
        self.base_margin = base_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.mining_strategy = mining_strategy
        self.semi_hard_ratio = semi_hard_ratio
        self.device = device
        
        self.current_margin = base_margin
        self.semi_hard_statistics = []
        self.pairwise_distance = pairwise_distance
        
    def _pairwise_distance(self, embeddings, squared=True):
        """计算成对距离"""
        
        if self.pairwise_distance is not None and False:
            distances = pairwise_distances(embeddings,distance_func=self.pairwise_distance)
        else:
            dot_product = torch.matmul(embeddings, embeddings.t())
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
            distances = F.relu(distances)
            
            if not squared:
                distances = torch.sqrt(distances + 1e-8)
            
        return distances
    
    def _adaptive_semi_hard_mining(self, distances, labels):
        """
        自适应半软样本挖掘
        动态调整阈值以确保有足够比例的半软样本
        """
        batch_size = labels.size(0)
        
        # 获取有效三元组mask
        mask_positive = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        mask_negative = ~mask_positive
        mask_negative.fill_diagonal_(False)
        
        # 对每个anchor，计算所有有效三元组
        all_losses = []
        semi_hard_counts = []
        
        for i in range(batch_size):
            # 获取当前anchor的正样本距离
            pos_mask = mask_positive[i]
            pos_distances = distances[i][pos_mask]
            
            if pos_distances.numel() == 0:
                continue
                
            # 使用平均正样本距离（或最难的正样本）
            d_ap = torch.mean(pos_distances)
            
            # 获取负样本距离
            neg_mask = mask_negative[i]
            neg_distances = distances[i][neg_mask]
            
            if neg_distances.numel() == 0:
                continue
            
            # 自适应阈值：寻找满足条件的负样本
            adaptive_margin = self.current_margin
            
            # 计算半软条件
            semi_hard_condition = (d_ap < neg_distances) & (neg_distances < d_ap + adaptive_margin)
            
            # 检查是否有足够的半软样本
            semi_hard_ratio = torch.sum(semi_hard_condition).float() / neg_distances.numel()
            
            # 调整margin以接近目标比例
            if self.training:
                if semi_hard_ratio < self.semi_hard_ratio * 0.5:
                    # 半软样本太少，增加margin
                    adaptive_margin = min(self.current_margin * 1.1, self.max_margin)
                elif semi_hard_ratio > self.semi_hard_ratio * 1.5:
                    # 半软样本太多，减小margin
                    adaptive_margin = max(self.current_margin * 0.9, self.min_margin)
                
                # 更新当前margin（指数移动平均）
                self.current_margin = 0.9 * self.current_margin + 0.1 * adaptive_margin
            
            # 重新计算半软条件
            semi_hard_condition = (d_ap < neg_distances) & (neg_distances < d_ap + adaptive_margin)
            semi_hard_count = torch.sum(semi_hard_condition)
            semi_hard_counts.append(semi_hard_count.item())
            
            if semi_hard_count > 0:
                # 选择半软样本
                semi_hard_distances = neg_distances[semi_hard_condition]
                
                # 可以选择最难的半软样本（距离最小的）
                hardest_semi_hard = torch.min(semi_hard_distances)
                
                # 计算损失
                loss = F.relu(d_ap - hardest_semi_hard + adaptive_margin)
                all_losses.append(loss)
            else:
                # 如果没有半软样本，回退到最难的负样本
                hardest_negative = torch.min(neg_distances)
                loss = F.relu(d_ap - hardest_negative + adaptive_margin)
                all_losses.append(loss)
        
        # 更新统计信息
        if self.training and len(semi_hard_counts) > 0:
            avg_semi_hard = np.mean(semi_hard_counts)
            total_possible = batch_size * (batch_size - 1)  # 近似
            current_ratio = avg_semi_hard / total_possible if total_possible > 0 else 0
            self.semi_hard_statistics.append(current_ratio)
            
            # 保持统计长度
            if len(self.semi_hard_statistics) > 100:
                self.semi_hard_statistics.pop(0)
        
        if len(all_losses) > 0:
            return torch.stack(all_losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def forward(self, embeddings, labels):
        if self.mining_strategy == 'adaptive':
            distances = self._pairwise_distance(embeddings, squared=False)
            return self._adaptive_semi_hard_mining(distances, labels)
        else:
            raise ValueError(f"不支持的挖掘策略: {self.mining_strategy}")
    
    def get_statistics(self):
        """获取训练统计信息"""
        if len(self.semi_hard_statistics) == 0:
            return {}
        
        return {
            'current_margin': self.current_margin,
            'avg_semi_hard_ratio': np.mean(self.semi_hard_statistics),
            'num_batches_tracked': len(self.semi_hard_statistics)
        }
        
class ContinuousSemiHardTripletLoss(nn.Module):
    """
    连续标签的自适应半软间隔三元组损失
    适用于回归任务、相似度学习等连续标签场景
    """
    
    def __init__(self, 
                 margin: float = 1.0,
                 similarity_threshold: float = 0.5,  # 相似度阈值，高于此值视为"正样本"
                 mining_strategy: str = 'semi-hard',  # 'semi-hard', 'adaptive', 'dynamic'
                 temperature: float = 0.1,  # 相似度计算的温度参数
                 adaptive_margin: bool = True,  # 是否使用自适应margin
                 pairwise_distance=None,
                 device: str = 'cuda'):
        super().__init__()
        
        self.base_margin = margin
        self.similarity_threshold = similarity_threshold
        self.mining_strategy = mining_strategy
        self.temperature = temperature
        self.adaptive_margin = adaptive_margin
        self.device = device
        self.pairwise_distance = pairwise_distance
        
        # 自适应参数
        self.current_margin = margin
        self.margin_history = []
        self.semi_hard_ratio_history = []
        
    def compute_label_similarity(self, labels_i, labels_j, method='gaussian'):
        """
        计算连续标签之间的相似度
        
        Args:
            labels_i: 标签张量1 [batch_size] 或 [batch_size, 1]
            labels_j: 标签张量2 [batch_size] 或 [batch_size, 1]
            method: 相似度计算方法
            
        Returns:
            similarity: 相似度矩阵 [batch_size, batch_size]，值在[0,1]之间
        """
        # 确保标签是2D的
        if labels_i.dim() == 1:
            labels_i = labels_i.unsqueeze(1)
        if labels_j.dim() == 1:
            labels_j = labels_j.unsqueeze(1)
        
        batch_size_i = labels_i.size(0)
        batch_size_j = labels_j.size(0)
        
        # 扩展为矩阵计算
        labels_i_exp = labels_i.unsqueeze(1).expand(batch_size_i, batch_size_j, 1)
        labels_j_exp = labels_j.unsqueeze(0).expand(batch_size_i, batch_size_j, 1)
        
        if method == 'gaussian':
            # 高斯相似度
            diff = labels_i_exp - labels_j_exp
            similarity = torch.exp(-diff.pow(2).sum(dim=2) / (2 * self.temperature ** 2))
            
        elif method == 'linear':
            # 线性相似度
            diff = torch.abs(labels_i_exp - labels_j_exp).sum(dim=2)
            max_diff = torch.max(diff)
            similarity = 1.0 - diff / (max_diff + 1e-8)
            similarity = torch.clamp(similarity, 0.0, 1.0)
            
        elif method == 'cosine':
            # 余弦相似度（假设标签是向量）
            similarity = F.cosine_similarity(
                labels_i_exp.squeeze(-1), 
                labels_j_exp.squeeze(-1), 
                dim=2
            )
            # 映射到[0,1]
            similarity = (similarity + 1) / 2
            
        else:
            raise ValueError(f"未知的相似度计算方法: {method}")
        
        return similarity
    
    def compute_feature_distance(self, embeddings, squared=False):
        """
        计算特征空间的距离矩阵
        
        Args:
            embeddings: 特征嵌入 [batch_size, embedding_dim]
            squared: 是否返回平方距离
            
        Returns:
            distance_matrix: 距离矩阵 [batch_size, batch_size]
        """
        # 使用欧氏距离
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2 * dot_product
        
        # 数值稳定性
        distances = F.relu(distances)
        
        if not squared:
            distances = torch.sqrt(distances + 1e-8)
            
        return distances
    
    def _adaptive_margin_for_continuous_labels(self, label_similarity):
        """
        为连续标签计算自适应margin
        
        基本思想：标签越相似，margin应该越小（因为更容易混淆）
                标签差异越大，margin应该越大（因为更容易区分）
        """
        # 将相似度转换为"差异度"
        label_difference = 1.0 - label_similarity
        
        # 自适应margin：差异越大，margin越大
        adaptive_margin = self.base_margin * (1.0 + label_difference)
        
        # 限制margin范围
        adaptive_margin = torch.clamp(adaptive_margin, 
                                      min=self.base_margin * 0.5,
                                      max=self.base_margin * 2.0)
        
        return adaptive_margin
    
    def _find_semi_hard_triplets_continuous(self, 
                                           embeddings, 
                                           labels, 
                                           similarity_matrix):
        """
        在连续标签下寻找半软三元组
        
        Args:
            embeddings: 特征嵌入 [batch_size, embedding_dim]
            labels: 连续标签 [batch_size]
            similarity_matrix: 标签相似度矩阵 [batch_size, batch_size]
            
        Returns:
            losses: 三元组损失列表
            mining_stats: 挖掘统计信息
        """
        batch_size = embeddings.size(0)
        
        # 计算特征距离矩阵
        feature_distances = self.compute_feature_distance(embeddings, squared=False)
        
        # 创建正负样本掩码
        # 正样本：标签相似度高于阈值
        positive_mask = similarity_matrix > self.similarity_threshold
        
        # 负样本：标签相似度低于阈值，但也不能太低（避免太简单的负样本）
        # 可以设置负样本的相似度上限，确保负样本不是完全不相关的
        negative_upper_bound = self.similarity_threshold * 0.8  # 负样本相似度上限
        negative_mask = similarity_matrix < negative_upper_bound
        
        # 排除对角线（自己）
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        positive_mask = positive_mask & (~eye_mask)
        negative_mask = negative_mask & (~eye_mask)
        
        losses = []
        mining_stats = {
            'semi_hard_found': 0,
            'hard_found': 0,
            'easy_skipped': 0,
            'no_positive': 0,
            'no_negative': 0
        }
        
        for i in range(batch_size):
            # 获取当前anchor的正样本索引
            pos_indices = torch.where(positive_mask[i])[0]
            
            if len(pos_indices) == 0:
                mining_stats['no_positive'] += 1
                continue
            
            # 选择最难的正样本（特征距离最远的）
            pos_distances = feature_distances[i, pos_indices]
            hardest_pos_idx = pos_indices[torch.argmax(pos_distances)]
            d_ap = pos_distances[torch.argmax(pos_distances)]
            
            # 获取负样本索引
            neg_indices = torch.where(negative_mask[i])[0]
            
            if len(neg_indices) == 0:
                mining_stats['no_negative'] += 1
                continue
            
            neg_distances = feature_distances[i, neg_indices]
            
            # 计算自适应margin（基于标签相似度）
            if self.adaptive_margin:
                # 使用anchor和最难正样本的标签相似度来计算margin
                pos_similarity = similarity_matrix[i, hardest_pos_idx]
                adaptive_margin = self._adaptive_margin_for_continuous_labels(pos_similarity)
                current_margin = adaptive_margin
            else:
                current_margin = self.current_margin
            
            # 半软条件：d_ap < d_an < d_ap + margin
            semi_hard_mask = (d_ap < neg_distances) & (neg_distances < d_ap + current_margin)
            
            if torch.any(semi_hard_mask):
                # 找到半软样本，选择最难的那个（距离最小的）
                semi_hard_distances = neg_distances[semi_hard_mask]
                hardest_semi_hard = torch.min(semi_hard_distances)
                
                # 计算损失
                loss = F.relu(d_ap - hardest_semi_hard + current_margin)
                losses.append(loss)
                mining_stats['semi_hard_found'] += 1
                
            else:
                # 没有半软样本，尝试找硬样本
                hard_mask = neg_distances < d_ap
                
                if torch.any(hard_mask):
                    # 找到硬样本，选择最难的那个（距离最大的硬样本）
                    hard_distances = neg_distances[hard_mask]
                    hardest_hard = torch.max(hard_distances)
                    
                    # 计算损失（使用较小的权重，避免梯度爆炸）
                    loss = F.relu(d_ap - hardest_hard + current_margin) * 0.5
                    losses.append(loss)
                    mining_stats['hard_found'] += 1
                else:
                    # 只有简单样本，跳过
                    mining_stats['easy_skipped'] += 1
        
        return losses, mining_stats
    
    def _dynamic_threshold_adjustment(self, similarity_matrix, mining_stats):
        """
        动态调整相似度阈值，以保持半软样本的比例
        
        Args:
            similarity_matrix: 标签相似度矩阵
            mining_stats: 当前批次的挖掘统计
        """
        total_samples = sum(mining_stats.values())
        
        if total_samples == 0:
            return
        
        # 计算半软样本比例
        semi_hard_ratio = mining_stats['semi_hard_found'] / total_samples
        self.semi_hard_ratio_history.append(semi_hard_ratio)
        
        # 保持历史长度
        if len(self.semi_hard_ratio_history) > 100:
            self.semi_hard_ratio_history.pop(0)
        
        # 目标半软样本比例
        target_ratio = 0.3
        
        if len(self.semi_hard_ratio_history) >= 10:
            avg_ratio = np.mean(self.semi_hard_ratio_history[-10:])
            
            # 动态调整阈值
            if avg_ratio < target_ratio * 0.5:
                # 半软样本太少，降低阈值（让更多样本成为正样本）
                self.similarity_threshold *= 0.95
                print(f"降低相似度阈值到: {self.similarity_threshold:.4f}")
                
            elif avg_ratio > target_ratio * 1.5:
                # 半软样本太多，提高阈值
                self.similarity_threshold *= 1.05
                print(f"提高相似度阈值到: {self.similarity_threshold:.4f}")
            
            # 限制阈值范围
            self.similarity_threshold = np.clip(self.similarity_threshold, 0.1, 0.9)
    
    def forward(self, embeddings, labels, similarity_method='gaussian'):
        """
        前向传播
        
        Args:
            embeddings: 特征嵌入 [batch_size, embedding_dim]
            labels: 连续标签 [batch_size] 或 [batch_size, 1]
            similarity_method: 标签相似度计算方法
            
        Returns:
            loss: 三元组损失
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 3:
            # 批次太小，无法形成三元组
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算标签相似度矩阵
        similarity_matrix = self.compute_label_similarity(
            labels, labels, method=similarity_method
        )
        
        # 寻找三元组并计算损失
        losses, mining_stats = self._find_semi_hard_triplets_continuous(
            embeddings, labels, similarity_matrix
        )
        
        # 动态调整阈值（仅在训练时）
        if self.training and self.mining_strategy == 'adaptive':
            self._dynamic_threshold_adjustment(similarity_matrix, mining_stats)
        
        # 计算平均损失
        if losses:
            loss = torch.stack(losses).mean()
            
            # 记录margin历史
            if self.adaptive_margin:
                self.margin_history.append(self.current_margin)
                if len(self.margin_history) > 100:
                    self.margin_history.pop(0)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss
    
    def get_mining_statistics(self):
        """获取挖掘统计信息"""
        if not self.semi_hard_ratio_history:
            return {}
        
        return {
            'similarity_threshold': self.similarity_threshold,
            'current_margin': self.current_margin,
            'avg_semi_hard_ratio': np.mean(self.semi_hard_ratio_history[-10:]) if len(self.semi_hard_ratio_history) >= 10 else 0,
            'margin_history_mean': np.mean(self.margin_history) if self.margin_history else self.base_margin
        }        