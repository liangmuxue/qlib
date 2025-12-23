import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import heapq
import collections
from collections import deque
import random
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ContinuousTripletConfig:
    """连续标签三元组损失配置"""
    memory_size: int = 4096  # 记忆库大小
    embedding_dim: int = 128  # 嵌入维度
    base_margin: float = 1.0  # 基础间隔
    min_margin: float = 0.3  # 最小间隔
    max_margin: float = 2.0  # 最大间隔
    similarity_threshold: float = 0.5  # 相似度阈值
    similarity_metric: str = 'gaussian'  # 相似度度量方法
    temperature: float = 0.1  # 高斯相似度温度参数
    mining_strategy: str = 'semi-hard'  # 挖掘策略
    semi_hard_ratio_target: float = 0.3  # 目标半软样本比例
    device: str = 'cuda'  # 设备
    adaptive_margin: bool = True  # 自适应间隔
    memory_update_strategy: str = 'fifo'  # 记忆库更新策略：fifo, lru, fifo
    hard_negative_weight: float = 0.5  # 硬样本权重
    max_hard_loss: float = 5.0  # 最大硬样本损失
    
    
class ContinuousSimilarity:
    """连续标签相似度计算"""
    
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
        
    @staticmethod
    def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
        """归一化标签到[0,1]范围"""
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        # 计算最小值和最大值
        min_val = labels.min(dim=0, keepdim=True)[0]
        max_val = labels.max(dim=0, keepdim=True)[0]
        range_val = max_val - min_val
        
        # 防止除零
        range_val[range_val == 0] = 1.0
        
        # 归一化
        normalized = (labels - min_val) / range_val
        return normalized
    
    def compute_gaussian_similarity(self, labels_i: torch.Tensor, labels_j: torch.Tensor) -> torch.Tensor:
        """
        高斯相似度：exp(-|labels_i - labels_j|² / (2*temperature²))
        
        适用于标签差异服从高斯分布的情况
        """
        if labels_i.dim() == 1:
            labels_i = labels_i.unsqueeze(1)
        if labels_j.dim() == 1:
            labels_j = labels_j.unsqueeze(1)
        
        batch_i, batch_j = labels_i.size(0), labels_j.size(0)
        
        # 计算标签差异矩阵
        labels_i_exp = labels_i.unsqueeze(1).expand(batch_i, batch_j, labels_i.size(1))
        labels_j_exp = labels_j.unsqueeze(0).expand(batch_i, batch_j, labels_j.size(1))
        
        diff = labels_i_exp - labels_j_exp
        squared_diff = diff.pow(2).sum(dim=2)  # 欧氏距离平方
        
        # 高斯相似度
        similarity = torch.exp(-squared_diff / (2 * self.temperature ** 2))
        
        return similarity
    
    def compute_cosine_similarity(self, labels_i: torch.Tensor, labels_j: torch.Tensor) -> torch.Tensor:
        """
        余弦相似度：labels_i·labels_j / (|labels_i| * |labels_j|)
        
        适用于方向性标签
        """
        if labels_i.dim() == 1:
            labels_i = labels_i.unsqueeze(1)
        if labels_j.dim() == 1:
            labels_j = labels_j.unsqueeze(1)
        
        # 归一化到单位长度
        labels_i_norm = F.normalize(labels_i, p=2, dim=1)
        labels_j_norm = F.normalize(labels_j, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(labels_i_norm, labels_j_norm.T)
        
        # 映射到[0,1]范围（余弦相似度范围是[-1,1]）
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def compute_linear_similarity(self, labels_i: torch.Tensor, labels_j: torch.Tensor, 
                                 max_diff: float = 1.0) -> torch.Tensor:
        """
        线性相似度：1 - min(|labels_i - labels_j| / max_diff, 1)
        
        适用于线性差异
        """
        if labels_i.dim() == 1:
            labels_i = labels_i.unsqueeze(1)
        if labels_j.dim() == 1:
            labels_j = labels_j.unsqueeze(1)
        
        batch_i, batch_j = labels_i.size(0), labels_j.size(0)
        
        labels_i_exp = labels_i.unsqueeze(1).expand(batch_i, batch_j, labels_i.size(1))
        labels_j_exp = labels_j.unsqueeze(0).expand(batch_i, batch_j, labels_j.size(1))
        
        diff = torch.abs(labels_i_exp - labels_j_exp).sum(dim=2)
        similarity = 1.0 - torch.clamp(diff / max_diff, 0, 1)
        
        return similarity
    
    def compute_percentile_similarity(self, labels_i: torch.Tensor, labels_j: torch.Tensor) -> torch.Tensor:
        """
        百分位相似度：基于标签在分布中的百分位差异
        
        适用于非均匀分布标签
        """
        if labels_i.dim() == 1:
            labels_i = labels_i.unsqueeze(1)
        if labels_j.dim() == 1:
            labels_j = labels_j.unsqueeze(1)
        
        # 计算百分位
        def compute_percentile(labels):
            sorted_labels, _ = torch.sort(labels, dim=0)
            percentiles = torch.zeros_like(labels)
            n = labels.size(0)
            
            for i in range(labels.size(1)):
                for j in range(n):
                    # 计算小于等于当前值的比例
                    percentile = torch.sum(sorted_labels[:, i] <= labels[j, i]).float() / n
                    percentiles[j, i] = percentile
                    
            return percentiles
        
        percentiles_i = compute_percentile(labels_i)
        percentiles_j = compute_percentile(labels_j)
        
        # 计算百分位差异
        diff = torch.abs(percentiles_i.unsqueeze(1) - percentiles_j.unsqueeze(0))
        similarity = 1.0 - diff.mean(dim=2)
        
        return similarity
    
    def compute_similarity(self, labels_i: torch.Tensor, labels_j: torch.Tensor, 
                          method: str = 'gaussian') -> torch.Tensor:
        """
        计算连续标签相似度
        
        Args:
            labels_i: 标签张量1
            labels_j: 标签张量2
            method: 相似度计算方法
            
        Returns:
            similarity: 相似度矩阵，范围[0,1]
        """
        if method == 'gaussian':
            return self.compute_gaussian_similarity(labels_i, labels_j)
        elif method == 'cosine':
            return self.compute_cosine_similarity(labels_i, labels_j)
        elif method == 'linear':
            return self.compute_linear_similarity(labels_i, labels_j)
        elif method == 'percentile':
            return self.compute_percentile_similarity(labels_i, labels_j)
        else:
            raise ValueError(f"未知的相似度计算方法: {method}")
    
    def compute_dynamic_similarity_threshold(self, similarities: torch.Tensor, 
                                            target_density: float = 0.3) -> float:
        """
        动态计算相似度阈值，以保持一定比例的正样本
        
        Args:
            similarities: 相似度矩阵
            target_density: 目标正样本密度
            
        Returns:
            threshold: 动态计算的阈值
        """
        if similarities.numel() == 0:
            return 0.5
        
        # 展平相似度矩阵（排除对角线）
        n = similarities.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=similarities.device)
        flat_similarities = similarities[mask].cpu().numpy()
        
        if len(flat_similarities) == 0:
            return 0.5
        
        # 计算百分位数阈值
        threshold = np.percentile(flat_similarities, (1 - target_density) * 100)
        
        return float(threshold)
        
class ContinuousMemoryBank:
    """
    连续标签记忆库
    支持基于标签相似度的智能检索和更新
    """
    
    def __init__(self, 
                 memory_size: int = 4096,
                 embedding_dim: int = 128,
                 label_dim: int = 1,
                 update_strategy: str = 'fifo',  # fifo, lru, priority
                 similarity_metric: str = 'gaussian',
                 device: str = 'cuda'):
        
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.label_dim = label_dim
        self.update_strategy = update_strategy
        self.similarity_metric = similarity_metric
        self.device = device
        
        # 初始化记忆库
        self.embeddings = torch.zeros(memory_size, embedding_dim, device=device).double()
        self.labels = torch.zeros(memory_size, label_dim, device=device).double()
        self.usage_counts = torch.zeros(memory_size, dtype=torch.long, device=device)
        self.timestamps = torch.zeros(memory_size, dtype=torch.long, device=device)
        
        # 指针和计数器
        self.pointer = 0
        self.current_size = 0
        self.time_counter = 0
        
        # 相似度计算器
        self.similarity_calculator = ContinuousSimilarity()
        
        # LRU缓存（如果使用LRU策略）
        self.lru_queue = deque(maxlen=memory_size)
        
        # 优先级队列（如果使用优先级策略）
        self.priority_queue = []
        
        # 统计信息
        self.stats = {
            'updates': 0,
            'queries': 0,
            'hits': 0,
            'misses': 0,
            'avg_similarity': 0.0
        }
    
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        更新记忆库
        
        Args:
            embeddings: 新嵌入 [batch_size, embedding_dim]
            labels: 新标签 [batch_size, label_dim]
        """
        batch_size = embeddings.size(0)
        self.time_counter += 1
        
        if batch_size == 0:
            return
        
        # 确保标签是2D的
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        # 根据更新策略选择位置
        if self.update_strategy == 'fifo':
            # 先进先出
            indices = torch.arange(self.pointer, self.pointer + batch_size) % self.memory_size
            
            self.embeddings[indices] = embeddings.detach()
            self.labels[indices] = labels.detach()
            self.timestamps[indices] = self.time_counter
            self.usage_counts[indices] += 1
            
            self.pointer = (self.pointer + batch_size) % self.memory_size
            
        elif self.update_strategy == 'lru':
            # 最近最少使用
            for i in range(batch_size):
                if self.current_size < self.memory_size:
                    # 还有空位
                    idx = self.current_size
                    self.current_size += 1
                else:
                    # 找到最久未使用的
                    idx = min(range(self.memory_size), 
                             key=lambda x: self.timestamps[x])
                
                self.embeddings[idx] = embeddings[i].detach()
                self.labels[idx] = labels[i].detach()
                self.timestamps[idx] = self.time_counter
                self.usage_counts[idx] += 1
                
                # 更新LRU队列
                if idx in self.lru_queue:
                    self.lru_queue.remove(idx)
                self.lru_queue.append(idx)
        
        elif self.update_strategy == 'priority':
            # 基于使用频率的优先级
            for i in range(batch_size):
                if self.current_size < self.memory_size:
                    # 还有空位
                    idx = self.current_size
                    self.current_size += 1
                    priority = 0  # 新样本初始优先级
                else:
                    # 找到优先级最低的（使用次数最少的）
                    idx = torch.argmin(self.usage_counts).item()
                    priority = self.usage_counts[idx].item()
                
                self.embeddings[idx] = embeddings[i].detach()
                self.labels[idx] = labels[i].detach()
                self.timestamps[idx] = self.time_counter
                self.usage_counts[idx] = priority + 1  # 增加优先级
        
        # 更新统计信息
        self.stats['updates'] += batch_size
        self.current_size = min(self.current_size + batch_size, self.memory_size)
    
    def query(self, 
              query_labels: torch.Tensor, 
              query_type: str = 'positive',
              k: int = 10,
              similarity_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于标签相似度查询记忆库
        
        Args:
            query_labels: 查询标签 [batch_size, label_dim] 或 [label_dim]
            query_type: 'positive', 'negative', 'semi-hard'
            k: 返回的样本数量
            similarity_threshold: 相似度阈值
            
        Returns:
            embeddings: 查询到的嵌入 [min(k, available), embedding_dim]
            labels: 查询到的标签 [min(k, available), label_dim]
        """
        self.stats['queries'] += 1
        
        if self.current_size == 0:
            return torch.empty(0, self.embedding_dim, device=self.device), \
                   torch.empty(0, self.label_dim, device=self.device)
        
        # 确保查询标签是2D的
        if query_labels.dim() == 1:
            query_labels = query_labels.unsqueeze(0)
        
        batch_size = query_labels.size(0)
        
        # 获取有效的记忆库样本
        valid_indices = torch.arange(self.memory_size, device=self.device)[:self.current_size]
        memory_embeddings = self.embeddings[valid_indices]
        memory_labels = self.labels[valid_indices]
        
        # 计算相似度
        similarities = self.similarity_calculator.compute_similarity(
            query_labels, memory_labels, self.similarity_metric
        )  # [batch_size, memory_size]
        
        all_results_embeddings = []
        all_results_labels = []
        
        for i in range(batch_size):
            sim = similarities[i]  # [memory_size]
            
            if query_type == 'positive':
                # 正样本：相似度高
                mask = sim > similarity_threshold
            elif query_type == 'negative':
                # 负样本：相似度低
                mask = sim < similarity_threshold * 0.8  # 负样本阈值更低
            elif query_type == 'semi-hard':
                # 半硬样本：中等相似度
                lower = similarity_threshold * 0.5
                upper = similarity_threshold * 1.2
                mask = (sim > lower) & (sim < upper)
            else:
                raise ValueError(f"未知的查询类型: {query_type}")
            
            # 获取满足条件的样本
            candidate_indices = torch.where(mask)[0]
            
            if len(candidate_indices) == 0:
                # 如果没有满足条件的样本，放宽条件
                if query_type == 'positive':
                    # 选择最相似的
                    _, topk_indices = torch.topk(sim, min(k, len(sim)))
                    candidate_indices = topk_indices
                elif query_type == 'negative':
                    # 选择最不相似的
                    _, topk_indices = torch.topk(-sim, min(k, len(sim)))
                    candidate_indices = topk_indices
                elif query_type == 'semi-hard':
                    # 选择中等相似的
                    mid_sim = (sim.max() + sim.min()) / 2
                    diff = torch.abs(sim - mid_sim)
                    _, topk_indices = torch.topk(-diff, min(k, len(sim)))
                    candidate_indices = topk_indices
            
            # 限制数量
            if len(candidate_indices) > k:
                candidate_indices = candidate_indices[:k]
            
            if len(candidate_indices) > 0:
                # 获取嵌入和标签
                result_embeddings = memory_embeddings[candidate_indices]
                result_labels = memory_labels[candidate_indices]
                
                all_results_embeddings.append(result_embeddings)
                all_results_labels.append(result_labels)
                
                # 更新使用计数
                actual_indices = valid_indices[candidate_indices]
                self.usage_counts[actual_indices] += 1
                self.timestamps[actual_indices] = self.time_counter
                
                self.stats['hits'] += len(candidate_indices)
            else:
                self.stats['misses'] += 1
        
        # 合并结果
        if all_results_embeddings:
            embeddings = torch.cat(all_results_embeddings, dim=0)
            labels = torch.cat(all_results_labels, dim=0)
        else:
            embeddings = torch.empty(0, self.embedding_dim, device=self.device)
            labels = torch.empty(0, self.label_dim, device=self.device)
        
        # 更新平均相似度统计
        if similarities.numel() > 0:
            self.stats['avg_similarity'] = 0.9 * self.stats['avg_similarity'] + 0.1 * similarities.mean().item()
        
        return embeddings, labels
    
    def query_by_feature_distance(self, 
                                 query_embedding: torch.Tensor,
                                 query_label: torch.Tensor,
                                 k: int = 10,
                                 margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于特征距离查询半硬样本
        
        Args:
            query_embedding: 查询嵌入 [embedding_dim]
            query_label: 查询标签 [label_dim]
            k: 返回样本数量
            margin: 三元组间隔
            
        Returns:
            positive_embeddings: 正样本嵌入
            negative_embeddings: 负样本嵌入
        """
        if self.current_size == 0:
            return None, None
        
        # 获取记忆库样本
        valid_indices = torch.arange(self.memory_size, device=self.device)[:self.current_size]
        memory_embeddings = self.embeddings[valid_indices]
        memory_labels = self.labels[valid_indices]
        
        # 计算相似度
        similarities = self.similarity_calculator.compute_similarity(
            query_label.unsqueeze(0), memory_labels, self.similarity_metric
        ).squeeze(0)  # [memory_size]
        
        # 计算特征距离
        distances = F.pairwise_distance(
            query_embedding.unsqueeze(0).expand(memory_embeddings.size(0), -1),
            memory_embeddings,
            p=2
        )  # [memory_size]
        
        # 分离正负样本
        positive_mask = similarities > 0.5  # 正样本阈值
        negative_mask = similarities < 0.3  # 负样本阈值
        
        if not torch.any(positive_mask):
            # 没有正样本，选择最相似的
            _, topk_pos = torch.topk(similarities, min(1, len(similarities)))
            positive_mask = torch.zeros_like(positive_mask)
            positive_mask[topk_pos] = True
        
        if not torch.any(negative_mask):
            # 没有负样本，选择最不相似的
            _, topk_neg = torch.topk(-similarities, min(1, len(similarities)))
            negative_mask = torch.zeros_like(negative_mask)
            negative_mask[topk_neg] = True
        
        # 获取正样本距离
        positive_distances = distances[positive_mask]
        if len(positive_distances) == 0:
            return None, None
        
        # 选择最难的正样本（距离最远的）
        hardest_positive_idx = torch.argmax(positive_distances)
        positive_indices = torch.where(positive_mask)[0]
        hardest_positive_embedding = memory_embeddings[positive_indices[hardest_positive_idx]]
        d_ap = positive_distances[hardest_positive_idx]
        
        # 寻找半硬负样本
        negative_distances = distances[negative_mask]
        negative_indices = torch.where(negative_mask)[0]
        
        # 半硬条件：d_ap < d_an < d_ap + margin
        semi_hard_mask = (d_ap < negative_distances) & (negative_distances < d_ap + margin)
        
        if torch.any(semi_hard_mask):
            # 选择最难的半硬负样本（距离最小的）
            semi_hard_distances = negative_distances[semi_hard_mask]
            semi_hard_indices = negative_indices[semi_hard_mask]
            
            hardest_semi_hard_idx = torch.argmin(semi_hard_distances)
            hardest_negative_embedding = memory_embeddings[semi_hard_indices[hardest_semi_hard_idx]]
            
            return hardest_positive_embedding.unsqueeze(0), hardest_negative_embedding.unsqueeze(0)
        
        return None, None
    
    def get_statistics(self) -> Dict:
        """获取记忆库统计信息"""
        return {
            'current_size': self.current_size,
            'memory_utilization': self.current_size / self.memory_size,
            'avg_usage': self.usage_counts[:self.current_size].float().mean().item() if self.current_size > 0 else 0,
            'query_hit_rate': self.stats['hits'] / max(self.stats['queries'], 1),
            'avg_similarity': self.stats['avg_similarity']
        }
    
    def clear(self):
        """清空记忆库"""
        self.embeddings.zero_()
        self.labels.zero_()
        self.usage_counts.zero_()
        self.timestamps.zero_()
        self.pointer = 0
        self.current_size = 0
        self.time_counter = 0
        self.lru_queue.clear()
        self.priority_queue.clear()
        self.stats = {'updates': 0, 'queries': 0, 'hits': 0, 'misses': 0, 'avg_similarity': 0.0}


class AdaptiveSemiHardMiner:
    """
    自适应半软挖掘策略
    根据训练状态动态调整挖掘参数
    """
    
    def __init__(self,
                 base_margin: float = 1.0,
                 min_margin: float = 0.3,
                 max_margin: float = 2.0,
                 similarity_threshold: float = 0.5,
                 semi_hard_ratio_target: float = 0.3,
                 adaptive_margin: bool = True,
                 device: str = 'cuda'):
        
        self.base_margin = base_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.similarity_threshold = similarity_threshold
        self.semi_hard_ratio_target = semi_hard_ratio_target
        self.adaptive_margin = adaptive_margin
        self.device = device
        
        # 当前状态
        self.current_margin = base_margin
        self.current_threshold = similarity_threshold
        
        # 历史记录
        self.margin_history = []
        self.threshold_history = []
        self.semi_hard_ratio_history = []
        
        # 相似度计算器
        self.similarity_calculator = ContinuousSimilarity()
        
        # 统计信息
        self.stats = {
            'triplets_mined': 0,
            'semi_hard_found': 0,
            'hard_found': 0,
            'easy_skipped': 0,
            'no_positive': 0,
            'no_negative': 0
        }
    
    def compute_adaptive_margin(self, label_similarity: float) -> float:
        """
        根据标签相似度计算自适应margin
        
        Args:
            label_similarity: 标签相似度，范围[0,1]
            
        Returns:
            adaptive_margin: 自适应margin
        """
        if not self.adaptive_margin:
            return self.current_margin
        
        # 标签越相似，margin应该越小（因为更难区分）
        # 标签差异越大，margin应该越大（更容易区分）
        
        # 相似度高 -> margin小
        # 相似度低 -> margin大
        margin_scale = 1.0 + (1.0 - label_similarity)  # 范围[1.0, 2.0]
        
        adaptive_margin = self.base_margin * margin_scale
        
        # 限制范围
        adaptive_margin = max(self.min_margin, min(self.max_margin, adaptive_margin))
        
        return adaptive_margin
    
    def mine_semi_hard_triplets(self,
                               embeddings: torch.Tensor,
                               labels: torch.Tensor,
                               similarity_matrix: torch.Tensor = None,
                               memory_bank: Optional[ContinuousMemoryBank] = None) -> List[Tuple[int, int, int]]:
        """
        挖掘半软三元组
        
        Args:
            embeddings: 嵌入向量 [batch_size, embedding_dim]
            labels: 连续标签 [batch_size, label_dim] 或 [batch_size]
            similarity_matrix: 预计算的相似度矩阵 [batch_size, batch_size]
            memory_bank: 记忆库（可选）
            
        Returns:
            triplets: 三元组列表，每个元组为(anchor_idx, positive_idx, negative_idx)
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 3:
            return []
        
        # 确保标签是2D的
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        # 计算相似度矩阵（如果没有提供）
        if similarity_matrix is None:
            similarity_matrix = self.similarity_calculator.compute_similarity(
                labels, labels, method='gaussian'
            )
        
        # 计算特征距离矩阵
        distance_matrix = self._compute_distance_matrix(embeddings)
        
        triplets = []
        batch_stats = self.stats.copy()
        
        for anchor_idx in range(batch_size):
            anchor_label = labels[anchor_idx]
            anchor_embedding = embeddings[anchor_idx]
            
            # 在当前批次中寻找正负样本
            batch_triplets = self._mine_from_batch(
                anchor_idx, embeddings, labels, similarity_matrix, distance_matrix
            )
            
            # 如果记忆库可用，从中挖掘
            if memory_bank is not None:
                memory_triplets = self._mine_from_memory(
                    anchor_embedding, anchor_label, memory_bank
                )
                batch_triplets.extend(memory_triplets)
            
            # 选择最佳的三元组
            if batch_triplets:
                # 根据损失值排序，选择最困难的三元组
                scored_triplets = []
                for pos_idx, neg_idx, margin in batch_triplets:
                    d_ap = distance_matrix[anchor_idx, pos_idx] if pos_idx < batch_size else None
                    d_an = distance_matrix[anchor_idx, neg_idx] if neg_idx < batch_size else None
                    
                    # 如果是记忆库样本，需要计算距离
                    if d_ap is None or d_an is None:
                        if pos_idx >= batch_size:
                            # 正样本来自记忆库
                            pos_embedding = memory_bank.embeddings[pos_idx - batch_size]
                            d_ap = F.pairwise_distance(anchor_embedding.unsqueeze(0), 
                                                      pos_embedding.unsqueeze(0), p=2).item()
                        if neg_idx >= batch_size:
                            # 负样本来自记忆库
                            neg_embedding = memory_bank.embeddings[neg_idx - batch_size]
                            d_an = F.pairwise_distance(anchor_embedding.unsqueeze(0), 
                                                      neg_embedding.unsqueeze(0), p=2).item()
                    
                    loss = max(0, d_ap - d_an + margin)
                    scored_triplets.append((loss, (anchor_idx, pos_idx, neg_idx, margin)))
                
                # 选择损失最大的（最困难的）
                scored_triplets.sort(key=lambda x: x[0], reverse=True)
                if scored_triplets:
                    _, best_triplet = scored_triplets[0]
                    triplets.append(best_triplet)
                    
                    # 更新统计
                    loss = scored_triplets[0][0]
                    if loss > 0:
                        if d_ap < d_an:
                            batch_stats['semi_hard_found'] += 1
                        else:
                            batch_stats['hard_found'] += 1
                    else:
                        batch_stats['easy_skipped'] += 1
        
        # 更新全局统计
        for key in self.stats:
            self.stats[key] += batch_stats[key]
        
        # 动态调整参数
        self._adjust_parameters()
        
        return triplets
    
    def _mine_from_batch(self,
                        anchor_idx: int,
                        embeddings: torch.Tensor,
                        labels: torch.Tensor,
                        similarity_matrix: torch.Tensor,
                        distance_matrix: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        从当前批次中挖掘三元组
        """
        batch_size = embeddings.size(0)
        anchor_label = labels[anchor_idx]
        
        # 计算与anchor的标签相似度
        similarities = similarity_matrix[anchor_idx]  # [batch_size]
        
        # 正样本：相似度高
        positive_mask = similarities > self.current_threshold
        positive_mask[anchor_idx] = False  # 排除自身
        
        # 负样本：相似度低
        negative_mask = similarities < self.current_threshold * 0.7
        
        positive_indices = torch.where(positive_mask)[0]
        negative_indices = torch.where(negative_mask)[0]
        
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            self.stats['no_positive'] += len(positive_indices) == 0
            self.stats['no_negative'] += len(negative_indices) == 0
            return []
        
        # 选择最难的正样本（距离最远的）
        pos_distances = distance_matrix[anchor_idx, positive_indices]
        hardest_pos_idx = positive_indices[torch.argmax(pos_distances)]
        d_ap = torch.max(pos_distances)
        
        # 计算自适应margin
        label_similarity = similarities[hardest_pos_idx]
        margin = self.compute_adaptive_margin(label_similarity.item())
        
        # 寻找半硬负样本
        neg_distances = distance_matrix[anchor_idx, negative_indices]
        
        # 半硬条件：d_ap < d_an < d_ap + margin
        semi_hard_mask = (d_ap < neg_distances) & (neg_distances < d_ap + margin)
        
        if torch.any(semi_hard_mask):
            # 找到半硬样本，选择最难的那个（距离最小的）
            semi_hard_indices = negative_indices[semi_hard_mask]
            semi_hard_distances = neg_distances[semi_hard_mask]
            
            hardest_semi_hard_idx = semi_hard_indices[torch.argmin(semi_hard_distances)]
            
            return [(int(hardest_pos_idx), int(hardest_semi_hard_idx), margin)]
        else:
            # 寻找硬样本
            hard_mask = neg_distances < d_ap
            if torch.any(hard_mask):
                hard_indices = negative_indices[hard_mask]
                hard_distances = neg_distances[hard_mask]
                
                # 选择最难的正样本（距离最大的）
                hardest_hard_idx = hard_indices[torch.argmax(hard_distances)]
                
                return [(int(hardest_pos_idx), int(hardest_hard_idx), margin)]
        
        return []
    
    def _mine_from_memory(self,
                         anchor_embedding: torch.Tensor,
                         anchor_label: torch.Tensor,
                         memory_bank: ContinuousMemoryBank) -> List[Tuple[int, int, float]]:
        """
        从记忆库中挖掘三元组
        
        Returns:
            列表，每个元素为(positive_idx, negative_idx, margin)
            其中idx是原始索引 + batch_size（以区分批次样本）
        """
        # 查询记忆库获取半硬负样本
        pos_embedding, neg_embedding = memory_bank.query_by_feature_distance(
            anchor_embedding, anchor_label, k=1, margin=self.current_margin
        )
        
        if pos_embedding is None or neg_embedding is None:
            return []
        
        # 计算标签相似度（近似）
        # 注意：这里需要从记忆库中获取标签，但我们的接口没有返回标签
        # 在实际实现中，可能需要修改query_by_feature_distance以返回标签
        
        # 使用默认margin
        margin = self.current_margin
        
        # 返回三元组索引（使用负数索引表示记忆库样本）
        return [(-1, -2, margin)]  # 简化版本
    
    def _compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算特征距离矩阵"""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
        distances = F.relu(distances)
        distances = torch.sqrt(distances + 1e-8)
        
        return distances
    
    def _adjust_parameters(self):
        """动态调整挖掘参数"""
        # 更新半硬样本比例历史
        total_mined = self.stats['triplets_mined']
        if total_mined > 0:
            semi_hard_ratio = self.stats['semi_hard_found'] / total_mined
            self.semi_hard_ratio_history.append(semi_hard_ratio)
            
            # 保持历史长度
            if len(self.semi_hard_ratio_history) > 100:
                self.semi_hard_ratio_history.pop(0)
        
        # 根据半硬样本比例调整阈值
        if len(self.semi_hard_ratio_history) >= 10:
            avg_ratio = np.mean(self.semi_hard_ratio_history[-10:])
            
            if avg_ratio < self.semi_hard_ratio_target * 0.8:
                # 半硬样本太少，降低阈值（让更多样本成为正样本）
                self.current_threshold *= 0.95
                self.current_threshold = max(0.1, self.current_threshold)
                
            elif avg_ratio > self.semi_hard_ratio_target * 1.2:
                # 半硬样本太多，提高阈值
                self.current_threshold *= 1.05
                self.current_threshold = min(0.9, self.current_threshold)
            
            # 记录阈值历史
            self.threshold_history.append(self.current_threshold)
            
            # 调整margin（如果启用）
            if self.adaptive_margin:
                # 根据半硬样本比例调整margin
                if avg_ratio < 0.1:
                    # 半硬样本太少，增加margin（让更多样本满足半硬条件）
                    self.current_margin = min(self.max_margin, self.current_margin * 1.1)
                elif avg_ratio > 0.5:
                    # 半硬样本太多，减少margin
                    self.current_margin = max(self.min_margin, self.current_margin * 0.9)
                
                self.margin_history.append(self.current_margin)
    
    def get_statistics(self) -> Dict:
        """获取挖掘统计信息"""
        total = sum([self.stats[k] for k in ['semi_hard_found', 'hard_found', 'easy_skipped']])
        
        if total > 0:
            semi_hard_ratio = self.stats['semi_hard_found'] / total
        else:
            semi_hard_ratio = 0.0
        
        return {
            'current_margin': self.current_margin,
            'current_threshold': self.current_threshold,
            'semi_hard_ratio': semi_hard_ratio,
            'total_triplets_mined': self.stats['triplets_mined'],
            'no_positive_count': self.stats['no_positive'],
            'no_negative_count': self.stats['no_negative']
        }
        
class ContinuousTripletLossWithMemory(nn.Module):
    """
    连续标签三元组损失（带记忆库和半软挖掘）
    
    适用于回归任务、相似度学习等连续标签场景
    """
    
    def __init__(self, config: ContinuousTripletConfig):
        super().__init__()
        
        self.config = config
        self.device = config.device
        
        # 记忆库
        self.memory_bank = ContinuousMemoryBank(
            memory_size=config.memory_size,
            embedding_dim=config.embedding_dim,
            label_dim=1,  # 假设标签是标量
            update_strategy=config.memory_update_strategy,
            similarity_metric=config.similarity_metric,
            device=config.device
        )
        
        # 半软挖掘器
        self.miner = AdaptiveSemiHardMiner(
            base_margin=config.base_margin,
            min_margin=config.min_margin,
            max_margin=config.max_margin,
            similarity_threshold=config.similarity_threshold,
            semi_hard_ratio_target=config.semi_hard_ratio_target,
            adaptive_margin=config.adaptive_margin,
            device=config.device
        )
        
        # 相似度计算器
        self.similarity_calculator = ContinuousSimilarity(
            temperature=config.temperature
        )
        
        # 损失权重
        self.hard_negative_weight = config.hard_negative_weight
        self.max_hard_loss = config.max_hard_loss
        
        # 训练历史
        self.training_history = {
            'loss': [],
            'margin': [],
            'threshold': [],
            'semi_hard_ratio': [],
            'memory_utilization': []
        }
        
    def forward(self, 
                embeddings: torch.Tensor, 
                labels: torch.Tensor,
                update_memory: bool = True) -> torch.Tensor:
        """
        计算三元组损失
        
        Args:
            embeddings: 嵌入向量 [batch_size, embedding_dim]
            labels: 连续标签 [batch_size] 或 [batch_size, 1]
            update_memory: 是否更新记忆库
            
        Returns:
            loss: 三元组损失
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 更新记忆库
        if update_memory and self.training:
            self.memory_bank.update(embeddings.detach(), labels.detach())
        
        # 确保标签是2D的
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        # 计算标签相似度矩阵
        similarity_matrix = self.similarity_calculator.compute_similarity(
            labels, labels, method=self.config.similarity_metric
        )
        
        # 挖掘三元组
        triplets = self.miner.mine_semi_hard_triplets(
            embeddings, labels, similarity_matrix, self.memory_bank
        )
        
        if not triplets:
            # 没有找到有效的三元组
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算损失
        losses = []
        semi_hard_count = 0
        hard_count = 0
        
        for anchor_idx, pos_idx, neg_idx, margin in triplets:
            # 获取嵌入向量
            if pos_idx < 0 or neg_idx < 0:
                # 来自记忆库的样本（简化处理）
                continue
            
            anchor_emb = embeddings[anchor_idx]
            pos_emb = embeddings[pos_idx]
            neg_emb = embeddings[neg_idx]
            
            # 计算距离
            d_ap = F.pairwise_distance(anchor_emb.unsqueeze(0), pos_emb.unsqueeze(0), p=2)
            d_an = F.pairwise_distance(anchor_emb.unsqueeze(0), neg_emb.unsqueeze(0), p=2)
            
            # 判断三元组类型
            if d_ap < d_an:
                # 半软或简单三元组
                loss = F.relu(d_ap - d_an + margin)
                semi_hard_count += 1
            else:
                # 硬三元组
                loss = F.relu(d_ap - d_an + margin) * self.hard_negative_weight
                hard_count += 1
            
            # 裁剪硬样本损失，防止梯度爆炸
            if loss.item() > self.max_hard_loss:
                loss = torch.clamp(loss, max=self.max_hard_loss)
            
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 平均损失
        loss = torch.stack(losses).mean()
        
        # 记录训练历史
        if self.training:
            self._update_training_history(loss, semi_hard_count, hard_count)
        
        return loss
    
    def _update_training_history(self, loss: torch.Tensor, 
                                semi_hard_count: int, hard_count: int):
        """更新训练历史"""
        self.training_history['loss'].append(loss.item())
        
        # 获取当前参数
        miner_stats = self.miner.get_statistics()
        memory_stats = self.memory_bank.get_statistics()
        
        self.training_history['margin'].append(miner_stats['current_margin'])
        self.training_history['threshold'].append(miner_stats['current_threshold'])
        
        # 计算半硬样本比例
        total_count = semi_hard_count + hard_count
        if total_count > 0:
            semi_hard_ratio = semi_hard_count / total_count
        else:
            semi_hard_ratio = 0.0
        
        self.training_history['semi_hard_ratio'].append(semi_hard_ratio)
        self.training_history['memory_utilization'].append(memory_stats['memory_utilization'])
    
    def get_training_statistics(self) -> Dict:
        """获取训练统计信息"""
        miner_stats = self.miner.get_statistics()
        memory_stats = self.memory_bank.get_statistics()
        
        stats = {
            'miner': miner_stats,
            'memory': memory_stats,
            'training': {
                'avg_loss': np.mean(self.training_history['loss'][-100:]) if self.training_history['loss'] else 0,
                'current_margin': self.training_history['margin'][-1] if self.training_history['margin'] else self.config.base_margin,
                'current_threshold': self.training_history['threshold'][-1] if self.training_history['threshold'] else self.config.similarity_threshold,
                'semi_hard_ratio': self.training_history['semi_hard_ratio'][-1] if self.training_history['semi_hard_ratio'] else 0,
                'memory_utilization': self.training_history['memory_utilization'][-1] if self.training_history['memory_utilization'] else 0
            }
        }
        
        return stats
    
    def reset_memory(self):
        """重置记忆库"""
        self.memory_bank.clear()
        
    def set_training_mode(self, training: bool):
        """设置训练模式"""
        self.training = training
        if not training:
            self.memory_bank.update_strategy = 'readonly'  # 评估模式下只读                        