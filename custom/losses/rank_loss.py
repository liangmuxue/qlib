import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from collections import defaultdict
from scipy.stats import rankdata
import warnings

class GlobalRankingLoss:
    """全局排名损失基类"""
    
    def __init__(self,
                 reduction: str = 'mean',
                 temperature: float = 1.0):
        """
        Args:
            reduction: 损失归约方式 ['mean', 'sum', 'none']
            temperature: 温度参数，控制分布的平滑度
        """
        self.reduction = reduction
        self.temperature = temperature
        
        # 全局排名统计
        self.global_ranks = {}
        self.rank_history = defaultdict(list)
        
    def compute_global_ranks(self,
                           scores: torch.Tensor,
                           global_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算全局排名
        Args:
            scores: 当前批次得分 [batch_size]
            global_scores: 全局得分池 [N]，如果为None则使用历史累积
        Returns:
            ranks: 全局排名 [batch_size]
        """
        if global_scores is None:
            # 如果没有提供全局得分池，使用历史累积
            if not hasattr(self, 'global_score_pool') or len(self.global_score_pool) == 0:
                # 首次调用，返回批次内排名
                return self._batch_ranks(scores)
            
            # 合并历史得分和当前得分
            all_scores = torch.cat([self.global_score_pool, scores.detach()])
        else:
            all_scores = torch.cat([global_scores, scores])
        
        # 计算全局排名
        sorted_indices = torch.argsort(all_scores, descending=True)
        ranks = torch.zeros_like(scores, dtype=torch.float32)
        
        for i, score in enumerate(scores):
            # 找到当前得分在全局中的位置
            rank = (sorted_indices == i + len(all_scores) - len(scores)).nonzero(as_tuple=True)[0].item()
            ranks[i] = rank + 1  # 从1开始计数
        
        return ranks
    
    def _batch_ranks(self, scores: torch.Tensor) -> torch.Tensor:
        """计算批次内排名"""
        sorted_indices = torch.argsort(scores, descending=True)
        ranks = torch.zeros_like(scores, dtype=torch.float32)
        
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
        
        return ranks
    
    def update_global_pool(self,
                          scores: torch.Tensor,
                          max_size: int = 10000):
        """更新全局得分池"""
        if not hasattr(self, 'global_score_pool'):
            self.global_score_pool = torch.tensor([], dtype=scores.dtype)
        
        # 分离梯度
        scores_detached = scores.detach().cpu()
        
        # 添加到池中
        self.global_score_pool = torch.cat([self.global_score_pool, scores_detached])
        
        # 限制池的大小
        if len(self.global_score_pool) > max_size:
            self.global_score_pool = self.global_score_pool[-max_size:]
    
    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """应用损失归约"""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class GlobalLambdaRankLoss(GlobalRankingLoss):
    """
    全局LambdaRank损失
    通过梯度加权直接优化排序指标
    """
    
    def __init__(self,
                 sigma: float = 1.0,
                 cost_function: str = 'ndcg',
                 **kwargs):
        super().__init__(**kwargs)
        
        self.sigma = sigma
        self.cost_function = cost_function  # 'ndcg', 'map', 'mrr'
        
    def forward(self,
                scores: torch.Tensor,
                labels: torch.Tensor,
                global_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        LambdaRank损失
        
        公式: λ_{ij} = |ΔNDCG_{ij}| / (1 + exp(σ * (s_i - s_j)))
        """
        batch_size = len(scores)
        
        # 计算pairwise的λ梯度
        lambda_grads = torch.zeros_like(scores)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and labels[i] != labels[j]:
                    # 计算交换i和j位置对NDCG的影响
                    delta_ndcg = self._compute_delta_ndcg(i, j, scores, labels)
                    
                    # 计算sigmoid梯度
                    score_diff = scores[i] - scores[j]
                    sigmoid_grad = -self.sigma / (1 + torch.exp(self.sigma * score_diff))
                    
                    # 如果标签i > 标签j，则i应该排名更高
                    if labels[i] > labels[j]:
                        lambda_grads[i] += delta_ndcg * sigmoid_grad
                        lambda_grads[j] -= delta_ndcg * sigmoid_grad
                    else:
                        lambda_grads[i] -= delta_ndcg * sigmoid_grad
                        lambda_grads[j] += delta_ndcg * sigmoid_grad
        
        # 计算损失（使用lambda梯度加权）
        loss = torch.sum(lambda_grads * scores)
        
        # 添加全局正则化
        if global_scores is not None:
            global_reg = self._global_lambda_regularization(scores, global_scores, labels)
            loss += 0.1 * global_reg
        
        # 更新全局池
        self.update_global_pool(scores)
        
        return self._apply_reduction(loss)
    
    def _compute_delta_ndcg(self,
                           i: int,
                           j: int,
                           scores: torch.Tensor,
                           labels: torch.Tensor,
                           topk: int = 10) -> float:
        """
        计算交换i和j位置对NDCG的影响
        """
        batch_size = len(scores)
        
        # 当前的排序顺序
        current_order = torch.argsort(scores, descending=True)
        
        # 创建交换后的顺序
        swapped_scores = scores.clone()
        swapped_scores[i], swapped_scores[j] = swapped_scores[j], swapped_scores[i]
        swapped_order = torch.argsort(swapped_scores, descending=True)
        
        # 计算当前NDCG
        current_ndcg = self._compute_truncated_ndcg(current_order, labels, topk)
        
        # 计算交换后的NDCG
        swapped_ndcg = self._compute_truncated_ndcg(swapped_order, labels, topk)
        
        return abs(swapped_ndcg - current_ndcg)
    
    def _compute_truncated_ndcg(self,
                               order: torch.Tensor,
                               labels: torch.Tensor,
                               topk: int) -> float:
        """计算top-k NDCG"""
        dcg = 0
        for pos, idx in enumerate(order[:topk]):
            gain = (2 ** labels[idx].float() - 1)
            discount = torch.log2(torch.tensor(pos + 2.0))
            dcg += gain / discount
        
        # 理想DCG
        ideal_order = torch.argsort(labels, descending=True)
        idcg = 0
        for pos, idx in enumerate(ideal_order[:topk]):
            gain = (2 ** labels[idx].float() - 1)
            discount = torch.log2(torch.tensor(pos + 2.0))
            idcg += gain / discount
        
        return dcg / idcg if idcg > 0 else 0
    
    def _global_lambda_regularization(self,
                                     scores: torch.Tensor,
                                     global_scores: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
        """
        全局Lambda正则化
        鼓励批次内的λ梯度与全局λ梯度一致
        """
        # 计算批次内λ梯度
        batch_lambda = self._compute_batch_lambda(scores, labels)
        
        # 计算全局λ梯度（使用全局池中的样本）
        if len(global_scores) > 0:
            # 采样全局样本
            n_samples = min(100, len(global_scores))
            indices = torch.randint(0, len(global_scores), (n_samples,))
            sampled_scores = global_scores[indices]
            
            # 为采样的全局样本生成伪标签（基于它们的排名）
            sampled_labels = torch.arange(n_samples, 0, -1, dtype=torch.float32) / n_samples
            
            # 计算全局λ梯度
            global_lambda = self._compute_batch_lambda(sampled_scores, sampled_labels)
            
            # 计算一致性损失
            consistency_loss = F.mse_loss(batch_lambda.mean(), global_lambda.mean())
            
            return consistency_loss
        
        return torch.tensor(0.0, device=scores.device)
    
    def _compute_batch_lambda(self,
                             scores: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
        """计算批次λ梯度"""
        batch_size = len(scores)
        lambda_grads = torch.zeros_like(scores)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and labels[i] != labels[j]:
                    delta_ndcg = self._compute_delta_ndcg(i, j, scores, labels)
                    score_diff = scores[i] - scores[j]
                    sigmoid_grad = -self.sigma / (1 + torch.exp(self.sigma * score_diff))
                    
                    if labels[i] > labels[j]:
                        lambda_grads[i] += delta_ndcg * sigmoid_grad
        
        return lambda_grads

class LambdaRankLoss(nn.Module):
    def __init__(self, k: int = None, max_rel: int = 5, lambda_scale: float = 10.0, temperature: float = 0.5):
        """
        优化后的LambdaRankLoss（解决梯度小问题）
        参数：
            k: 计算NDCG@k的k值
            max_rel: 最大相关度，避免2^rel数值爆炸
            lambda_scale: Lambda值的放大系数，提升梯度量级
            temperature: 温度系数，调整预测分数的分布（越小越集中）
        """
        super(LambdaRankLoss, self).__init__()
        self.k = k
        self.max_rel = max_rel
        self.lambda_scale = lambda_scale  # Lambda值放大系数
        self.temperature = temperature    # 温度系数

    def forward(self, pred_scores: torch.Tensor, rel_labels: torch.Tensor) -> torch.Tensor:
        """
        计算LambdaRank损失（提升梯度量级）
        参数：
            pred_scores: [batch_size, num_docs] 模型预测分数
            rel_labels: [batch_size, num_docs] 真实相关度标签
        返回：
            loss: 标量张量，损失值（梯度量级显著提升）
        """
        batch_size, num_docs = pred_scores.shape
        k = self.k if self.k is not None else num_docs
        k = min(k, num_docs)
        device = pred_scores.device

        # -------------------------- 1. 预处理：数值稳定性 + 温度系数 --------------------------
        # 裁剪相关度标签，避免数值爆炸
        rel_labels = torch.clamp(rel_labels, 0, self.max_rel)
        # 对预测分数应用温度系数，让分数分布更集中，缓解sigmoid梯度饱和
        pred_scores = pred_scores / self.temperature

        # -------------------------- 2. 计算IDCG@k（仅用于防止除以0，不做过度归一化） --------------------------
        _, ideal_rank_indices = torch.sort(rel_labels, dim=1, descending=True)
        rel_sorted_by_ideal = torch.gather(rel_labels, dim=1, index=ideal_rank_indices[:, :k])
        discount = 1.0 / torch.log2(torch.arange(2, k+2, device=device, dtype=torch.float32))
        idcg = torch.sum((torch.pow(2.0, rel_sorted_by_ideal) - 1) * discount, dim=1)
        idcg = torch.clamp(idcg, min=1e-10)  # 仅防止除以0
        idcg_expand = idcg.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

        # -------------------------- 3. 计算文档排名和折扣因子 --------------------------
        _, pred_rank_indices = torch.sort(pred_scores, dim=1, descending=True)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_docs)
        rank = torch.zeros_like(pred_scores, dtype=torch.float32)
        rank[batch_idx, pred_rank_indices] = torch.arange(1, num_docs+1, device=device, dtype=torch.float32).unsqueeze(0)
        discount_rank = 1.0 / torch.log2(rank + 1.0)  # [B, D]

        # -------------------------- 4. 计算Lambda值（放大系数 + 减少归一化） --------------------------
        # 扩展维度用于广播
        rel_i = rel_labels.unsqueeze(2)  # [B, D, 1]
        rel_j = rel_labels.unsqueeze(1)  # [B, 1, D]
        discount_i = discount_rank.unsqueeze(2)  # [B, D, 1]
        discount_j = discount_rank.unsqueeze(1)  # [B, 1, D]
        s_i = pred_scores.unsqueeze(2)  # [B, D, 1]
        s_j = pred_scores.unsqueeze(1)  # [B, 1, D]

        # 核心计算：Lambda_ij（乘以放大系数，提升量级）
        rel_diff = torch.pow(2.0, rel_i) - torch.pow(2.0, rel_j)
        discount_diff = discount_i - discount_j
        # 可选：改用IDCG的平方根归一化，减少量级缩小（或直接除以1，仅用IDCG防止除以0）
        lambda_ij = discount_diff * rel_diff / torch.sqrt(idcg_expand)
        lambda_ij = lambda_ij * self.lambda_scale  # 放大Lambda值

        # -------------------------- 5. 过滤有效文档对 --------------------------
        # 只保留：rel_i > rel_j（有排序意义）
        mask = (rel_i > rel_j).float()
        lambda_ij = lambda_ij * mask
        # 统计有效文档对的数量（避免除以0）
        valid_pairs = torch.clamp(torch.sum(mask), min=1.0)

        # -------------------------- 6. 替换Sigmoid：用交叉熵逻辑提升梯度量级 --------------------------
        # 用 -log(sigmoid(s_i - s_j)) 替代单纯的sigmoid，梯度更敏感
        # 等价于：BCE损失，标签为1（因为rel_i > rel_j，期望s_i > s_j）
        loss_term = -torch.abs(lambda_ij) * F.logsigmoid(s_i - s_j)
        # 也可选用：lambda_ij * torch.nn.functional.softplus(s_j - s_i)（更平缓）

        # -------------------------- 7. 损失归一化：除以有效文档对数量 --------------------------
        loss = torch.sum(loss_term) / valid_pairs

        return loss
    
class GlobalListwiseRankingLoss(GlobalRankingLoss):
    """全局Listwise排名损失"""
    
    def __init__(self,
                 topk: int = 10,
                 ndcg_weight: float = 1.0,
                 precision_weight: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.topk = topk
        self.ndcg_weight = ndcg_weight
        self.precision_weight = precision_weight
        
    def forward(self,
                scores: torch.Tensor,
                labels: torch.Tensor,
                global_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算全局listwise排名损失
        
        Args:
            scores: 预测得分 [batch_size]
            labels: 相关度标签 [batch_size]，数值越大相关性越高
            global_scores: 全局得分池
            
        Returns:
            loss: 组合损失
        """
        # 计算NDCG损失
        ndcg_loss = self.ndcg_loss(scores, labels, global_scores)
        
        # 计算Precision@k损失
        precision_loss = self.precision_loss(scores, labels)
        
        # 组合损失
        total_loss = (self.ndcg_weight * ndcg_loss + 
                     self.precision_weight * precision_loss)
        
        # 更新全局池
        self.update_global_pool(scores)
        
        return total_loss
    
    def ndcg_loss(self,
                  scores: torch.Tensor,
                  labels: torch.Tensor,
                  global_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        NDCG（归一化折损累积增益）损失
        
        NDCG = DCG / IDCG
        DCG = sum_i (2^rel_i - 1) / log2(i + 1)
        """
        batch_size = len(scores)
        
        # 按得分排序
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        
        # 计算DCG
        dcg = 0
        for i in range(min(self.topk, batch_size)):
            gain = (2 ** sorted_labels[i].float() - 1)
            discount = torch.log2(torch.tensor(i + 2.0))  # i+2因为log2(1)=0
            dcg += gain / discount
        
        # 按标签排序（理想情况）
        ideal_indices = torch.argsort(labels, descending=True)
        ideal_labels = labels[ideal_indices]
        
        # 计算IDCG
        idcg = 0
        for i in range(min(self.topk, batch_size)):
            gain = (2 ** ideal_labels[i].float() - 1)
            discount = torch.log2(torch.tensor(i + 2.0))
            idcg += gain / discount
        
        if idcg == 0:
            return torch.tensor(0.0, device=scores.device)
        
        # 计算NDCG
        ndcg = dcg / idcg
        
        # NDCG损失（1 - NDCG）
        ndcg_loss = 1 - ndcg
        
        # 添加全局一致性惩罚
        if global_scores is not None:
            global_consistency = self._global_consistency_penalty(scores, global_scores)
            ndcg_loss += 0.1 * global_consistency
        
        return ndcg_loss
    
    def precision_loss(self,
                      scores: torch.Tensor,
                      labels: torch.Tensor,
                      threshold: float = 0.5) -> torch.Tensor:
        """
        Precision@k损失
        """
        batch_size = len(scores)
        
        # 按得分排序
        sorted_indices = torch.argsort(scores, descending=True)
        
        # 计算top-k中的相关文档数
        relevant_count = 0
        for i in range(min(self.topk, batch_size)):
            if labels[sorted_indices[i]] >= threshold:
                relevant_count += 1
        
        # 计算precision
        precision = relevant_count / min(self.topk, batch_size)
        
        # Precision损失（1 - Precision）
        precision_loss = 1 - precision
        
        return precision_loss
    
    def _global_consistency_penalty(self,
                                   scores: torch.Tensor,
                                   global_scores: torch.Tensor) -> torch.Tensor:
        """
        全局一致性惩罚 - 鼓励批次排名与全局排名一致
        """
        # 计算批次排名
        batch_ranks = self._batch_ranks(scores)
        
        # 计算全局排名
        global_ranks = self.compute_global_ranks(scores, global_scores)
        
        # 计算排名差异
        rank_diff = torch.abs(batch_ranks - global_ranks)
        
        return rank_diff.mean()
