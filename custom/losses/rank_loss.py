import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------- 带注意力加权的 ApproxNDCG 损失 --------------------------
def attention_approx_ndcg_loss(scores, attention_weights, y_true, temperature=0.1, top_k=10):
    """
    对前 top_k 样本的损失赋予更高权重，强化对高排名样本的优化
    """
    # 1. 近似可微 NDCG 损失（基础部分）
    n_docs = len(scores)
    scores_exp = scores.unsqueeze(1)
    scores_tile = scores.unsqueeze(0)
    approx_rank = 1.0 + torch.sum(torch.sigmoid((scores_tile - scores_exp) / temperature), dim=1)
    
    gain = 2 ** y_true - 1
    discount = torch.log2(approx_rank + 1.0)
    approx_dcg = torch.sum(gain / discount)
    best_dcg = dcg_score(y_true, y_true, top_k)
    approx_ndcg = approx_dcg / (best_dcg + 1e-10)
    base_loss = 1.0 - approx_ndcg
    
    # 2. 损失加权：前 top_k 样本的损失权重更高
    sorted_idx = torch.argsort(scores, descending=True)
    loss_weights = torch.ones_like(scores)
    loss_weights[sorted_idx[:top_k]] = 2.0  # 前 top_k 样本损失权重翻倍
    loss_weights = loss_weights * attention_weights  # 结合注意力权重
    
    # 3. 加权损失
    weighted_loss = base_loss * loss_weights.mean()
    return weighted_loss

# --------------------------  核心工具函数（新增反向NDCG/反向lambda） --------------------------
def dcg_score(y_true, y_score, k=None):
    """计算 DCG (Discounted Cumulative Gain)"""
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]
    if k is not None:
        y_true = y_true[:k]
    gain = 2 ** y_true - 1
    discount = torch.log2(torch.arange(1, len(y_true) + 1, dtype=torch.float64) + 1).to(y_true.device)
    return torch.sum(gain / discount)

def ndcg_score(y_true, y_score, k=None):
    """计算 NDCG (Normalized DCG)"""
    best_dcg = dcg_score(y_true, y_true, k)
    if best_dcg == 0:
        return torch.tensor(0.0)
    return dcg_score(y_true, y_score, k) / best_dcg

def compute_delta_ndcg(y_true, y_score, i, j):
    """计算交换样本 i 和 j 后的 NDCG 变化量 |delta NDCG|"""
    # 原始 NDCG
    original_ndcg = ndcg_score(y_true, y_score)
    
    # 交换 i 和 j 的分数
    y_score_swapped = y_score.clone()
    y_score_swapped[i], y_score_swapped[j] = y_score_swapped[j], y_score_swapped[i]
    
    # 交换后的 NDCG
    swapped_ndcg = ndcg_score(y_true, y_score_swapped)
    
    # 返回绝对变化量
    return torch.abs(original_ndcg - swapped_ndcg)