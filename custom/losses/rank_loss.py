import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------- 1. 核心工具函数：计算NDCG（排序质量评价指标） --------------------------
def dcg_score(y_true, y_score, k=3):
    """计算DCG@k（折损累积增益）"""
    # 按预测分数降序排序真实标签
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    
    # DCG公式：DCG@k = sum( (2^rel_i - 1) / log2(i+1) )
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)  # i从1开始，所以+2
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=3):
    """计算NDCG@k（归一化DCG）"""
    # 理想DCG（真实标签降序排列）
    best_dcg = dcg_score(y_true, y_true, k)
    if best_dcg == 0:
        return 0.0
    # 实际DCG / 理想DCG
    return dcg_score(y_true, y_score, k) / best_dcg

def compute_lambdas(y_true, y_score, k=3):
    """计算每个候选者的lambda值（核心：反映排序错误对NDCG的影响）"""
    n = len(y_true)
    lambdas = np.zeros(n)
    ndcg_base = ndcg_score(y_true, y_score, k)
    
    # 遍历每一对候选者(i,j)，计算交换i和j后的NDCG变化
    for i in range(n):
        for j in range(n):
            if y_true[i] == y_true[j]:
                continue  # 真实标签相同，交换无意义
            
            # 交换i和j的预测分数
            y_score_swap = y_score.copy()
            y_score_swap[i], y_score_swap[j] = y_score_swap[j], y_score_swap[i]
            ndcg_swap = ndcg_score(y_true, y_score_swap, k)
            
            # 计算lambda值：NDCG变化量 * 符号（反映i和j的优劣关系）
            delta_ndcg = ndcg_swap - ndcg_base
            s_ij = 1 if y_true[i] > y_true[j] else -1  # i比j优则为1，否则为-1
            lambdas[i] += 0.5 * delta_ndcg * s_ij
            lambdas[j] -= 0.5 * delta_ndcg * s_ij
    
    return lambdas

# -------------------------- 2. 定义LambdaRank Loss类 --------------------------
class LambdaRankLoss(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k  # 关注前k名的排序质量
    
    def forward(self, pred_scores, y_true):
        """
        计算LambdaRank损失
        :param pred_scores: 模型预测分数 (batch_size, n_candidates)
        :param y_true: 真实标签（反映候选者优先级）(batch_size, n_candidates)
        :return: 批次的平均损失
        """
        batch_loss = 0.0
        batch_size = pred_scores.shape[0]
        
        for i in range(batch_size):
            # 转numpy计算lambda值（单条样本）
            scores_np = pred_scores[i].detach().cpu().numpy()
            true_np = y_true[i].cpu().numpy()
            
            # 计算当前样本的lambda值
            lambdas = compute_lambdas(true_np, scores_np, self.k)
            lambdas = torch.tensor(lambdas, dtype=torch.float32, device=pred_scores.device)
            
            # LambdaRank损失核心：lambda值 * （sigmoid(score_j - score_i)）
            # 简化实现：loss = sum( lambda_i * log(1 + exp(-s_ij*(score_i - score_j))) )
            # 这里用更易实现的形式：基于lambda值和预测分数的交叉熵类损失
            loss = torch.sum(lambdas * torch.log(1 + torch.exp(-pred_scores[i])))
            batch_loss += loss
        
        return batch_loss / batch_size