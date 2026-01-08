import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------- 1. 核心工具函数（新增反向NDCG/反向lambda） --------------------------
def dcg_score(y_true, y_score, k=3):
    """计算DCG@k（折损累积增益）"""
    order = np.argsort(y_score)[::-1]  # 按预测分数降序
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=3):
    """计算正向NDCG@k（优→劣）"""
    best_dcg = dcg_score(y_true, y_true, k)
    if best_dcg == 0:
        return 0.0
    return dcg_score(y_true, y_score, k) / best_dcg

def reverse_ndcg_score(y_true, y_score, k=3):
    """计算反向NDCG@k（劣→优）：反转真实标签后计算NDCG"""
    # 反转标签：优→劣，劣→优（比如[5,3,1]→[1,3,5]）
    y_true_rev = np.max(y_true) - y_true + np.min(y_true)
    return ndcg_score(y_true_rev, y_score, k)

def compute_bidirectional_lambdas(y_true, y_score, k=3):
    """计算双向lambda值：正向lambda + 反向lambda"""
    n = len(y_true)
    # 1. 计算正向lambda（优→劣）
    lambdas_forward = np.zeros(n)
    ndcg_forward_base = ndcg_score(y_true, y_score, k)
    for i in range(n):
        for j in range(n):
            if y_true[i] == y_true[j]:
                continue
            # 交换i和j的预测分数
            y_score_swap = y_score.copy()
            y_score_swap[i], y_score_swap[j] = y_score_swap[j], y_score_swap[i]
            ndcg_swap = ndcg_score(y_true, y_score_swap, k)
            delta_ndcg = ndcg_swap - ndcg_forward_base
            s_ij = 1 if y_true[i] > y_true[j] else -1
            lambdas_forward[i] += 0.5 * delta_ndcg * s_ij
            lambdas_forward[j] -= 0.5 * delta_ndcg * s_ij
    
    # 2. 计算反向lambda（劣→优）
    lambdas_backward = np.zeros(n)
    ndcg_backward_base = reverse_ndcg_score(y_true, y_score, k)
    y_true_rev = np.max(y_true) - y_true + np.min(y_true)  # 反转真实标签
    for i in range(n):
        for j in range(n):
            if y_true_rev[i] == y_true_rev[j]:
                continue
            # 交换i和j的预测分数
            y_score_swap = y_score.copy()
            y_score_swap[i], y_score_swap[j] = y_score_swap[j], y_score_swap[i]
            ndcg_swap = reverse_ndcg_score(y_true, y_score_swap, k)
            delta_ndcg = ndcg_swap - ndcg_backward_base
            s_ij_rev = 1 if y_true_rev[i] > y_true_rev[j] else -1
            lambdas_backward[i] += 0.5 * delta_ndcg * s_ij_rev
            lambdas_backward[j] -= 0.5 * delta_ndcg * s_ij_rev
    
    # 3. 双向lambda：正向 + 反向（加权平衡，可调整alpha）
    alpha = 0.5  # 正向/反向权重，总和为1
    lambdas_bi = alpha * lambdas_forward + (1 - alpha) * lambdas_backward
    return lambdas_bi

# -------------------------- 2. 定义双向LambdaRank Loss类 --------------------------
class BidirectionalLambdaRankLoss(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k  # 关注前k名的排序质量
    
    def forward(self, pred_scores, y_true):
        """
        计算双向LambdaRank损失
        :param pred_scores: 模型预测分数 (batch_size, n_candidates)
        :param y_true: 真实优先级标签 (batch_size, n_candidates)
        :return: 批次平均损失
        """
        batch_loss = 0.0
        batch_size = pred_scores.shape[0]
        
        for i in range(batch_size):
            # 转numpy计算双向lambda
            scores_np = pred_scores[i].detach().cpu().numpy()
            true_np = y_true[i].cpu().numpy()
            
            # 计算双向lambda值
            lambdas_bi = compute_bidirectional_lambdas(true_np, scores_np, self.k)
            lambdas_bi = torch.tensor(lambdas_bi, dtype=torch.float32, device=pred_scores.device)
            
            # 双向损失计算：lambda值 * 对数损失项（约束排序方向）
            loss = torch.sum(lambdas_bi * torch.log(1 + torch.exp(-pred_scores[i])))
            batch_loss += loss
        
        return batch_loss / batch_size

# -------------------------- 3. 排序模型（复用之前的MLP） --------------------------
class RankModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        batch_size, n_candidates, feature_dim = x.shape
        x_flat = x.reshape(-1, feature_dim)
        scores_flat = self.mlp(x_flat)
        scores = scores_flat.reshape(batch_size, n_candidates)
        return scores.squeeze(-1)

# -------------------------- 4. 训练与测试（对比单向/双向效果） --------------------------
if __name__ == "__main__":
    # 超参数
    feature_dim = 8
    n_candidates = 10
    batch_size = 4
    k = 3
    epochs = 100
    
    # 初始化模型、双向损失函数、优化器
    model = RankModel(feature_dim)
    criterion_bi = BidirectionalLambdaRankLoss(k=k)  # 双向损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练数据
    X_train = torch.randn(batch_size, n_candidates, feature_dim)
    y_train = torch.tensor([
        [5, 3, 4, 2, 1, 0, 4, 1, 5, 2],
        [4, 5, 2, 3, 1, 0, 3, 2, 4, 1],
        [3, 4, 5, 2, 1, 0, 2, 1, 3, 0],
        [2, 3, 4, 5, 1, 0, 1, 0, 2, 1]
    ], dtype=torch.float32)
    
    # 训练过程
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_scores = model(X_train)
        loss = criterion_bi(pred_scores, y_train)
        loss.backward()
        optimizer.step()
        
        # 每20轮打印结果（计算正向NDCG）
        if (epoch + 1) % 20 == 0:
            ndcg_avg = 0.0
            for i in range(batch_size):
                ndcg_avg += ndcg_score(y_train[i].numpy(), pred_scores[i].detach().numpy(), k=k)
            ndcg_avg /= batch_size
            print(f"Epoch {epoch+1:3d} | Bi-Loss: {loss.item():.4f} | NDCG@3: {ndcg_avg:.4f}")
    
    # 最终筛选前3名
    model.eval()
    with torch.no_grad():
        final_pred = model(X_train)
        for i in range(batch_size):
            top3_indices = torch.argsort(final_pred[i], descending=True)[:k]
            print(f"\n样本{i+1}前3名索引：{top3_indices.tolist()}")
            print(f"对应真实优先级：{y_train[i][top3_indices].tolist()}")