import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class PCGrad:
    """
    PCGrad: 通过梯度投影解决多任务学习的梯度冲突
    论文: "Gradient Surgery for Multi-Task Learning" (ICLR 2020)
    """
    
    def __init__(self, 
                 reduction: str = 'mean',
                 normalize: bool = False,
                 eps: float = 1e-8):
        """
        Args:
            reduction: 梯度聚合方式 ('mean', 'sum', 'weighted')
            normalize: 是否在投影前归一化梯度
            eps: 数值稳定性常数
        """
        assert reduction in ['mean', 'sum', 'weighted']
        self.reduction = reduction
        self.normalize = normalize
        self.eps = eps
        
        # 记录统计信息
        self.stats = {
            'conflict_rate': [],
            'cosine_similarities': [],
            'gradient_norms': [],
            'projection_count': [],
        }
    
    def apply(self, task_gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        应用PCGrad算法
        
        Args:
            task_gradients: 各任务的梯度列表
                [
                    [param1_grad, param2_grad, ...],  # 任务1
                    [param1_grad, param2_grad, ...],  # 任务2
                    ...
                ]
        
        Returns:
            处理后的聚合梯度列表 [param1_grad, param2_grad, ...]
        """
        num_tasks = len(task_gradients)
        
        if num_tasks == 1:
            return task_gradients[0]
        
        # 将每个任务的梯度展平为向量
        flat_gradients = []
        original_shapes = []
        
        for task_grad in task_gradients:
            flat_grad = []
            shapes = []
            
            for grad in task_grad:
                if grad is not None:
                    shapes.append(grad.shape)
                    flat_grad.append(grad.flatten())
                else:
                    shapes.append(None)
                    flat_grad.append(torch.tensor([], device=task_grad[0].device))
            
            flat_gradients.append(torch.cat(flat_grad))
            original_shapes.append(shapes)
        
        # 应用PCGrad投影
        processed_grads = self._pcgrad_projection(flat_gradients)
        
        # 将处理后的梯度恢复为原始形状
        return self._unflatten_gradients(processed_grads, original_shapes[0])
    
    def _pcgrad_projection(self, flat_gradients: List[torch.Tensor]) -> torch.Tensor:
        """PCGrad核心投影算法"""
        num_tasks = len(flat_gradients)
        processed_grads = [grad.clone() for grad in flat_gradients]
        
        conflict_count = 0
        total_pairs = 0
        cosine_sims = []
        
        # 对每对任务应用PCGrad
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                
                total_pairs += 1
                g_i = processed_grads[i]
                g_j = flat_gradients[j]
                
                # 计算梯度点积
                dot_product = torch.dot(g_i, g_j)
                
                # 计算余弦相似度
                norm_i = torch.norm(g_i) + self.eps
                norm_j = torch.norm(g_j) + self.eps
                cos_sim = dot_product / (norm_i * norm_j)
                cosine_sims.append(cos_sim.item())
                
                # 如果梯度冲突（点积 < 0）
                if dot_product < 0:
                    conflict_count += 1
                    
                    # 投影：g_i = g_i - (g_i·g_j / ||g_j||²) * g_j
                    g_j_norm_sq = torch.dot(g_j, g_j) + self.eps
                    proj_coef = dot_product / g_j_norm_sq
                    
                    # 应用投影
                    processed_grads[i] = g_i - proj_coef * g_j
        
        # 记录统计信息
        conflict_rate = conflict_count / max(total_pairs, 1)
        self.stats['conflict_rate'].append(conflict_rate)
        self.stats['cosine_similarities'].append(cosine_sims)
        self.stats['projection_count'].append(conflict_count)
        
        # 聚合处理后的梯度
        if self.reduction == 'mean':
            aggregated = torch.stack(processed_grads).mean(dim=0)
        elif self.reduction == 'sum':
            aggregated = torch.stack(processed_grads).sum(dim=0)
        else:  # 'weighted'
            # 根据梯度范数加权
            weights = []
            for grad in processed_grads:
                norm = torch.norm(grad) + self.eps
                weights.append(1.0 / norm)
            
            weights = torch.tensor(weights, device=processed_grads[0].device)
            weights = weights / weights.sum()
            
            aggregated = torch.stack([w * g for w, g in zip(weights, processed_grads)]).sum(dim=0)
        
        return aggregated
    
    def _unflatten_gradients(self, 
                           flat_gradient: torch.Tensor, 
                           original_shapes: List[Optional[torch.Size]]) -> List[torch.Tensor]:
        """将展平的梯度恢复为原始形状"""
        unflattened = []
        idx = 0
        
        for shape in original_shapes:
            if shape is None:
                unflattened.append(None)
            else:
                size = np.prod(shape).item()
                grad = flat_gradient[idx:idx+size].view(shape)
                unflattened.append(grad)
                idx += size
        
        return unflattened
    
    def get_statistics(self) -> Dict:
        """获取PCGrad统计信息"""
        if not self.stats['conflict_rate']:
            return {}
        
        return {
            'avg_conflict_rate': np.mean(self.stats['conflict_rate']),
            'max_conflict_rate': np.max(self.stats['conflict_rate']),
            'min_conflict_rate': np.min(self.stats['conflict_rate']),
            'avg_cosine_similarity': np.mean([np.mean(sims) for sims in self.stats['cosine_similarities']]),
            'total_projections': sum(self.stats['projection_count']),
            'conflict_rate_history': self.stats['conflict_rate'],
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'conflict_rate': [],
            'cosine_similarities': [],
            'gradient_norms': [],
            'projection_count': [],
        }

class EnhancedPCGrad(PCGrad):
    """
    增强版PCGrad：支持多种投影策略和优化
    """
    
    def __init__(self,
                 projection_type: str = 'pcgrad',
                 reduction: str = 'mean',
                 soft_threshold: float = 0.0,
                 momentum: float = 0.0,
                 **kwargs):
        """
        Args:
            projection_type: 投影策略
                'pcgrad': 原始PCGrad
                'pcgrad_sym': 对称PCGrad（双向投影）
                'mgda': 多梯度下降算法
                'cagrad': 冲突避免梯度下降
            soft_threshold: 软阈值（当余弦相似度低于此值时进行投影）
            momentum: 动量项，平滑梯度变化
        """
        super().__init__(reduction, **kwargs)
        
        assert projection_type in ['pcgrad', 'pcgrad_sym', 'mgda', 'cagrad']
        self.projection_type = projection_type
        self.soft_threshold = soft_threshold
        self.momentum = momentum
        
        # 动量缓冲区
        self.momentum_buffer = None
        
    def _pcgrad_projection(self, flat_gradients: List[torch.Tensor]) -> torch.Tensor:
        """增强的投影算法，支持多种策略"""
        num_tasks = len(flat_gradients)
        
        if self.projection_type == 'pcgrad':
            return self._original_pcgrad(flat_gradients)
        elif self.projection_type == 'pcgrad_sym':
            return self._symmetric_pcgrad(flat_gradients)
        elif self.projection_type == 'mgda':
            return self._mgda_projection(flat_gradients)
        elif self.projection_type == 'cagrad':
            return self._cagrad_projection(flat_gradients)
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
    
    def _original_pcgrad(self, flat_gradients: List[torch.Tensor]) -> torch.Tensor:
        """原始PCGrad算法"""
        num_tasks = len(flat_gradients)
        processed_grads = [grad.clone() for grad in flat_gradients]
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue
                
                g_i = processed_grads[i]
                g_j = flat_gradients[j]
                
                # 计算点积和余弦相似度
                dot_product = torch.dot(g_i, g_j)
                
                if self.normalize:
                    norm_i = torch.norm(g_i) + self.eps
                    norm_j = torch.norm(g_j) + self.eps
                    cos_sim = dot_product / (norm_i * norm_j)
                else:
                    cos_sim = dot_product / (torch.norm(g_j) ** 2 + self.eps)
                
                # 判断是否进行投影（可配置软阈值）
                if dot_product < self.soft_threshold:
                    g_j_norm_sq = torch.dot(g_j, g_j) + self.eps
                    proj_coef = dot_product / g_j_norm_sq
                    processed_grads[i] = g_i - proj_coef * g_j
        
        # 应用动量（如果启用）
        if self.momentum > 0:
            aggregated = self._apply_momentum(processed_grads)
        else:
            aggregated = self._aggregate_gradients(processed_grads)
        
        return aggregated
    
    def _symmetric_pcgrad(self, flat_gradients: List[torch.Tradient]) -> torch.Tensor:
        """对称PCGrad：对两个任务同时进行投影"""
        num_tasks = len(flat_gradients)
        processed_grads = [grad.clone() for grad in flat_gradients]
        
        for i in range(num_tasks):
            for j in range(i+1, num_tasks):
                g_i = processed_grads[i]
                g_j = processed_grads[j]
                
                dot_product = torch.dot(g_i, g_j)
                
                if dot_product < self.soft_threshold:
                    g_i_norm_sq = torch.dot(g_i, g_i) + self.eps
                    g_j_norm_sq = torch.dot(g_j, g_j) + self.eps
                    
                    # 对称投影：两个梯度都投影
                    proj_coef_ij = dot_product / g_j_norm_sq
                    proj_coef_ji = dot_product / g_i_norm_sq
                    
                    processed_grads[i] = g_i - proj_coef_ij * g_j
                    processed_grads[j] = g_j - proj_coef_ji * g_i
        
        return self._aggregate_gradients(processed_grads)
    
    def _mgda_projection(self, flat_gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        MGDA（Multiple Gradient Descent Algorithm）投影
        寻找帕累托最优解
        """
        num_tasks = len(flat_gradients)
        
        # 构建梯度矩阵 G ∈ R^{num_tasks × dim}
        G = torch.stack(flat_gradients)  # [num_tasks, dim]
        
        # 使用Frank-Wolfe算法寻找最优凸组合系数
        # 目标：min_α ||Σ α_i g_i||^2, s.t. Σ α_i = 1, α_i ≥ 0
        
        # 初始化均匀权重
        alpha = torch.ones(num_tasks, device=G.device) / num_tasks
        
        # Frank-Wolfe迭代
        for t in range(10):  # 通常10次迭代足够
            # 计算当前聚合梯度
            aggregated = torch.matmul(alpha, G)
            
            # 找到最小点积的任务（最冲突的任务）
            dot_products = torch.matmul(G, aggregated)
            min_idx = torch.argmin(dot_products)
            
            # 计算步长
            gamma = 2.0 / (t + 2.0)
            
            # 更新alpha
            new_alpha = alpha * (1 - gamma)
            new_alpha[min_idx] += gamma
            
            # 投影到单纯形（确保非负且和为1）
            new_alpha = torch.clamp(new_alpha, min=0)
            new_alpha = new_alpha / new_alpha.sum()
            
            alpha = new_alpha
        
        # 计算最终聚合梯度
        aggregated = torch.matmul(alpha, G)
        
        # 记录alpha权重
        if 'mgda_weights' not in self.stats:
            self.stats['mgda_weights'] = []
        self.stats['mgda_weights'].append(alpha.cpu().numpy())
        
        return aggregated
    
    def _cagrad_projection(self, 
                          flat_gradients: List[torch.Tensor], 
                          c: float = 0.5) -> torch.Tensor:
        """
        CAGrad（Conflict-Averse Gradient Descent）投影
        """
        num_tasks = len(flat_gradients)
        
        # 计算平均梯度
        avg_grad = torch.stack(flat_gradients).mean(dim=0)
        
        # 归一化梯度方向
        grad_directions = []
        for g in flat_gradients:
            norm = torch.norm(g)
            if norm > self.eps:
                grad_directions.append(g / norm)
            else:
                grad_directions.append(g)
        
        # 构建方向矩阵
        D = torch.stack(grad_directions)  # [num_tasks, dim]
        
        # 计算与平均方向最冲突的任务
        u = avg_grad / (torch.norm(avg_grad) + self.eps)
        conflicts = torch.matmul(D, u)
        min_conflict_idx = torch.argmin(conflicts)
        
        # 如果冲突超过阈值c，进行调整
        d_min = grad_directions[min_conflict_idx]
        if torch.dot(u, d_min) < c:
            # 调整方向：u_new = u + (c - u·d_min) * d_min
            u_new = u + (c - torch.dot(u, d_min)) * d_min
            u_new = u_new / (torch.norm(u_new) + self.eps)
            
            # 缩放回原始大小
            aggregated = torch.norm(avg_grad) * u_new
        else:
            aggregated = avg_grad
        
        return aggregated
    
    def _apply_momentum(self, processed_grads: List[torch.Tensor]) -> torch.Tensor:
        """应用动量到聚合梯度"""
        current_aggregated = self._aggregate_gradients(processed_grads)
        
        if self.momentum_buffer is None:
            self.momentum_buffer = current_aggregated
        else:
            self.momentum_buffer = self.momentum * self.momentum_buffer + \
                                 (1 - self.momentum) * current_aggregated
        
        return self.momentum_buffer
    
    def _aggregate_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """根据reduction策略聚合梯度"""
        if self.reduction == 'mean':
            return torch.stack(gradients).mean(dim=0)
        elif self.reduction == 'sum':
            return torch.stack(gradients).sum(dim=0)
        elif self.reduction == 'weighted':
            weights = []
            for grad in gradients:
                norm = torch.norm(grad) + self.eps
                weights.append(1.0 / norm)
            
            weights = torch.tensor(weights, device=gradients[0].device)
            weights = weights / weights.sum()
            
            return torch.stack([w * g for w, g in zip(weights, gradients)]).sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def get_enhanced_statistics(self) -> Dict:
        """获取增强统计信息"""
        base_stats = super().get_statistics()
        
        if 'mgda_weights' in self.stats and self.stats['mgda_weights']:
            mgda_weights = np.array(self.stats['mgda_weights'])
            base_stats['mgda_weight_mean'] = mgda_weights.mean(axis=0).tolist()
            base_stats['mgda_weight_std'] = mgda_weights.std(axis=0).tolist()
        
        base_stats.update({
            'projection_type': self.projection_type,
            'soft_threshold': self.soft_threshold,
            'momentum': self.momentum,
        })
        
        return base_stats