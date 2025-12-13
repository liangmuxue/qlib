import numpy as np
import torch
from typing import List, Dict,Union
from .enhence_grad import EnhancedPCGrad

class PCGradOptimizer:
    """
    PCGrad优化器包装器：将PCGrad集成到标准PyTorch优化器中
    """
    
    def __init__(self, 
                 params,
                 pcgrad_config: Dict,
                 base_optimizer: str = 'Adam',
                 base_optimizer_config: Dict = None):
        """
        Args:
            params: 模型参数
            pcgrad_config: PCGrad配置
            base_optimizer: 基础优化器 ('Adam', 'SGD', 'AdamW')
            base_optimizer_config: 基础优化器配置
        """
        self.pcgrad = EnhancedPCGrad(**pcgrad_config)
        
        # 创建基础优化器
        if base_optimizer_config is None:
            base_optimizer_config = {'lr': 1e-3}
        
        if base_optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params, **base_optimizer_config)
        elif base_optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params, **base_optimizer_config)
        elif base_optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, **base_optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer: {base_optimizer}")
        
        # 存储任务损失函数
        self.task_losses = []
        self.task_names = []
        
        # 训练历史
        self.history = {
            'losses': [],
            'gradient_stats': [],
        }
    
    def zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()
        self.task_losses = []
    
    def set_task_names(self, task_names: List[str]):
        """设置任务名称"""
        self.task_names = task_names
    
    def backward(self, losses: Union[torch.Tensor, List[torch.Tensor]], 
                task_indices: List[int] = None):
        """
        计算梯度并应用PCGrad
        
        Args:
            losses: 任务损失列表或单个损失张量
            task_indices: 任务索引列表（如果为None，则假设每个损失对应一个任务）
        """
        if isinstance(losses, torch.Tensor):
            losses = [losses]
        
        # 存储任务损失
        self.task_losses = losses
        
        if task_indices is None:
            task_indices = list(range(len(losses)))
        
        # 为每个任务计算梯度
        task_gradients = []
        
        for i, (loss, task_idx) in enumerate(zip(losses, task_indices)):
            # 清空梯度
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param.grad.zero_()
            
            # 计算当前任务的梯度
            loss.backward(retain_graph=(i < len(losses) - 1))
            
            # 收集梯度
            grads = []
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    grads.append(param.grad.clone())
                else:
                    grads.append(None)
            
            task_gradients.append(grads)
        
        # 应用PCGrad
        aggregated_gradients = self.pcgrad.apply(task_gradients)
        
        # 将聚合梯度赋回模型参数
        for param, grad in zip(self.optimizer.param_groups[0]['params'], aggregated_gradients):
            if grad is not None:
                param.grad = grad
        
        # 记录统计信息
        self._record_stats(losses)
    
    def step(self):
        """执行优化步骤"""
        self.optimizer.step()
    
    def _record_stats(self, losses: List[torch.Tensor]):
        """记录统计信息"""
        loss_dict = {}
        for i, loss in enumerate(losses):
            task_name = self.task_names[i] if i < len(self.task_names) else f'task_{i}'
            loss_dict[task_name] = loss.item()
        
        grad_stats = self.pcgrad.get_enhanced_statistics()
        
        self.history['losses'].append(loss_dict)
        self.history['gradient_stats'].append(grad_stats)
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        if not self.history['losses']:
            return {}
        
        # 计算平均损失
        avg_losses = {}
        for task_name in self.history['losses'][0].keys():
            task_losses = [step[task_name] for step in self.history['losses']]
            avg_losses[task_name] = np.mean(task_losses)
        
        # 获取PCGrad统计
        pcgrad_stats = self.pcgrad.get_enhanced_statistics()
        
        return {
            'avg_losses': avg_losses,
            'pcgrad_stats': pcgrad_stats,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        if not self.history['losses']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 任务损失曲线
        for task_name in self.history['losses'][0].keys():
            task_losses = [step[task_name] for step in self.history['losses']]
            axes[0, 0].plot(task_losses, label=task_name)
        
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Task Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 梯度冲突率
        if 'conflict_rate_history' in self.pcgrad.stats:
            axes[0, 1].plot(self.pcgrad.stats['conflict_rate_history'])
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Conflict Rate')
            axes[0, 1].set_title('Gradient Conflict Rate')
            axes[0, 1].grid(True)
        
        # 余弦相似度分布
        if self.pcgrad.stats['cosine_similarities']:
            all_cos_sims = []
            for sims in self.pcgrad.stats['cosine_similarities']:
                all_cos_sims.extend(sims)
            
            axes[0, 2].hist(all_cos_sims, bins=50, alpha=0.7)
            axes[0, 2].set_xlabel('Cosine Similarity')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Gradient Cosine Similarity Distribution')
            axes[0, 2].axvline(x=0, color='r', linestyle='--', label='Conflict Threshold')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # MGDA权重（如果使用）
        if 'mgda_weights' in self.pcgrad.stats and self.pcgrad.stats['mgda_weights']:
            mgda_weights = np.array(self.pcgrad.stats['mgda_weights'])
            
            for i in range(mgda_weights.shape[1]):
                axes[1, 0].plot(mgda_weights[:, i], label=f'Task {i}')
            
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('MGDA Weight')
            axes[1, 0].set_title('MGDA Task Weights')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 投影次数
        if 'projection_count' in self.pcgrad.stats and self.pcgrad.stats['projection_count']:
            axes[1, 1].plot(self.pcgrad.stats['projection_count'])
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Projection Count')
            axes[1, 1].set_title('PCGrad Projections per Step')
            axes[1, 1].grid(True)
        
        # 损失与冲突率散点图
        if len(self.history['losses']) > 10 and 'conflict_rate_history' in self.pcgrad.stats:
            primary_losses = [step.get('primary', step.get('task_0', 0)) 
                            for step in self.history['losses']]
            conflict_rates = self.pcgrad.stats['conflict_rate_history']
            
            min_len = min(len(primary_losses), len(conflict_rates))
            axes[1, 2].scatter(primary_losses[:min_len], conflict_rates[:min_len], alpha=0.5)
            axes[1, 2].set_xlabel('Primary Task Loss')
            axes[1, 2].set_ylabel('Conflict Rate')
            axes[1, 2].set_title('Loss vs Conflict Rate')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()