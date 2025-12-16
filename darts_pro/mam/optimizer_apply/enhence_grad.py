import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AdaptiveAuxiliaryAllocator:
    """
    自适应辅助任务分配器
    根据辅助任务对主任务的帮助程度动态调整
    """
    
    def __init__(self,
                 primary_tasks: List[int],
                 auxiliary_tasks: List[int],
                 device: torch.device):
        
        self.primary_tasks = primary_tasks
        self.auxiliary_tasks = auxiliary_tasks
        self.device = device
        
        # 辅助任务权重（可学习）
        self.auxiliary_weights = nn.ParameterDict()
        self._init_weights()
        
        # 历史信息
        self.history_losses = {task_idx: [] for task_idx in range(len(primary_tasks) + len(auxiliary_tasks))}
        
        self.info = {
            'auxiliary_weights': [],
            'helpfulness_scores': []
        }
    
    def _init_weights(self):
        """初始化辅助任务权重"""
        for task_idx in self.auxiliary_tasks:
            weight = nn.Parameter(torch.ones(1, device=self.device) * 0.5)  # 初始权重0.5
            self.auxiliary_weights[f'w_{task_idx}'] = weight
    
    def allocate(self,
                model: nn.Module,
                task_losses: List[torch.Tensor],
                optimizer: torch.optim.Optimizer) -> torch.Tensor:
        
        if optimizer is None:
            raise ValueError("需要优化器")
        
        # 记录损失
        for task_idx, loss in enumerate(task_losses):
            self.history_losses[task_idx].append(loss.item())
        
        # 计算辅助任务对主任务的帮助程度
        helpfulness = self._compute_helpfulness()
        
        # 计算所有任务梯度
        all_gradients = {}
        for task_idx in range(len(task_losses)):
            optimizer.zero_grad()
            task_losses[task_idx].backward(retain_graph=(task_idx != len(task_losses)-1))
            
            all_gradients[task_idx] = []
            for param in model.parameters():
                if param.grad is not None:
                    all_gradients[task_idx].append(param.grad.clone())
            
            if task_idx != len(task_losses)-1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
        
        # 自适应调整辅助任务梯度
        adjusted_gradients = {}
        
        for task_idx in range(len(task_losses)):
            if task_idx in self.auxiliary_tasks:
                # 获取当前辅助任务权重
                weight = torch.sigmoid(self.auxiliary_weights[f'w_{task_idx}'])
                
                # 根据帮助程度调整权重
                if task_idx in helpfulness:
                    helpfulness_score = helpfulness[task_idx]
                    # 帮助程度越高，权重越大
                    adjusted_weight = weight * (1.0 + helpfulness_score * 0.5)
                else:
                    adjusted_weight = weight
                
                # 应用权重
                weighted_grads = []
                for grad in all_gradients[task_idx]:
                    weighted_grads.append(grad * adjusted_weight)
                
                adjusted_gradients[task_idx] = weighted_grads
                
                # 记录帮助程度和权重
                if task_idx in helpfulness:
                    self.info['helpfulness_scores'].append(helpfulness[task_idx])
            else:
                # 主任务梯度保持不变
                adjusted_gradients[task_idx] = all_gradients[task_idx]
        
        # 合并梯度
        optimizer.zero_grad()
        for task_idx in range(len(task_losses)):
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    if param.grad is None:
                        param.grad = adjusted_gradients[task_idx][param_idx]
                    else:
                        param.grad += adjusted_gradients[task_idx][param_idx]
                    param_idx += 1
        
        # 更新辅助任务权重
        self._update_weights(helpfulness)
        
        # 记录权重
        aux_weights = []
        for task_idx in self.auxiliary_tasks:
            aux_weights.append(torch.sigmoid(self.auxiliary_weights[f'w_{task_idx}']).item())
        
        self.info['auxiliary_weights'].append(aux_weights)
        
        total_loss = torch.stack(task_losses).sum()
        return total_loss
    
    def _compute_helpfulness(self) -> Dict[int, float]:
        """计算辅助任务对主任务的帮助程度"""
        if len(self.history_losses[0]) < 2:
            return {}
        
        helpfulness = {}
        
        # 计算主任务损失变化
        primary_losses = []
        for task_idx in self.primary_tasks:
            if len(self.history_losses[task_idx]) >= 2:
                recent_loss = np.mean(self.history_losses[task_idx][-5:])  # 最近5步平均
                prev_loss = np.mean(self.history_losses[task_idx][-10:-5]) if len(self.history_losses[task_idx]) >= 10 else recent_loss
                loss_change = prev_loss - recent_loss  # 正数表示损失下降
                primary_losses.append(loss_change)
        
        if not primary_losses:
            return {}
        
        avg_primary_change = np.mean(primary_losses)
        
        # 计算每个辅助任务的帮助程度
        for task_idx in self.auxiliary_tasks:
            if len(self.history_losses[task_idx]) >= 2:
                recent_loss = np.mean(self.history_losses[task_idx][-5:])
                prev_loss = np.mean(self.history_losses[task_idx][-10:-5]) if len(self.history_losses[task_idx]) >= 10 else recent_loss
                aux_loss_change = prev_loss - recent_loss
                
                # 帮助程度定义：辅助任务损失下降且主任务损失也下降
                if avg_primary_change > 0 and aux_loss_change > 0:
                    helpfulness[task_idx] = min(aux_loss_change, avg_primary_change)
                elif avg_primary_change > 0:
                    helpfulness[task_idx] = avg_primary_change * 0.1  # 小幅正影响
                else:
                    helpfulness[task_idx] = -0.1  # 负影响
        
        return helpfulness
    
    def _update_weights(self, helpfulness: Dict[int, float]):
        """更新辅助任务权重"""
        with torch.no_grad():
            for task_idx in self.auxiliary_tasks:
                weight_param = self.auxiliary_weights[f'w_{task_idx}']
                
                if task_idx in helpfulness:
                    helpful_score = helpfulness[task_idx]
                    # 根据帮助程度调整权重
                    if helpful_score > 0:
                        # 正向帮助，增加权重
                        adjustment = helpful_score * 0.01
                    else:
                        # 负向影响，减少权重
                        adjustment = helpful_score * 0.02
                else:
                    # 默认小幅随机调整
                    adjustment = torch.randn(1, device=self.device).item() * 0.005
                
                weight_param.data += adjustment
                # 限制在合理范围
                weight_param.data = torch.clamp(weight_param.data, -3, 3)
    
    def get_info(self):
        return self.info
    
class PrimaryTaskGradientManager:
    """
    主任务优先的梯度管理器
    专门处理主任务与辅助任务的梯度分配
    
    Args:
        num_tasks (int): 总任务数
        primary_task_idx (int or list): 主任务的索引（可以多个）
        grad_method (str): 梯度分配方法
        device (torch.device): 设备
    """
    def __init__(self,
                 num_tasks: int,
                 primary_task_idx: Union[int, List[int]] = 0,
                 grad_method: str = 'primary_first',
                 device: torch.device = None):
        
        self.num_tasks = num_tasks
        self.grad_method = grad_method
        
        if isinstance(primary_task_idx, int):
            self.primary_tasks = [primary_task_idx]
        else:
            self.primary_tasks = primary_task_idx
        
        self.auxiliary_tasks = [i for i in range(num_tasks) if i not in self.primary_tasks]
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # 初始化
        self._init_method()
        
    def _init_method(self):
        """初始化特定方法"""
        if self.grad_method == 'primary_first':
            self.allocator = PrimaryFirstAllocator(
                primary_tasks=self.primary_tasks,
                auxiliary_tasks=self.auxiliary_tasks,
                device=self.device
            )
        elif self.grad_method == 'auxiliary_project':
            self.allocator = AuxiliaryProjectAllocator(
                primary_tasks=self.primary_tasks,
                auxiliary_tasks=self.auxiliary_tasks,
                device=self.device
            )
        elif self.grad_method == 'gradient_masking':
            self.allocator = GradientMaskingAllocator(
                primary_tasks=self.primary_tasks,
                auxiliary_tasks=self.auxiliary_tasks,
                device=self.device
            )
        elif self.grad_method == 'adaptive_auxiliary':
            self.allocator = AdaptiveAuxiliaryAllocator(
                primary_tasks=self.primary_tasks,
                auxiliary_tasks=self.auxiliary_tasks,
                device=self.device
            )
        elif self.grad_method == 'hierarchical_grad':
            self.allocator = HierarchicalGradientAllocator(
                primary_tasks=self.primary_tasks,
                auxiliary_tasks=self.auxiliary_tasks,
                device=self.device
            )
        else:
            raise ValueError(f"未知的梯度分配方法: {self.grad_method}")
    
    def allocate_gradients(self,
                          model: nn.Module,
                          task_losses: List[torch.Tensor],
                          optimizer: Optional[torch.optim.Optimizer] = None) -> torch.Tensor:
        """
        分配梯度并返回总损失
        
        Args:
            model: 模型
            task_losses: 各任务损失列表
            optimizer: 优化器
            
        Returns:
            total_loss: 总损失
        """
        return self.allocator.allocate(model, task_losses, optimizer)
    
    def get_allocation_info(self) -> Dict:
        """获取当前梯度分配信息"""
        return self.allocator.get_info()    
    