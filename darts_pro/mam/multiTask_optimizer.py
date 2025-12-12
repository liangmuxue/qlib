import numpy as np
from  torch.optim import Adam
import torch
import torch.nn as nn
from typing import List, Tuple
from rqalpha import interface

def calculate_gradient_norm(gradient):
    total_norm = 0.0
    for p in list(gradient.values()):
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def pc_grad(gradient_components, task_weights,direction=0):
    """
    PCGrad: 通过投影消除梯度冲突
    """
    
    processed_grads = {}
    param_names = interact_grad_names(gradient_components,interact=False)
    #TODO param name conflict
    for param_name in param_names:
        # 收集所有任务在该参数上的梯度
        task_grads = [comp[param_name] for comp in gradient_components]
        # 对每对任务进行冲突消除
        for i in range(len(task_grads)):
            if param_name not in gradient_components[i]:
                continue
            for j in range(len(task_grads)):
                if param_name not in gradient_components[j]:
                    continue
                
                if i != j:
                    # 计算两个梯度的点积
                    dot_product = torch.dot(
                        task_grads[i].flatten(), 
                        task_grads[j].flatten()
                    )
                    
                    # 如果梯度方向冲突（点积为负）
                    if dot_product < 0 and direction>=0:
                        if direction==0:
                            # 将梯度j投影到梯度i的正交补空间
                            projection = (dot_product / 
                                        (torch.norm(task_grads[i]) ** 2 + 1e-8)) * task_grads[i]
                            task_grads[j] = task_grads[j] - projection
                        else:
                            # 将梯度i投影到梯度j的正交补空间
                            projection = (dot_product / 
                                        (torch.norm(task_grads[j]) ** 2 + 1e-8)) * task_grads[j]
                            task_grads[i] = task_grads[i] - projection                            
        # 只需要使用被投影的梯度  
        processed_grads[param_name] = task_grads[direction]
    
    return processed_grads

def grad_combine(gradient_components,task_grad_norms, task_weights,grad_limits):
    """不进行梯度手术，直接合并"""

    processed_grads = {}
    param_names = interact_grad_names(gradient_components,interact=False)
    
    # 通过原来的分任务梯度范数，倒推需要剪裁的比例
    grad_rate = [grad_limits[i]/task_grad_norms[i] if task_grad_norms[i]>grad_limits[i] else 1 for i in range(len(gradient_components))]
    
    gradient_components_after = gradient_components.copy()
    for param_name in param_names:   
        final_grad = None 
        for i in range(len(gradient_components)):
            # 不同子模型，梯度有可能不一致
            if param_name in gradient_components[i]:
                item_grad = gradient_components[i][param_name]
                # 不同任务不同梯度剪裁
                item_grad = item_grad * grad_rate[i] * task_weights[i]  
                gradient_components_after[i][param_name] = item_grad
                # 加权合并处理后的梯度
                if final_grad is None:
                    final_grad = item_grad                   
                else:
                    final_grad = final_grad + item_grad

        processed_grads[param_name] = final_grad
        
    return processed_grads,gradient_components_after
   
def adaptive_gradient_processing(gradients, conflict_analysis):
    """
    根据冲突分析自适应处理梯度
    """
    processed_gradients = {}
    
    for param_name, grad in gradients.items():
        conflict_info = conflict_analysis.get(param_name, {})
        conflict_count = conflict_info.get('conflict_count', 0)
        
        # 根据冲突程度采取不同策略
        if conflict_count > len(gradients) // 2:  # 高冲突
            # 梯度裁剪 + 归一化
            grad_norm = torch.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm  # 归一化
                
        elif conflict_count > 0:  # 中等冲突
            # 轻微梯度裁剪
            max_norm = 2.0
            grad_norm = torch.norm(grad)
            if grad_norm > max_norm:
                grad = grad * (max_norm / grad_norm)
        
        processed_gradients[param_name] = grad
    
    return processed_gradients

def interact_grad_names(gradient_components,interact=True):
    
    param_names = None
    for comp in gradient_components:
        param_name = np.array(list(comp.keys()))
        if param_names is None:
            param_names = param_name
        else:
            if interact:
                param_names = np.intersect1d(param_names,param_name)
            else:
                param_names = np.union1d(param_names,param_name)
    return param_names

def analyze_gradient_conflicts(gradient_components):
    """
    分析多任务间的梯度冲突
    """
    
    param_names = interact_grad_names(gradient_components,interact=True)
    conflict_analysis = {}
    
    for param_name in param_names:
        # 收集各任务在该参数上的梯度方向
        task_grads = [comp[param_name].flatten() for comp in gradient_components]
        
        # 计算梯度余弦相似度矩阵
        similarity_matrix = torch.zeros(len(task_grads), len(task_grads))
        for i in range(len(task_grads)):
            for j in range(len(task_grads)):
                if i != j:
                    cos_sim = torch.cosine_similarity(
                        task_grads[i], task_grads[j], dim=0
                    )
                    similarity_matrix[i, j] = cos_sim
        
        # 检测冲突（余弦相似度为负表示方向冲突）
        conflict_mask = similarity_matrix < -0.1
        conflict_count = conflict_mask.sum().item()
        
        conflict_analysis[param_name] = {
            'similarity_matrix': similarity_matrix,
            'conflict_count': conflict_count,
            'avg_similarity': similarity_matrix.mean().item()
        }
    
    return conflict_analysis

class MultiTaskGradientCalculator():
    def __init__(self, model, task_weights=None,grad_limits=None):
        self.model = model
        self.task_weights = task_weights
        self.grad_limits = grad_limits
        
    def compute_gradients(self, task_losses: List[torch.Tensor],return_components: bool = True):
        """
        计算多任务学习的梯度分解
        
        Args:
            task_losses: 各个任务的损失值列表
            return_components: 是否返回梯度分量
            
        Returns:
            total_gradients: 总梯度
            gradient_components: 各任务的梯度分量（可选）
        """

               
        if return_components:
            # 单独计算损失和梯度
            gradient_components,loss_total = self._compute_gradient_components(task_losses)
            main_grads = self._get_parameter_gradients()
            # 执行此步骤是为了清空计算图，由此计算出的参数梯度没有实际意义，后续需要用实际梯度替换--cancel
            # task_losses[-1].backward()            
            return main_grads, gradient_components,loss_total
        else:
            # 直接计算总损失
            self.model.zero_grad()           
            loss_total = 0
            for i, loss in enumerate(task_losses):
                
                loss = self.task_weights[i] * loss   
                loss_total = loss_total + loss        
            loss_total.backward()          
            return self._get_parameter_gradients()
    
    def _compute_gradient_components(self, task_losses):
        """计算每个任务对共享参数的梯度贡献"""
        gradient_components = []
        loss_total = []
        for i, task_loss in enumerate(task_losses):
            # 首先清空梯度才能真正体现单个损失的回传梯度
            self.model.zero_grad()              
            # 计算单个任务的梯度，保留计算图，后续统一处理
            weighted_loss = task_loss
            weighted_loss.backward(retain_graph=True)     
            loss_total.append(weighted_loss)
            # 获取该任务的梯度贡献
            task_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    task_grads[name] = param.grad.clone()
            
            gradient_components.append(task_grads)
        
        return gradient_components,loss_total
    
    def _get_parameter_gradients(self):
        """获取模型参数的当前梯度"""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

class MultiTaskOptimizer(Adam):
    def __init__(self, params, defaults_dict,model=None,task_weights=None,grad_limits=None,use_gradient_surgery=True,use_adaptive_clip=False,use_pcgrad=False):
        super().__init__(params, **defaults_dict)
        self.model = model
        self.task_weights = task_weights
        self.grad_limits = grad_limits
        self.use_gradient_surgery = use_gradient_surgery
        self.use_adaptive_clip = use_adaptive_clip
        self.use_pcgrad = use_pcgrad
        self.gradient_calculator = MultiTaskGradientCalculator(model, task_weights,grad_limits)
        self.accumulation_steps = 4
        self.gradients_recorder = []
        self.loss_recorder = []

    def step_with_batch(self, task_losses,batch_idx=0,total_batch_number=0):
        """
        执行多任务学习的一步参数更新,Batch Keep Mode
        """
        
        update_info = None
        # accumulate gradients first
        self.acc_grad(task_losses)
        # apply gradients interval
        if ((batch_idx+1) % self.accumulation_steps==0) or ((total_batch_number-batch_idx)==1):
            update_info = self.apply_grad()
            self.clear_record()
        return update_info
    
    def clear_record(self):
        self.gradients_recorder = []
        self.loss_recorder = []
         
    def acc_grad(self,task_losses):
        
        show_loss = 0
        self.model.zero_grad() 
        # compute grad for two losses
        loss = task_losses[0] / self.accumulation_steps
        loss.backward(retain_graph=True)
        show_loss = show_loss + loss.item()
        gradients_cls = self.gradient_calculator._get_parameter_gradients()
        self.model.zero_grad() 
        loss = task_losses[1] / self.accumulation_steps
        loss.backward(retain_graph=True)        
        show_loss = show_loss + loss.item()
        gradients_ce = self.gradient_calculator._get_parameter_gradients()
        # pc grad
        total_gradients = pc_grad([gradients_cls,gradients_ce], self.task_weights,direction=0)
        # Store Grad
        self.gradients_recorder.append(total_gradients)
        self.loss_recorder.append(show_loss)
 
    def apply_grad(self):
        """apply grad separately"""

        total_gradients = {}
        for gradients in self.gradients_recorder:
            for name in gradients.keys():
                grad = gradients[name]
                if name not in total_gradients:
                    total_gradients[name] = grad
                else:
                    total_gradients[name] = total_gradients[name] + grad   
        # 手动设置梯度并更新参数
        self._set_gradients(total_gradients)
        # Grad Cut
        cut_norm_grad = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)                           
        # Step Action    
        super().step()     
        
        return {
            'total_loss': sum(self.loss_recorder)/len(self.loss_recorder),
            'total_grad_norm': self._compute_total_grad_norm(cut_norm_grad),
        }        
           
    def compute_single_grad(self,loss_arr,loss_seq=0):
        
        for loss in loss_arr:
            loss = loss / self.accumulation_steps
            loss = self.task_weights[loss_seq] * loss
            # 计算单个任务的梯度，保留计算图，后续统一处理
            loss.backward(retain_graph=True)
        # 获取该任务的梯度贡献
        task_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                task_grads[name] = param.grad.clone()
        current_loss = loss.item() * self.accumulation_steps 
        return task_grads , current_loss   
                    
    def step(self, task_losses):
        """
        执行多任务学习的一步参数更新
        """
        # 1. 计算梯度
        if self.use_gradient_surgery:
            # 需要梯度分量进行梯度手术
            total_gradients, gradient_components,loss_total = self.gradient_calculator.compute_gradients(
                task_losses, return_components=True
            )
            
            # 分析梯度冲突
            conflict_analysis = analyze_gradient_conflicts(gradient_components)
            # 2. 冲突处理
            if self.use_adaptive_clip:
                # 自适应梯度处理
                total_gradients = adaptive_gradient_processing(total_gradients, conflict_analysis)     
            elif self.use_pcgrad:
                # 应用梯度手术
                total_gradients = pc_grad(gradient_components, self.task_weights,direction=0)
            else:
                # 标准多任务梯度计算
                # grad_norm_arr = [self._compute_total_grad_norm(grad) for grad in gradient_components]
                # 取得每个分任务的梯度范数，用于后续梯度剪裁
                task_grad_norms = [self._compute_grad_norm(comp) for comp in gradient_components] 
                total_gradients,gradient_components = grad_combine(gradient_components,task_grad_norms, self.task_weights,self.grad_limits)
        else:
            # 标准梯度计算
            total_gradients = self.gradient_calculator.compute_gradients(task_losses,return_components=False)
            gradient_components = None
            conflict_analysis = {}
            return {
                'total_grad_norm': self._compute_total_grad_norm(total_gradients),
                'total_loss': task_losses[0].item()
            }    
                  
        if not self.use_gradient_surgery:
            self._set_gradients(total_gradients)
            super().step()
            return
            
        def combine_conflict_analysis(conflicts):
            
            conflict_count = 0
            avg_similarity_total = 0
            for name in conflicts.keys():
                count = conflicts[name]['conflict_count']
                conflict_count += count
                avg_similarity = conflicts[name]['avg_similarity']
                avg_similarity_total += avg_similarity
            return conflict_count/len(conflicts.keys()),avg_similarity_total/len(conflicts.keys())
            
        conflict_analysis_total = analyze_gradient_conflicts(gradient_components)
        conflict_count,similarity = combine_conflict_analysis(conflict_analysis_total)
        conflict_analysis_total_cls = analyze_gradient_conflicts([gradient_components[0],total_gradients])
        conflict_analysis_total_ce = analyze_gradient_conflicts([gradient_components[1],total_gradients])
        cls_conflict_count,cls_similarity= combine_conflict_analysis(conflict_analysis_total_cls)
        ce_conflict_count,ce_similarity= combine_conflict_analysis(conflict_analysis_total_ce)
        
        # 3. 手动设置梯度并更新参数
        self._set_gradients(total_gradients)
        super().step()
        
        show_loss = 0
        for loss in loss_total:
            show_loss = show_loss + loss
        return {
            'total_loss': show_loss,
            'total_grad_norm': self._compute_total_grad_norm(total_gradients),
            'conflict_analysis': {'conflict_count':conflict_count,'similarity':similarity},
            'task_grad_norms': [self._compute_grad_norm(comp) for comp in gradient_components] 
                if gradient_components else None
        }
    
    def _set_gradients(self, gradients_dict):
        
        """手动设置模型参数的梯度"""
        for name, param in self.model.named_parameters():
            if name in gradients_dict:
                param.grad = gradients_dict[name]
            else:
                param.grad = None
                
    def _compute_total_grad_norm(self, gradients):
        """计算总梯度范数"""
        total_norm = 0
        for grad in gradients.values():
            total_norm += torch.norm(grad) ** 2
        return torch.sqrt(total_norm).item()
    
    def _compute_grad_norm(self, gradient_dict):
        """计算梯度字典的范数"""
        total_norm = 0
        for grad in gradient_dict.values():
            total_norm += torch.norm(grad) ** 2
        return torch.sqrt(total_norm).item()