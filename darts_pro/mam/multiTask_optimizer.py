import numpy as np
from  torch.optim import Adam
import torch
import torch.nn as nn
from typing import List, Dict
from rqalpha import interface

def calculate_gradient_norm(gradient):
    total_norm = 0.0
    for p in list(gradient.values()):
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def pc_grad(gradient_components, main_task_seq_arr=[0]):
    """
    PCGrad: 通过投影消除梯度冲突，主辅任务模式
    """
    
    processed_grads = {}
    param_names = interact_grad_names(gradient_components,interact=False)
    asis_task_seq = [i for i in range(len(gradient_components))]
    for main_task_seq in main_task_seq_arr:
        asis_task_seq.remove(main_task_seq)
        
    for main_task_seq in main_task_seq_arr:   
        for param_name in param_names:
            if param_name not in gradient_components[main_task_seq]:
                continue        
            # 主任务投影到其他子任务
            grads_main = gradient_components[main_task_seq][param_name]
            # 对每对任务进行冲突消除
            for j in asis_task_seq:
                if param_name not in gradient_components[j]:
                    continue
                grads_j = gradient_components[j][param_name]
                # 计算两个梯度的点积
                dot_product = torch.dot(
                    grads_main.flatten(), 
                    grads_j.flatten()
                )
                
                # 如果梯度方向冲突（点积为负）
                if dot_product < 0:
                    # 将梯度i投影到梯度j的正交补空间
                    projection = (dot_product / 
                                (torch.norm(grads_main) ** 2 + 1e-8)) * grads_main
                    gradient_components[j][param_name] = grads_j - projection      
    
    return processed_grads

def grad_combine(gradient_components,task_grad_norms, task_weights,grad_limits):
    """合并梯度"""

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
            if i==len(task_losses)-1:
                weighted_loss.backward()     
            else:
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
    
    def __init__(self, params, defaults_dict,model=None,task_weights=None,grad_limits=None,use_gradient_surgery=True,use_adaptive_clip=False,use_pcgrad=False,device=None):
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
        self.device = device
        
        self.primary_tasks = [0]
        self.auxiliary_tasks = [i for i in range(len(task_weights)) if i not in self.primary_tasks]
        # 辅助任务权重（可学习）
        self.auxiliary_weights = nn.ParameterDict()
        self._init_weights()  
              
        # 记录历史损失信息
        self.history_losses = {task_idx: [] for task_idx in range(len(task_weights))}
        # 日志记录
        self.info = {
            'auxiliary_weights': [],
            'helpfulness_scores': []
        }
        
    def _init_weights(self):
        """初始化辅助任务权重"""
        
        for task_idx in self.auxiliary_tasks:
            weight = nn.Parameter(torch.ones(1, device=self.device) * 0.5)  # 初始权重0.5
            self.auxiliary_weights[f'w_{task_idx}'] = weight
            
    def step_with_auto_weights(self, task_losses):
        """辅助任务自适应权重"""
        
        # 记录历史损失数据
        for task_idx, loss in enumerate(task_losses):
            self.history_losses[task_idx].append(loss.item())
        # 计算辅助任务对主任务的帮助程度
        helpfulness = self._compute_helpfulness()
        
        # 计算所有任务梯度
        total_gradients, gradient_components,loss_total = self.gradient_calculator.compute_gradients(
            task_losses, return_components=True
        )
        # 统计梯度冲突情况
        conflict_analysis_total = analyze_gradient_conflicts(gradient_components)
        conflict_count,similarity = self.combine_conflict_analysis(conflict_analysis_total)    
            
        all_gradients = [self._compute_grad_norm(comp) for comp in gradient_components] 
        # 自适应调整辅助任务梯度
        # adjusted_gradients = self._compute_auto_weights(task_losses, helpfulness, all_gradients)
        # 应用梯度手术,合并梯度
        if self.use_pcgrad:
            pc_grad(gradient_components, main_task_seq_arr=[1])
        # 多个任务的梯度相加（带权重）
        total_gradients,gradient_components = self.grad_combine(gradient_components,)
        # 统计梯度范数
        task_grad_norms = [self._compute_grad_norm(comp) for comp in gradient_components] 
        # 更新辅助任务权重
        self._update_weights(helpfulness)
                
        # 手动设置梯度并更新参数
        self._set_gradients(total_gradients)
        super().step()
        
        show_loss = 0
        for loss in loss_total:
            show_loss = show_loss + loss
        return {
            'total_loss': show_loss,
            'total_grad_norm': self._compute_total_grad_norm(total_gradients),
            'conflict_analysis': {'conflict_count':conflict_count,'similarity':similarity},
            'task_grad_norms': task_grad_norms
                if gradient_components else None
        }

    def grad_combine(self,gradient_components):
        """合并多个任务梯度"""
    
        processed_grads = {}
        param_names = interact_grad_names(gradient_components,interact=False)
        
        gradient_components_after = gradient_components.copy()
        for param_name in param_names:   
            final_grad = None 
            for i in range(len(gradient_components)):
                # 不同子模型，梯度有可能不一致
                if param_name in gradient_components[i]:
                    item_grad = gradient_components[i][param_name]
                    # 不同任务不同梯度剪裁
                    item_grad = item_grad  * self.task_weights[i]  
                    gradient_components_after[i][param_name] = item_grad
                    # 加权合并处理后的梯度
                    if final_grad is None:
                        final_grad = item_grad                   
                    else:
                        final_grad = final_grad + item_grad
    
            processed_grads[param_name] = final_grad
            
        return processed_grads,gradient_components_after

    def _compute_auto_weights(self,task_losses,helpfulness,all_gradients):
        
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
                
                adjusted_gradients[task_idx] = all_gradients[task_idx] * adjusted_weight
                
                # 记录帮助程度和权重
                if task_idx in helpfulness:
                    self.info['helpfulness_scores'].append(helpfulness[task_idx])
            else:
                # 主任务梯度保持不变
                adjusted_gradients[task_idx] = all_gradients[task_idx]
        
        return adjusted_gradients
                
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
        total_gradients = pc_grad([gradients_cls,gradients_ce], self.task_weights)
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

    def combine_conflict_analysis(self,conflicts):
        
        conflict_count = 0
        avg_similarity_total = 0
        for name in conflicts.keys():
            count = conflicts[name]['conflict_count']
            conflict_count += count
            avg_similarity = conflicts[name]['avg_similarity']
            avg_similarity_total += avg_similarity
        return conflict_count/len(conflicts.keys()),avg_similarity_total/len(conflicts.keys())  
                    
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
                pc_grad(gradient_components, self.task_weights)
                task_grad_norms = [self._compute_grad_norm(comp) for comp in gradient_components] 
                # 多个任务的梯度相加（带权重）
                total_gradients,gradient_components = grad_combine(gradient_components,task_grad_norms, self.task_weights,self.grad_limits)
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
                  
        if not self.use_gradient_surgery:
            self._set_gradients(total_gradients)
            super().step()
            conflict_analysis = {}
            return {
                'total_grad_norm': self._compute_total_grad_norm(total_gradients),
                'total_loss': task_losses[0].item()
            }              
            
        conflict_analysis_total = analyze_gradient_conflicts(gradient_components)
        conflict_count,similarity = self.combine_conflict_analysis(conflict_analysis_total)
        
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
        
        step_range = 10
        
        # 计算每个辅助任务的帮助程度
        for task_idx in self.auxiliary_tasks:
            if len(self.history_losses[task_idx]) >= 2:
                recent_loss = np.mean(self.history_losses[task_idx][-step_range:])
                prev_loss = np.mean(self.history_losses[task_idx][-2*step_range:-step_range]) if len(self.history_losses[task_idx]) >= 2*step_range else recent_loss
                aux_loss_change = prev_loss - recent_loss
                
                # 帮助程度定义：辅助任务损失下降且主任务损失也下降
                if avg_primary_change > 0 and aux_loss_change > 0:
                    helpfulness[task_idx] = min(aux_loss_change, avg_primary_change)
                elif avg_primary_change > 0:
                    helpfulness[task_idx] = avg_primary_change * 0.1  # 小幅正影响
                else:
                    helpfulness[task_idx] = -0.02  # 负影响
        
        return helpfulness
    
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
        rtn = torch.sqrt(total_norm).item()
        return rtn
    
    def _compute_grad_norm(self, gradient_dict):
        """计算梯度字典的范数"""
        total_norm = 0
        for grad in gradient_dict.values():
            total_norm += torch.norm(grad) ** 2
        return torch.sqrt(total_norm).item()