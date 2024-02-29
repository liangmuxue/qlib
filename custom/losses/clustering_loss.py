import numpy as np
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import pairwise_distances,pairwise_compare

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class ClusteringLoss(UncertaintyLoss):
    """基于聚类的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(ClusteringLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0):
        """套用dinknet中的聚类损失计算，使用具有梯度的聚类中心参数进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,target_info,y_transform,past_target) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        kl_loss = torch.Tensor(np.array([1 for i in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for i in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        fake_loss = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                target_item = target[:,:,i]
                # 全模式下，会有2次模型处理，得到2组数据
                output_item,out_again = output[i] 
                # 如果属于特征值阶段，则只比较特征距离
                if out_again is None:
                    x_bar, _, _, _ = output_item
                    corr_loss_combine[i] += self.ccc_loss_comp(x_bar,past_target[:,:,i])
                else:  
                    _, tmp_q, _, _ = output_item
                    tmp_q = tmp_q.data
                    p = target_distribution(tmp_q)          
                    x_bar, q, pred, _ =  out_again         
                    # 实现损失计算
                    corr_loss_combine[i] += self.ccc_loss_comp(x_bar,past_target[:,:,i])
                    # DNN结果与聚类簇心的KL散度计算
                    kl_loss[i] = 5 * F.kl_div(q.log(), p, reduction='batchmean')
                    # GCN结果与聚类簇心的KL散度计算
                    ce_loss[i] = F.kl_div(pred.log(), p, reduction='batchmean')
                loss_sum = loss_sum + corr_loss_combine[i] + kl_loss[i] + ce_loss[i]
        return loss_sum,[corr_loss_combine,kl_loss,ce_loss]
            
    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    
    
class DistanceMetricLoss(UncertaintyLoss):
    """基于距离矩阵的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(DistanceMetricLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,cluster_centers=None):

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,target_info,y_transform) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        triplet_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i][:,:,0]
                target_item = target[:,:,i]    
                # 分别计算输出值与目标值的距离矩阵
                mat_output = pairwise_distances(output_item,distance_func=self.ccc_distance_torch)
                mat_target = pairwise_distances(target_item,distance_func=self.ccc_distance_torch)
                # 计算距离矩阵的MSE损失作为实际损失
                corr_loss_combine[i] = self.mse_loss(mat_output, mat_target)                
                loss_sum += corr_loss_combine[i] 
        return loss_sum,[corr_loss_combine,triplet_loss_combine,similarity_value[i]]
            
